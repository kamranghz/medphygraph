#!/usr/bin/env python3
"""Evaluate MedPhyGraph across seeds 0-4 on frozen verification targets."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_setup_spec = importlib.util.spec_from_file_location(
  "eval._path_setup", Path(__file__).with_name("_path_setup.py")
)
assert _setup_spec and _setup_spec.loader
_setup_mod = importlib.util.module_from_spec(_setup_spec)
_setup_spec.loader.exec_module(_setup_mod)

import copy
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation._benchmark_utils import gt_from, load_scene, load_transitions, present, prf
from evaluation._candidate_dropout_common import load_stage6_targets, load_state_context
from evaluation._config import DYNAMIC, HARD_DS, RATIO, THR, TRANSFER, require_dynamic_corpora
from evaluation._core_scoring import (
  delta_counts,
  pack_dynamic,
  recompute_feats,
  score_raw,
  static_macro_pooled,
)
from evaluation._multiseed_common import (
  OUT_DIR,
  ROOT,
  SEEDS,
  STAGE6_DIR,
  VERIFIED_TARGETS,
  assert_no_eval_leakage_in_training_dataset,
  checkpoint_path_for_seed,
  sha256_file,
  verify_frozen_policy,
)
from evaluation.candidate_dropout import evaluate_case
from medphygraph.consistency import (
  DeltaUnionV2Config,
  apply_consistency,
  apply_transition_aware_consistency_v2,
  count_violations,
)
from medphygraph.data import load_dataset, split_samples
from medphygraph.metrics import scene_graph_edit_f1
from medphygraph.model import HealthDyPhyGraph, ModelConfig
from medphygraph.scene_graph import PhysicalSceneGraph


def fit_health_scorer(ckpt_path: Path, device: torch.device) -> dict[str, Any]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HealthDyPhyGraph(ModelConfig(hidden=int(blob.get("config", {}).get("hidden", 64)))).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()

    @torch.no_grad()
    def _score_feat(feats: np.ndarray) -> float:
        x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(device)
        _, prob = model(x)
        return float(prob.item())

    def probs_for(_name: str, samples: list[dict]) -> dict[tuple[str, str], float]:
        out = {}
        for s in samples:
            feats = np.asarray(s["features_partial"][str(RATIO)], dtype=np.float32)
            out[(s["subject_id"], s["host_id"])] = _score_feat(feats)
        return out

    def score_feat_matrix(_name: str, feats: np.ndarray) -> float:
        return _score_feat(np.asarray(feats, dtype=np.float32))

    return {
        "probs_for": probs_for,
        "score_feat_matrix": score_feat_matrix,
        "model": model,
        "n_params": sum(p.numel() for p in model.parameters()),
        "ckpt_meta": {k: blob[k] for k in blob if k != "model_state"},
    }


def full_population_static_ge_f1(scene_level: list[dict], corpus: str) -> dict[str, Any]:
    """A1 rule: TA rows for transition-current + SC rows for origin states."""
    ta = [r for r in scene_level if r["corpus"] == corpus and r["mode"] == "transition_aware"]
    sc = [r for r in scene_level if r["corpus"] == corpus and r["mode"] == "state_consistency"]
    ta_ids = {r["scene_id"] for r in ta}
    origin_sc = [r for r in sc if r["scene_id"] not in ta_ids]
    combined = ta + origin_sc
    return {
        "n_states": len(combined),
        "n_transition_aware": len(ta),
        "n_origin_state_consistency": len(origin_sc),
        **static_macro_pooled(combined),
    }


def evaluate_suite_a(scorers: dict[str, Any], targets_doc: dict[str, Any]) -> dict[str, Any]:
    """Canonical Suite A for health_dyphygraph only (independent / SC / Full)."""
    cfg_v2 = DeltaUnionV2Config()
    targets_by_key = {
        (t["corpus"], t["previous_state_id"], t["current_state_id"]): t for t in targets_doc["targets"]
    }
    scorer = "health_dyphygraph"
    scene_level: list[dict] = []
    transition_level: list[dict] = []
    out_corpora: dict[str, Any] = {}

    for corpus, cfg in DYNAMIC.items():
        scenes = cfg["scenes"]
        raw = load_dataset(cfg["dataset"])
        by_scene: dict[str, list[dict]] = defaultdict(list)
        for s in raw["samples"]:
            by_scene[s["scene_id"]].append(s)
        transitions = load_transitions(scenes)
        state_ids = sorted({t["previous_state_id"] for t in transitions} | {t["current_state_id"] for t in transitions})

        g_init_c: dict[str, PhysicalSceneGraph] = {}
        scene_c = {}
        gt_c: dict[str, set[tuple[str, str]]] = {}
        ent_c: dict[str, set[str]] = {}
        cand_c: dict[str, set[tuple[str, str]]] = {}
        feat_cache: dict[tuple[str, str, str], np.ndarray] = {}

        for sid in state_ids:
            g_init_c[sid] = PhysicalSceneGraph.load(scenes / sid / "graph_initial.json")
            scene_c[sid] = load_scene(scenes, sid)
            gt_c[sid] = gt_from(by_scene.get(sid, []), PhysicalSceneGraph.load(scenes / sid / "graph_gt.json"))
            ent_c[sid] = {e.entity_id for e in scene_c[sid].entities if e.entity_type != "zone"}
            cand_c[sid] = set(g_init_c[sid].edges.keys())

        ds_probs: dict[str, dict[tuple[str, str], float]] = {}
        raw_g: dict[str, PhysicalSceneGraph] = {}
        state_g: dict[str, PhysicalSceneGraph] = {}
        for sid in state_ids:
            samples = by_scene.get(sid, [])
            probs = scorers["probs_for"](scorer, samples)
            ds_probs[sid] = probs
            g_raw = score_raw(g_init_c[sid], samples, probs)
            raw_g[sid] = g_raw
            g_st = PhysicalSceneGraph.from_dict(copy.deepcopy(g_raw.to_dict()))
            apply_consistency(g_st)
            state_g[sid] = g_st
            for mode, g in (("independent", g_raw), ("state_consistency", g_st)):
                pred = present(g)
                gt = gt_c[sid]
                tp, fp, fn = len(pred & gt), len(pred - gt), len(gt - pred)
                edit = scene_graph_edit_f1(pred, gt)
                viol = count_violations(g)
                scene_level.append(
                    {
                        "corpus": corpus,
                        "scorer": scorer,
                        "mode": mode,
                        "scene_id": sid,
                        "f1": edit["graph_edit_f1"],
                        "precision": edit["precision"],
                        "recall": edit["recall"],
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "n_gt": len(gt),
                        "n_pred": len(pred),
                        "false_support_rate": fp / max(len(pred), 1),
                        "violations": viol["total"],
                    }
                )

        mode_dyn_rows: dict[str, list] = defaultdict(list)
        false_nt = 0
        graph_viol_total = 0

        for t in transitions:
            a, b = t["previous_state_id"], t["current_state_id"]
            op = t.get("operation_type") or t.get("operation") or "unknown"
            target = targets_by_key[(corpus, a, b)]
            eligible = bool(target.get("primary_metric_eligible", True))
            include_primary = not (op == TRANSFER and not eligible)
            gt_add = {(x[0], x[1]) for x in target["gt_add"]}
            gt_rem = {(x[0], x[1]) for x in target["gt_remove"]}
            sample_map_a = {(s["subject_id"], s["host_id"]): s for s in by_scene.get(a, [])}
            sample_map_b = {(s["subject_id"], s["host_id"]): s for s in by_scene.get(b, [])}

            c_union = set(cand_c[a]) | set(cand_c[b]) | present(raw_g[b]) | present(state_g[a])
            prev_real: dict[tuple[str, str], float] = {}
            curr_real: dict[tuple[str, str], float] = {}

            def fill(sid, out, sample_map, ents):
                nodes = scene_c[sid].entity_map()
                for s, h in c_union:
                    if s not in ents or h not in ents or s not in nodes or h not in nodes:
                        continue
                    if (s, h) in ds_probs[sid]:
                        out[(s, h)] = ds_probs[sid][(s, h)]
                        continue
                    if (s, h) in sample_map:
                        feats = np.asarray(sample_map[(s, h)]["features_partial"][str(RATIO)], dtype=np.float32)
                        out[(s, h)] = scorers["score_feat_matrix"](scorer, feats)
                        continue
                    fkey = (sid, s, h)
                    if fkey not in feat_cache:
                        feats = recompute_feats(scene_c[sid], s, h)
                        if feats is None:
                            continue
                        feat_cache[fkey] = feats
                    out[(s, h)] = scorers["score_feat_matrix"](scorer, feat_cache[fkey])

            fill(a, prev_real, sample_map_a, ent_c[a])
            fill(b, curr_real, sample_map_b, ent_c[b])

            g_ta = PhysicalSceneGraph.from_dict(copy.deepcopy(raw_g[b].to_dict()))
            for (s, h), pr in curr_real.items():
                if (s, h) not in g_ta.edges:
                    g_ta.add_candidate(s, h)
                g_ta.edges[(s, h)].score = pr
                g_ta.edges[(s, h)].confidence = pr
                g_ta.edges[(s, h)].present = pr >= THR
            apply_transition_aware_consistency_v2(
                g_ta,
                prev_confidence=prev_real,
                curr_confidence=curr_real,
                prev_refined=state_g[a],
                prev_entity_ids=ent_c[a],
                curr_entity_ids=ent_c[b],
                curr_scene_entities=list(scene_c[b].entities),
                config=cfg_v2,
            )
            pred_ta = present(g_ta)
            gt_b = gt_c[b]
            edit_ta = scene_graph_edit_f1(pred_ta, gt_b)
            viol_ta = count_violations(g_ta)
            graph_viol_total += viol_ta["total"]
            tp, fp, fn = len(pred_ta & gt_b), len(pred_ta - gt_b), len(gt_b - pred_ta)
            scene_level.append(
                {
                    "corpus": corpus,
                    "scorer": scorer,
                    "mode": "transition_aware",
                    "scene_id": b,
                    "f1": edit_ta["graph_edit_f1"],
                    "precision": edit_ta["precision"],
                    "recall": edit_ta["recall"],
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "n_gt": len(gt_b),
                    "n_pred": len(pred_ta),
                    "false_support_rate": fp / max(len(pred_ta), 1),
                    "violations": viol_ta["total"],
                }
            )

            mode_preds = {
                "independent": present(raw_g[b]),
                "state_consistency": present(state_g[b]),
                "transition_aware": pred_ta,
            }
            pred_prev_map = {
                "independent": present(raw_g[a]),
                "state_consistency": present(state_g[a]),
                "transition_aware": present(state_g[a]),
            }
            for mode, pred_curr in mode_preds.items():
                pred_prev = pred_prev_map[mode]
                pred_add = pred_curr - pred_prev
                pred_rem = pred_prev - pred_curr
                counts = delta_counts(gt_add, gt_rem, pred_add, pred_rem)
                # false non-transfer switches: for non-TRANSFER ops, any add/rem on transfer subjects is N/A;
                # for TRANSFER, switches on subjects other than intervention subject.
                if mode == "transition_aware" and include_primary:
                    if op == TRANSFER and eligible:
                        subj = target.get("intervention_subject")
                        false_nt += sum(1 for (s, _h) in (pred_add | pred_rem) if s != subj)
                    elif op != TRANSFER:
                        # any predicted supporter switch outside no-op expectations counted via delta FPs
                        pass

                row = {
                    "corpus": corpus,
                    "scorer": scorer,
                    "mode": mode,
                    "previous_state_id": a,
                    "current_state_id": b,
                    "operation": op,
                    "primary_eligible": include_primary,
                    "combined": counts["combined"],
                    "added": counts["added"],
                    "removed": counts["removed"],
                    "unchanged_preservation": 1.0,
                    "unchanged_retention_cond": 1.0,
                    "n_gt_add": len(gt_add),
                    "n_gt_rem": len(gt_rem),
                    "n_pred_add": len(pred_add),
                    "n_pred_rem": len(pred_rem),
                }
                transition_level.append(row)
                if include_primary:
                    mode_dyn_rows[mode].append(row)

        static_indep = static_macro_pooled(
            [r for r in scene_level if r["corpus"] == corpus and r["mode"] == "independent"]
        )
        static_sc = static_macro_pooled(
            [r for r in scene_level if r["corpus"] == corpus and r["mode"] == "state_consistency"]
        )
        static_ta = static_macro_pooled(
            [r for r in scene_level if r["corpus"] == corpus and r["mode"] == "transition_aware"]
        )
        full_pop = full_population_static_ge_f1(scene_level, corpus)
        dyn_pack = {m: pack_dynamic(mode_dyn_rows[m]) for m in ("independent", "state_consistency", "transition_aware")}

        # Transfer-specific exact success on eligible transfers (Full only)
        transfer_rows = [
            r
            for r in mode_dyn_rows["transition_aware"]
            if r["operation"] == TRANSFER
        ]
        # Recompute exact transfer success from targets
        exact_ok = 0
        n_transfer = 0
        for t in transitions:
            op = t.get("operation_type") or t.get("operation") or "unknown"
            if op != TRANSFER:
                continue
            a, b = t["previous_state_id"], t["current_state_id"]
            target = targets_by_key[(corpus, a, b)]
            if not target.get("primary_metric_eligible", True):
                continue
            n_transfer += 1
            # find matching transition_level Full row
            row = next(
                r
                for r in transition_level
                if r["corpus"] == corpus
                and r["mode"] == "transition_aware"
                and r["previous_state_id"] == a
                and r["current_state_id"] == b
            )
            subj = target["intervention_subject"]
            dest = target["new_direct_host"]
            prev_h = target["previous_direct_host"]
            # reconstruct from counts: exact success if add TP includes dest and rem TP includes prev
            # Use stored pred lists if available — we didn't store them; recompute from dyn pack isn't enough.
            # Re-score success from gt_add/gt_rem perfect F1 as proxy is wrong.
            # Instead: check added/removed TP counts == 1 and FP/FN == 0 for both sides for single-edge transfers.
            if (
                row["added"]["tp"] >= 1
                and row["removed"]["tp"] >= 1
                and row["added"]["fn"] == 0
                and row["removed"]["fn"] == 0
            ):
                # still may have extras; for canonical Transfer F1 the pooled metric is primary.
                # Exact required-transfer success: both required edges correct (no FN on those two).
                exact_ok += 1 if (row["added"]["fn"] == 0 and row["removed"]["fn"] == 0 and row["added"]["tp"] >= 1 and row["removed"]["tp"] >= 1) else 0

        out_corpora[corpus] = {
            "n_states_full_pop": full_pop["n_states"],
            "n_transition_current": static_ta.get("n_scenes"),
            "static_independent_pooled_ge_f1": static_indep.get("pooled_ge_f1"),
            "static_state_consistency_full_pop_pooled_ge_f1": static_sc.get("pooled_ge_f1"),
            "static_full_medphygraph_full_pop_pooled_ge_f1": full_pop.get("pooled_ge_f1"),
            "static_full_medphygraph_transition_current_pooled_ge_f1": static_ta.get("pooled_ge_f1"),
            "dynamic": dyn_pack,
            "false_non_transfer_switches_full": false_nt,
            "graph_constraint_violations_total_ta_current": graph_viol_total,
            "n_eligible_transfers": n_transfer,
            "exact_transfer_success_proxy": exact_ok,
            "transfer_f1_full": dyn_pack["transition_aware"].get("transfer_pooled_delta_micro_f1"),
            "add_f1_full": dyn_pack["transition_aware"].get("add_pooled_delta_micro_f1"),
            "remove_f1_full": dyn_pack["transition_aware"].get("remove_pooled_delta_micro_f1"),
            "transition_macro_dyn_f1_full": dyn_pack["transition_aware"].get("transition_macro_dyn_f1"),
            "pooled_delta_micro_f1_full": dyn_pack["transition_aware"].get("pooled_delta_micro_f1"),
            # scorer vs SC vs Full transfer F1
            "transfer_f1_independent": dyn_pack["independent"].get("transfer_pooled_delta_micro_f1"),
            "transfer_f1_state_consistency": dyn_pack["state_consistency"].get("transfer_pooled_delta_micro_f1"),
            "transition_macro_dyn_f1_independent": dyn_pack["independent"].get("transition_macro_dyn_f1"),
            "transition_macro_dyn_f1_state_consistency": dyn_pack["state_consistency"].get("transition_macro_dyn_f1"),
        }

    return {"corpora": out_corpora, "scene_level": scene_level, "transition_level": transition_level}


def evaluate_suite_b(scorers: dict[str, Any]) -> dict[str, Any]:
    """Expanded-transfer suite of 217 transfers with seed-specific MedPhyGraph scorer."""
    cfg_v2 = DeltaUnionV2Config()
    targets = load_stage6_targets()
    state_ctx: dict[str, dict] = {}
    for t in targets:
        for sid in (t["previous_state_id"], t["current_state_id"]):
            if sid not in state_ctx:
                state_ctx[sid] = load_state_context(t["corpus"], t["case_id"], sid)

    feat_cache: dict[tuple[str, str, str], np.ndarray | None] = {}

    def prob_fn(sid: str, s: str, h: str) -> float | None:
        key = (sid, s, h)
        if key not in feat_cache:
            feat_cache[key] = recompute_feats(state_ctx[sid]["scene"], s, h)
        feats = feat_cache[key]
        if feats is None:
            return None
        return scorers["score_feat_matrix"]("health_dyphygraph", feats)

    rows_ta: list[dict] = []
    rows_sc: list[dict] = []
    rows_indep: list[dict] = []
    per_template: dict[str, list[dict]] = defaultdict(list)

    for target in targets:
        a, b = target["previous_state_id"], target["current_state_id"]
        ctx_a, ctx_b = state_ctx[a], state_ctx[b]
        cand_a, cand_b = ctx_a["native_cand"], ctx_b["native_cand"]
        case_rows = evaluate_case(target, cand_a, cand_b, ctx_a, ctx_b, prob_fn, cfg_v2)
        for r in case_rows:
            if r["mode"] == "transition_aware":
                rows_ta.append(r)
                per_template[r["semantic_template"]].append(r)
            elif r["mode"] == "state_consistency":
                rows_sc.append(r)

        # Independent mode: score without consistency (reuse evaluate_case's state_consistency
        # rows aren't independent). Build a lightweight independent prediction here.
        # evaluate_case already returns state_consistency; for independent we score raw
        # candidate edges at thr without apply_consistency — approximate via rebuilding.
        # For final decomposition we need independent on the expanded set too.
        # Reconstruct from evaluate_case is hard; run a thin independent pass:
        subj = target["subject_id"]
        dest = target["new_direct_host"]
        prev_host = target["previous_direct_host"]
        eligible = target["primary_metric_eligible"]
        gt_add = {tuple(x) for x in target["gt_add"]}
        gt_rem = {tuple(x) for x in target["gt_remove"]}

        def present_raw(scene, cands, sid):
            edges = set()
            for s, h in cands:
                pr = prob_fn(sid, s, h)
                if pr is not None and pr >= THR:
                    edges.add((s, h))
            return edges

        pred_a = present_raw(ctx_a["scene"], cand_a, a)
        pred_b = present_raw(ctx_b["scene"], cand_b, b)
        pred_add = pred_b - pred_a
        pred_rem = pred_a - pred_b
        counts = delta_counts(gt_add, gt_rem, pred_add, pred_rem) if eligible else None
        transfer_success = bool(eligible and (subj, dest) in pred_add and (subj, prev_host) in pred_rem)
        rows_indep.append(
            {
                "case_id": target["case_id"],
                "corpus": target["corpus"],
                "semantic_template": target["semantic_template"],
                "transfer_success": transfer_success,
                "add_tp": counts["added"]["tp"] if counts else None,
                "add_fp": counts["added"]["fp"] if counts else None,
                "add_fn": counts["added"]["fn"] if counts else None,
                "rem_tp": counts["removed"]["tp"] if counts else None,
                "rem_fp": counts["removed"]["fp"] if counts else None,
                "rem_fn": counts["removed"]["fn"] if counts else None,
                "false_switches": len({(s2, h2) for (s2, h2) in (pred_add | pred_rem) if s2 != subj}),
                "graph_constraint_violations": 0,
            }
        )

    def pack_rows(rows: list[dict]) -> dict[str, Any]:
        n = len(rows)
        exact = sum(1 for r in rows if r["transfer_success"])
        tp = sum((r["add_tp"] or 0) + (r["rem_tp"] or 0) for r in rows)
        fp = sum((r["add_fp"] or 0) + (r["rem_fp"] or 0) for r in rows)
        fn = sum((r["add_fn"] or 0) + (r["rem_fn"] or 0) for r in rows)
        pooled = prf(tp, fp, fn)
        add = prf(sum(r["add_tp"] or 0 for r in rows), sum(r["add_fp"] or 0 for r in rows), sum(r["add_fn"] or 0 for r in rows))
        rem = prf(sum(r["rem_tp"] or 0 for r in rows), sum(r["rem_fp"] or 0 for r in rows), sum(r["rem_fn"] or 0 for r in rows))
        by_corp: dict[str, list] = defaultdict(list)
        for r in rows:
            by_corp[r["corpus"]].append(r)

        def corp_f1(rs):
            return prf(
                sum((r["add_tp"] or 0) + (r["rem_tp"] or 0) for r in rs),
                sum((r["add_fp"] or 0) + (r["rem_fp"] or 0) for r in rs),
                sum((r["add_fn"] or 0) + (r["rem_fn"] or 0) for r in rs),
            )["f1"]

        return {
            "n": n,
            "exact_transfer_success": exact,
            "exact_transfer_success_rate": exact / n if n else None,
            "transfer_f1_pooled": pooled["f1"],
            "add_f1_pooled": add["f1"],
            "remove_f1_pooled": rem["f1"],
            "false_switches_total": sum(r.get("false_switches") or 0 for r in rows),
            "graph_constraint_violations_total": sum(r.get("graph_constraint_violations") or 0 for r in rows),
            "procedural_transfer_f1": corp_f1(by_corp.get("procedural", [])),
            "isaac_hc_transfer_f1": corp_f1(by_corp.get("isaac_hc", [])),
        }

    tmpl_rows = []
    for tmpl, rs in sorted(per_template.items()):
        succ = sum(1 for r in rs if r["transfer_success"])
        tmpl_rows.append(
            {
                "semantic_template": tmpl,
                "n": len(rs),
                "exact_transfer_success": succ,
                "exact_transfer_success_rate": succ / len(rs),
            }
        )

    return {
        "full_medphygraph": pack_rows(rows_ta),
        "state_consistency": pack_rows(rows_sc),
        "independent": pack_rows(rows_indep),
        "per_template": tmpl_rows,
        "per_case_full": [
            {
                "case_id": r["case_id"],
                "corpus": r["corpus"],
                "semantic_template": r["semantic_template"],
                "transfer_success": r["transfer_success"],
                "transfer_f1": r["transfer_f1"],
                "failure_class": r.get("failure_class"),
            }
            for r in rows_ta
        ],
    }


def main() -> int:
    import argparse

    argparse.ArgumentParser(
        description="Evaluate MedPhyGraph across seeds 0-4 on core and expanded procedural suites."
    ).parse_args()

    t0 = time.time()
    from medphygraph.paths import new_run_dir
    import evaluation._multiseed_common as _ms_common

    run_dir = new_run_dir("multiseed", protocol_id="final_multiseed")
    _ms_common.OUT_DIR = run_dir
    OUT_DIR = run_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Stage 9: Final MedPhyGraph Multi-Seed Evaluation ===")

    policy = verify_frozen_policy()
    print("Frozen policy check:", "OK" if policy["ok"] else policy["mismatches"])
    if not policy["ok"]:
        raise SystemExit("Frozen DeltaUnionV2Config mismatch — refusing to evaluate")

    leakage = assert_no_eval_leakage_in_training_dataset()
    print("Leakage check:", "PASS" if not leakage["leakage_detected"] else "FAIL", leakage)
    if leakage["leakage_detected"]:
        raise SystemExit("Training/eval leakage detected")

    seed0_sha = sha256_file(checkpoint_path_for_seed(0))
    print(f"Seed-0 frozen SHA256: {seed0_sha}")
    if seed0_sha != "e0b34529745399ecc5da5341ed7a162173611e12c8bd50dec121b0c575c5b789":
        raise SystemExit("Frozen seed-0 checkpoint hash mismatch")

    targets_doc = json.loads(VERIFIED_TARGETS.read_text(encoding="utf-8"))
    # Suite A rebuilds scores from TwinWorld dynamic CF datasets (not on public HF).
    require_dynamic_corpora(require_all=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_seed_results: dict[str, Any] = {}
    for seed in SEEDS:
        ckpt = checkpoint_path_for_seed(seed)
        print(f"\n===== SEED {seed}  ckpt={ckpt} =====")
        if not ckpt.exists():
            raise SystemExit(f"Missing checkpoint for seed {seed}: {ckpt}")
        scorers = fit_health_scorer(ckpt, device)
        print("  Suite A (canonical)...")
        suite_a = evaluate_suite_a(scorers, targets_doc)
        print("  Suite B (expanded 217)...")
        suite_b = evaluate_suite_b(scorers)
        all_seed_results[str(seed)] = {
            "seed": seed,
            "checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha256_file(ckpt),
            "n_params": scorers["n_params"],
            "suite_a": {
                "corpora": suite_a["corpora"],
            },
            "suite_b": {
                "full_medphygraph": suite_b["full_medphygraph"],
                "state_consistency": suite_b["state_consistency"],
                "independent": suite_b["independent"],
                "per_template": suite_b["per_template"],
            },
        }
        # Print seed-0 anchors for gate
        if seed == 0:
            isaac = suite_a["corpora"]["isaac_hc"]
            proc = suite_a["corpora"]["procedural"]
            print("  SEED-0 ANCHORS:")
            print(f"    Isaac full-pop Full static GE-F1: {isaac['static_full_medphygraph_full_pop_pooled_ge_f1']}")
            print(f"    Isaac TA transition-current:      {isaac['static_full_medphygraph_transition_current_pooled_ge_f1']}")
            print(f"    Isaac Transfer F1:                {isaac['transfer_f1_full']}")
            print(f"    Isaac Trans-Macro:                {isaac['transition_macro_dyn_f1_full']}")
            print(f"    Proc full-pop Full static GE-F1:  {proc['static_full_medphygraph_full_pop_pooled_ge_f1']}")
            print(f"    Proc TA transition-current:       {proc['static_full_medphygraph_transition_current_pooled_ge_f1']}")
            print(f"    Proc Transfer F1:                 {proc['transfer_f1_full']}")
            print(f"    Expanded exact success:           {suite_b['full_medphygraph']['exact_transfer_success']}/217")
            print(f"    Expanded Transfer F1:             {suite_b['full_medphygraph']['transfer_f1_pooled']}")

    payload = {
        "policy": policy,
        "leakage_check": leakage,
        "seeds": all_seed_results,
        "elapsed_s": time.time() - t0,
    }
    (OUT_DIR / "per_seed_raw.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'per_seed_raw.json'} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    from evaluation._runtime import configure_repo_paths

    configure_repo_paths()
    raise SystemExit(main())
