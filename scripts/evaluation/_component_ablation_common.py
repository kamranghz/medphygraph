#!/usr/bin/env python3
"""Shared helpers for multi-seed component ablation (core + expanded 217)."""

from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation._config import FROZEN_ROOT
from evaluation._shared import COMPONENT_ABLATION_OUT, load_health_dyphygraph_model

OUT_DIR = COMPONENT_ABLATION_OUT
VERIFIED_TARGETS = FROZEN_ROOT / "evaluation_targets/operation_consistent_dynamic_verified.json"
REFINEMENT_ANCHOR = FROZEN_ROOT / "dynamic/processed/refinement_ablation.json"

ABLATION_STAGES = (
    "independent",
    "state_consistency",
    "union_rescoring",
    "direct_support_gate",
    "union_and_gate",
    "medphygraph",
)

STAGE_LABELS = {
    "independent": "Independent CF-SupportNet",
    "state_consistency": "CF-SupportNet + State Consistency",
    "union_rescoring": "+ Union rescoring only",
    "direct_support_gate": "+ Direct-Support Gate only",
    "union_and_gate": "+ Union and Gate",
    "medphygraph": "MedPhyGraph",
}


def fit_health_scorer(ckpt_path: Path, device: torch.device) -> dict[str, Any]:
    from evaluation._config import RATIO
    from evaluation._core_scoring import _score_feat
    model, _blob = load_health_dyphygraph_model(ckpt_path, device)

    @torch.no_grad()
    def score_feat_matrix(_name: str, feats: np.ndarray) -> float:
        return _score_feat(model, np.asarray(feats, dtype=np.float32), device)

    def probs_for(_name: str, samples: list[dict]) -> dict[tuple[str, str], float]:
        out = {}
        for s in samples:
            feats = np.asarray(s["features_partial"][str(RATIO)], dtype=np.float32)
            out[(s["subject_id"], s["host_id"])] = score_feat_matrix("health_dyphygraph", feats)
        return out

    return {
        "probs_for": probs_for,
        "score_feat_matrix": score_feat_matrix,
        "model": model,
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def summarize_ablation_block(
    ablation_rows: list[dict],
    scene_level: list[dict],
    *,
    corpus: str | None = None,
    suite: str,
) -> list[dict]:
    """Aggregate per-transition/case rows into the 6-stage component matrix."""
    from evaluation._core_scoring import pack_dynamic, static_macro_pooled

    rows = [r for r in ablation_rows if r.get("primary_eligible", True)]
    if corpus is not None:
        rows = [r for r in rows if r["corpus"] == corpus]
        scene_level = [r for r in scene_level if r["corpus"] == corpus]

    block = []
    for st in ABLATION_STAGES:
        dyn_rows = [r for r in rows if r["mode"] == st]
        dyn = pack_dynamic(dyn_rows)
        if st == "independent":
            mode_s = "independent"
        elif st == "state_consistency":
            mode_s = "state_consistency"
        elif st == "medphygraph":
            mode_s = "transition_aware"
        else:
            mode_s = "ablation_static"

        if st in ("union_rescoring", "direct_support_gate", "union_and_gate"):
            static_rows = [r for r in scene_level if r["mode"] == mode_s and r.get("stage") == st]
            static_note = "not_independently_evaluated"
            static_pooled = None
            mean_viol = round(
                float(np.mean([r.get("violations", 0.0) for r in dyn_rows])), 6
            ) if dyn_rows else None
        elif st == "medphygraph" and suite == "core":
            static_rows = [r for r in scene_level if r["mode"] == "transition_aware"]
            static = static_macro_pooled(static_rows)
            static_note = "exact_transition_current_subset"
            static_pooled = static.get("pooled_ge_f1")
            mean_viol = static.get("mean_violations")
        elif st == "medphygraph" and suite == "expanded":
            static_rows = [r for r in scene_level if r["mode"] == st]
            static = static_macro_pooled(static_rows)
            static_note = "exact_current_state_per_case"
            static_pooled = static.get("pooled_ge_f1")
            mean_viol = static.get("mean_violations")
        else:
            static_rows = [r for r in scene_level if r["mode"] == mode_s]
            static = static_macro_pooled(static_rows)
            static_note = "exact"
            static_pooled = static.get("pooled_ge_f1")
            mean_viol = static.get("mean_violations")

        block.append(
            {
                "stage": st,
                "label": STAGE_LABELS[st],
                "static_pooled_ge_f1": static_pooled,
                "static_note": static_note,
                "mean_violations": mean_viol,
                "transition_macro_dyn_f1": dyn.get("transition_macro_dyn_f1"),
                "pooled_delta_micro_f1": dyn.get("pooled_delta_micro_f1"),
                "add": dyn.get("add_transition_macro_dyn_f1"),
                "remove": dyn.get("remove_transition_macro_dyn_f1"),
                "transfer": dyn.get("transfer_transition_macro_dyn_f1"),
                "n_transitions": dyn.get("n_transitions"),
                "transfer_pooled_micro_f1": dyn.get("transfer_pooled_delta_micro_f1"),
            }
        )
    return block


def evaluate_core_component_ablation(
    scorers: dict[str, Any],
    targets_doc: dict[str, Any],
    *,
    corpus_filter: str | None = None,
    limit_transitions: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Run 6-stage ablation on canonical core transitions (health_dyphygraph only)."""
    from medphygraph.consistency import (
        DeltaUnionV2Config,
        apply_consistency,
        apply_transition_aware_consistency_v2,
        count_violations,
    )
    from medphygraph.data import load_dataset
    from medphygraph.metrics import scene_graph_edit_f1
    from medphygraph.scene_graph import PhysicalSceneGraph

    from evaluation._benchmark_utils import gt_from, load_scene, load_transitions, present
    from evaluation._config import DYNAMIC, RATIO, THR, TRANSFER
    from evaluation._core_scoring import (
        apply_direct_gate_only,
        apply_union_and_gate,
        apply_union_rescoring,
        delta_counts,
        recompute_feats,
        score_raw,
    )

    cfg_v2 = DeltaUnionV2Config()
    targets_by_key = {
        (t["corpus"], t["previous_state_id"], t["current_state_id"]): t for t in targets_doc["targets"]
    }
    scorer = "health_dyphygraph"
    ablation_rows: list[dict] = []
    scene_level: list[dict] = []
    feat_cache: dict[tuple[str, str, str], np.ndarray] = {}

    corpora = [corpus_filter] if corpus_filter else list(DYNAMIC.keys())
    for corpus in corpora:
        cfg = DYNAMIC[corpus]
        scenes = cfg["scenes"]
        raw = load_dataset(cfg["dataset"])
        by_scene: dict[str, list[dict]] = defaultdict(list)
        for s in raw["samples"]:
            by_scene[s["scene_id"]].append(s)
        transitions = load_transitions(scenes)
        if limit_transitions:
            transitions = transitions[:limit_transitions]
        state_ids = sorted({t["previous_state_id"] for t in transitions} | {t["current_state_id"] for t in transitions})

        g_init_c: dict[str, PhysicalSceneGraph] = {}
        scene_c = {}
        gt_c: dict[str, set[tuple[str, str]]] = {}
        ent_c: dict[str, set[str]] = {}
        cand_c: dict[str, set[tuple[str, str]]] = {}

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

        ta_static_rows: list[dict] = []

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
                    if fkey in feat_cache:
                        feats = feat_cache[fkey]
                    else:
                        feats = recompute_feats(scene_c[sid], s, h)
                        if feats is None:
                            continue
                        feat_cache[fkey] = feats
                    out[(s, h)] = scorers["score_feat_matrix"](scorer, feats)

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
            tp, fp, fn = len(pred_ta & gt_b), len(pred_ta - gt_b), len(gt_b - pred_ta)
            ta_static_rows.append(
                {
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
            scene_level.append(
                {
                    "corpus": corpus,
                    "scorer": scorer,
                    "mode": "transition_aware",
                    "scene_id": b,
                    **ta_static_rows[-1],
                }
            )

            g_union = apply_union_rescoring(raw_g[b], curr_real)
            g_gate = apply_direct_gate_only(raw_g[b], scene_c[b])
            g_both = apply_union_and_gate(raw_g[b], curr_real, scene_c[b])
            graphs = {
                "independent": raw_g[b],
                "state_consistency": state_g[b],
                "union_rescoring": g_union,
                "direct_support_gate": g_gate,
                "union_and_gate": g_both,
                "medphygraph": g_ta,
            }
            mode_preds = {m: present(g) for m, g in graphs.items()}
            pred_prev_map = {
                "independent": present(raw_g[a]),
                "state_consistency": present(state_g[a]),
                "union_rescoring": present(state_g[a]),
                "direct_support_gate": present(state_g[a]),
                "union_and_gate": present(state_g[a]),
                "medphygraph": present(state_g[a]),
            }

            for mode, pred_curr in mode_preds.items():
                pred_prev = pred_prev_map[mode]
                pred_add = pred_curr - pred_prev
                pred_rem = pred_prev - pred_curr
                counts = delta_counts(gt_add, gt_rem, pred_add, pred_rem)
                if mode in ("union_rescoring", "direct_support_gate", "union_and_gate"):
                    g = graphs[mode]
                    pred_s = present(g)
                    gt = gt_b
                    edit = scene_graph_edit_f1(pred_s, gt)
                    viol = count_violations(g)
                    scene_level.append(
                        {
                            "corpus": corpus,
                            "scorer": scorer,
                            "mode": "ablation_static",
                            "stage": mode,
                            "scene_id": b,
                            "f1": edit["graph_edit_f1"],
                            "tp": len(pred_s & gt),
                            "fp": len(pred_s - gt),
                            "fn": len(gt - pred_s),
                            "violations": viol["total"],
                        }
                    )
                if include_primary:
                    ablation_rows.append(
                        {
                            "corpus": corpus,
                            "scorer": scorer,
                            "mode": mode,
                            "previous_state_id": a,
                            "current_state_id": b,
                            "base_scene_id": t.get("base_scene_id"),
                            "operation": op,
                            "primary_eligible": include_primary,
                            "combined": counts["combined"],
                            "added": counts["added"],
                            "removed": counts["removed"],
                            "unchanged_preservation": 0.0,
                            "unchanged_retention_cond": 0.0,
                            "n_gt_add": len(gt_add),
                            "n_gt_rem": len(gt_rem),
                            "n_pred_add": len(pred_add),
                            "n_pred_rem": len(pred_rem),
                        }
                    )

    return ablation_rows, scene_level


def evaluate_expanded_component_ablation(
    scorers: dict[str, Any],
    targets: list[dict],
    *,
    limit_cases: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Run 6-stage ablation on frozen expanded-transfer cases."""
    from medphygraph.consistency import (
        DeltaUnionV2Config,
        apply_consistency,
        apply_transition_aware_consistency_v2,
        count_violations,
    )
    from medphygraph.metrics import scene_graph_edit_f1
    from medphygraph.scene_graph import PhysicalSceneGraph

    from evaluation._config import RATIO, THR
    from evaluation._benchmark_utils import present
    from evaluation._core_scoring import (
        apply_direct_gate_only,
        apply_union_and_gate,
        apply_union_rescoring,
        delta_counts,
        recompute_feats,
        score_raw,
    )
    from _candidate_dropout_common import load_state_context
    from procedural_transfer import load_frozen_targets_and_states

    if limit_cases:
        targets = targets[:limit_cases]

    ent_c, scene_c, cand_c, g_init_c, by_scene = load_frozen_targets_and_states(targets)
    cfg_v2 = DeltaUnionV2Config()
    scorer = "health_dyphygraph"
    ablation_rows: list[dict] = []
    scene_level: list[dict] = []
    feat_cache: dict[tuple[str, str, str], np.ndarray] = {}
    gt_c: dict[str, set[tuple[str, str]]] = {}

    state_ids = sorted(ent_c.keys())
    ds_probs: dict[str, dict[tuple[str, str], float]] = {}
    raw_g: dict[str, PhysicalSceneGraph] = {}
    state_g: dict[str, PhysicalSceneGraph] = {}

    for target in targets:
        for sid in (target["previous_state_id"], target["current_state_id"]):
            if sid in gt_c:
                continue
            ctx = load_state_context(target["corpus"], target["case_id"], sid)
            gt_c[sid] = ctx["gt"]

    for sid in state_ids:
        samples = by_scene.get(sid, [])
        probs = scorers["probs_for"](scorer, samples)
        ds_probs[sid] = probs
        g_raw = score_raw(g_init_c[sid], samples, probs)
        raw_g[sid] = g_raw
        g_st = PhysicalSceneGraph.from_dict(copy.deepcopy(g_raw.to_dict()))
        apply_consistency(g_st)
        state_g[sid] = g_st

    for ci, target in enumerate(targets):
        a, b = target["previous_state_id"], target["current_state_id"]
        corpus = target["corpus"]
        eligible = target["primary_metric_eligible"]
        gt_add = {tuple(x) for x in target["gt_add"]}
        gt_rem = {tuple(x) for x in target["gt_remove"]}
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
                if fkey in feat_cache:
                    feats = feat_cache[fkey]
                else:
                    feats = recompute_feats(scene_c[sid], s, h)
                    if feats is None:
                        continue
                    feat_cache[fkey] = feats
                out[(s, h)] = scorers["score_feat_matrix"](scorer, feats)

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

        g_union = apply_union_rescoring(raw_g[b], curr_real)
        g_gate = apply_direct_gate_only(raw_g[b], scene_c[b])
        g_both = apply_union_and_gate(raw_g[b], curr_real, scene_c[b])
        graphs = {
            "independent": raw_g[b],
            "state_consistency": state_g[b],
            "union_rescoring": g_union,
            "direct_support_gate": g_gate,
            "union_and_gate": g_both,
            "medphygraph": g_ta,
        }
        mode_preds = {m: present(g) for m, g in graphs.items()}
        pred_prev_map = {
            "independent": present(raw_g[a]),
            "state_consistency": present(state_g[a]),
            "union_rescoring": present(state_g[a]),
            "direct_support_gate": present(state_g[a]),
            "union_and_gate": present(state_g[a]),
            "medphygraph": present(state_g[a]),
        }
        gt_b = gt_c[b]

        for mode, pred_curr in mode_preds.items():
            pred_prev = pred_prev_map[mode]
            pred_add = pred_curr - pred_prev
            pred_rem = pred_prev - pred_curr
            counts = delta_counts(gt_add, gt_rem, pred_add, pred_rem)
            edit = scene_graph_edit_f1(pred_curr, gt_b)
            viol = count_violations(graphs[mode])["total"]
            tp_s, fp_s, fn_s = len(pred_curr & gt_b), len(pred_curr - gt_b), len(gt_b - pred_curr)
            scene_level.append(
                {
                    "corpus": corpus,
                    "case_id": target["case_id"],
                    "mode": mode,
                    "scene_id": b,
                    "f1": edit["graph_edit_f1"],
                    "precision": edit["precision"],
                    "recall": edit["recall"],
                    "tp": tp_s,
                    "fp": fp_s,
                    "fn": fn_s,
                    "n_gt": len(gt_b),
                    "n_pred": len(pred_curr),
                    "false_support_rate": fp_s / max(len(pred_curr), 1),
                    "violations": viol,
                }
            )
            if eligible:
                ablation_rows.append(
                    {
                        "corpus": corpus,
                        "case_id": target["case_id"],
                        "semantic_template": target["semantic_template"],
                        "mode": mode,
                        "operation": "transfer_support",
                        "primary_eligible": True,
                        "combined": counts["combined"],
                        "added": counts["added"],
                        "removed": counts["removed"],
                        "unchanged_preservation": 0.0,
                        "unchanged_retention_cond": 0.0,
                        "n_gt_add": len(gt_add),
                        "n_gt_rem": len(gt_rem),
                        "n_pred_add": len(pred_add),
                        "n_pred_rem": len(pred_rem),
                        "transfer_success": (target["subject_id"], target["new_direct_host"]) in pred_add
                        and (target["subject_id"], target["previous_direct_host"]) in pred_rem,
                    }
                )
        if (ci + 1) % 25 == 0 or (ci + 1) == len(targets):
            print(f"    expanded cases {ci + 1}/{len(targets)}")

    return ablation_rows, scene_level
