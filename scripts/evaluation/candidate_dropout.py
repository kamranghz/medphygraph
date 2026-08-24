#!/usr/bin/env python3
"""Candidate-availability stress test on frozen expanded-transfer cases."""

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
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation._core_scoring import delta_counts, fit_scorers, recompute_feats
from evaluation._candidate_dropout_common import (
  DROPOUT_RATES,
  DROPOUT_SEEDS,
  OUT_DIR,
  STAGE6_DIR,
  THR,
  destination_targeted_survivors,
  load_state_context,
  load_stage6_targets,
  random_dropout_survivors,
  sha256_file,
)
from medphygraph.consistency import (
  DeltaUnionV2Config,
  apply_consistency,
  apply_transition_aware_consistency_v2,
  count_violations,
)
from medphygraph.scene_graph import PhysicalSceneGraph

# THR imported from evaluation._candidate_dropout_common (frozen presence threshold 0.5).


def present(g: PhysicalSceneGraph) -> set[tuple[str, str]]:
    return {(e.subject_id, e.host_id) for e in g.edges.values() if e.present}


def build_graph(scene, cand_edges: set[tuple[str, str]], sid: str, prob_fn) -> PhysicalSceneGraph:
    g = PhysicalSceneGraph(scene=scene)
    for s, h in sorted(cand_edges):
        pr = prob_fn(sid, s, h)
        if pr is None:
            continue
        g.add_candidate(s, h)
        g.edges[(s, h)].score = pr
        g.edges[(s, h)].confidence = pr
        g.edges[(s, h)].present = pr >= THR
    return g


def refined_copy(g: PhysicalSceneGraph) -> PhysicalSceneGraph:
    g2 = PhysicalSceneGraph.from_dict(copy.deepcopy(g.to_dict()))
    apply_consistency(g2)
    return g2


FAILURE_CATEGORIES = (
    "destination_candidate_dropped",
    "destination_candidate_available_below_threshold",
    "previous_probability_unavailable",
    "temporal_replacement_condition_failed",
    "direct_support_gate_failed",
    "final_refinement_changed_decision",
    "other",
)

_TEMPORAL_REASONS = {
    "no_previous_refined_host",
    "subject_not_temporally_comparable",
    "old_host_not_temporally_comparable",
    "no_comparable_alternative_host",
    "gain_below_threshold",
    "switch_score_below_threshold",
    "absolute_margin_not_met",
}


def classify_failure(*, subj: str, dest: str, c_union: set, cand_b: set, trep) -> tuple[str, str]:
    if (subj, dest) not in c_union:
        return "destination_candidate_dropped", "destination absent from candidate union (dropped by this condition)"
    decision = next((d for d in trep.subject_decisions if d.get("subject_id") == subj), None)
    rr = decision.get("rejection_reason") if decision else "no_subject_decision_recorded"
    if rr is None:
        return "final_refinement_changed_decision", "switch accepted at subject-decision level but net transfer still incorrect"
    if rr == "destination_absent_from_candidates":
        return "destination_candidate_dropped", "destination absent from scored candidate graph at decision time"
    if rr == "destination_below_presence_threshold":
        return "destination_candidate_available_below_threshold", f"rejection_reason={rr}"
    if rr == "missing_real_p_prev_old":
        return "previous_probability_unavailable", f"rejection_reason={rr}"
    if rr in _TEMPORAL_REASONS:
        return "temporal_replacement_condition_failed", f"rejection_reason={rr}"
    if isinstance(rr, str) and rr.startswith("direct_support_gate_"):
        return "direct_support_gate_failed", f"rejection_reason={rr}"
    return "other", f"rejection_reason={rr}"


def evaluate_case(
    target: dict,
    cand_a: set[tuple[str, str]],
    cand_b: set[tuple[str, str]],
    ctx_a: dict,
    ctx_b: dict,
    prob_fn,
    cfg_v2: DeltaUnionV2Config,
) -> list[dict[str, Any]]:
    a, b = target["previous_state_id"], target["current_state_id"]
    subj = target["subject_id"]
    dest = target["new_direct_host"]
    prev_host = target["previous_direct_host"]
    eligible = target["primary_metric_eligible"]
    gt_add = {tuple(x) for x in target["gt_add"]}
    gt_rem = {tuple(x) for x in target["gt_remove"]}
    ent_a, ent_b = ctx_a["ent"], ctx_b["ent"]
    scene_a, scene_b = ctx_a["scene"], ctx_b["scene"]

    raw_g_a = build_graph(scene_a, cand_a, a, prob_fn)
    raw_g_b = build_graph(scene_b, cand_b, b, prob_fn)
    state_g_a = refined_copy(raw_g_a)
    state_g_b = refined_copy(raw_g_b)

    c_union = set(cand_a) | set(cand_b)
    prev_real: dict[tuple[str, str], float] = {}
    curr_real: dict[tuple[str, str], float] = {}
    for s, h in c_union:
        if s in ent_a and h in ent_a:
            pr = prob_fn(a, s, h)
            if pr is not None:
                prev_real[(s, h)] = pr
        if s in ent_b and h in ent_b:
            pr = prob_fn(b, s, h)
            if pr is not None:
                curr_real[(s, h)] = pr

    g_ta = PhysicalSceneGraph.from_dict(copy.deepcopy(raw_g_b.to_dict()))
    for (s, h), pr in curr_real.items():
        if (s, h) not in g_ta.edges:
            g_ta.add_candidate(s, h)
        g_ta.edges[(s, h)].score = pr
        g_ta.edges[(s, h)].confidence = pr
        g_ta.edges[(s, h)].present = pr >= THR

    trep = apply_transition_aware_consistency_v2(
        g_ta,
        prev_confidence=prev_real,
        curr_confidence=curr_real,
        prev_refined=state_g_a,
        prev_entity_ids=ent_a,
        curr_entity_ids=ent_b,
        curr_scene_entities=list(scene_b.entities),
        config=cfg_v2,
    )

    mode_pred_curr = {"state_consistency": present(state_g_b), "transition_aware": present(g_ta)}
    mode_pred_prev = {"state_consistency": present(state_g_a), "transition_aware": present(state_g_a)}

    dest_in_cand_a = (subj, dest) in cand_a
    dest_in_cand_b = (subj, dest) in cand_b
    dest_in_union = (subj, dest) in c_union
    prevhost_in_cand_a = (subj, prev_host) in cand_a

    rows = []
    for mode in ("state_consistency", "transition_aware"):
        pred_curr = mode_pred_curr[mode]
        pred_prev = mode_pred_prev[mode]
        pred_add = pred_curr - pred_prev
        pred_rem = pred_prev - pred_curr
        counts = delta_counts(gt_add, gt_rem, pred_add, pred_rem) if eligible else None
        transfer_success = bool(eligible and (subj, dest) in pred_add and (subj, prev_host) in pred_rem)
        false_switch_edges = {(s2, h2) for (s2, h2) in (pred_add | pred_rem) if s2 != subj}

        failure_class = None
        failure_evidence = None
        if mode == "transition_aware" and eligible and not transfer_success:
            failure_class, failure_evidence = classify_failure(
                subj=subj, dest=dest, c_union=c_union, cand_b=cand_b, trep=trep
            )

        rows.append(
            {
                "case_id": target["case_id"],
                "corpus": target["corpus"],
                "semantic_template": target["semantic_template"],
                "subject_id": subj,
                "previous_host": prev_host,
                "destination_host": dest,
                "mode": mode,
                "eligible": eligible,
                "transfer_success": transfer_success,
                "add_f1": counts["added"]["f1"] if counts else None,
                "remove_f1": counts["removed"]["f1"] if counts else None,
                "transfer_f1": counts["combined"]["f1"] if counts else None,
                "add_tp": counts["added"]["tp"] if counts else None,
                "add_fp": counts["added"]["fp"] if counts else None,
                "add_fn": counts["added"]["fn"] if counts else None,
                "rem_tp": counts["removed"]["tp"] if counts else None,
                "rem_fp": counts["removed"]["fp"] if counts else None,
                "rem_fn": counts["removed"]["fn"] if counts else None,
                "false_switches": len(false_switch_edges),
                "graph_constraint_violations": count_violations(g_ta)["total"] if mode == "transition_aware" else None,
                "dest_in_cand_prev": dest_in_cand_a,
                "dest_in_cand_curr": dest_in_cand_b,
                "dest_in_union": dest_in_union,
                "prevhost_in_cand_prev": prevhost_in_cand_a,
                "candidate_covered": dest_in_union,
                "failure_class": failure_class,
                "failure_evidence": failure_evidence,
            }
        )
    return rows


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="debug only: cap number of cases evaluated")
    args = p.parse_args()

    t0 = time.time()
    from medphygraph.paths import new_run_dir
    import evaluation._candidate_dropout_common as _cdc

    run_dir = new_run_dir("candidate_dropout", protocol_id="candidate_dropout")
    _cdc.OUT_DIR = run_dir
    global OUT_DIR
    OUT_DIR = run_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Stage 7: Candidate-Availability Stress Test ===")
    print("Reading frozen expanded-transfer targets (read-only)...")
    targets = load_stage6_targets()
    if args.limit:
        targets = targets[: args.limit]
        print(f"[DEBUG] limiting to first {args.limit} cases")
    n_targets = len(targets)
    print(f"  loaded {n_targets} predeclared expanded-transfer cases")

    print("Fitting frozen-protocol scorer (CF-SupportNet checkpoint only; no retraining)...")
    device = torch.device("cpu")
    scorers = fit_scorers(device)

    print("Loading frozen expanded-transfer scene/candidate/GT context for every referenced state (read-only)...")
    state_ctx: dict[str, dict] = {}
    for t in targets:
        for sid in (t["previous_state_id"], t["current_state_id"]):
            if sid in state_ctx:
                continue
            state_ctx[sid] = load_state_context(t["corpus"], t["case_id"], sid)
    print(f"  loaded {len(state_ctx)} unique states")

    print("Precomputing frozen-physics features for every native candidate edge (one-time; reused by ALL dropout conditions)...")
    feat_cache: dict[tuple[str, str, str], np.ndarray | None] = {}
    n_edges = 0
    for sid, ctx in state_ctx.items():
        for s, h in sorted(ctx["native_cand"]):
            feat_cache[(sid, s, h)] = recompute_feats(ctx["scene"], s, h)
            n_edges += 1
    print(f"  computed features for {n_edges} native (state, subject, host) candidate edges")
    print(
        "  (edges reachable ONLY via cross-state union augmentation, e.g. a pair that is "
        "native at the current state but not the previous one, are recomputed lazily and "
        "cached on first use below -- this exactly reproduces the frozen MedPhyGraph "
        "union mechanism, which also recomputes p_prev/p_curr for such pairs.)"
    )

    def prob_fn(sid: str, s: str, h: str) -> float | None:
        key = (sid, s, h)
        if key not in feat_cache:
            feat_cache[key] = recompute_feats(state_ctx[sid]["scene"], s, h)
        feats = feat_cache[key]
        if feats is None:
            return None
        return scorers["score_feat_matrix"]("health_dyphygraph", feats)

    cfg_v2 = DeltaUnionV2Config()

    all_rows: list[dict[str, Any]] = []
    condition_meta: list[dict[str, Any]] = []

    def run_condition(condition_id: str, condition_kind: str, rate: float, seed: int | None, cand_fn) -> None:
        rows_this = []
        for target in targets:
            a, b = target["previous_state_id"], target["current_state_id"]
            ctx_a, ctx_b = state_ctx[a], state_ctx[b]
            cand_a = cand_fn(target, a, ctx_a["native_cand"])
            cand_b = cand_fn(target, b, ctx_b["native_cand"])
            rows = evaluate_case(target, cand_a, cand_b, ctx_a, ctx_b, prob_fn, cfg_v2)
            for r in rows:
                r["condition_id"] = condition_id
                r["condition_kind"] = condition_kind
                r["dropout_rate"] = rate
                r["dropout_seed"] = seed
                r["total_candidates_prev"] = len(cand_a)
                r["total_candidates_curr"] = len(cand_b)
            rows_this.extend(rows)
        all_rows.extend(rows_this)
        condition_meta.append(
            {"condition_id": condition_id, "condition_kind": condition_kind, "dropout_rate": rate, "dropout_seed": seed}
        )
        print(f"  condition={condition_id:32s} rows={len(rows_this)}")

    print("\n[1/3] Native reference condition (unperturbed expanded-transfer candidate sets)...")
    run_condition("native", "native", 0.0, None, lambda t, sid, native: set(native))

    print("\n[2/3] Stress Test A: targeted destination-candidate removal (per case)...")

    def stress_a_fn(t, sid, native):
        return destination_targeted_survivors(native, subject=t["subject_id"], dest=t["new_direct_host"])

    run_condition("stress_a_targeted_removal", "stress_a", None, None, stress_a_fn)

    print("\n[3/3] Stress Test B: random candidate dropout (rates x seeds)...")
    for rate in DROPOUT_RATES:
        if rate == 0.0:
            for seed in DROPOUT_SEEDS:
                cid = f"stress_b_rate{int(rate*100):02d}_seed{seed}"
                run_condition(cid, "stress_b", rate, seed, lambda t, sid, native: set(native))
            continue
        for seed in DROPOUT_SEEDS:
            cid = f"stress_b_rate{int(rate*100):02d}_seed{seed}"

            def rf(t, sid, native, _rate=rate, _seed=seed):
                return random_dropout_survivors(native, rate=_rate, seed=_seed, state_id=sid)

            run_condition(cid, "stress_b", rate, seed, rf)

    (OUT_DIR / "per_case_raw.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    (OUT_DIR / "condition_meta.json").write_text(json.dumps(condition_meta, indent=2), encoding="utf-8")
    print(f"\nWrote {len(all_rows)} raw rows across {len(condition_meta)} conditions to {OUT_DIR / 'per_case_raw.json'}")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    from evaluation._runtime import configure_repo_paths

    configure_repo_paths()
    raise SystemExit(main())
