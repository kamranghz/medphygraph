#!/usr/bin/env python3
"""Core evaluation scoring for the MedPhyGraph benchmark."""

from __future__ import annotations

import copy
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from evaluation._benchmark_utils import (
  direct_support_host,
  edge_f1_sets,
  gt_from,
  load_scene,
  load_transitions,
  present,
  prf,
)
from evaluation._config import (
  ABLATION_STAGES,
  CFG,
  CKPT,
  DYNAMIC,
  HARD_DS,
  RATIO,
  SCORERS,
  SEED,
  THR,
  TRANSFER,
)
from medphygraph.analytic_physics import run_counterfactual_pair
from medphygraph.baselines import (
    fit_logistic,
    fit_mlp,
    fit_rf,
    geometry_rule_prob,
    pooled_vector,
    predict_sklearn,
)
from medphygraph.candidates import _xy_sep
from medphygraph.consistency import (
    DeltaUnionV2Config,
    apply_consistency,
    apply_transition_aware_consistency_v2,
    count_violations,
)
from medphygraph.data import load_dataset, split_samples
from medphygraph.features import trajectory_features
from medphygraph.metrics import scene_graph_edit_f1
from medphygraph.model import HealthDyPhyGraph, ModelConfig
from medphygraph.schema import STRUCTURAL_TYPES, HealthScene
from medphygraph.scene_graph import PhysicalSceneGraph


@torch.no_grad()
def _score_feat(model: HealthDyPhyGraph, feats: np.ndarray, device: torch.device) -> float:
    x = torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0).to(device)
    _, prob = model(x)
    return float(prob.item())


def fit_scorers(device: torch.device) -> dict[str, Any]:
    parts = split_samples(load_dataset(HARD_DS))
    train = parts["train"]
    logistic = fit_logistic(train, ratio=RATIO)
    rf = fit_rf(train, ratio=RATIO, seed=SEED)
    mlp = fit_mlp(train, ratio=RATIO, seed=SEED)
    blob = torch.load(CKPT, map_location=device, weights_only=False)
    model = HealthDyPhyGraph(ModelConfig(hidden=int(blob.get("config", {}).get("hidden", 64)))).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    def probs_for(name: str, samples: list[dict]) -> dict[tuple[str, str], float]:
        if name == "geometry_rule":
            return {(s["subject_id"], s["host_id"]): geometry_rule_prob(s, ratio=RATIO) for s in samples}
        if name == "health_dyphygraph":
            out = {}
            for s in samples:
                feats = np.asarray(s["features_partial"][str(RATIO)], dtype=np.float32)
                out[(s["subject_id"], s["host_id"])] = _score_feat(model, feats, device)
            return out
        clf = {"logistic": logistic, "random_forest": rf, "mlp": mlp}[name]
        if not samples:
            return {}
        p = predict_sklearn(clf, samples, ratio=RATIO)
        return {(s["subject_id"], s["host_id"]): float(p[i]) for i, s in enumerate(samples)}

    def score_feat_matrix(name: str, feats: np.ndarray) -> float:
        sample = {
            "features_partial": {str(RATIO): np.asarray(feats, dtype=np.float32).tolist()},
            "features_full": np.asarray(feats, dtype=np.float32).tolist(),
        }
        if name == "geometry_rule":
            return float(geometry_rule_prob(sample, ratio=RATIO))
        if name == "health_dyphygraph":
            return _score_feat(model, np.asarray(feats, dtype=np.float32), device)
        clf = {"logistic": logistic, "random_forest": rf, "mlp": mlp}[name]
        X = pooled_vector(np.asarray(feats, dtype=np.float64)).reshape(1, -1)
        if X.shape[1] == 9:
            X = X[:, :-1]
        if hasattr(clf, "predict_proba"):
            return float(clf.predict_proba(X)[0, 1])
        return float(clf.decision_function(X)[0])

    return {"probs_for": probs_for, "score_feat_matrix": score_feat_matrix, "model": model, "n_params": n_params}


def recompute_feats(scene: HealthScene, subject_id: str, host_id: str) -> np.ndarray | None:
    nodes = scene.entity_map()
    if subject_id not in nodes or host_id not in nodes:
        return None
    result = run_counterfactual_pair(scene, subject_id=subject_id, host_id=host_id, n_frames=60)
    cf_pos = np.asarray(result["counterfactual"]["positions"][subject_id], dtype=float)
    contact = np.asarray(result["counterfactual"]["contact_subject_host"], dtype=float)
    s, h = nodes[subject_id], nodes[host_id]
    return trajectory_features(
        positions_subject=cf_pos,
        contact=contact,
        host_removed=True,
        geom_xy_sep=_xy_sep(s, h),
        geom_vertical_gap=float(s.aabb_min()[2] - h.aabb_max()[2]),
    )


def score_raw(g_init, samples, probs) -> PhysicalSceneGraph:
    g = PhysicalSceneGraph.from_dict(copy.deepcopy(g_init.to_dict()))
    for s in samples:
        key = (s["subject_id"], s["host_id"])
        if key not in g.edges:
            g.add_candidate(*key)
        pr = float(probs.get(key, 0.0))
        g.edges[key].score = pr
        g.edges[key].confidence = pr
        g.edges[key].present = pr >= THR
    return g


def apply_union_rescoring(
    g_raw: PhysicalSceneGraph,
    curr_real: dict[tuple[str, str], float],
) -> PhysicalSceneGraph:
    """Rescore with union probabilities, then State Consistency (max confidence)."""
    g = PhysicalSceneGraph.from_dict(copy.deepcopy(g_raw.to_dict()))
    for (s, h), pr in curr_real.items():
        if (s, h) not in g.edges:
            g.add_candidate(s, h)
        g.edges[(s, h)].score = pr
        g.edges[(s, h)].confidence = pr
        g.edges[(s, h)].present = pr >= THR
    apply_consistency(g)
    return g


def apply_direct_gate_only(
    g_raw: PhysicalSceneGraph,
    scene: HealthScene,
) -> PhysicalSceneGraph:
    """Prefer geometrically direct host among present multi-support, then consistency."""
    g = PhysicalSceneGraph.from_dict(copy.deepcopy(g_raw.to_dict()))
    by_subj: dict[str, list] = defaultdict(list)
    for e in g.edges.values():
        if e.present:
            by_subj[e.subject_id].append(e)
    nodes = scene.entity_map()
    for sid, elist in by_subj.items():
        subj = nodes.get(sid)
        if subj is None or subj.entity_type in STRUCTURAL_TYPES or subj.entity_type == "zone":
            continue
        if len(elist) < 2:
            continue
        hosts = [e.host_id for e in elist]
        prefer = direct_support_host(scene, sid, hosts)
        if prefer is None:
            continue
        for e in elist:
            e.present = e.host_id == prefer
            if e.present:
                e.confidence = max(float(e.confidence), THR)
    apply_consistency(g)
    return g


def apply_union_and_gate(
    g_raw: PhysicalSceneGraph,
    curr_real: dict[tuple[str, str], float],
    scene: HealthScene,
) -> PhysicalSceneGraph:
    g = apply_union_rescoring(g_raw, curr_real)
    # Re-apply gate preference on union-rescored graph before final consistency already done;
    # run gate on present edges then consistency again.
    by_subj: dict[str, list] = defaultdict(list)
    for e in g.edges.values():
        if e.present:
            by_subj[e.subject_id].append(e)
    nodes = scene.entity_map()
    for sid, elist in by_subj.items():
        subj = nodes.get(sid)
        if subj is None or subj.entity_type in STRUCTURAL_TYPES or subj.entity_type == "zone":
            continue
        if len(elist) < 2:
            continue
        hosts = [e.host_id for e in elist]
        prefer = direct_support_host(scene, sid, hosts)
        if prefer is None:
            continue
        for e in elist:
            e.present = e.host_id == prefer
    apply_consistency(g)
    return g


def delta_counts(gt_add, gt_rem, pred_add, pred_rem) -> dict[str, Any]:
    a = edge_f1_sets(pred_add, gt_add)
    r = edge_f1_sets(pred_rem, gt_rem)
    pooled = prf(a["tp"] + r["tp"], a["fp"] + r["fp"], a["fn"] + r["fn"])
    return {"added": a, "removed": r, "combined": pooled}


def static_macro_pooled(rows: list[dict]) -> dict[str, Any]:
    """rows: per-scene with tp,fp,fn,f1,precision,recall,violations,n_gt,n_pred"""
    if not rows:
        return {}
    macro_f1 = float(np.mean([r["f1"] for r in rows]))
    macro_p = float(np.mean([r["precision"] for r in rows]))
    macro_r = float(np.mean([r["recall"] for r in rows]))
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    pooled = prf(tp, fp, fn)
    return {
        "scene_macro_ge_f1": round(macro_f1, 6),
        "scene_macro_precision": round(macro_p, 6),
        "scene_macro_recall": round(macro_r, 6),
        "pooled_ge_f1": round(pooled["f1"], 6),
        "pooled_precision": round(pooled["precision"], 6),
        "pooled_recall": round(pooled["recall"], 6),
        "mean_false_support_rate": round(float(np.mean([r["false_support_rate"] for r in rows])), 6),
        "mean_violations": round(float(np.mean([r["violations"] for r in rows])), 6),
        "n_scenes": len(rows),
        "total_gt_edges": int(sum(r["n_gt"] for r in rows)),
        "total_pred_edges": int(sum(r["n_pred"] for r in rows)),
        "pooled_tp": tp,
        "pooled_fp": fp,
        "pooled_fn": fn,
    }


def pack_dynamic(rows: list[dict]) -> dict[str, Any]:
    """rows each have combined/added/removed metrics and operation."""
    if not rows:
        return {
            "transition_macro_dyn_f1": None,
            "pooled_delta_micro_f1": None,
            "n": 0,
        }
    trans_f1 = [r["combined"]["f1"] for r in rows]
    tp = sum(r["combined"]["tp"] for r in rows)
    fp = sum(r["combined"]["fp"] for r in rows)
    fn = sum(r["combined"]["fn"] for r in rows)
    pooled = prf(tp, fp, fn)
    add_rows = [r for r in rows if r["operation"] == "add_object"]
    rem_rows = [r for r in rows if r["operation"] == "remove_object"]
    tr_rows = [r for r in rows if r["operation"] == TRANSFER]

    def op_macro(rs):
        if not rs:
            return None
        return round(float(np.mean([r["combined"]["f1"] for r in rs])), 6)

    def op_pooled(rs):
        if not rs:
            return None
        return round(prf(sum(r["combined"]["tp"] for r in rs), sum(r["combined"]["fp"] for r in rs), sum(r["combined"]["fn"] for r in rs))["f1"], 6)

    # Operation-wise: for Add, use ADD-edge F1 only when ADD is relevant (gt_add or pred_add non-empty OR both empty)
    def op_side_macro(rs, side: str):
        if not rs:
            return None
        return round(float(np.mean([r[side]["f1"] for r in rs])), 6)

    return {
        "transition_macro_dyn_f1": round(float(np.mean(trans_f1)), 6),
        "pooled_delta_micro_f1": round(pooled["f1"], 6),
        "pooled_delta_precision": round(pooled["precision"], 6),
        "pooled_delta_recall": round(pooled["recall"], 6),
        "add_transition_macro_dyn_f1": op_macro(add_rows),
        "remove_transition_macro_dyn_f1": op_macro(rem_rows),
        "transfer_transition_macro_dyn_f1": op_macro(tr_rows),
        "add_pooled_delta_micro_f1": op_pooled(add_rows),
        "remove_pooled_delta_micro_f1": op_pooled(rem_rows),
        "transfer_pooled_delta_micro_f1": op_pooled(tr_rows),
        "add_edge_macro_f1": op_side_macro(add_rows, "added"),
        "remove_edge_macro_f1": op_side_macro(rem_rows, "removed"),
        "unchanged_preservation_mean": round(float(np.mean([r["unchanged_preservation"] for r in rows])), 6),
        "unchanged_retention_cond_prev_correct_mean": round(
            float(np.mean([r["unchanged_retention_cond"] for r in rows])), 6
        ),
        "n_transitions": len(rows),
        "n_add": len(add_rows),
        "n_remove": len(rem_rows),
        "n_transfer": len(tr_rows),
        "gt_add_edges": int(sum(r["n_gt_add"] for r in rows)),
        "gt_remove_edges": int(sum(r["n_gt_rem"] for r in rows)),
        "pred_add_edges": int(sum(r["n_pred_add"] for r in rows)),
        "pred_remove_edges": int(sum(r["n_pred_rem"] for r in rows)),
        "pooled_tp": tp,
        "pooled_fp": fp,
        "pooled_fn": fn,
    }


def _preservation_and_retention(
    *,
    op: str,
    eligible: bool,
    target: dict[str, Any],
    gt_prev: set[tuple[str, str]],
    gt_curr: set[tuple[str, str]],
    gt_rem: set[tuple[str, str]],
    pred_prev: set[tuple[str, str]],
    pred_curr: set[tuple[str, str]],
) -> tuple[float, float]:
    """Compute unchanged preservation and conditional retention.

    Extracted from the `evaluate_all` transition scoring loop to keep that
    loop's orchestration readable while preserving exact numerics.
    """
    # For transfer ops under eligibility, the "unchanged" set is the
    # previous-current GT intersection except that the expected direct-host
    # edge is resolved to the target's new destination host.
    if op == TRANSFER and eligible:
        unchanged_gt = set(gt_prev)  # override any "approx" term from caller
        if target.get("previous_direct_host"):
            unchanged_gt.discard(
                (target["intervention_subject"], target["previous_direct_host"])
            )
        if target.get("new_direct_host"):
            unchanged_gt |= {(target["intervention_subject"], target["new_direct_host"])}
        u_gt = gt_prev & unchanged_gt
    else:
        # Non-transfer ops (or non-eligible transfers): unchanged = prev & curr GT.
        u_gt = gt_prev & gt_curr

    u_pred = pred_prev & pred_curr
    preserv = len(u_gt & u_pred) / max(len(u_gt), 1)

    prev_correct = pred_prev & gt_prev
    u_cond_denom = prev_correct & u_gt
    u_cond = len(u_cond_denom & pred_curr) / max(len(u_cond_denom), 1)
    # `gt_rem` is accepted for parity with the original in-loop signature, but
    # is not needed after the transfer-eligible override.
    _ = gt_rem
    return float(preserv), float(u_cond)


def _fill_union_feature_scores(
    *,
    use_cache: bool,
    local_cache: dict[tuple[str, str, str], np.ndarray],
    a: str,
    b: str,
    scorer: str,
    ds_probs: dict[str, dict[tuple[str, str], float]],
    scorers: dict[str, Any],
    scene_c: dict[str, HealthScene],
    ent_c: dict[str, set[str]],
    cand_c: dict[str, set[tuple[str, str]]],
    raw_g: dict[str, PhysicalSceneGraph],
    state_g: dict[str, PhysicalSceneGraph],
    sample_map_a: dict[tuple[str, str], dict],
    sample_map_b: dict[tuple[str, str], dict],
    feat_cache: dict[tuple[str, str, str], np.ndarray],
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], float], dict[tuple[str, str], float], dict[str, float]]:
    """Build prev/curr feature-confidence maps for all candidate edges in the union."""
    t_comp: dict[str, float] = {}
    t0 = time.perf_counter()
    c_union = set(cand_c[a]) | set(cand_c[b]) | present(raw_g[b]) | present(state_g[a])
    t_comp["union_s"] = time.perf_counter() - t0

    prev_real: dict[tuple[str, str], float] = {}
    curr_real: dict[tuple[str, str], float] = {}

    def fill(sid: str, out: dict[tuple[str, str], float], sample_map: dict[tuple[str, str], dict], ents: set[str], key_name: str):
        t1 = time.perf_counter()
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
            if use_cache and fkey in local_cache:
                feats = local_cache[fkey]
            elif fkey in feat_cache and use_cache:
                feats = feat_cache[fkey]
            else:
                feats = recompute_feats(scene_c[sid], s, h)
                if feats is None:
                    continue
                if use_cache:
                    feat_cache[fkey] = feats
                    local_cache[fkey] = feats
            out[(s, h)] = scorers["score_feat_matrix"](scorer, feats)

        t_comp[key_name] = time.perf_counter() - t1

    fill(a, prev_real, sample_map_a, ent_c[a], "prev_feat_score_s")
    fill(b, curr_real, sample_map_b, ent_c[b], "curr_feat_score_s")
    return c_union, prev_real, curr_real, t_comp


def _append_static_mode_rows(
    *,
    corpus: str,
    scorer: str,
    mode: str,
    sid: str,
    g: PhysicalSceneGraph,
    gt: set[tuple[str, str]],
    scene_level: list[dict],
    all_state_preds: list[dict],
) -> None:
    """Append per-scene metrics for independent/state-consistency modes."""
    pred = present(g)
    tp, fp, fn = len(pred & gt), len(pred - gt), len(gt - pred)
    edit = scene_graph_edit_f1(pred, gt)
    viol = count_violations(g)
    row = {
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
    scene_level.append(row)
    all_state_preds.append(
        {
            "corpus": corpus,
            "scorer": scorer,
            "mode": mode,
            "scene_id": sid,
            "predicted_edges": [list(x) for x in sorted(pred)],
            "gt_edges": [list(x) for x in sorted(gt)],
            "scores": {
                f"{s}|{h}": float(g.edges[(s, h)].confidence)
                for (s, h) in g.edges
                if (s, h) in pred or True
            }
            if False
            else {f"{e.subject_id}|{e.host_id}": float(e.confidence) for e in g.edges.values()},
        }
    )


def evaluate_all(scorers: dict[str, Any], targets_doc: dict[str, Any], *, cold_samples: int = 8) -> dict[str, Any]:
    cfg_v2 = DeltaUnionV2Config()
    targets_by_key = {
        (t["corpus"], t["previous_state_id"], t["current_state_id"]): t for t in targets_doc["targets"]
    }

    all_state_preds: list[dict] = []
    all_trans_preds: list[dict] = []
    all_edge_probs: list[dict] = []
    all_switch: list[dict] = []
    transfer_traces: list[dict] = []
    false_switch_traces: list[dict] = []
    scene_level: list[dict] = []
    transition_level: list[dict] = []
    runtime_warm: list[dict] = []
    runtime_cold: list[dict] = []
    component_rows: list[dict] = []
    candidate_rows: list[dict] = []
    ablation_rows: list[dict] = []

    final_methods: dict[str, Any] = {"corpora": {}}

    for corpus, cfg in DYNAMIC.items():
        print(f"[verified] corpus={corpus}")
        scenes = cfg["scenes"]
        raw = load_dataset(cfg["dataset"])
        by_scene: dict[str, list[dict]] = defaultdict(list)
        for s in raw["samples"]:
            by_scene[s["scene_id"]].append(s)
        transitions = load_transitions(scenes)
        state_ids = sorted({t["previous_state_id"] for t in transitions} | {t["current_state_id"] for t in transitions})

        g_init_c: dict[str, PhysicalSceneGraph] = {}
        scene_c: dict[str, HealthScene] = {}
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

        corpus_methods: dict[str, Any] = {}

        for scorer in SCORERS:
            print(f"  scorer={scorer}")
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
                    _append_static_mode_rows(
                        corpus=corpus,
                        scorer=scorer,
                        mode=mode,
                        sid=sid,
                        g=g,
                        gt=gt_c[sid],
                        scene_level=scene_level,
                        all_state_preds=all_state_preds,
                    )

            # Dynamic modes + ablation for HealthDyPhyGraph
            mode_dyn_rows: dict[str, list] = defaultdict(list)
            switch_stats = {"accepted_transfers": 0, "false_non_transfer_switches": 0, "accepted_all": 0}
            ta_static_rows: list[dict] = []

            cold_budget = cold_samples if scorer == "health_dyphygraph" else 0

            for ti, t in enumerate(transitions):
                a, b = t["previous_state_id"], t["current_state_id"]
                op = t.get("operation_type") or t.get("operation") or "unknown"
                target = targets_by_key[(corpus, a, b)]
                eligible = bool(target.get("primary_metric_eligible", True))
                # Exclude ambiguous/invalid transfers from primary metrics
                if op == TRANSFER and not eligible:
                    include_primary = False
                else:
                    include_primary = True

                gt_add = {(x[0], x[1]) for x in target["gt_add"]}
                gt_rem = {(x[0], x[1]) for x in target["gt_remove"]}

                sample_map_a = {(s["subject_id"], s["host_id"]): s for s in by_scene.get(a, [])}
                sample_map_b = {(s["subject_id"], s["host_id"]): s for s in by_scene.get(b, [])}

                def fill_union(use_cache: bool, local_cache: dict) -> tuple[set, dict, dict, dict]:
                    return _fill_union_feature_scores(
                        use_cache=use_cache,
                        local_cache=local_cache,
                        a=a,
                        b=b,
                        scorer=scorer,
                        ds_probs=ds_probs,
                        scorers=scorers,
                        scene_c=scene_c,
                        ent_c=ent_c,
                        cand_c=cand_c,
                        raw_g=raw_g,
                        state_g=state_g,
                        sample_map_a=sample_map_a,
                        sample_map_b=sample_map_b,
                        feat_cache=feat_cache,
                    )

                # Warm path (shared cache)
                t_wall0 = time.perf_counter()
                c_union, prev_real, curr_real, t_comp = fill_union(True, {})
                t_sel0 = time.perf_counter()
                g_ta = PhysicalSceneGraph.from_dict(copy.deepcopy(raw_g[b].to_dict()))
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
                    prev_refined=state_g[a],
                    prev_entity_ids=ent_c[a],
                    curr_entity_ids=ent_c[b],
                    curr_scene_entities=list(scene_c[b].entities),
                    config=cfg_v2,
                )
                t_comp["supporter_selection_s"] = time.perf_counter() - t_sel0
                pred_ta = present(g_ta)
                warm_s = time.perf_counter() - t_wall0

                # Static for transition-aware current state
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

                # Ablation graphs (HealthDyPhyGraph only)
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
                if scorer == "health_dyphygraph":
                    g_union = apply_union_rescoring(raw_g[b], curr_real)
                    g_gate = apply_direct_gate_only(raw_g[b], scene_c[b])
                    g_both = apply_union_and_gate(raw_g[b], curr_real, scene_c[b])
                    mode_preds["union_rescoring"] = present(g_union)
                    mode_preds["direct_support_gate"] = present(g_gate)
                    mode_preds["union_and_gate"] = present(g_both)
                    mode_preds["medphygraph"] = pred_ta
                    for m in ("union_rescoring", "direct_support_gate", "union_and_gate", "medphygraph"):
                        pred_prev_map[m] = present(state_g[a])

                for mode, pred_curr in mode_preds.items():
                    pred_prev = pred_prev_map[mode]
                    pred_add = pred_curr - pred_prev
                    pred_rem = pred_prev - pred_curr
                    counts = delta_counts(gt_add, gt_rem, pred_add, pred_rem)
                    preserv, u_cond = _preservation_and_retention(
                        op=op,
                        eligible=eligible,
                        target=target,
                        gt_prev=gt_c[a],
                        gt_curr=gt_c[b],
                        gt_rem=gt_rem,
                        pred_prev=pred_prev,
                        pred_curr=pred_curr,
                    )

                    row = {
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
                        "unchanged_preservation": preserv,
                        "unchanged_retention_cond": u_cond,
                        "n_gt_add": len(gt_add),
                        "n_gt_rem": len(gt_rem),
                        "n_pred_add": len(pred_add),
                        "n_pred_rem": len(pred_rem),
                        "pred_add": [list(x) for x in sorted(pred_add)],
                        "pred_rem": [list(x) for x in sorted(pred_rem)],
                        "gt_add": [list(x) for x in sorted(gt_add)],
                        "gt_rem": [list(x) for x in sorted(gt_rem)],
                    }
                    transition_level.append(row)
                    if include_primary and mode in ("independent", "state_consistency", "transition_aware"):
                        mode_dyn_rows[mode].append(row)
                    if include_primary and scorer == "health_dyphygraph" and mode in ABLATION_STAGES:
                        ablation_rows.append(row)

                    all_trans_preds.append(
                        {
                            "corpus": corpus,
                            "scorer": scorer,
                            "mode": mode,
                            "previous_state_id": a,
                            "current_state_id": b,
                            "operation": op,
                            "pred_add": row["pred_add"],
                            "pred_remove": row["pred_rem"],
                            "gt_add": row["gt_add"],
                            "gt_remove": row["gt_rem"],
                        }
                    )

                # Edge probs for MedPhyGraph traces
                if scorer == "health_dyphygraph":
                    for (s, h), pr in curr_real.items():
                        all_edge_probs.append(
                            {
                                "corpus": corpus,
                                "state_id": b,
                                "subject": s,
                                "host": h,
                                "p_curr": pr,
                                "p_prev": prev_real.get((s, h)),
                            }
                        )
                    runtime_warm.append(
                        {
                            "corpus": corpus,
                            "a": a,
                            "b": b,
                            "operation": op,
                            "seconds": warm_s,
                            "n_union": len(c_union),
                            "cache": "warm",
                            **t_comp,
                        }
                    )
                    component_rows.append({"corpus": corpus, "a": a, "b": b, "cache": "warm", **t_comp, "total_s": warm_s})

                    # Cold-cache samples
                    if cold_budget > 0 and ti < cold_budget:
                        local = {}
                        # Ensure we don't reuse feat_cache for this measurement
                        keys = [k for k in list(feat_cache.keys()) if k[0] in (a, b)]
                        for k in keys:
                            feat_cache.pop(k, None)
                        t0 = time.perf_counter()
                        c2, pr2, cr2, tc2 = fill_union(False, local)
                        g2 = PhysicalSceneGraph.from_dict(copy.deepcopy(raw_g[b].to_dict()))
                        for (s, h), pr in cr2.items():
                            if (s, h) not in g2.edges:
                                g2.add_candidate(s, h)
                            g2.edges[(s, h)].score = pr
                            g2.edges[(s, h)].confidence = pr
                            g2.edges[(s, h)].present = pr >= THR
                        apply_transition_aware_consistency_v2(
                            g2,
                            prev_confidence=pr2,
                            curr_confidence=cr2,
                            prev_refined=state_g[a],
                            prev_entity_ids=ent_c[a],
                            curr_entity_ids=ent_c[b],
                            curr_scene_entities=list(scene_c[b].entities),
                            config=cfg_v2,
                        )
                        cold_s = time.perf_counter() - t0
                        runtime_cold.append(
                            {
                                "corpus": corpus,
                                "a": a,
                                "b": b,
                                "operation": op,
                                "seconds": cold_s,
                                "n_union": len(c2),
                                "cache": "cold",
                                **tc2,
                            }
                        )

                # Switch stats
                for d in trep.differs_from_legacy:
                    switch_stats["accepted_all"] += 1
                    if op == TRANSFER:
                        if d.get("track_c_v2_host") == target.get("new_direct_host"):
                            switch_stats["accepted_transfers"] += 1
                    else:
                        switch_stats["false_non_transfer_switches"] += 1
                        if scorer == "health_dyphygraph":
                            false_switch_traces.append(
                                {
                                    "corpus": corpus,
                                    "a": a,
                                    "b": b,
                                    "operation": op,
                                    "diff": d,
                                }
                            )

                if scorer == "health_dyphygraph":
                    for dec in trep.subject_decisions:
                        if not (dec.get("attempted") or dec.get("accepted") or op == TRANSFER):
                            continue
                        best = dec.get("best_candidate") or {}
                        tr = {
                            "corpus": corpus,
                            "previous_state_id": a,
                            "current_state_id": b,
                            "operation": op,
                            "subject": dec.get("subject_id"),
                            "previous_host": dec.get("h_old"),
                            "destination_host": target.get("new_direct_host"),
                            "selected_host": dec.get("selected_host"),
                            "dest_in_prev_native": (
                                (dec.get("subject_id"), target.get("new_direct_host")) in cand_c[a]
                                if target.get("new_direct_host")
                                else None
                            ),
                            "dest_in_curr_native": (
                                (dec.get("subject_id"), target.get("new_direct_host")) in cand_c[b]
                                if target.get("new_direct_host")
                                else None
                            ),
                            "dest_in_union": (
                                (dec.get("subject_id"), target.get("new_direct_host")) in c_union
                                if target.get("new_direct_host")
                                else None
                            ),
                            "real_p_prev": best.get("p_prev"),
                            "real_p_curr": best.get("p_t"),
                            "gain": best.get("gain_new"),
                            "drop_old": dec.get("drop_old"),
                            "switch_score": best.get("switch_score"),
                            "gate": dec.get("direct_support_gate"),
                            "accepted": dec.get("accepted"),
                            "reason": dec.get("rejection_reason"),
                            "zero_fill_used": False,
                            "comparability": "real_prev_required",
                            "pred_add": [list(x) for x in sorted(pred_ta - present(state_g[a]))],
                            "pred_rem": [list(x) for x in sorted(present(state_g[a]) - pred_ta)],
                            "gt_add": target["gt_add"],
                            "gt_remove": target["gt_remove"],
                            "primary_eligible": eligible,
                        }
                        all_switch.append(tr)
                        if op == TRANSFER:
                            transfer_traces.append(tr)

                    if op == TRANSFER:
                        dest = (target.get("intervention_subject"), target.get("new_direct_host"))
                        prev_h = (target.get("intervention_subject"), target.get("previous_direct_host"))
                        candidate_rows.append(
                            {
                                "corpus": corpus,
                                "transition": f"{a}->{b}",
                                "eligible": eligible,
                                "dest": list(dest) if dest[0] else None,
                                "dest_in_prev": dest in cand_c[a] if dest[0] else None,
                                "dest_in_curr": dest in cand_c[b] if dest[0] else None,
                                "dest_in_union": dest in c_union if dest[0] else None,
                                "prev_host_in_prev": prev_h in cand_c[a] if prev_h[0] else None,
                                "p_prev_dest": prev_real.get(dest) if dest[0] else None,
                                "p_curr_dest": curr_real.get(dest) if dest[0] else None,
                                "p_prev_old": prev_real.get(prev_h) if prev_h[0] else None,
                                "p_curr_old": curr_real.get(prev_h) if prev_h[0] else None,
                                "static_gt_edge_in_cand_prev": sum(1 for e in gt_c[a] if e in cand_c[a]),
                                "n_static_gt_prev": len(gt_c[a]),
                            }
                        )

            # Pack method summary (primary = transition_aware final)
            static_indep = static_macro_pooled(
                [r for r in scene_level if r["corpus"] == corpus and r["scorer"] == scorer and r["mode"] == "independent"]
            )
            static_state = static_macro_pooled(
                [
                    r
                    for r in scene_level
                    if r["corpus"] == corpus and r["scorer"] == scorer and r["mode"] == "state_consistency"
                ]
            )
            static_ta = static_macro_pooled(ta_static_rows)
            dyn_pack = {m: pack_dynamic(mode_dyn_rows[m]) for m in ("independent", "state_consistency", "transition_aware")}
            corpus_methods[scorer] = {
                "label": scorer,
                "paper_name": "MedPhyGraph" if scorer == "health_dyphygraph" else scorer,
                "static_independent": static_indep,
                "static_state_consistency": static_state,
                "static_final": static_ta if scorer != "geometry_rule" else static_state,
                "dynamic": dyn_pack,
                "switch_stats": switch_stats,
            }
            # Geometry may lack meaningful union probs; still report TA static
            if scorer == "health_dyphygraph":
                corpus_methods[scorer]["static_final"] = static_ta

        final_methods["corpora"][corpus] = {"methods": corpus_methods}

    return {
        "final_methods": final_methods,
        "all_state_predictions": all_state_preds,
        "all_transition_predictions": all_trans_preds,
        "all_edge_probabilities": all_edge_probs,
        "all_switch_decisions": all_switch,
        "transfer_traces": transfer_traces,
        "false_switch_traces": false_switch_traces,
        "scene_level": scene_level,
        "transition_level": transition_level,
        "ablation_rows": ablation_rows,
        "candidate_rows": candidate_rows,
        "runtime_warm": runtime_warm,
        "runtime_cold": runtime_cold,
        "component_rows": component_rows,
        "n_params": scorers["n_params"],
        "config": cfg_v2.to_dict(),
    }
