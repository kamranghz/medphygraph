#!/usr/bin/env python3
"""Generate and evaluate the frozen expanded-transfer case list."""

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
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation._benchmark_utils import direct_support_host, edge_f1_sets, prf
from evaluation._config import CFG, RATIO, REPO_ROOT, TRANSFER
from evaluation._core_scoring import delta_counts, fit_scorers, recompute_feats, score_raw
from evaluation._procedural_transfer_common import BASE_CORPORA, author_generic_transfer, clone_scene
from evaluation.build_counterfactual_dataset import process_scene_analytic
from medphygraph.paths import expanded_transfer_root, new_run_dir
from medphygraph.consistency import (
  DeltaUnionV2Config,
  apply_consistency,
  apply_transition_aware_consistency_v2,
  count_violations,
)
from medphygraph.candidates import _xy_sep
from medphygraph.features import trajectory_features
from medphygraph.schema import HealthScene
from medphygraph.scene_graph import PhysicalSceneGraph

# Downloaded suite inputs (targets, scenes, manifests) — read-only for --eval-only.
INPUT_DIR = expanded_transfer_root()
SCENES_DIR = INPUT_DIR / "scenes"
MANIFEST_PATH = INPUT_DIR / "predeclared_manifest.json"
# Legacy alias used by helpers that still say OUT_DIR for input paths.
OUT_DIR = INPUT_DIR

SCORERS = ("geometry_rule", "logistic", "random_forest", "mlp", "health_dyphygraph")
FINAL_LABEL = {
    "geometry_rule": "Geometry Rule",
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest",
    "mlp": "MLP",
    "health_dyphygraph": "MedPhyGraph",
}
MODE_LABEL = {
    "independent": "Independent (scorer only)",
    "state_consistency": "CF-SupportNet + State Consistency",
    "transition_aware": "MedPhyGraph (Union-Based Transition-Aware Consistency)",
}


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_case_scenes(case: dict) -> dict[str, Any]:
    """Author + label the frozen (s0, s1) pair for one accepted case. Returns paths + samples."""
    corpus = case["corpus"]
    base_dir = BASE_CORPORA[corpus] / case["base_scene_id"]
    base_scene = HealthScene.from_dict(json.loads((base_dir / "scene.json").read_text(encoding="utf-8")))

    case_dir = SCENES_DIR / corpus / case["case_id"]
    s0_id = f"{case['case_id']}__s0"
    s1_id = f"{case['case_id']}__s1"
    s0_dir = case_dir / s0_id
    s1_dir = case_dir / s1_id
    s0_dir.mkdir(parents=True, exist_ok=True)
    s1_dir.mkdir(parents=True, exist_ok=True)

    s0 = clone_scene(base_scene, s0_id)
    (s0_dir / "scene.json").write_text(json.dumps(s0.to_dict(), indent=2), encoding="utf-8")

    s1 = clone_scene(base_scene, s1_id)
    op1 = author_generic_transfer(s1, case["subject_id"], case["destination_id"])
    if not op1.get("ok"):
        raise RuntimeError(f"authoring failed for {case['case_id']}: {op1}")
    (s1_dir / "scene.json").write_text(json.dumps(s1.to_dict(), indent=2), encoding="utf-8")
    (s1_dir / "phase2_state_meta.json").write_text(
        json.dumps(
            {
                "base_scene_id": case["base_scene_id"],
                "operation": op1["operation"],
                "op_meta": op1,
                "generation_note": "expanded_transfer: generic authoring mechanism, no model used",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report0 = process_scene_analytic(s0_dir, n_frames=60)
    report1 = process_scene_analytic(s1_dir, n_frames=60)

    return {
        "s0_id": s0_id,
        "s1_id": s1_id,
        "s0_dir": s0_dir,
        "s1_dir": s1_dir,
        "samples0": report0["samples"],
        "samples1": report1["samples"],
    }


def build_target_for_case(case: dict, gt_a: set[tuple[str, str]], scene_a: HealthScene) -> dict[str, Any]:
    """Same target-construction rule as build_verified_targets() for op==TRANSFER."""
    subject = case["subject_id"]
    dest = case["destination_id"]
    prev_hosts = sorted(h for (s, h) in gt_a if s == subject)
    n_prev = len(prev_hosts)
    ambiguous = False
    invalid = False
    eligible = True
    prev_direct = None
    resolution = None
    if n_prev == 1:
        prev_direct = prev_hosts[0]
        resolution = "unique_previous_gt_host"
    elif n_prev == 0:
        invalid = True
        eligible = False
        resolution = "invalid_zero_previous_gt_hosts"
    else:
        geom = direct_support_host(scene_a, subject, prev_hosts)
        if geom is not None:
            prev_direct = geom
            resolution = "geometry_direct_support_among_gt_hosts"
        else:
            ambiguous = True
            eligible = False
            resolution = "ambiguous_multiple_prev_hosts_no_unique_direct"

    if eligible and prev_direct and dest:
        gt_add = [[subject, dest]]
        gt_rem = [[subject, prev_direct]]
    else:
        gt_add, gt_rem = [], []
        eligible = False

    return {
        "corpus": case["corpus"],
        "case_id": case["case_id"],
        "base_scene_id": case["base_scene_id"],
        "semantic_template": case["semantic_template"],
        "subject_id": subject,
        "previous_direct_host": prev_direct,
        "new_direct_host": dest,
        "n_previous_gt_hosts": n_prev,
        "previous_gt_hosts": prev_hosts,
        "ambiguous": ambiguous,
        "invalid": invalid,
        "primary_metric_eligible": eligible,
        "resolution": resolution,
        "gt_add": gt_add,
        "gt_remove": gt_rem,
    }


def sample_from_pair_json(pair_path: Path, scene: HealthScene) -> dict[str, Any] | None:
    """Rebuild one CF-SupportNet sample from a frozen expanded-transfer rollout pair.json."""
    subj, host = pair_path.parent.name.split("__", 1)
    nodes = scene.entity_map()
    if subj not in nodes or host not in nodes:
        return None
    result = json.loads(pair_path.read_text(encoding="utf-8"))
    cf_pos = np.asarray(result["counterfactual"]["positions"][subj], dtype=float)
    contact = np.asarray(result["counterfactual"]["contact_subject_host"], dtype=float)
    s_ent, h_ent = nodes[subj], nodes[host]
    feats = trajectory_features(
        positions_subject=cf_pos,
        contact=contact,
        host_removed=True,
        geom_xy_sep=_xy_sep(s_ent, h_ent),
        geom_vertical_gap=float(s_ent.aabb_min()[2] - h_ent.aabb_max()[2]),
    )
    feat_arr = np.asarray(feats, dtype=np.float32)
    return {
        "scene_id": scene.scene_id,
        "subject_id": subj,
        "host_id": host,
        "label": int(result.get("label", {}).get("positive", 0)),
        "features_partial": {str(RATIO): feat_arr.tolist()},
        "features_full": feat_arr.tolist(),
    }


def load_samples_from_state_dir(sdir: Path, scene: HealthScene) -> list[dict[str, Any]]:
    """Read-only: load analytic CF samples from frozen rollouts (no scene regeneration)."""
    rollouts_dir = sdir / "rollouts"
    if not rollouts_dir.exists():
        return []
    samples: list[dict[str, Any]] = []
    for pair_path in sorted(rollouts_dir.glob("*/pair.json")):
        sample = sample_from_pair_json(pair_path, scene)
        if sample is not None:
            samples.append(sample)
    return samples


def load_frozen_targets_and_states(targets: list[dict]) -> tuple[
    dict[str, set[str]],
    dict[str, HealthScene],
    dict[str, set[tuple[str, str]]],
    dict[str, PhysicalSceneGraph],
    dict[str, list[dict]],
]:
    """Load frozen expanded-transfer scenes/targets from disk without regenerating anything."""
    ent_c: dict[str, set[str]] = {}
    scene_c: dict[str, HealthScene] = {}
    cand_c: dict[str, set[tuple[str, str]]] = {}
    g_init_c: dict[str, PhysicalSceneGraph] = {}
    by_scene: dict[str, list[dict]] = defaultdict(list)

    for i, target in enumerate(targets):
        corpus = target["corpus"]
        case_id = target["case_id"]
        for sid in (target["previous_state_id"], target["current_state_id"]):
            if sid in scene_c:
                continue
            sdir = SCENES_DIR / corpus / case_id / sid
            if not sdir.exists():
                raise FileNotFoundError(f"missing frozen expanded-transfer state directory: {sdir}")
            scene = HealthScene.from_dict(json.loads((sdir / "scene.json").read_text(encoding="utf-8")))
            scene_c[sid] = scene
            ent_c[sid] = {e.entity_id for e in scene.entities if e.entity_type != "zone"}
            g_init_c[sid] = PhysicalSceneGraph.load(sdir / "graph_initial.json")
            cand_c[sid] = set(g_init_c[sid].edges.keys())
            by_scene[sid] = load_samples_from_state_dir(sdir, scene)
        if (i + 1) % 25 == 0 or (i + 1) == len(targets):
            print(f"  loaded {i + 1}/{len(targets)} frozen cases")

    return ent_c, scene_c, cand_c, g_init_c, by_scene


def print_suite_b_summary(per_case_rows: list[dict]) -> None:
    """Print headline Suite-B numbers for MedPhyGraph (health_dyphygraph, transition_aware)."""
    rows = [r for r in per_case_rows if r["scorer"] == "health_dyphygraph" and r["mode"] == "transition_aware"]
    n = len(rows)
    exact = sum(1 for r in rows if r["transfer_success"])
    pooled_f1_vals = [r["transfer_f1"] for r in rows if r["transfer_f1"] is not None]
    pooled_mean = float(np.mean(pooled_f1_vals)) if pooled_f1_vals else 0.0
    violations = sum(r.get("graph_constraint_violations") or 0 for r in rows)
    dest_union = sum(1 for r in rows if r.get("dest_in_union"))
    real_prev = sum(1 for r in rows if r.get("real_p_prev_dest_available"))
    print("\n=== SUITE B (MedPhyGraph / transition_aware) ===")
    print(f"Cases evaluated: {n}")
    print(f"Exact transfer success: {exact}/{n} ({exact / n if n else 0:.3f})")
    print(f"Mean per-case transfer F1: {pooled_mean:.6f}")
    print(f"Destination in union: {dest_union}/{n}")
    print(f"Real p_prev(dest) available: {real_prev}/{n}")
    print(f"Graph constraint violations (total): {violations}")
    tmpl: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        tmpl[r["semantic_template"]].append(r)
    print("\nPer-template exact success:")
    for tmpl_name in sorted(tmpl):
        rs = tmpl[tmpl_name]
        succ = sum(1 for r in rs if r["transfer_success"])
        print(f"  {tmpl_name}: {succ}/{len(rs)}")
    fails = [r for r in rows if not r["transfer_success"]]
    if fails:
        print(f"\nFailures ({len(fails)}):")
        for r in fails[:10]:
            print(f"  {r['case_id']} ({r['semantic_template']}, {r['corpus']})")
        if len(fails) > 10:
            print(f"  ... and {len(fails) - 10} more")


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="debug only: cap number of cases evaluated")
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="evaluate frozen expanded-transfer scenes/targets from disk (no scene regeneration)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: runs/expanded_transfer/<utc-timestamp>/)",
    )
    args = p.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = new_run_dir(
            "expanded_transfer",
            protocol_id="expanded_transfer_eval_only" if args.eval_only else "expanded_transfer_generate",
        )

    ent_c: dict[str, set[str]] = {}
    scene_c: dict[str, HealthScene] = {}
    gt_c: dict[str, set[tuple[str, str]]] = {}
    cand_c: dict[str, set[tuple[str, str]]] = {}
    g_init_c: dict[str, PhysicalSceneGraph] = {}
    by_scene: dict[str, list[dict]] = defaultdict(list)
    targets: list[dict] = []

    if args.eval_only:
        targets_path = INPUT_DIR / "targets.json"
        if not targets_path.exists():
            raise SystemExit(f"missing frozen targets: {targets_path}")
        targets = json.loads(targets_path.read_text(encoding="utf-8"))
        if args.limit:
            targets = targets[: args.limit]
            print(f"[DEBUG] limiting to first {args.limit} targets")
        print(f"Loaded frozen targets: {len(targets)} cases from {targets_path}")
        print("Loading frozen expanded-transfer scenes + rollouts (read-only)...")
        ent_c, scene_c, cand_c, g_init_c, by_scene = load_frozen_targets_and_states(targets)
    else:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        recorded_hash = manifest.pop("manifest_sha256_self")
        manifest["manifest_sha256_self"] = recorded_hash  # restore in-memory copy
        accepted = manifest["accepted_cases"]
        if args.limit:
            accepted = accepted[: args.limit]
            print(f"[DEBUG] limiting to first {args.limit} cases")
        print(f"Loaded frozen manifest: {len(accepted)} accepted cases (hash on file: {recorded_hash[:16]}...)")

        print(f"Generating + labeling {len(accepted)} frozen expanded cases (analytic physics; no model used)...")
        for i, case in enumerate(accepted):
            built = build_case_scenes(case)
            s0_id, s1_id = built["s0_id"], built["s1_id"]

            for sid, sdir, samples in (
                (s0_id, built["s0_dir"], built["samples0"]),
                (s1_id, built["s1_dir"], built["samples1"]),
            ):
                scene_c[sid] = HealthScene.from_dict(json.loads((sdir / "scene.json").read_text(encoding="utf-8")))
                ent_c[sid] = {e.entity_id for e in scene_c[sid].entities if e.entity_type != "zone"}
                g_init_c[sid] = PhysicalSceneGraph.load(sdir / "graph_initial.json")
                cand_c[sid] = set(g_init_c[sid].edges.keys())
                gt_graph = PhysicalSceneGraph.load(sdir / "graph_gt.json")
                gt_c[sid] = {(e.subject_id, e.host_id) for e in gt_graph.edges.values() if e.is_gt is True}
                by_scene[sid] = samples

            target = build_target_for_case(case, gt_c[s0_id], scene_c[s0_id])
            target["previous_state_id"] = s0_id
            target["current_state_id"] = s1_id
            targets.append(target)
            if (i + 1) % 25 == 0 or (i + 1) == len(accepted):
                print(f"  generated {i + 1}/{len(accepted)}")

        n_gen_mismatch = sum(
            1
            for t, c in zip(targets, accepted)
            if t["primary_metric_eligible"] != c["eligible"] or t["previous_direct_host"] != c["previous_direct_host"]
        )
        print(f"Post-generation eligibility re-check mismatches vs manifest: {n_gen_mismatch}")

    device = torch.device("cpu")
    print("Fitting frozen-protocol scorers (train split only; no leakage from expanded scenes)...")
    scorers = fit_scorers(device)

    state_ids = list(ent_c.keys())
    print(f"Scoring {len(state_ids)} states x {len(SCORERS)} scorers...")

    ds_probs_by_scorer: dict[str, dict[str, dict[tuple[str, str], float]]] = {}
    raw_g_by_scorer: dict[str, dict[str, PhysicalSceneGraph]] = {}
    state_g_by_scorer: dict[str, dict[str, PhysicalSceneGraph]] = {}

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
        ds_probs_by_scorer[scorer] = ds_probs
        raw_g_by_scorer[scorer] = raw_g
        state_g_by_scorer[scorer] = state_g

    def present(g: PhysicalSceneGraph) -> set[tuple[str, str]]:
        return {(e.subject_id, e.host_id) for e in g.edges.values() if e.present}

    cfg_v2 = DeltaUnionV2Config()
    per_case_rows: list[dict] = []
    failure_rows: list[dict] = []

    feat_cache: dict[tuple[str, str, str], np.ndarray] = {}

    print("Evaluating transition-aware / state-consistency / independent modes per case...")
    for ci, target in enumerate(targets):
        a, b = target["previous_state_id"], target["current_state_id"]
        subj = target["subject_id"]
        dest = target["new_direct_host"]
        prev_host = target["previous_direct_host"]
        eligible = target["primary_metric_eligible"]
        gt_add = {tuple(x) for x in target["gt_add"]}
        gt_rem = {tuple(x) for x in target["gt_remove"]}

        sample_map_a = {(s["subject_id"], s["host_id"]): s for s in by_scene.get(a, [])}
        sample_map_b = {(s["subject_id"], s["host_id"]): s for s in by_scene.get(b, [])}

        for scorer in SCORERS:
            ds_probs = ds_probs_by_scorer[scorer]
            raw_g = raw_g_by_scorer[scorer]
            state_g = state_g_by_scorer[scorer]

            c_union = set(cand_c[a]) | set(cand_c[b]) | present(raw_g[b]) | present(state_g[a])
            prev_real: dict[tuple[str, str], float] = {}
            curr_real: dict[tuple[str, str], float] = {}

            def fill(sid, out, sample_map, ents):
                for s, h in c_union:
                    if s not in ents or h not in ents or s not in scene_c[sid].entity_map() or h not in scene_c[sid].entity_map():
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
                g_ta.edges[(s, h)].present = pr >= 0.5

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
            pred_ta = present(g_ta)

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

            dest_in_prev_native = (subj, dest) in cand_c[a] if dest else None
            dest_in_curr_native = (subj, dest) in cand_c[b] if dest else None
            dest_in_union = (subj, dest) in c_union if dest else None
            real_p_prev_dest = (subj, dest) in prev_real if dest else None
            prev_host_candidate_available = (subj, prev_host) in cand_c[a] if prev_host else None
            violations_ta = count_violations(g_ta)["total"]

            for mode, pred_curr in mode_preds.items():
                pred_prev = pred_prev_map[mode]
                pred_add = pred_curr - pred_prev
                pred_rem = pred_prev - pred_curr
                counts = delta_counts(gt_add, gt_rem, pred_add, pred_rem) if eligible else None
                transfer_correct = eligible and (subj, dest) in pred_add and (subj, prev_host) in pred_rem

                row = {
                    "case_id": target["case_id"],
                    "corpus": target["corpus"],
                    "semantic_template": target["semantic_template"],
                    "base_scene_id": target["base_scene_id"],
                    "subject_id": subj,
                    "previous_host": prev_host,
                    "destination_host": dest,
                    "scorer": scorer,
                    "mode": mode,
                    "primary_eligible": eligible,
                    "transfer_success": transfer_correct,
                    "add_f1": counts["added"]["f1"] if counts else None,
                    "remove_f1": counts["removed"]["f1"] if counts else None,
                    "transfer_f1": counts["combined"]["f1"] if counts else None,
                    "pred_add": sorted(pred_add),
                    "pred_rem": sorted(pred_rem),
                    "dest_in_prev_native_candidates": dest_in_prev_native,
                    "dest_in_curr_native_candidates": dest_in_curr_native,
                    "dest_in_union": dest_in_union,
                    "real_p_prev_dest_available": real_p_prev_dest,
                    "previous_host_candidate_available": prev_host_candidate_available,
                    "graph_constraint_violations": violations_ta if mode == "transition_aware" else None,
                }
                per_case_rows.append(row)

                if mode == "transition_aware" and scorer == "health_dyphygraph" and eligible and not transfer_correct:
                    reason = "other"
                    evidence = ""
                    if dest_in_union is False:
                        reason = "destination_candidate_missing"
                        evidence = "destination absent from candidate union"
                    elif real_p_prev_dest is False:
                        reason = "previous_probability_unavailable"
                        evidence = "no real p_prev for destination edge"
                    else:
                        dec = next((d for d in trep.subject_decisions if d.get("subject_id") == subj), None)
                        rr = dec.get("rejection_reason") if dec else "no_subject_decision_recorded"
                        if rr == "destination_below_presence_threshold":
                            reason = "alternative_edge_below_presence_threshold"
                        elif rr in ("gain_below_threshold", "switch_score_below_threshold", "no_comparable_alternative_host"):
                            reason = "temporal_replacement_condition_failed"
                        elif rr and rr.startswith("direct_support_gate_"):
                            reason = "direct_support_gate_failed"
                        elif rr in ("no_previous_refined_host", "subject_not_temporally_comparable", "old_host_not_temporally_comparable", "missing_real_p_prev_old"):
                            reason = "temporal_replacement_condition_failed"
                        else:
                            reason = "other"
                        evidence = f"apply_transition_aware_consistency_v2 rejection_reason={rr}"
                    failure_rows.append(
                        {
                            "case_id": target["case_id"],
                            "semantic_template": target["semantic_template"],
                            "failure_class": reason,
                            "evidence": evidence,
                        }
                    )

        if (ci + 1) % 25 == 0 or (ci + 1) == len(targets):
            print(f"  evaluated {ci + 1}/{len(targets)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        (out_dir / "targets.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")
    (out_dir / "per_case_raw.json").write_text(json.dumps(per_case_rows, indent=2), encoding="utf-8")
    (out_dir / "failure_analysis.json").write_text(json.dumps(failure_rows, indent=2), encoding="utf-8")
    print(f"n_targets={len(targets)} n_rows={len(per_case_rows)} n_failures={len(failure_rows)}")
    print(f"wrote {out_dir / 'per_case_raw.json'}, {out_dir / 'failure_analysis.json'}")
    print_suite_b_summary(per_case_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
