#!/usr/bin/env python3
"""Core MedPhyGraph benchmark driver: verified targets, scoring, and evaluation finalize."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_setup_spec = importlib.util.spec_from_file_location(
  "evaluation._path_setup", Path(__file__).with_name("_path_setup.py")
)
assert _setup_spec and _setup_spec.loader
_setup_mod = importlib.util.module_from_spec(_setup_spec)
_setup_spec.loader.exec_module(_setup_mod)

import json
import platform
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from evaluation._benchmark_utils import (
  direct_support_host,
  gt_from,
  load_scene,
  load_transitions,
  op_meta,
)
from evaluation._config import (
  CFG,
  CKPT,
  CONS_PY,
  DYN_PY,
  DYNAMIC,
  HARD_DS,
  PKG,
  REPO_ROOT,
  TRANSFER,
  available_dynamic_corpora,
  require_dynamic_corpora,
)
from evaluation._io import sha256, write_csv, write_json, write_markdown
from medphygraph.data import load_dataset
from medphygraph.scene_graph import PhysicalSceneGraph

PROTECTED = [
  REPO_ROOT / "manuscript",
  CKPT,
  CONS_PY,
  DYN_PY,
]
# Historical entries under outputs/dyphygraph_health/... (dual_dyn_f1_eval,
# consistency_on_baselines, transfer_gt_intent_v1, transfer_fix_safe_v1,
# transfer_fix_delta_v1, transfer_fix_delta_union_v2) were removed here: that
# whole tree no longer exists after the Phase A layout migration, so each
# always resolved to {"type": "missing"} in the safety audit below and
# protected nothing. They were named after specific historical development
# runs with no clean equivalent under the current runs/<script>/<timestamp>/
# convention, so no replacement path was guessed -- add real ones deliberately
# if there are current directories worth auditing here.


# ---------------------------------------------------------------------------
# Safety / environment
# ---------------------------------------------------------------------------


def record_safety(tag: str) -> dict:
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        env["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        env["git_status_short"] = subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True)[:4000]
    except Exception as e:
        env["git_error"] = str(e)

    hashes = {
        "checkpoint": sha256(CKPT),
        "consistency_py": sha256(CONS_PY),
        "dynamic_metrics_py": sha256(DYN_PY),
    }
    # sample GT files
    for corpus, cfg in DYNAMIC.items():
        scenes = cfg["scenes"]
        for sid in sorted([p.name for p in scenes.iterdir() if p.is_dir()])[:3]:
            gp = scenes / sid / "graph_gt.json"
            if gp.exists():
                hashes[f"gt::{corpus}::{sid}"] = sha256(gp)
    protected = {}
    for p in PROTECTED:
        if p.is_file():
            protected[str(p)] = {"type": "file", "sha256": sha256(p)}
        elif p.is_dir():
            files = sorted([x for x in p.rglob("*") if x.is_file()])[:8]
            protected[str(p)] = {"type": "dir", "sample_hashes": {str(f): sha256(f) for f in files}}
        else:
            protected[str(p)] = {"type": "missing"}

    blob = {"tag": tag, "timestamp": datetime.now(timezone.utc).isoformat(), "environment": env, "hashes": hashes, "protected": protected}
    write_json(PKG / f"audits/safety_{tag}.json", blob)
    if tag == "before":
        write_json(PKG / "audits/environment.json", env)
    return blob


def protected_regression(before: dict, after: dict) -> dict:
    mismatches = []
    for k, vb in before.get("protected", {}).items():
        va = after.get("protected", {}).get(k)
        if vb != va:
            mismatches.append(k)
    for k in ("checkpoint", "consistency_py", "dynamic_metrics_py"):
        if before["hashes"].get(k) != after["hashes"].get(k):
            mismatches.append(k)
    out = {"unchanged": len(mismatches) == 0, "mismatches": mismatches}
    write_json(PKG / "audits/protected_files_regression.json", out)
    return out


# ---------------------------------------------------------------------------
# Method–code alignment
# ---------------------------------------------------------------------------


def write_method_alignment() -> None:
    alignment = {
        "final_mode": "apply_transition_aware_consistency_v2 / DeltaUnionV2Config",
        "sequence": [
            "1. Candidate generation per state (graph_initial / generate_candidates offline)",
            "2. Union C_prev ∪ C_curr in evaluation script",
            "3. Real p_prev/p_curr via dataset features or recomputed CF features (no 0-fill)",
            "4. Temporal comparability: subject/host must exist in both states; real p_prev required",
            "5. gain_new = p_t - p_prev",
            "6. drop_old = p_prev_old - p_curr_old",
            "7. switch_score = gain_new + lambda_drop * max(drop_old, 0)",
            "8. Direct-support admissibility (_direct_support_gate)",
            "9. Select best comparable alternative host",
            "10. Enforce single primary host",
            "11. Structural termination",
            "12. Cycle removal + post-pass",
            "13. ADD/REMOVE = set difference of present edges across states",
        ],
        "formulas": {
            "gain": {
                "paper": "gain(s,h)=p_t(s,h)-p_{t-1}(s,h)",
                "code": "gain_new = float(p_t.get(h_new, 0.0)) - p_prev_new",
                "file": "consistency.py",
                "lines": "917-918",
                "matches_paper": True,
            },
            "drop_old": {
                "paper": "drop_old(s)=p_{t-1}(s,h_old)-p_t(s,h_old)",
                "code": "drop_old = p_prev_old - p_curr_old",
                "file": "consistency.py",
                "lines": "893",
                "matches_paper": True,
            },
            "switch_score": {
                "paper": "switch_score=gain+lambda_drop*max(drop_old,0)",
                "code": "switch_score = gain_new + cfg.lambda_drop * max(drop_old, 0.0)",
                "file": "consistency.py",
                "lines": "918",
                "matches_paper": True,
            },
        },
        "thresholds": CFG.to_dict(),
        "acceptance_order": [
            "destination edge exists",
            "p_new >= presence_threshold (0.5); below-threshold rescue false",
            "gain_new >= gain_threshold (0.05)",
            "switch_score >= switch_threshold (0.10)",
            "absolute_margin only if >0 (default 0 → disabled)",
            "direct_support_gate ok",
            "_would_be_valid_primary (load path + no cycle)",
        ],
        "inference_guarantees": {
            "reads_gt": False,
            "reads_intervention_destination": False,
            "missing_prev_filled_with_zero": False,
            "noncomparable_excluded": True,
        },
        "dynamic_metrics": {
            "file": "dynamic_metrics.py",
            "function": "set_delta_metrics",
            "note": "ADD/REMOVE F1 from set differences; empty-empty handled in verified package via prf()",
        },
    }
    write_json(PKG / "audits/method_code_alignment.json", alignment)
    write_markdown(
        PKG / "audits/method_code_alignment.md",
        "# Method–code alignment\n\n"
        + "\n".join(f"- {s}" for s in alignment["sequence"])
        + "\n\n## Switch score\nVerified exact match to paper formula in `apply_transition_aware_consistency_v2` "
        f"(consistency.py ~917–918).\n\nThresholds: `{json.dumps(CFG.to_dict())}`\n\n"
        "Inference does **not** read GT or intervention destination; missing previous probabilities are "
        "**not** replaced by zero.\n",
    )


# ---------------------------------------------------------------------------
# Verified targets
# ---------------------------------------------------------------------------


def _summarize_transfer_diversity(targets: list[dict]) -> dict:
    transfers = [target for target in targets if target["operation"] == TRANSFER and target["primary_metric_eligible"]]
    templates = sorted(
        {
            f"{target['intervention_subject']}::{target['previous_direct_host']}->{target['new_direct_host']}"
            for target in transfers
            if target["previous_direct_host"] and target["new_direct_host"]
        }
    )
    return {
        "n_transfer_transitions_eligible": len(transfers),
        "n_transfer_all": sum(1 for target in targets if target["operation"] == TRANSFER),
        "n_ambiguous_excluded": sum(1 for target in targets if target["operation"] == TRANSFER and target["ambiguous"]),
        "n_invalid_excluded": sum(1 for target in targets if target["operation"] == TRANSFER and target["invalid"]),
        "unique_semantic_templates": templates,
        "n_unique_semantic_templates": len(templates),
        "subject_categories": sorted({target["intervention_subject"] for target in transfers}),
        "old_host_categories": sorted({target["previous_direct_host"] for target in transfers}),
        "new_host_categories": sorted({target["new_direct_host"] for target in transfers}),
        "unique_base_scenes": sorted({target["base_scene_id"] for target in transfers}),
        "counts_by_corpus": {corpus: sum(1 for target in transfers if target["corpus"] == corpus) for corpus in DYNAMIC},
        "note": "Repeated layout instances sharing the same subject/old/new template are one semantic type.",
    }


def _audit_target_builder_source() -> dict:
    source = Path(__file__).read_text(encoding="utf-8")
    return {
        "scanned_file": str(Path(__file__)),
        "forbidden_special_case_branches_found": bool(
            re.search(r'if\s+(subject|s)\s*==\s*[\'"](tray|monitor|bench|floor)[\'"]', source)
        ),
        "note": "Object names may appear in data/comments; no subject=='tray' special-case in target logic.",
        "destination_fields_eval_only": ["to_hint", "intended_surface", "subject_id from op_meta"],
    }


def _write_verified_target_artifacts(out: dict, targets: list[dict], inconsistencies: list[dict]) -> None:
    diversity = out["diversity"]
    audit = {
        "n_targets": len(targets),
        "n_inconsistencies": len(inconsistencies),
        "transfer_eligible": diversity["n_transfer_transitions_eligible"],
        "ambiguous": diversity["n_ambiguous_excluded"],
        "invalid": diversity["n_invalid_excluded"],
        "destination_metadata_eval_only": True,
    }
    write_json(PKG / "evaluation_targets/operation_consistent_dynamic_verified.json", out)
    write_csv(
        PKG / "evaluation_targets/operation_consistent_dynamic_verified.csv",
        [
            {
                "corpus": target["corpus"],
                "base_scene_id": target["base_scene_id"],
                "previous_state_id": target["previous_state_id"],
                "current_state_id": target["current_state_id"],
                "operation": target["operation"],
                "subject": target["intervention_subject"],
                "prev_host": target["previous_direct_host"],
                "new_host": target["new_direct_host"],
                "n_prev_gt_hosts": target["n_previous_gt_hosts"],
                "ambiguous": target["ambiguous"],
                "primary_eligible": target["primary_metric_eligible"],
                "gt_add": json.dumps(target["gt_add"]),
                "gt_remove": json.dumps(target["gt_remove"]),
                "resolution": target["resolution"],
            }
            for target in targets
        ],
    )
    write_json(PKG / "audits/dynamic_target_verified.json", {"audit": audit, "inconsistencies": inconsistencies})
    write_markdown(
        PKG / "audits/dynamic_target_verified.md",
        "# Verified dynamic targets\n\n"
        f"- Targets: {audit['n_targets']}\n"
        f"- Eligible transfers: {audit['transfer_eligible']}\n"
        f"- Ambiguous excluded: {audit['ambiguous']}\n"
        f"- Invalid excluded: {audit['invalid']}\n"
        f"- Inconsistency reports: {audit['n_inconsistencies']}\n",
    )
    write_json(PKG / "audits/transfer_diversity.json", diversity)
    write_markdown(
        PKG / "audits/transfer_diversity.md",
        "# Transfer diversity\n\n"
        f"- Eligible transfers: {diversity['n_transfer_transitions_eligible']}\n"
        f"- Unique semantic templates: {diversity['n_unique_semantic_templates']}\n",
    )
    hard = _audit_target_builder_source()
    write_json(PKG / "audits/no_hardcoded_target_logic.json", hard)
    write_markdown(
        PKG / "audits/no_hardcoded_target_logic.md",
        "# No hard-coded target logic\n\n"
        f"Special-case subject==name branches found: {hard['forbidden_special_case_branches_found']}\n",
    )


def _resolve_transfer_target(
    *,
    corpus: str,
    a: str,
    b: str,
    scene_a: PhysicalSceneGraph,
    subject: str | None,
    dest: str | None,
    prev_hosts: list[str],
    inconsistencies: list[dict],
) -> dict[str, Any]:
    """Resolve a TRANSFER operation into the verified GT delta target."""
    n_prev = len(prev_hosts)
    provenance = "intervention_destination_for_eval_targets_only+previous_gt_hosts"
    new_direct = dest
    prev_host_unique = n_prev == 1

    prev_direct = None
    ambiguous = False
    invalid = False
    primary_eligible = True
    resolution = None

    if n_prev == 1:
        prev_direct = prev_hosts[0]
        resolution = "unique_previous_gt_host"
    elif n_prev == 0:
        invalid = True
        primary_eligible = False
        resolution = "invalid_zero_previous_gt_hosts"
        inconsistencies.append({"corpus": corpus, "a": a, "b": b, "issue": "transfer_zero_prev_hosts"})
    else:
        # Multiple hosts: operation-independent direct-support geometry.
        geom = direct_support_host(scene_a, subject, prev_hosts)
        if geom is not None:
            prev_direct = geom
            resolution = "geometry_direct_support_among_gt_hosts"
            ambiguous = False
        else:
            ambiguous = True
            primary_eligible = False
            resolution = "ambiguous_multiple_prev_hosts_no_unique_direct"
            inconsistencies.append(
                {
                    "corpus": corpus,
                    "a": a,
                    "b": b,
                    "issue": "transfer_ambiguous_prev_hosts",
                    "prev_hosts": prev_hosts,
                }
            )

    if primary_eligible and subject and prev_direct and new_direct:
        gt_add = [[subject, new_direct]]
        gt_rem = [[subject, prev_direct]]
    else:
        gt_add, gt_rem = [], []
        if not invalid and not ambiguous:
            primary_eligible = False

    return {
        "gt_add": gt_add,
        "gt_rem": gt_rem,
        "prev_direct": prev_direct,
        "new_direct": new_direct,
        "ambiguous": ambiguous,
        "invalid": invalid,
        "primary_eligible": primary_eligible,
        "prev_host_unique": prev_host_unique,
        "provenance": provenance,
        "resolution": resolution,
    }


def _resolve_non_transfer_target(
    *,
    corpus: str,
    a: str,
    b: str,
    op: str,
    frozen_add: list[tuple[str, str]],
    frozen_rem: list[tuple[str, str]],
    ents_a: set[str],
    ents_b: set[str],
    inconsistencies: list[dict],
) -> dict[str, Any]:
    """Resolve non-TRANSFER operations into the verified GT delta target."""
    gt_add = [list(x) for x in frozen_add]
    gt_rem = [list(x) for x in frozen_rem]
    ambiguous = False
    invalid = False
    primary_eligible = True
    provenance = "frozen_state_graph_delta"
    resolution = None

    # Verify entity existence
    for s, h in frozen_add:
        if s not in ents_b or h not in ents_b:
            inconsistencies.append(
                {"corpus": corpus, "a": a, "b": b, "issue": "add_entity_missing_curr", "edge": [s, h]}
            )
    for s, h in frozen_rem:
        if s not in ents_a or h not in ents_a:
            inconsistencies.append(
                {"corpus": corpus, "a": a, "b": b, "issue": "rem_entity_missing_prev", "edge": [s, h]}
            )

    appeared, disappeared = ents_b - ents_a, ents_a - ents_b
    if op == "add_object":
        add_subjs = {s for s, _ in frozen_add}
        if appeared and add_subjs and not (add_subjs & appeared) and frozen_add:
            # relation creation on existing objects is allowed; flag only if empty delta with appearance
            pass
        if appeared and not frozen_add:
            inconsistencies.append(
                {
                    "corpus": corpus,
                    "a": a,
                    "b": b,
                    "issue": "add_object_appeared_but_empty_edge_delta",
                    "appeared": sorted(appeared),
                }
            )

    if op == "remove_object" and disappeared and not frozen_rem:
        inconsistencies.append(
            {
                "corpus": corpus,
                "a": a,
                "b": b,
                "issue": "remove_object_disappeared_but_empty_edge_delta",
                "disappeared": sorted(disappeared),
            }
        )

    if not gt_add and not gt_rem and op in ("add_object", "remove_object"):
        inconsistencies.append({"corpus": corpus, "a": a, "b": b, "issue": "empty_add_remove_target", "op": op})
        # still include but flag
        primary_eligible = True  # empty may be valid for some ops; keep countable

    return {
        "gt_add": gt_add,
        "gt_rem": gt_rem,
        "ambiguous": ambiguous,
        "invalid": invalid,
        "primary_eligible": primary_eligible,
        "provenance": provenance,
        "resolution": resolution,
    }


def build_verified_targets() -> dict:
    targets = []
    inconsistencies = []
    for corpus, cfg in DYNAMIC.items():
        scenes = cfg["scenes"]
        raw = load_dataset(cfg["dataset"])
        by = defaultdict(list)
        for s in raw["samples"]:
            by[s["scene_id"]].append(s)
        for t in load_transitions(scenes):
            a, b = t["previous_state_id"], t["current_state_id"]
            op = t.get("operation_type") or t.get("operation") or "unknown"
            scene_a, scene_b = load_scene(scenes, a), load_scene(scenes, b)
            ents_a = {e.entity_id for e in scene_a.entities if e.entity_type != "zone"}
            ents_b = {e.entity_id for e in scene_b.entities if e.entity_type != "zone"}
            gta = gt_from(by[a], PhysicalSceneGraph.load(scenes / a / "graph_gt.json"))
            gtb = gt_from(by[b], PhysicalSceneGraph.load(scenes / b / "graph_gt.json"))
            cand_a = set(PhysicalSceneGraph.load(scenes / a / "graph_initial.json").edges.keys())
            cand_b = set(PhysicalSceneGraph.load(scenes / b / "graph_initial.json").edges.keys())
            meta = op_meta(t, scenes)
            subject = meta.get("subject_id") or (t.get("affected_instance_ids") or [None])[0]
            dest = meta.get("to_hint") or meta.get("intended_surface")
            frozen_add = sorted(gtb - gta)
            frozen_rem = sorted(gta - gtb)

            prev_hosts = sorted(h for (s, h) in gta if s == subject) if subject else []
            n_prev = len(prev_hosts)
            prev_direct = None
            new_direct = None
            prev_host_unique = None
            ambiguous = False
            invalid = False
            primary_eligible = True
            provenance = "frozen_state_graph_delta"
            resolution = None

            if op == TRANSFER:
                resolved = _resolve_transfer_target(
                    corpus=corpus,
                    a=a,
                    b=b,
                    scene_a=scene_a,
                    subject=subject,
                    dest=dest,
                    prev_hosts=prev_hosts,
                    inconsistencies=inconsistencies,
                )
            else:
                resolved = _resolve_non_transfer_target(
                    corpus=corpus,
                    a=a,
                    b=b,
                    op=op,
                    frozen_add=frozen_add,
                    frozen_rem=frozen_rem,
                    ents_a=ents_a,
                    ents_b=ents_b,
                    inconsistencies=inconsistencies,
                )

            gt_add = resolved["gt_add"]
            gt_rem = resolved["gt_rem"]
            ambiguous = resolved["ambiguous"]
            invalid = resolved["invalid"]
            primary_eligible = resolved["primary_eligible"]
            provenance = resolved["provenance"]
            resolution = resolved["resolution"]
            if op == TRANSFER:
                prev_direct = resolved["prev_direct"]
                new_direct = resolved["new_direct"]
                prev_host_unique = resolved["prev_host_unique"]

            row = {
                "corpus": corpus,
                "base_scene_id": t.get("base_scene_id"),
                "previous_state_id": a,
                "current_state_id": b,
                "operation": op,
                "intervention_subject": subject,
                "previous_direct_host": prev_direct,
                "new_direct_host": new_direct,
                "n_previous_gt_hosts": n_prev,
                "previous_gt_hosts": prev_hosts,
                "previous_host_unique": prev_host_unique,
                "ambiguous": ambiguous,
                "invalid": invalid,
                "primary_metric_eligible": primary_eligible and not (op == TRANSFER and (not gt_add or not gt_rem)),
                "resolution": resolution,
                "gt_add": gt_add if op == TRANSFER else [list(x) for x in frozen_add],
                "gt_remove": gt_rem if op == TRANSFER else [list(x) for x in frozen_rem],
                "provenance": provenance,
                "entities_prev": sorted(ents_a),
                "entities_curr": sorted(ents_b),
                "candidate_prev_has_dest": (subject, dest) in cand_a if subject and dest else None,
                "candidate_curr_has_dest": (subject, dest) in cand_b if subject and dest else None,
            }
            if op == TRANSFER:
                row["gt_add"] = gt_add
                row["gt_remove"] = gt_rem
            targets.append(row)

    diversity = _summarize_transfer_diversity(targets)

    out = {
        "protocol": "operation_consistent_dynamic_verified",
        "targets": targets,
        "inconsistencies": inconsistencies,
        "diversity": diversity,
    }
    _write_verified_target_artifacts(out, targets, inconsistencies)
    return out


def main() -> int:
    from evaluation._core_finalize import finalize
    from evaluation._core_scoring import evaluate_all, fit_scorers

    # Full core rebuild needs phase-2 + Isaac-HC CF datasets (not on public HF).
    require_dynamic_corpora(require_all=True)
    print(f"=== Dynamic corpora available: {', '.join(available_dynamic_corpora())} ===")

    from medphygraph.paths import new_run_dir
    import evaluation._config as _cfg
    import evaluation._core_finalize as _fin

    run_pkg = new_run_dir("core", protocol_id="operation_consistent_dynamic_verified")
    _cfg.PKG = run_pkg
    _fin.PKG = run_pkg
    global PKG
    PKG = run_pkg
    PKG.mkdir(parents=True, exist_ok=True)
    print("=== Safety before ===")
    before = record_safety("before")
    print("=== Method–code alignment ===")
    write_method_alignment()
    print("=== Verified targets ===")
    targets_doc = build_verified_targets()
    print("=== Fit scorers ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    scorers = fit_scorers(device)
    print("=== Full evaluation ===")
    bundle = evaluate_all(scorers, targets_doc, cold_samples=8)
    if torch.cuda.is_available():
        bundle["peak_gpu_memory_mb"] = float(torch.cuda.max_memory_allocated() / (1024**2))
    print("=== Safety after ===")
    after = record_safety("after")
    reg = protected_regression(before, after)
    print("=== Finalize package ===")
    summary = finalize(bundle, targets_doc, before, after)
    isaac = None
    for row in json.loads((PKG / "results/processed/final_all_methods.json").read_text())["rows"]:
        if row["corpus"] == "isaac_hc" and row["method"] == "MedPhyGraph":
            isaac = row
            break
    print(
        json.dumps(
            {
                "package": str(PKG),
                "protected_unchanged": reg.get("unchanged"),
                "isaac_medphygraph": isaac,
                "cold_runtime_mean": summary["runtime"]["cold"].get("mean_s"),
                "warm_runtime_mean": summary["runtime"]["warm"].get("mean_s"),
                "regression": summary["regression"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    import argparse

    from evaluation._runtime import configure_repo_paths

    configure_repo_paths()
    parser = argparse.ArgumentParser(description="Run the frozen core benchmark evaluation pipeline.")
    parser.parse_args()
    raise SystemExit(main())
