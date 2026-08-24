#!/usr/bin/env python3
"""Multi-seed component ablation on core and expanded transfer suites."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_setup_spec = importlib.util.spec_from_file_location(
  "eval._path_setup", Path(__file__).with_name("_path_setup.py")
)
assert _setup_spec and _setup_spec.loader
_setup_mod = importlib.util.module_from_spec(_setup_spec)
_setup_spec.loader.exec_module(_setup_mod)

import json
import time
from pathlib import Path
from typing import Any

import torch

from evaluation._candidate_dropout_common import load_stage6_targets
from evaluation._component_ablation_common import (
  OUT_DIR,
  REFINEMENT_ANCHOR,
  VERIFIED_TARGETS,
  evaluate_core_component_ablation,
  evaluate_expanded_component_ablation,
  fit_health_scorer,
  summarize_ablation_block,
)
from evaluation._multiseed_common import (
  FROZEN_SEED0_SHA256,
  ROOT,
  SEEDS,
  assert_no_eval_leakage_in_training_dataset,
  checkpoint_path_for_seed,
  sha256_file,
  verify_frozen_policy,
)
from evaluation._config import require_dynamic_corpora
from medphygraph.paths import new_run_dir
import evaluation._component_ablation_common as _cab_common


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2,3,4", help="comma-separated seeds")
    p.add_argument("--suite", choices=("core", "expanded", "all"), default="all")
    p.add_argument("--limit-transitions", type=int, default=0, help="debug: cap core transitions")
    p.add_argument("--limit-cases", type=int, default=0, help="debug: cap expanded cases")
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    run_dir = new_run_dir("component_analysis", protocol_id="component_analysis")
    _cab_common.OUT_DIR = run_dir
    OUT_DIR = run_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== component_analysis_multiseed ===")
    policy = verify_frozen_policy()
    if not policy["ok"]:
        raise SystemExit(f"Frozen policy mismatch: {policy['mismatches']}")
    leakage = assert_no_eval_leakage_in_training_dataset()
    if leakage.get("leakage_detected"):
        raise SystemExit(f"Leakage detected: {leakage}")

    if sha256_file(checkpoint_path_for_seed(0)) != FROZEN_SEED0_SHA256:
        raise SystemExit("Frozen seed-0 checkpoint SHA256 mismatch")

    if args.suite in ("core", "all"):
        # Core ablation needs TwinWorld dynamic CF datasets (not on public HF).
        require_dynamic_corpora(require_all=True)

    targets_doc = json.loads(VERIFIED_TARGETS.read_text(encoding="utf-8"))
    expanded_targets = load_stage6_targets()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    t0 = time.time()
    all_results: dict[str, Any] = {
        "protocol": "component_analysis_multiseed",
        "baseline_id": "FROZEN_BASELINE",
        "suites": {
            "core": {
                "description": "Canonical 60-transition protocol (15 eligible transfers)",
                "corpora": ["procedural", "isaac_hc"],
            },
            "expanded": {
                "description": "Frozen expanded-transfer 217-case transfer suite",
                "n_cases": len(expanded_targets),
            },
        },
        "stages": list(
            __import__("_component_ablation_common", fromlist=["ABLATION_STAGES"]).ABLATION_STAGES
        ),
        "seeds": {},
        "elapsed_s": None,
    }

    for seed in seeds:
        ckpt = checkpoint_path_for_seed(seed)
        if not ckpt.exists():
            raise SystemExit(f"Missing checkpoint for seed {seed}: {ckpt}")
        print(f"\n===== SEED {seed} =====")
        scorers = fit_health_scorer(ckpt, device)
        seed_block: dict[str, Any] = {
            "seed": seed,
            "checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
            "checkpoint_sha256": sha256_file(ckpt),
            "n_params": scorers["n_params"],
            "core": {},
            "expanded": {},
        }

        if args.suite in ("core", "all"):
            print("  Core ablation (both corpora)...")
            ab_rows, scene_rows = evaluate_core_component_ablation(
                scorers,
                targets_doc,
                limit_transitions=args.limit_transitions,
            )
            for corpus in ("procedural", "isaac_hc"):
                seed_block["core"][corpus] = summarize_ablation_block(
                    ab_rows, scene_rows, corpus=corpus, suite="core"
                )
            (OUT_DIR / f"seed{seed}_core_ablation_rows.json").write_text(
                json.dumps(ab_rows, indent=2), encoding="utf-8"
            )

        if args.suite in ("expanded", "all"):
            print("  Expanded ablation (217 cases)...")
            exp_rows, exp_scene = evaluate_expanded_component_ablation(
                scorers,
                expanded_targets,
                limit_cases=args.limit_cases,
            )
            seed_block["expanded"]["all"] = summarize_ablation_block(
                exp_rows, exp_scene, corpus=None, suite="expanded"
            )
            for corpus in ("procedural", "isaac_hc"):
                seed_block["expanded"][corpus] = summarize_ablation_block(
                    exp_rows, exp_scene, corpus=corpus, suite="expanded"
                )
            (OUT_DIR / f"seed{seed}_expanded_ablation_rows.json").write_text(
                json.dumps(exp_rows, indent=2), encoding="utf-8"
            )

        all_results["seeds"][str(seed)] = seed_block

    all_results["elapsed_s"] = time.time() - t0
    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path} in {all_results['elapsed_s']:.1f}s")

    # Quick seed-0 Isaac core anchor vs refinement_ablation.json
    if "0" in all_results["seeds"] and args.suite in ("core", "all"):
        anchor_path = REFINEMENT_ANCHOR
        if anchor_path.exists():
            anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
            got = {r["stage"]: r for r in all_results["seeds"]["0"]["core"]["isaac_hc"]}
            ref = {r["stage"]: r for r in anchor["corpora"]["isaac_hc"]}
            print("\nSeed-0 Isaac core anchor check (transfer):")
            for st in got:
                g = got[st].get("transfer")
                r = ref[st].get("transfer")
                ok = "OK" if g == r else f"DIFF (got {g}, ref {r})"
                print(f"  {st}: {ok}")

    return 0


if __name__ == "__main__":
    from evaluation._runtime import configure_repo_paths

    configure_repo_paths()
    raise SystemExit(main())
