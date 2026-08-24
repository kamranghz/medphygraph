#!/usr/bin/env python3
"""Final MedPhyGraph multi-seed: shared constants / helpers.

Writes ONLY under runs/multiseed/ (see MULTISEED_OUT / medphygraph.paths.new_run_dir).
Never writes into frozen checkpoint dirs or evaluation-only artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation._config import REPO_ROOT, SEED0_SHA256
from evaluation._shared import (
  FROZEN_SEED0_CKPT,
  HARD_DATASET_JSON,
  HARD_SPLIT_JSON,
  MULTISEED_CKPT_DIR,
  MULTISEED_OUT,
  SEEDS,
  TRAIN_SEEDS,
  checkpoint_path_for_seed,
  sha256_file,
)
from evaluation._stage6_paths import STAGE6_DIR, STAGE6_MANIFEST, STAGE6_PER_CASE, STAGE6_SCENES, STAGE6_TARGETS

ROOT = REPO_ROOT
from medphygraph.paths import expanded_transfer_root, results_frozen_root, runs_root  # noqa: E402

OUT_DIR = MULTISEED_OUT
CKPT_DIR = MULTISEED_CKPT_DIR
FROZEN_SEED0_SHA256 = SEED0_SHA256
HARD_DS = HARD_DATASET_JSON
HARD_SPLIT = HARD_SPLIT_JSON

# Exact frozen seed-0 training recipe (Phase 9A).
TRAIN_RECIPE = {
    "script": "scripts/train_dyphygraph.py",
    "dataset": str(HARD_DS.relative_to(ROOT)).replace("\\", "/"),
    "ratio": 1.0,
    "epochs": 40,
    "batch_size": 64,
    "lr": 1e-3,
    "hidden": 64,
    "optimizer": "Adam",
    "loss": "BCEWithLogitsLoss(pos_weight=n_neg/n_pos on train)",
    "checkpoint_selection": "best validation edge F1 @ thr=0.5",
    "early_stopping": False,
    "normalization": "none (incompleteness mask only; ratio=1.0 uses full features)",
    "split": "scene_level_only seed=20260730; train/val/test edge counts 324/101/108",
    "dynamic_transfer_in_selection": False,
}

# Evaluation-only trees that must NEVER enter training/val/selection.
EVAL_ONLY_TREES = [
    expanded_transfer_root(),
    runs_root() / "candidate_dropout",
    runs_root() / "expanded_transfer",
]

VERIFIED_TARGETS = (
    results_frozen_root() / "evaluation_targets/operation_consistent_dynamic_verified.json"
)

# Std definition used everywhere in Stage 9 summaries.
STD_DEFINITION = "sample standard deviation (numpy ddof=1) across the 5 seeds"


def verify_frozen_policy() -> dict[str, Any]:
    from medphygraph.consistency import DeltaUnionV2Config

    cfg = DeltaUnionV2Config()
    expected = {
        "presence_threshold": 0.5,
        "gain_threshold": 0.05,
        "switch_threshold": 0.10,
        "lambda_drop": 1.0,
        "absolute_margin": 0.0,
        "allow_below_threshold_rescue": False,
        "direct_gap_lo": -0.02,
        "direct_gap_hi": 0.15,
        "direct_xy_expand": 0.08,
        "direct_contact_eps": 0.08,
        "require_real_prev_prob": True,
    }
    actual = {k: getattr(cfg, k) for k in expected}
    mismatches = {k: (actual[k], expected[k]) for k in expected if actual[k] != expected[k]}
    return {"ok": not mismatches, "actual": actual, "mismatches": mismatches}


def assert_no_eval_leakage_in_training_dataset() -> dict[str, Any]:
    """Programmatic leakage check: hard train dataset IDs must not intersect expanded/eval artifacts."""
    hard = json.loads(HARD_DS.read_text(encoding="utf-8"))
    hard_scene_ids = {s["scene_id"] for s in hard["samples"]}
    hard_sample_keys = {(s["scene_id"], s["subject_id"], s["host_id"]) for s in hard["samples"]}

    stage6_scene_ids: set[str] = set()
    if STAGE6_TARGETS.exists():
        targets = json.loads(STAGE6_TARGETS.read_text(encoding="utf-8"))
        if isinstance(targets, list):
            for t in targets:
                stage6_scene_ids.add(t.get("previous_state_id", ""))
                stage6_scene_ids.add(t.get("current_state_id", ""))
                stage6_scene_ids.add(t.get("case_id", ""))
        elif isinstance(targets, dict):
            for t in targets.get("targets", targets if isinstance(targets, list) else []):
                if isinstance(t, dict):
                    stage6_scene_ids.add(t.get("previous_state_id", ""))
                    stage6_scene_ids.add(t.get("current_state_id", ""))
                    stage6_scene_ids.add(t.get("case_id", ""))
    stage6_scene_ids.discard("")

    overlap_scenes = sorted(hard_scene_ids & stage6_scene_ids)
    # Expanded-transfer scenes live under data/expanded_transfer/scenes/ with case_id prefixes —
    # hard scenes are hard_NNN. Intersection should be empty.
    return {
        "n_hard_train_val_test_samples": len(hard["samples"]),
        "n_hard_unique_scenes": len(hard_scene_ids),
        "n_hard_unique_edges": len(hard_sample_keys),
        "stage6_state_overlap_with_hard": overlap_scenes,
        "leakage_detected": bool(overlap_scenes),
        "eval_only_trees": [str(p.relative_to(ROOT)).replace("\\", "/") for p in EVAL_ONLY_TREES],
        "note": (
            "Training uses ONLY dataset_hard (hard_000..hard_019 analytic CF edges). "
            "Expanded-transfer / candidate-dropout artifacts are evaluation-only and were not passed "
            "to train_dyphygraph.py."
        ),
    }


def sample_stats(values: list[float]) -> dict[str, float | None]:
    import numpy as np

    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(len(arr)),
        "std_definition": STD_DEFINITION,
    }
