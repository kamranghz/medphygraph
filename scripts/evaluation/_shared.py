"""Shared runtime paths and scorer helpers for evaluation drivers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from evaluation._config import REPO_ROOT, SCORERS, SEED0_SHA256, THR
from medphygraph.paths import (
  checkpoint_seed0,
  dataset_hard_json,
  dataset_hard_split,
  expanded_transfer_root,
  multiseed_checkpoints_dir,
  new_run_dir,
  runs_root,
)

# Runtime outputs go under runs/<script>/<utc>/ (see new_run_dir).
# Downloaded inputs live under data/ and checkpoints/ via medphygraph.paths.
MULTISEED_CKPT_DIR = multiseed_checkpoints_dir()
PROCEDURAL_TRANSFER_INPUT = expanded_transfer_root()
HARD_DATASET_JSON = dataset_hard_json()
HARD_SPLIT_JSON = dataset_hard_split()
FROZEN_SEED0_CKPT = checkpoint_seed0()

SEEDS = (0, 1, 2, 3, 4)
TRAIN_SEEDS = (1, 2, 3, 4)


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def checkpoint_path_for_seed(seed: int) -> Path:
  if seed == 0:
    return FROZEN_SEED0_CKPT
  return MULTISEED_CKPT_DIR / f"health_dyphygraph_r1.0_seed{seed}.pt"


def fit_health_scorer(ckpt_path: Path, device: torch.device):
  model, blob = load_health_dyphygraph_model(ckpt_path, device)

  @torch.no_grad()
  def score_feat(features):
    tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    _, prob = model(tensor)
    return float(prob.item())

  return {"model": model, "score_feat": score_feat, "blob": blob}


def load_health_dyphygraph_model(ckpt_path: Path, device: torch.device):
  """Load the frozen HealthDyPhyGraph checkpoint for CF-SupportNet scoring."""
  from medphygraph.model import HealthDyPhyGraph, ModelConfig

  blob = torch.load(ckpt_path, map_location=device, weights_only=False)
  model = HealthDyPhyGraph(ModelConfig(hidden=int(blob.get("config", {}).get("hidden", 64)))).to(device)
  model.load_state_dict(blob["model_state"])
  model.eval()
  return model, blob


def verify_seed0_checkpoint() -> None:
  actual = sha256_file(FROZEN_SEED0_CKPT)
  if actual != SEED0_SHA256:
    raise SystemExit(f"Seed-0 checkpoint SHA256 mismatch (expected {SEED0_SHA256}, got {actual})")


# Backward-compatible names used by older helpers (now point at data/ or runs/).
MULTISEED_OUT = runs_root() / "multiseed"
CANDIDATE_DROPOUT_OUT = runs_root() / "candidate_dropout"
COMPONENT_ABLATION_OUT = runs_root() / "component_analysis"
PROCEDURAL_TRANSFER_OUT = runs_root() / "expanded_transfer"


__all__ = [
  "CANDIDATE_DROPOUT_OUT",
  "COMPONENT_ABLATION_OUT",
  "FROZEN_SEED0_CKPT",
  "HARD_DATASET_JSON",
  "HARD_SPLIT_JSON",
  "MULTISEED_CKPT_DIR",
  "MULTISEED_OUT",
  "PROCEDURAL_TRANSFER_INPUT",
  "PROCEDURAL_TRANSFER_OUT",
  "SCORERS",
  "SEEDS",
  "THR",
  "TRAIN_SEEDS",
  "checkpoint_path_for_seed",
  "fit_health_scorer",
  "load_health_dyphygraph_model",
  "new_run_dir",
  "sha256_file",
  "verify_seed0_checkpoint",
]
