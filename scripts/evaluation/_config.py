"""Frozen evaluation configuration shared across core benchmark modules."""

from __future__ import annotations

from pathlib import Path

from medphygraph.paths import (
  checkpoint_seed0,
  dataset_hard_json,
  results_frozen_root,
  twinworld_i4h_dynamic_dataset,
  twinworld_i4h_dynamic_scenes,
  twinworld_phase2_dataset,
  twinworld_phase2_scenes,
  verification_paper_root,
)
from medphygraph.consistency import DeltaUnionV2Config

REPO_ROOT = Path(__file__).resolve().parents[2]


def verification_paper_root_local() -> Path:
  return verification_paper_root()


# Read-only frozen paper bundle (never write here from eval drivers).
FROZEN_ROOT = results_frozen_root()
# Write destination for core rebuilds; set to a runs/<stamp> dir in core.main().
PKG = FROZEN_ROOT
RATIO = 1.0
THR = 0.5
SEED = 0
BOOT_N = 10000
TRANSFER = "transfer_support"
CFG = DeltaUnionV2Config()

CKPT = checkpoint_seed0()
HARD_DS = dataset_hard_json()
CONS_PY = REPO_ROOT / "src/medphygraph/consistency.py"
DYN_PY = REPO_ROOT / "src/medphygraph/dynamic_metrics.py"

DYNAMIC = {
  "procedural": {
    "label": "Procedural",
    "dataset": twinworld_phase2_dataset(),
    "scenes": twinworld_phase2_scenes(),
  },
  "isaac_hc": {
    "label": "Isaac for Healthcare",
    "dataset": twinworld_i4h_dynamic_dataset(),
    "scenes": twinworld_i4h_dynamic_scenes(),
  },
}


def corpus_asset_status(corpus: str) -> dict[str, bool | str]:
  """Return presence flags for a DYNAMIC corpus (scenes dir + CF dataset)."""
  cfg = DYNAMIC[corpus]
  scenes = Path(cfg["scenes"])
  dataset = Path(cfg["dataset"])
  return {
    "corpus": corpus,
    "scenes_ok": scenes.is_dir(),
    "dataset_ok": dataset.is_file(),
    "scenes": str(scenes),
    "dataset": str(dataset),
  }


def available_dynamic_corpora() -> list[str]:
  """Corpora that have both a scene tree and a counterfactual dataset on disk."""
  return [
    name
    for name in DYNAMIC
    if Path(DYNAMIC[name]["scenes"]).is_dir() and Path(DYNAMIC[name]["dataset"]).is_file()
  ]


def missing_dynamic_asset_messages() -> list[str]:
  messages: list[str] = []
  for name, cfg in DYNAMIC.items():
    scenes = Path(cfg["scenes"])
    dataset = Path(cfg["dataset"])
    if not scenes.is_dir():
      messages.append(f"{name}: missing scenes directory ({scenes})")
    if not dataset.is_file():
      messages.append(f"{name}: missing CF dataset ({dataset})")
  return messages


def require_dynamic_corpora(*, require_all: bool = True) -> list[str]:
  """Ensure TwinWorld dynamic corpora exist; raise SystemExit with guidance if not.

  Public Hugging Face downloads include procedural *scenes* and the expanded-transfer
  subset, but not the full phase-2 / Isaac-HC CF datasets needed to rebuild core
  benchmark targets. Frozen paper metrics live under results/.
  """
  available = available_dynamic_corpora()
  if require_all and len(available) == len(DYNAMIC):
    return available
  if not require_all and available:
    return available

  missing = missing_dynamic_asset_messages()
  lines = [
    "Core TwinWorld dynamic corpora are not fully available in this checkout.",
    "Paper-frozen metrics are already shipped under results/.",
    "Public HF download covers procedural scenes + expanded transfer; full core",
    "re-run additionally needs twinworld_phase2 and Isaac-HC dynamic trees under data/.",
    "Missing:",
    *[f"  - {m}" for m in missing],
    "See README.md (Download artifacts / Reproduce the paper numbers).",
  ]
  raise SystemExit("\n".join(lines))


SCORERS = ("geometry_rule", "logistic", "random_forest", "mlp", "health_dyphygraph")
SCORER_ORDER = SCORERS
PAPER_NAMES = {
  "geometry_rule": "Geometry Rule",
  "logistic": "Logistic Regression",
  "random_forest": "Random Forest",
  "mlp": "MLP",
  "health_dyphygraph": "MedPhyGraph",
}
FINAL_LABEL = dict(PAPER_NAMES)
LABELS = {
  **PAPER_NAMES,
  "health_dyphygraph": "HealthDyPhyGraph",
}

ABLATION_STAGES = (
  "independent",
  "state_consistency",
  "union_rescoring",
  "direct_support_gate",
  "union_and_gate",
  "medphygraph",
)

MODE_LABEL = {
  "independent": "Independent (scorer only)",
  "state_consistency": "CF-SupportNet + State Consistency",
  "transition_aware": "MedPhyGraph (Union-Based Transition-Aware Consistency)",
}

SEED0_SHA256 = "e0b34529745399ecc5da5341ed7a162173611e12c8bd50dec121b0c575c5b789"
