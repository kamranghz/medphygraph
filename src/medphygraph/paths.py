"""Repository path resolution — single source of truth for MedPhyGraph (Phase A).

Layout (repo-relative defaults, all overridable via environment variables):

  DATA_ROOT      data/                 MEDPHYGRAPH_DATA_ROOT
  CKPT_ROOT      checkpoints/          MEDPHYGRAPH_CKPT_ROOT
  RUNS_ROOT      runs/                 MEDPHYGRAPH_RUNS_ROOT
  RESULTS_ROOT   results/              MEDPHYGRAPH_RESULTS_ROOT

Do not hard-code outputs/dyphygraph_health or outputs/reviewer_response paths
in new code — route through this module.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _root() -> Path:
    override = os.environ.get("MEDPHYGRAPH_REPO_ROOT")
    return Path(override).resolve() if override else REPO_ROOT


def data_root() -> Path:
    """Root for downloaded datasets and scenes (default: ``data/``)."""
    override = os.environ.get("MEDPHYGRAPH_DATA_ROOT")
    return Path(override).resolve() if override else (_root() / "data")


def ckpt_root() -> Path:
    """Root for downloaded checkpoints (default: ``checkpoints/``)."""
    override = os.environ.get("MEDPHYGRAPH_CKPT_ROOT")
    return Path(override).resolve() if override else (_root() / "checkpoints")


def runs_root() -> Path:
    """Root for new evaluation run outputs (default: ``runs/``)."""
    override = os.environ.get("MEDPHYGRAPH_RUNS_ROOT")
    return Path(override).resolve() if override else (_root() / "runs")


def results_root() -> Path:
    """Frozen paper results (default: ``results/``)."""
    override = os.environ.get("MEDPHYGRAPH_RESULTS_ROOT")
    return Path(override).resolve() if override else (_root() / "results")


def dataset_hard_dir() -> Path:
    return data_root() / "training"


def dataset_hard_json() -> Path:
    return dataset_hard_dir() / "dataset.json"


def dataset_hard_split() -> Path:
    return dataset_hard_dir() / "split.json"


def checkpoint_seed0() -> Path:
    return ckpt_root() / "health_dyphygraph_r1.0_seed0.pt"


def multiseed_checkpoints_dir() -> Path:
    return ckpt_root() / "multiseed"


def multiseed_checkpoint(seed: int) -> Path:
    return multiseed_checkpoints_dir() / f"health_dyphygraph_r1.0_seed{seed}.pt"


def procedural_scenes_root() -> Path:
    return data_root() / "procedural_scenes"


def expanded_transfer_root() -> Path:
    """Downloaded expanded-transfer suite inputs (targets, scenes, manifests)."""
    return data_root() / "expanded_transfer"


def twinworld_phase2_dataset() -> Path:
    return data_root() / "twinworld_phase2" / "dataset.json"


def twinworld_phase2_scenes() -> Path:
    return data_root() / "twinworld_phase2" / "scenes"


def twinworld_i4h_dynamic_dataset() -> Path:
    return data_root() / "twinworld_i4h_dynamic" / "dataset.json"


def twinworld_i4h_dynamic_scenes() -> Path:
    return data_root() / "twinworld_i4h_dynamic" / "scenes"


def results_frozen_root() -> Path:
    return results_root()


def verification_paper_root() -> Path:
    """Alias for frozen paper results (Phase A rename of verification/paper)."""
    return results_root()


def git_sha(repo: Path | None = None) -> str:
    root = repo or _root()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def new_run_dir(script_name: str, *, protocol_id: str | None = None, seed: int | None = None) -> Path:
    """Create ``runs/<script_name>/<utc-timestamp>/`` with metadata.json."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = runs_root() / script_name / stamp
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "script": script_name,
        "protocol_id": protocol_id,
        "seed": seed,
        "git_sha": git_sha(),
        "created_utc": stamp,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out
