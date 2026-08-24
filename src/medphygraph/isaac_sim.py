"""Isaac Sim path resolution for DyPhyGraph-Health (standalone, not Isaac Lab)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ISAAC_SIM_ROOT = Path(r"C:\isaac-sim-standalone-6.0.1-windows-x86_64")
ENV_ISAAC_SIM_ROOT = "ISAAC_SIM_ROOT"


@dataclass(frozen=True)
class HealthIsaacConfig:
    root: Path | None
    python_bat: Path | None
    isaac_sim_bat: Path | None
    available: bool
    notes: str


def resolve_health_isaac_config() -> HealthIsaacConfig:
    """Prefer ISAAC_SIM_ROOT; fall back to known local standalone install if present."""
    raw = os.environ.get(ENV_ISAAC_SIM_ROOT, "").strip()
    root = Path(raw) if raw else None
    if root is None and DEFAULT_ISAAC_SIM_ROOT.is_dir():
        root = DEFAULT_ISAAC_SIM_ROOT
    if root is None or not root.is_dir():
        return HealthIsaacConfig(
            root=None,
            python_bat=None,
            isaac_sim_bat=None,
            available=False,
            notes="Isaac Sim root not found. Set ISAAC_SIM_ROOT.",
        )
    py = root / "python.bat"
    bat = root / "isaac-sim.bat"
    ok = py.is_file() and bat.is_file()
    return HealthIsaacConfig(
        root=root,
        python_bat=py if py.is_file() else None,
        isaac_sim_bat=bat if bat.is_file() else None,
        available=ok,
        notes="Isaac Sim standalone for Health scene generation (not Isaac Lab).",
    )
