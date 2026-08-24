"""Pytest path configuration for MedPhyGraph."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "scripts"):
  path = str(entry)
  if path not in sys.path:
    sys.path.insert(0, path)
