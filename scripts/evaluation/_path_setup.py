"""Bootstrap repository import paths before the evaluate package is imported."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def configure_repo_paths() -> Path:
  for entry in (_REPO_ROOT / "scripts", _REPO_ROOT / "src"):
    path = str(entry)
    if path not in sys.path:
      sys.path.insert(0, path)
  return _REPO_ROOT


configure_repo_paths()
