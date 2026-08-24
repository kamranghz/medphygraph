"""One-shot path configuration for running evaluation entry points from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path

_CONFIGURED = False


def configure_repo_paths() -> Path:
  """Ensure ``src`` and ``scripts`` are importable; return repository root."""
  global _CONFIGURED
  root = Path(__file__).resolve().parents[2]
  if not _CONFIGURED:
    for entry in (root / "src", root / "scripts"):
      path = str(entry)
      if path not in sys.path:
        sys.path.insert(0, path)
    _CONFIGURED = True
  return root
