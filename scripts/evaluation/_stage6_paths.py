from __future__ import annotations

from pathlib import Path

from medphygraph.paths import expanded_transfer_root

# Shared filesystem locations for the frozen expanded-transfer suite used by the
# candidate-dropout and component-analysis public evaluation scripts.

STAGE6_DIR = expanded_transfer_root()
STAGE6_TARGETS = STAGE6_DIR / "targets.json"
STAGE6_MANIFEST = STAGE6_DIR / "predeclared_manifest.json"
STAGE6_SCENES = STAGE6_DIR / "scenes"
STAGE6_PER_CASE = STAGE6_DIR / "per_case_raw.json"

__all__ = [
  "STAGE6_DIR",
  "STAGE6_TARGETS",
  "STAGE6_MANIFEST",
  "STAGE6_SCENES",
  "STAGE6_PER_CASE",
]
