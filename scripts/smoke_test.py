#!/usr/bin/env python3
"""CPU-only smoke test for the public MedPhyGraph release (<60s, no Isaac)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from medphygraph import (
  CFSupportNet,
  HealthScene,
  ModelConfig,
  PhysicalSceneGraph,
  apply_consistency,
  apply_transition_aware_consistency_v2,
  count_violations,
)
from medphygraph.paths import checkpoint_seed0, dataset_hard_json, procedural_scenes_root
from medphygraph.schema import example_rehab_scene

CHECKPOINT = checkpoint_seed0()
DATASET_JSON = dataset_hard_json()
PROCEDURAL_ROOT = procedural_scenes_root()


def main() -> int:
  start = time.monotonic()

  if not CHECKPOINT.is_file():
    raise SystemExit(f"Checkpoint not found: {CHECKPOINT}. Run scripts/download.py first.")
  if not DATASET_JSON.is_file():
    raise SystemExit(f"Dataset not found: {DATASET_JSON}. Run scripts/download.py first.")

  ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
  model = CFSupportNet(ModelConfig(hidden=int(ckpt.get("config", {}).get("hidden", 64))))
  model.load_state_dict(ckpt["model_state"])
  model.eval()

  raw = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
  sample = raw["samples"][0]
  features = np.asarray(sample["features_partial"]["1.0"], dtype=np.float32)
  with torch.no_grad():
    _, prob = model(torch.from_numpy(features).unsqueeze(0))
  probability = float(prob.item())
  if not np.isfinite(probability) or probability <= 0.0 or probability >= 1.0:
    raise RuntimeError(f"model forward produced invalid probability: {probability}")

  scene = example_rehab_scene()
  graph = PhysicalSceneGraph(scene=scene)
  graph.add_candidate("bed", "walker")
  graph.add_candidate("walker", "bed")
  graph.edges[("bed", "walker")].present = True
  graph.edges[("bed", "walker")].confidence = 0.4
  graph.edges[("walker", "bed")].present = True
  graph.edges[("walker", "bed")].confidence = 0.9
  apply_consistency(graph)
  if count_violations(graph)["cycle_rate_flag"] != 0:
    raise RuntimeError("State Consistency failed to remove cycle")

  graph_prev = PhysicalSceneGraph(scene=scene)
  graph_curr = PhysicalSceneGraph(scene=scene)
  graph_prev.add_candidate("bed", "walker")
  graph_curr.add_candidate("bed", "walker")
  graph_prev.edges[("bed", "walker")].present = True
  graph_prev.edges[("bed", "walker")].confidence = 0.9
  graph_curr.edges[("bed", "walker")].present = True
  graph_curr.edges[("bed", "walker")].confidence = 0.85
  transition_report = apply_transition_aware_consistency_v2(
    graph_curr,
    prev_confidence={("bed", "walker"): 0.9},
    curr_confidence={("bed", "walker"): 0.85},
    prev_refined=graph_prev,
  )
  if transition_report is None:
    raise RuntimeError("Union-Based Transition-Aware Consistency returned no report")

  if PROCEDURAL_ROOT.is_dir():
    for scene_dir in sorted(path for path in PROCEDURAL_ROOT.iterdir() if path.is_dir()):
      scene_json = scene_dir / "scene.json"
      if scene_json.is_file():
        HealthScene.from_dict(json.loads(scene_json.read_text(encoding="utf-8")))
        break

  elapsed = time.monotonic() - start
  if elapsed > 60.0:
    raise RuntimeError(f"smoke test exceeded 60s budget ({elapsed:.1f}s)")

  print("SMOKE TEST PASSED")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
