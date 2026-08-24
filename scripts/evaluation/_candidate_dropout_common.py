#!/usr/bin/env python3
"""Candidate-dropout stress-test helpers (read-only).

Reads frozen expanded-transfer artifacts from disk and reconstructs
features/probabilities deterministically in memory, without regenerating scenes,
GT labels, or re-fitting models.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from evaluation._config import THR
from evaluation._shared import CANDIDATE_DROPOUT_OUT, sha256_file
from evaluation._stage6_paths import (
  STAGE6_DIR,
  STAGE6_TARGETS,
  STAGE6_MANIFEST,
  STAGE6_SCENES,
)

import numpy as np

from medphygraph.schema import HealthScene  # noqa: E402
from medphygraph.scene_graph import PhysicalSceneGraph  # noqa: E402

OUT_DIR = CANDIDATE_DROPOUT_OUT

DROPOUT_RATES = (0.0, 0.10, 0.20, 0.30, 0.50)
DROPOUT_SEEDS = (0, 1, 2, 3, 4)


def load_stage6_targets() -> list[dict[str, Any]]:
    return json.loads(STAGE6_TARGETS.read_text(encoding="utf-8"))


def state_scene_dir(corpus: str, case_id: str, state_id: str) -> Path:
    return STAGE6_SCENES / corpus / case_id / state_id


def load_state_context(corpus: str, case_id: str, state_id: str) -> dict[str, Any]:
    """Read-only load of one expanded-transfer generated state (scene + native candidates + GT)."""
    sdir = state_scene_dir(corpus, case_id, state_id)
    scene = HealthScene.from_dict(json.loads((sdir / "scene.json").read_text(encoding="utf-8")))
    g_init = PhysicalSceneGraph.load(sdir / "graph_initial.json")
    native_cand = set(g_init.edges.keys())
    gt_graph = PhysicalSceneGraph.load(sdir / "graph_gt.json")
    gt = {(e.subject_id, e.host_id) for e in gt_graph.edges.values() if e.is_gt is True}
    ent = {e.entity_id for e in scene.entities if e.entity_type != "zone"}
    return {"scene": scene, "native_cand": native_cand, "gt": gt, "ent": ent}


def stable_seed_int(*parts: Any) -> int:
    material = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def random_dropout_survivors(edges: set[tuple[str, str]], *, rate: float, seed: int, state_id: str) -> set[tuple[str, str]]:
    """Deterministically drop `rate` fraction of candidate edges for one state.

    Reproducible given (rate, seed, state_id, sorted edge list): a fresh RNG is
    seeded from a SHA-256 digest of (seed, state_id), then each candidate edge
    (visited in sorted order) draws one uniform random number and is dropped
    if that draw is < rate. rate=0.0 is a no-op by construction (survivors ==
    edges) regardless of seed.
    """
    if rate <= 0.0:
        return set(edges)
    rng = np.random.default_rng(stable_seed_int(seed, state_id))
    survivors = set()
    for e in sorted(edges):
        r = rng.random()
        if r >= rate:
            survivors.add(e)
    return survivors


def destination_targeted_survivors(edges: set[tuple[str, str]], *, subject: str, dest: str) -> set[tuple[str, str]]:
    """Stress Test A: remove exactly the (subject, dest) edge, nothing else."""
    return {e for e in edges if e != (subject, dest)}
