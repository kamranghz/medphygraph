"""Tests for Health predictor, consistency, twin write-back."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from medphygraph.consistency import apply_consistency, count_violations
from medphygraph.model import HealthDyPhyGraph, ModelConfig
from medphygraph.schema import example_rehab_scene
from medphygraph.scene_graph import PhysicalSceneGraph
from medphygraph.twin_store import write_twin


def test_model_forward_and_param_count() -> None:
    m = HealthDyPhyGraph(ModelConfig(hidden=32))
    x = torch.randn(4, 20, 9)
    x[:, :, -1] = 1.0
    logits, probs = m(x)
    assert logits.shape == (4,)
    assert probs.shape == (4,)
    assert 0.0 <= float(probs.min()) <= float(probs.max()) <= 1.0
    assert m.count_parameters() > 1000


def test_consistency_removes_cycle() -> None:
    scene = example_rehab_scene()
    g = PhysicalSceneGraph(scene=scene)
    g.add_candidate("bed", "walker")
    g.add_candidate("walker", "bed")
    g.edges[("bed", "walker")].present = True
    g.edges[("bed", "walker")].confidence = 0.4
    g.edges[("walker", "bed")].present = True
    g.edges[("walker", "bed")].confidence = 0.9
    before = count_violations(g)["cycle_rate_flag"]
    assert before == 1
    rep = apply_consistency(g)
    assert count_violations(g)["cycle_rate_flag"] == 0
    assert rep.n_cycles_broken >= 1


def test_consistency_one_primary_support() -> None:
    scene = example_rehab_scene()
    g = PhysicalSceneGraph(scene=scene)
    g.add_candidate("bed", "floor")
    g.add_candidate("bed", "walker")
    g.edges[("bed", "floor")].present = True
    g.edges[("bed", "floor")].confidence = 0.9
    g.edges[("bed", "walker")].present = True
    g.edges[("bed", "walker")].confidence = 0.2
    apply_consistency(g)
    assert g.edges[("bed", "floor")].present is True
    assert g.edges[("bed", "walker")].present is False


def test_twin_write_back(tmp_path: Path) -> None:
    scene = example_rehab_scene()
    g = PhysicalSceneGraph(scene=scene)
    g.add_candidate("bed", "floor")
    g.edges[("bed", "floor")].present = True
    g.edges[("bed", "floor")].confidence = 0.95
    g.edges[("bed", "floor")].evidence_source = "test"
    res = write_twin(g, out_dir=tmp_path / "twin", evidence_source="test")
    assert res.twin_path.exists()
    assert res.provenance_path.exists()
    assert (tmp_path / "twin" / "twin_latest.json").exists()
    assert res.version >= 1
