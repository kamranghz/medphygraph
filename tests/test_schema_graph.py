"""Unit tests for DyPhyGraph-Health schema and PhysicalSceneGraph (P0-1, P0-2)."""

from __future__ import annotations

import json
from pathlib import Path

from medphygraph.isaac_sim import resolve_health_isaac_config
from medphygraph.schema import (
    REQUIRED_OBJECT_INVENTORY,
    HealthEntity,
    example_rehab_scene,
)
from medphygraph.scene_graph import (
    PhysicalSceneGraph,
    demo_graph_with_candidates,
)


def test_example_scene_has_required_types() -> None:
    scene = example_rehab_scene()
    types = {e.entity_type for e in scene.entities}
    for req in REQUIRED_OBJECT_INVENTORY:
        assert req in types, f"missing {req}"
    assert len(scene.entities) >= 8


def test_structural_not_movable() -> None:
    e = HealthEntity("floor", "floor", (0, 0, 0), (1, 1, 0.05), parent_zone="z")
    assert e.movable is False
    assert e.is_structural is True


def test_scene_roundtrip(tmp_path: Path) -> None:
    scene = example_rehab_scene()
    p = tmp_path / "scene.json"
    p.write_text(json.dumps(scene.to_dict()), encoding="utf-8")
    loaded = type(scene).from_dict(json.loads(p.read_text(encoding="utf-8")))
    assert loaded.scene_id == scene.scene_id
    assert len(loaded.entities) == len(scene.entities)


def test_physical_scene_graph_candidates_and_gt() -> None:
    g = demo_graph_with_candidates()
    assert len(g.candidate_edges()) >= 4
    assert any(e.is_gt for e in g.edges.values())
    assert g.edges[("bed", "floor")].is_gt is True
    assert g.edges[("walker", "bed")].is_gt is False


def test_graph_diff_add_remove(tmp_path: Path) -> None:
    g = demo_graph_with_candidates()
    for e in g.edges.values():
        e.present = False
    diff = g.apply_probabilities(
        {("bed", "floor"): 0.9, ("walker", "bed"): 0.1},
        evidence_source="test",
    )
    assert g.version == 1
    assert g.edges[("bed", "floor")].present is True
    assert g.edges[("walker", "bed")].present is False
    actions = {op["action"] for op in diff.operations}
    assert "ADD" in actions
    g.save(tmp_path / "g.json")
    g2 = PhysicalSceneGraph.load(tmp_path / "g.json")
    assert g2.edges[("bed", "floor")].present is True


def test_isaac_sim_resolves_local_install() -> None:
    cfg = resolve_health_isaac_config()
    # On this machine the standalone install exists; still assert shape
    assert cfg.notes
    if cfg.available:
        assert cfg.root is not None
        assert cfg.python_bat is not None


def test_geometry_candidates_cover_gt() -> None:
    from medphygraph.candidates import (
        candidate_recall_vs_gt,
        generate_candidates,
    )
    from medphygraph.scene_graph import demo_graph_with_candidates

    gt = demo_graph_with_candidates()
    cand, stats = generate_candidates(gt.scene)
    assert stats.n_candidates >= 4
    recall = candidate_recall_vs_gt(cand, gt)
    assert recall >= 0.8
