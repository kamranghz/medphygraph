"""Tests for Health counterfactual labeling, analytic physics, USD, features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medphygraph.analytic_physics import run_counterfactual_pair
from medphygraph.features import OBSERVATION_RATIOS, apply_incompleteness, trajectory_features
from medphygraph.labeling import GT_DROP_M, label_from_rollout
from medphygraph.schema import example_rehab_scene
from medphygraph.usd_authoring import write_scene_usda


def test_label_positive_on_drop() -> None:
    zf = np.ones(30) * 1.0
    zc = np.linspace(1.0, 0.2, 30)
    d = label_from_rollout(
        subject_id="a",
        host_id="b",
        z_factual=zf,
        z_counterfactual=zc,
        contact_factual=np.ones(30),
        contact_counterfactual=np.zeros(30),
    )
    assert d.positive is True
    assert d.drop_m >= GT_DROP_M


def test_label_negative_when_stable() -> None:
    z = np.ones(30) * 0.5
    d = label_from_rollout(
        subject_id="a",
        host_id="b",
        z_factual=z,
        z_counterfactual=z.copy(),
        contact_factual=np.ones(30),
        contact_counterfactual=np.ones(30),
    )
    assert d.positive is False


def test_analytic_floor_removal_drops_bed() -> None:
    scene = example_rehab_scene()
    result = run_counterfactual_pair(scene, subject_id="bed", host_id="floor", n_frames=45)
    assert result["label"]["positive"] is True


def test_analytic_false_bed_host_no_fall() -> None:
    scene = example_rehab_scene()
    result = run_counterfactual_pair(scene, subject_id="walker", host_id="bed", n_frames=45)
    # walker still has floor; removing bed should not make walker fall
    assert result["label"]["positive"] is False


def test_wall_rail_positive_when_wall_removed() -> None:
    """Hidden wall support: floor must not cancel rail→wall GT."""
    from medphygraph.hard_scenes import build_hard_scene

    scene, _ = build_hard_scene(0, seed=0)
    wall = run_counterfactual_pair(scene, subject_id="rail", host_id="wall_n", n_frames=45)
    floor = run_counterfactual_pair(scene, subject_id="rail", host_id="floor", n_frames=45)
    assert wall["label"]["positive"] is True, wall["label"]
    assert floor["label"]["positive"] is False, floor["label"]
    assert wall["counterfactual"]["meta"]["will_fall"] is True


def test_ceiling_lift_positive_when_ceiling_removed() -> None:
    from medphygraph.hard_scenes import build_hard_scene

    scene, _ = build_hard_scene(0, seed=0)
    ceil = run_counterfactual_pair(scene, subject_id="lift", host_id="ceiling", n_frames=45)
    floor = run_counterfactual_pair(scene, subject_id="lift", host_id="floor", n_frames=45)
    assert ceil["label"]["positive"] is True, ceil["label"]
    assert floor["label"]["positive"] is False, floor["label"]


def test_direct_furniture_stack_monitor_on_bed_is_positive() -> None:
    """MedPhyGraph direct load-bearing support semantics: furniture is a valid direct host.

    monitor rests directly on bed -> monitor->bed is the direct load-bearing
    edge; monitor->floor is ancestral-only and must NOT be positive (floor is
    not retained through the intermediate bed). See
    repair_sr1_support_semantics/CANONICAL_DEFINITION.md.
    """
    from medphygraph.hard_scenes import build_hard_scene

    scene, _ = build_hard_scene(0, seed=0)
    bed = run_counterfactual_pair(scene, subject_id="monitor", host_id="bed", n_frames=45)
    fl = run_counterfactual_pair(scene, subject_id="monitor", host_id="floor", n_frames=45)
    assert bed["label"]["positive"] is True, bed["label"]
    assert fl["label"]["positive"] is False, fl["label"]


def test_multi_support_both_anchors_positive() -> None:
    from medphygraph.hard_scenes import build_hard_scene

    scene, _ = build_hard_scene(0, seed=0)
    fl = run_counterfactual_pair(scene, subject_id="cart", host_id="floor", n_frames=45)
    wall = run_counterfactual_pair(scene, subject_id="cart", host_id="wall_w", n_frames=45)
    assert fl["label"]["positive"] is True, fl["label"]
    assert wall["label"]["positive"] is True, wall["label"]


def test_incompleteness_ratios() -> None:
    feats = trajectory_features(
        positions_subject=np.linspace([0, 0, 1], [0, 0, 0.2], 40),
        contact=np.zeros(40),
        host_removed=True,
        geom_xy_sep=0.1,
        geom_vertical_gap=0.0,
    )
    for r in OBSERVATION_RATIOS:
        part = apply_incompleteness(feats, ratio=r, seed=0)
        assert part.shape == feats.shape
        assert part[:, -1].sum() >= 1


def test_usda_written(tmp_path: Path) -> None:
    scene = example_rehab_scene()
    path = write_scene_usda(scene, tmp_path / "s.usda")
    text = path.read_text(encoding="utf-8")
    assert "#usda 1.0" in text
    assert 'def Cube "bed"' in text
    assert "PhysicsScene" in text


def test_gui_usda_has_camera_and_color(tmp_path: Path) -> None:
    scene = example_rehab_scene()
    path = write_scene_usda(scene, tmp_path / "gui.usda", gui=True)
    text = path.read_text(encoding="utf-8")
    assert 'def Camera "DemoCamera"' in text
    assert 'def DistantLight "KeyLight"' in text
    assert "primvars:displayColor" in text


def test_stylized_gui_assets_have_parts() -> None:
    from medphygraph.gui_assets import iter_entity_world_parts, room_shell_parts

    shell = room_shell_parts()
    assert len(shell) >= 6
    bed = iter_entity_world_parts("bed", "therapy_bed", (0, 0, 0.45), (2.0, 0.9, 0.5))
    assert len(bed) >= 4
    assert any("mattress" in p[0] for p in bed)


def test_i4h_visual_map_resolves_when_installed() -> None:
    from medphygraph.i4h_visuals import inventory, usd_for_entity_type

    inv = inventory()
    if inv.get("root") is None:
        return  # optional dependency
    bed = usd_for_entity_type("therapy_bed")
    chair = usd_for_entity_type("wheelchair")
    assert bed is not None and bed.is_file()
    assert chair is not None and chair.is_file()
