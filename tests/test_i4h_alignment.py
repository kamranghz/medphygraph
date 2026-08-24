"""Tests for i4h visual alignment (display-only layer)."""

from __future__ import annotations

from medphygraph.i4h_alignment import (
    CM_TO_M,
    compute_visual_xform,
    logical_center,
    logical_floor_anchor,
    placement_debug_row,
)


def test_therapy_bed_scale_maps_native_cm_bounds_to_logical_meters():
    visual = compute_visual_xform("therapy_bed", (2.0, 0.9, 0.5))
    # Native ~2.2m x 0.69m x 0.94m after cm→m → logical 2.0 x 0.9 x 0.5 m
    assert 0.85 < visual.scale[0] < 0.95
    assert 1.2 < visual.scale[1] < 1.4
    assert 0.5 < visual.scale[2] < 0.6


def test_therapy_bench_not_hundred_meter_wide():
    visual = compute_visual_xform("therapy_bench", (1.2, 0.4, 0.45))
    bmin, bmax = visual.visual_bbox_in_parent((1.2, 0.4, 0.45))
    assert abs((bmax[0] - bmin[0]) - 1.2) < 1e-6
    assert abs((bmax[1] - bmin[1]) - 0.4) < 1e-6
    assert bmin[2] == 0.0


def test_logical_center_unchanged():
    pos = (1.8, 0.8, 0.7)
    assert logical_center(pos) == pos


def test_floor_anchor_uses_logical_pose_not_visual():
    anchor = logical_floor_anchor((0.0, 0.5, 0.45), (2.0, 0.9, 0.5), "therapy_bed")
    assert anchor[0] == 0.0
    assert anchor[1] == 0.5
    assert abs(anchor[2] - 0.2) < 1e-6


def test_placement_debug_row_fields():
    row = placement_debug_row(
        "bench",
        "therapy_bench",
        (0.5, -1.8, 0.25),
        (1.2, 0.4, 0.45),
        "/fake/bench.usd",
    )
    assert row["entity"] == "bench"
    assert row["logical_position"] == [0.5, -1.8, 0.25]
    assert "visual_world_bbox_min" in row
    assert "scale" in row
    assert all(0.1 < s < 1.5 for s in row["scale"])
