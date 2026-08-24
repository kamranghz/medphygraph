"""Tests for tray prep Isaac visualization (cabinet → side table beside monitor)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medphygraph.tray_transfer_demo import (
    TRAY_SUPPLY_PLACEMENTS,
    TRAY_TRANSFER_ENTITY_TYPES,
    TRAY_TRANSFER_USD_REL,
    build_clinician_motion,
    build_tray_transfer_layout,
    build_tray_transfer_timeline,
    build_tray_translate_samples,
    resolve_clinician_usd,
    tray_transfer_demo_usda,
)


def test_tray_transfer_assets_defined():
    assert "side_table" in TRAY_TRANSFER_ENTITY_TYPES
    assert "instrument_tray" in TRAY_TRANSFER_USD_REL
    assert TRAY_TRANSFER_USD_REL["side_table"].endswith(".usd")


def test_timeline_single_transfer():
    tl = build_tray_transfer_timeline(fps=30)
    assert tl.end_frame == 360
    assert tl.duration_seconds == pytest.approx(12.03, abs=0.05)
    assert tl.approach_end < tl.hold_start_end < tl.lift_end
    assert tl.lift_end < tl.travel_end < tl.lower_end < tl.hold_place_end


def test_layout_no_floating_supports():
    layout = build_tray_transfer_layout()
    assert len(layout.props) == 3
    cabinet = layout.prop_by_type("cabinet")
    side_table = layout.prop_by_type("side_table")
    monitor = layout.prop_by_type("monitor_cart")

    assert cabinet.composed.bmin[2] == pytest.approx(0.0, abs=1e-6)
    assert side_table.composed.bmin[2] == pytest.approx(0.0, abs=1e-6)
    assert monitor.composed.bmin[2] == pytest.approx(0.0, abs=1e-6)

    gap = monitor.composed.bmin[0] - side_table.composed.bmax[0]
    assert 0.05 < gap < 0.25

    assert layout.tray_start_xyz[2] == pytest.approx(
        cabinet.tray_on_surface_world_xyz()[2], abs=1e-6
    )
    assert layout.tray_end_xyz[2] == pytest.approx(
        side_table.tray_on_surface_world_xyz()[2], abs=1e-6
    )


def test_tray_end_beside_monitor():
    layout = build_tray_transfer_layout()
    monitor = layout.prop_by_type("monitor_cart")
    end = layout.tray_end_xyz
    assert end[0] < monitor.composed.bmin[0]
    assert end[0] > layout.prop_by_type("side_table").composed.bmin[0]


def test_tray_animation_stable():
    layout = build_tray_transfer_layout()
    tl = build_tray_transfer_timeline()
    samples = build_tray_translate_samples(layout, tl)
    for i in range(3):
        assert samples[0][i] == pytest.approx(layout.tray_start_xyz[i], abs=1e-4)
        assert samples[tl.end_frame][i] == pytest.approx(layout.tray_end_xyz[i], abs=1e-4)
    assert samples[tl.lift_end][2] > samples[0][2]
    assert samples[tl.travel_end][0] > samples[tl.lift_end][0]


def test_usda_structure(tmp_path):
    usda = tray_transfer_demo_usda(i4h_root=tmp_path)
    assert "PhysicsScene" not in usda
    assert 'def Xform "Graph"' not in usda
    assert "Room_floor_accent" not in usda
    assert 'def Camera "DemoCamera"' in usda
    assert 'def Camera "HeroCamera"' in usda
    assert "RectLight" in usda
    assert "OverheadLeft" in usda
    assert 'def Xform "TRAY_ASSEMBLY"' in usda
    assert "PROP_side_table" in usda
    assert "PROP_cabinet" in usda
    assert "PROP_monitor_cart" in usda
    assert "Room_dado_n" in usda
    assert "Supply_syringe" in usda or "TrayBase" in usda
    tray_idx = usda.index('def Xform "TRAY_ASSEMBLY"')
    for prop in ("cabinet", "monitor_cart", "side_table"):
        block = usda[usda.index(f"PROP_{prop}") : tray_idx]
        assert "timeSamples" not in block


def test_room_is_compact_procedure_size():
    layout = build_tray_transfer_layout()
    assert layout.room_size == (5.5, 4.5, 3.0)
    bed = [b for b in layout.background if b.prop_id == "BG_surgical_table"][0]
    assert bed.parent_xyz[0] > 0.5
    assert bed.parent_xyz[1] > 0.8


def test_dome_not_overexposed():
    usda = tray_transfer_demo_usda(i4h_root=Path("."))
    assert "float inputs:intensity = 1400" in usda
    assert "float inputs:intensity = 4500" not in usda
    assert "ActionFill" in usda
    # Key is strong enough to read the room in RTX
    assert "float inputs:intensity = 6000" in usda


# --- Camera geometry: both shots must actually contain their story points,
# and both eye positions must stay inside the room walls. Written after
# discovering the previous hand-picked camera offsets did neither (verified
# by computing the actual angular offsets: tray_start was 39.6deg off-axis
# against an 18.5deg limit). ---

def _project_angles(eye, look_at, pt, eye_z_unused=None):
    import math

    def sub(a, b):
        return tuple(a[i] - b[i] for i in range(3))

    def norm(v):
        n = math.sqrt(sum(c * c for c in v))
        return tuple(c / n for c in v)

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def dot(a, b):
        return sum(a[i] * b[i] for i in range(3))

    forward = norm(sub(look_at, eye))
    right = norm(cross(forward, (0, 0, 1)))
    up = cross(right, forward)
    v = sub(pt, eye)
    vx, vy, vz = dot(v, right), dot(v, up), dot(v, forward)
    assert vz > 0, "point must be in front of the camera, not behind it"
    return math.degrees(math.atan2(vx, vz)), math.degrees(math.atan2(vy, vz))


def _fov_half_angles(focal_length, h_ap, v_ap):
    import math

    return (
        math.degrees(2 * math.atan(h_ap / (2 * focal_length))) / 2,
        math.degrees(2 * math.atan(v_ap / (2 * focal_length))) / 2,
    )


def test_demo_camera_contains_tray_start_and_end():
    from medphygraph.tray_transfer_demo import _solve_camera_for_points

    layout = build_tray_transfer_layout()
    p0, p1 = layout.tray_start_xyz, layout.tray_end_xyz
    eye, look_at = _solve_camera_for_points(
        [p0, p1], eye_z=1.35, focal_length=16.0,
        horizontal_aperture=20.955, vertical_aperture=18.0, azimuth_deg=30.0, margin=0.9,
    )
    h_half, v_half = _fov_half_angles(16.0, 20.955, 18.0)
    for pt in (p0, p1):
        h_ang, v_ang = _project_angles(eye, look_at, pt)
        assert abs(h_ang) < h_half, f"point {pt} outside horizontal FOV: {h_ang:.1f} vs limit {h_half:.1f}"
        assert abs(v_ang) < v_half, f"point {pt} outside vertical FOV: {v_ang:.1f} vs limit {v_half:.1f}"


def test_hero_camera_contains_tray_end_and_monitor():
    from medphygraph.tray_transfer_demo import _solve_camera_for_points

    layout = build_tray_transfer_layout()
    p1 = layout.tray_end_xyz
    monitor = layout.prop_by_type("monitor_cart")
    monitor_center = (*monitor.composed.center_xy, p1[2])
    eye, look_at = _solve_camera_for_points(
        [p1, monitor_center], eye_z=1.35, focal_length=20.0,
        horizontal_aperture=20.955, vertical_aperture=18.0, azimuth_deg=20.0, margin=0.9,
    )
    h_half, v_half = _fov_half_angles(20.0, 20.955, 18.0)
    for pt in (p1, monitor_center):
        h_ang, v_ang = _project_angles(eye, look_at, pt)
        assert abs(h_ang) < h_half
        assert abs(v_ang) < v_half


def test_both_cameras_stay_inside_room_walls():
    from medphygraph.tray_transfer_demo import _solve_camera_for_points

    layout = build_tray_transfer_layout()
    p0, p1 = layout.tray_start_xyz, layout.tray_end_xyz
    monitor = layout.prop_by_type("monitor_cart")
    monitor_center = (*monitor.composed.center_xy, p1[2])
    wall_clear = 0.15
    lx, ly, _ = layout.room_size
    xlim = lx / 2 - wall_clear
    ylim = ly / 2 - wall_clear

    demo_eye, _ = _solve_camera_for_points(
        [p0, p1], eye_z=1.35, focal_length=16.0,
        horizontal_aperture=20.955, vertical_aperture=18.0, azimuth_deg=30.0, margin=0.9,
    )
    hero_eye, _ = _solve_camera_for_points(
        [p1, monitor_center], eye_z=1.35, focal_length=20.0,
        horizontal_aperture=20.955, vertical_aperture=18.0, azimuth_deg=20.0, margin=0.9,
    )
    for name, eye in (("DemoCamera", demo_eye), ("HeroCamera", hero_eye)):
        assert -xlim <= eye[0] <= xlim, f"{name} eye X outside room: {eye}"
        assert -ylim <= eye[1] <= ylim, f"{name} eye Y outside room: {eye}"


# --- Materials: the room shell previously had displayColor only, with no
# real PBR shader for the light rig to actually interact with -- the
# documented ("Flat displayColor walls/floors") root cause. ---

def test_room_shell_has_real_pbr_materials_not_just_displaycolor():
    usda = tray_transfer_demo_usda(i4h_root=Path("/nonexistent"))
    assert usda.count('def Material "') == 8  # 6 room-shell + Mat_FallbackProp + Mat_MonitorFallback
    assert "UsdPreviewSurface" in usda
    for mat in ("Mat_Floor", "Mat_Wall", "Mat_WallBack", "Mat_Dado", "Mat_Ceiling", "Mat_Trim"):
        assert f'def Material "{mat}"' in usda
        assert f"</World/Looks/{mat}>" in usda, f"{mat} defined but never bound to a room-shell prim"


def test_monitor_cart_has_fallback_material_binding(tmp_path):
    # Fallback binding only matters (and only applies) when the real i4h
    # asset resolves -- that's the actual failure mode it's defending
    # against. Build a fake but genuinely-resolving asset tree so this test
    # exercises that code path instead of the unrelated no-asset-at-all cube.
    from medphygraph.tray_transfer_demo import TRAY_TRANSFER_USD_REL

    rel = TRAY_TRANSFER_USD_REL["monitor_cart"]
    fake_asset = tmp_path / rel
    fake_asset.parent.mkdir(parents=True, exist_ok=True)
    fake_asset.write_text('#usda 1.0\ndef Xform "Fake" {}\n')

    usda = tray_transfer_demo_usda(i4h_root=tmp_path)
    assert 'def Material "Mat_MonitorFallback"' in usda
    idx = usda.find('health:entity_type = "monitor_cart"')
    assert idx != -1, "monitor_cart should use the real-asset path once its USD resolves"
    nearby = usda[idx : idx + 700]
    assert "Mat_MonitorFallback" in nearby