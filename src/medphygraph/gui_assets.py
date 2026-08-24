"""Stylized multi-prim visual proxies for Isaac GUI (display only; CF uses AABB)."""

from __future__ import annotations

from typing import Callable

# Local offsets are in entity frame: origin at entity pose COM, extents ≈ size_xyz.
# Each part: (suffix, dx, dy, dz, sx, sy, sz, rgb)

Part = tuple[str, float, float, float, float, float, float, tuple[float, float, float]]


def _parts_for(entity_type: str, size_xyz: tuple[float, float, float]) -> list[Part]:
    sx, sy, sz = size_xyz
    if entity_type == "therapy_bed":
        return [
            ("frame", 0.0, 0.0, -0.12 * sz, sx * 0.98, sy * 0.95, sz * 0.35, (0.55, 0.58, 0.60)),
            ("mattress", 0.0, 0.0, 0.12 * sz, sx * 0.92, sy * 0.88, sz * 0.45, (0.25, 0.62, 0.66)),
            ("headboard", -0.45 * sx, 0.0, 0.25 * sz, sx * 0.08, sy * 0.95, sz * 0.9, (0.45, 0.48, 0.52)),
            ("leg_fl", -0.4 * sx, -0.35 * sy, -0.4 * sz, 0.08, 0.08, 0.35 * sz, (0.35, 0.35, 0.38)),
            ("leg_fr", -0.4 * sx, 0.35 * sy, -0.4 * sz, 0.08, 0.08, 0.35 * sz, (0.35, 0.35, 0.38)),
            ("leg_bl", 0.4 * sx, -0.35 * sy, -0.4 * sz, 0.08, 0.08, 0.35 * sz, (0.35, 0.35, 0.38)),
            ("leg_br", 0.4 * sx, 0.35 * sy, -0.4 * sz, 0.08, 0.08, 0.35 * sz, (0.35, 0.35, 0.38)),
        ]
    if entity_type == "walker":
        h, w, d = sz, sy, sx
        metal = (0.75, 0.78, 0.80)
        grip = (0.15, 0.15, 0.16)
        return [
            ("post_fl", -0.35 * d, -0.4 * w, 0.0, 0.05, 0.05, h * 0.95, metal),
            ("post_fr", -0.35 * d, 0.4 * w, 0.0, 0.05, 0.05, h * 0.95, metal),
            ("post_bl", 0.35 * d, -0.4 * w, 0.0, 0.05, 0.05, h * 0.85, metal),
            ("post_br", 0.35 * d, 0.4 * w, 0.0, 0.05, 0.05, h * 0.85, metal),
            ("top_l", 0.0, -0.4 * w, 0.35 * h, d * 0.75, 0.05, 0.05, grip),
            ("top_r", 0.0, 0.4 * w, 0.35 * h, d * 0.75, 0.05, 0.05, grip),
            ("front", -0.35 * d, 0.0, 0.25 * h, 0.05, w * 0.75, 0.05, metal),
            ("cross", 0.0, 0.0, -0.15 * h, d * 0.7, 0.05, 0.05, metal),
        ]
    if entity_type == "wheelchair":
        seat = (0.18, 0.42, 0.72)
        frame = (0.25, 0.25, 0.28)
        tire = (0.08, 0.08, 0.08)
        return [
            ("seat", 0.0, 0.0, -0.05 * sz, sx * 0.7, sy * 0.65, sz * 0.12, seat),
            ("back", -0.32 * sx, 0.0, 0.25 * sz, sx * 0.1, sy * 0.65, sz * 0.55, seat),
            ("base", 0.0, 0.0, -0.3 * sz, sx * 0.55, sy * 0.45, sz * 0.12, frame),
            ("wheel_l", 0.05 * sx, -0.42 * sy, -0.15 * sz, 0.12, 0.08, 0.45 * sz, tire),
            ("wheel_r", 0.05 * sx, 0.42 * sy, -0.15 * sz, 0.12, 0.08, 0.45 * sz, tire),
            ("foot", 0.35 * sx, 0.0, -0.35 * sz, sx * 0.25, sy * 0.4, 0.05, frame),
        ]
    if entity_type == "iv_pole":
        metal = (0.78, 0.80, 0.82)
        return [
            ("base", 0.0, 0.0, -0.45 * sz, sx * 0.9, sy * 0.9, 0.06, (0.35, 0.35, 0.38)),
            ("pole", 0.0, 0.0, 0.0, 0.06, 0.06, sz * 0.95, metal),
            ("hook", 0.0, 0.0, 0.42 * sz, sx * 0.7, 0.04, 0.04, metal),
            ("bag", 0.25 * sx, 0.0, 0.25 * sz, 0.12, 0.08, 0.2, (0.55, 0.75, 0.85)),
        ]
    if entity_type == "monitor_cart":
        return [
            ("base", 0.0, 0.0, -0.4 * sz, sx * 0.85, sy * 0.85, 0.08, (0.25, 0.25, 0.28)),
            ("column", 0.0, 0.0, -0.05 * sz, 0.1, 0.1, sz * 0.7, (0.4, 0.4, 0.42)),
            ("screen", 0.0, 0.0, 0.35 * sz, sx * 0.9, 0.08, sz * 0.35, (0.1, 0.12, 0.14)),
            ("bezel", 0.0, 0.05 * sy, 0.35 * sz, sx * 0.7, 0.02, sz * 0.25, (0.2, 0.55, 0.35)),
        ]
    if entity_type == "cabinet":
        wood = (0.55, 0.38, 0.22)
        return [
            ("body", 0.0, 0.0, 0.0, sx, sy, sz, wood),
            ("door", 0.0, 0.52 * sy, 0.0, sx * 0.9, 0.04, sz * 0.9, (0.48, 0.32, 0.18)),
            ("handle", 0.2 * sx, 0.55 * sy, 0.05 * sz, 0.04, 0.04, 0.12, (0.7, 0.7, 0.72)),
        ]
    if entity_type == "therapy_bench":
        return [
            ("top", 0.0, 0.0, 0.25 * sz, sx, sy, sz * 0.25, (0.30, 0.58, 0.35)),
            ("leg_l", -0.35 * sx, 0.0, -0.15 * sz, 0.08, sy * 0.7, sz * 0.7, (0.4, 0.4, 0.42)),
            ("leg_r", 0.35 * sx, 0.0, -0.15 * sz, 0.08, sy * 0.7, sz * 0.7, (0.4, 0.4, 0.42)),
        ]
    if entity_type == "wall_rail":
        return [("bar", 0.0, 0.0, 0.0, sx, max(sy, 0.06), max(sz, 0.08), (0.85, 0.70, 0.15))]
    if entity_type == "patient_lift":
        return [
            ("rail", 0.0, 0.0, 0.15 * sz, sx, 0.08, 0.08, (0.55, 0.55, 0.58)),
            ("hoist", 0.0, 0.0, -0.05 * sz, 0.35, sy, sz * 0.7, (0.75, 0.22, 0.18)),
            ("strap", 0.0, 0.0, -0.45 * sz, 0.05, 0.05, 0.35, (0.2, 0.2, 0.2)),
        ]
    if entity_type == "equipment_cart":
        return [
            ("shelf_b", 0.0, 0.0, -0.3 * sz, sx, sy, 0.06, (0.35, 0.48, 0.42)),
            ("shelf_t", 0.0, 0.0, 0.25 * sz, sx, sy, 0.06, (0.35, 0.48, 0.42)),
            ("post_fl", -0.4 * sx, -0.4 * sy, 0.0, 0.05, 0.05, sz * 0.9, (0.45, 0.45, 0.48)),
            ("post_fr", -0.4 * sx, 0.4 * sy, 0.0, 0.05, 0.05, sz * 0.9, (0.45, 0.45, 0.48)),
            ("post_bl", 0.4 * sx, -0.4 * sy, 0.0, 0.05, 0.05, sz * 0.9, (0.45, 0.45, 0.48)),
            ("post_br", 0.4 * sx, 0.4 * sy, 0.0, 0.05, 0.05, sz * 0.9, (0.45, 0.45, 0.48)),
        ]
    if entity_type == "wall":
        return [("panel", 0.0, 0.0, 0.0, sx, sy, sz, (0.82, 0.86, 0.88))]
    if entity_type == "ceiling":
        return [("slab", 0.0, 0.0, 0.0, sx, sy, sz, (0.92, 0.93, 0.94))]
    # fallback single box
    return [("body", 0.0, 0.0, 0.0, sx, sy, sz, (0.55, 0.55, 0.55))]


def room_shell_parts(room_size: tuple[float, float, float] = (6.0, 5.0, 3.0)) -> list[tuple[str, float, float, float, float, float, float, tuple[float, float, float]]]:
    """Full 4-wall room + floor (no ceiling — keeps view open). Positions are world."""
    L, W, H = room_size
    t = 0.08
    floor_c = (0.62, 0.66, 0.68)
    wall_c = (0.86, 0.89, 0.91)
    trim = (0.72, 0.74, 0.76)
    return [
        ("floor", 0.0, 0.0, t * 0.5, L, W, t, floor_c),
        ("wall_n", 0.0, W * 0.5, H * 0.5, L, t, H, wall_c),
        ("wall_s", 0.0, -W * 0.5, H * 0.5, L * 0.35, t, H, wall_c),  # partial = door opening
        ("wall_s_r", L * 0.35, -W * 0.5, H * 0.5, L * 0.3, t, H, wall_c),
        ("wall_w", -L * 0.5, 0.0, H * 0.5, t, W, H, wall_c),
        ("wall_e", L * 0.5, 0.0, H * 0.5, t, W, H, wall_c),
        ("base_n", 0.0, W * 0.5 - 0.02, 0.08, L * 0.98, 0.04, 0.12, trim),
        ("base_w", -L * 0.5 + 0.02, 0.0, 0.08, 0.04, W * 0.98, 0.12, trim),
        ("base_e", L * 0.5 - 0.02, 0.0, 0.08, 0.04, W * 0.98, 0.12, trim),
    ]


def iter_entity_world_parts(
    entity_id: str,
    entity_type: str,
    pose_xyz: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
) -> list[tuple[str, float, float, float, float, float, float, tuple[float, float, float]]]:
    """Return world-space parts (path_suffix, x,y,z, sx,sy,sz, rgb)."""
    px, py, pz = pose_xyz
    out = []
    for suffix, dx, dy, dz, sx, sy, sz, rgb in _parts_for(entity_type, size_xyz):
        out.append((f"{entity_id}__{suffix}", px + dx, py + dy, pz + dz, sx, sy, sz, rgb))
    return out
