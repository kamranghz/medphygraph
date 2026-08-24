"""Visualization-only alignment for Isaac for Healthcare (i4h) prop references.

Logical/scientific entity poses and AABB sizes from the scene graph are unchanged.
This module maps each entity type's **native asset bounds** (measured in centimeters
at identity) onto the logical meter-scale footprint via child-prim translate/scale.

Run ``scripts/isaac/measure_i4h_bounds.py`` (Isaac ``python.bat``) to refresh bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

CM_TO_M = 0.01


@dataclass(frozen=True)
class I4HNativeBounds:
    """Axis-aligned bounds of an i4h asset at identity (centimeters)."""

    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]

    @property
    def size_cm(self) -> tuple[float, float, float]:
        return tuple(self.bbox_max[i] - self.bbox_min[i] for i in range(3))

    @property
    def center_cm(self) -> tuple[float, float, float]:
        return tuple((self.bbox_min[i] + self.bbox_max[i]) / 2.0 for i in range(3))

    def size_m(self) -> tuple[float, float, float]:
        return tuple(s * CM_TO_M for s in self.size_cm)

    def center_m(self) -> tuple[float, float, float]:
        return tuple(c * CM_TO_M for c in self.center_cm)

    def min_m(self) -> tuple[float, float, float]:
        return tuple(v * CM_TO_M for v in self.bbox_min)


# Measured 2026-08-24 via scripts/isaac/measure_i4h_bounds.py (Isaac Sim pxr).
I4H_NATIVE_BOUNDS: dict[str, I4HNativeBounds] = {
    "therapy_bed": I4HNativeBounds(
        (-105.65393829345703, -34.30152130126953, 0.011794641613960266),
        (114.94393920898438, 34.315185546875, 93.85730743408203),
    ),
    "therapy_bench": I4HNativeBounds(
        (-151.50994873046875, -35.141300201416016, 9.999999747378752e-06),
        (149.04786682128906, 35.141300201416016, 72.8014907836914),
    ),
    "monitor_cart": I4HNativeBounds(
        (-28.442302703857422, -57.02725791931152, -2.86102294921875e-06),
        (28.442302703857422, 58.15000915527344, 122.16561889648438),
    ),
    "cabinet": I4HNativeBounds(
        (-37.51844024658203, -36.14737796783447, -9.059906005859375e-06),
        (37.51844024658203, 35.84634351730347, 100.0958251953125),
    ),
    "wheelchair": I4HNativeBounds(
        (-35.073856353759766, -39.59412384033203, 7.343292236328125e-05),
        (35.187744140625, 32.07368850708008, 126.8436279296875),
    ),
    "walker": I4HNativeBounds(
        (-48.20197172540205, -28.98101709892026, 4.991712643231949e-08),
        (57.47188655010042, 28.998230844623816, 88.94646835327148),
    ),
    "iv_pole": I4HNativeBounds(
        (-52.703895568847656, -82.59994506835938, -6.962596893310547),
        (88.51345825195312, 53.51499938964844, 181.4083709716797),
    ),
    "patient_lift": I4HNativeBounds(
        (-40.0, -40.0, 0.0),
        (40.0, 40.0, 200.0),
    ),
    "equipment_cart": I4HNativeBounds(
        (-50.0, -35.0, 0.0),
        (50.0, 35.0, 90.0),
    ),
    "wall_rail": I4HNativeBounds(
        (-60.0, -5.0, 0.0),
        (60.0, 5.0, 100.0),
    ),
}

# Optional yaw correction (degrees, Z-up) when native long axis ≠ logical layout.
I4H_ROTATE_Z_DEG: dict[str, float] = {
    "monitor_cart": 0.0,
    "walker": 0.0,
}


@dataclass(frozen=True)
class I4HVisualXform:
    """Child-prim correction under the animated entity Xform (visual only)."""

    translate: tuple[float, float, float]
    scale: tuple[float, float, float]
    rotate_z_deg: float = 0.0

    def visual_bbox_in_parent(
        self, logical_size_xyz: tuple[float, float, float]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Expected axis-aligned bounds in parent space after alignment."""
        sx, sy, sz = logical_size_xyz
        return (
            (-0.5 * sx, -0.5 * sy, 0.0),
            (0.5 * sx, 0.5 * sy, sz),
        )


def logical_floor_anchor(
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    entity_type: str,
) -> tuple[float, float, float]:
    """Parent Xform translate: logical XY center, floor-contact Z (meters)."""
    cx, cy, cz = pos
    _sx, _sy, sz = size
    floor_z = float(cz - 0.5 * sz)
    if entity_type in ("patient_lift",):
        floor_z = float(max(cz - 0.2, 1.8))
    return (float(cx), float(cy), max(0.0, floor_z))


def logical_center(pos: tuple[float, float, float]) -> tuple[float, float, float]:
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def compute_visual_xform(
    entity_type: str,
    logical_size_xyz: tuple[float, float, float],
) -> I4HVisualXform:
    """Map native i4h asset bounds onto the logical meter-scale AABB."""
    bounds = I4H_NATIVE_BOUNDS.get(entity_type)
    if bounds is None:
        return I4HVisualXform((0.0, 0.0, 0.0), (CM_TO_M, CM_TO_M, CM_TO_M))

    lx, ly, lz = logical_size_xyz
    nx_m, ny_m, nz_m = bounds.size_m()
    nmin_m = bounds.min_m()

    def _safe_ratio(target: float, native: float, fallback: float = 1.0) -> float:
        if native < 1e-6:
            return fallback
        return target / native

    scale = (
        _safe_ratio(lx, nx_m),
        _safe_ratio(ly, ny_m),
        _safe_ratio(lz, nz_m),
    )
    # Center XY on parent origin; floor-align native bbox min-Z to parent Z=0.
    translate = (
        -0.5 * lx,
        -0.5 * ly,
        -nmin_m[2] * scale[2],
    )
    return I4HVisualXform(
        translate=translate,
        scale=scale,
        rotate_z_deg=I4H_ROTATE_Z_DEG.get(entity_type, 0.0),
    )


def placement_debug_row(
    entity_id: str,
    entity_type: str,
    logical_pos: tuple[float, float, float],
    logical_size: tuple[float, float, float],
    asset_path: str | None,
) -> dict:
    """One row for the Isaac placement debug report."""
    anchor = logical_floor_anchor(logical_pos, logical_size, entity_type)
    visual = compute_visual_xform(entity_type, logical_size)
    bmin, bmax = visual.visual_bbox_in_parent(logical_size)
    world_min = (
        anchor[0] + bmin[0],
        anchor[1] + bmin[1],
        anchor[2] + bmin[2],
    )
    world_max = (
        anchor[0] + bmax[0],
        anchor[1] + bmax[1],
        anchor[2] + bmax[2],
    )
    center = tuple((world_min[i] + world_max[i]) / 2.0 for i in range(3))
    logical_center_xyz = logical_center(logical_pos)
    offset = tuple(center[i] - logical_center_xyz[i] for i in range(3))
    return {
        "entity": entity_id,
        "entity_type": entity_type,
        "logical_position": list(logical_center_xyz),
        "logical_size": list(logical_size),
        "parent_anchor": list(anchor),
        "visual_world_bbox_min": list(world_min),
        "visual_world_bbox_max": list(world_max),
        "visual_bbox_center": list(center),
        "visual_offset_from_logical_center": list(offset),
        "child_translate": list(visual.translate),
        "scale": list(visual.scale),
        "rotate_z_deg": visual.rotate_z_deg,
        "asset_path": asset_path,
    }
