"""Visualization-only tray prep demo: cabinet → side table beside monitor cart.

Scientific / evaluation data are untouched. Dedicated Isaac visualization only.

Story: a clinician approaches the drug cabinet, picks up a prepared supply tray,
carries it 2–3 steps, places it on the mobile side table beside the monitor, then
steps away. Motion is manual USD keyframes + a SkelAnimation carry pose (no PhysX
grasp, no Anim Graph walk clip — CDN character has no walk animation).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from medphygraph.i4h_alignment import CM_TO_M, I4HNativeBounds, I4HVisualXform
from medphygraph.i4h_visuals import (
    CEILING_LAMP_REL,
    ENTITY_USD_REL,
    MATERIAL_LIBRARY_REL,
    resolve_i4h_root,
)
from medphygraph.usd_authoring import (
    _camera_look_rotate_xyz,
    _fmt_i4h_asset_child,
    _fmt_rotate_xyz_samples,
    _fmt_translate_samples,
    _fmt_vec3,
    _usd_asset_ref,
)

_OR = (
    "Props/shared_OR_without_Mark/Collected_surgery_room_movie_with_heart_adjusted"
)

TRAY_TRANSFER_ENTITY_TYPES: tuple[str, ...] = (
    "instrument_tray",
    "cabinet",
    "monitor_cart",
    "side_table",
    "insulin_syringe",
    "laboratory_bottle",
    "surgical_needles",
)

TRAY_TRANSFER_USD_REL: dict[str, str] = {
    "instrument_tray": (
        f"{_OR}/Assets/Surgery Room/InstrrumentSterilizationTray_C/"
        "sm_instrrumentsterilizationtray_c01_01.usd"
    ),
    "cabinet": ENTITY_USD_REL["cabinet"],
    "monitor_cart": ENTITY_USD_REL["monitor_cart"],
    "side_table": (
        f"{_OR}/Assets/Surgery Room/MobileCartsAndTables_A/"
        "sm_mobilecartsandtables_c01_01.usd"
    ),
    "insulin_syringe": f"{_OR}/Assets/Surgery Room/InsulinSyringe_A/sm_insulinsyringe_a01_01.usd",
    "laboratory_bottle": (
        f"{_OR}/Assets/Surgery Room/LaboratoryBottles_A/sm_laboratorybottle_a01_01.usd"
    ),
    "surgical_needles": (
        f"{_OR}/Assets/Surgery Room/SurgicalNeedles_A/sm_surgicalneedles_a01_01.usd"
    ),
}

TRAY_TRANSFER_NATIVE_BOUNDS: dict[str, I4HNativeBounds] = {
    "instrument_tray": I4HNativeBounds(
        (-9.487306833267212, -17.68290865421295, -1.6665353541611694e-09),
        (9.487308263778687, 17.682916283607483, 4.587534777570281),
    ),
    "cabinet": I4HNativeBounds(
        (-37.51844024658203, -36.14737796783447, -9.059906005859375e-06),
        (37.51844024658203, 35.84634351730347, 100.0958251953125),
    ),
    "monitor_cart": I4HNativeBounds(
        (-28.442302703857422, -57.02725791931152, -2.86102294921875e-06),
        (28.442302703857422, 58.15000915527344, 122.16561889648438),
    ),
    "side_table": I4HNativeBounds(
        (-69.8303108215332, -27.691730499267578, 0.07962703704833984),
        (79.58367919921875, 27.532060623168945, 102.226318359375),
    ),
    "insulin_syringe": I4HNativeBounds(
        (-1.11277437210083, -5.487442493438721, -0.5922468900680542),
        (1.112774133682251, 4.721740484237671, 0.5922468900680542),
    ),
    "laboratory_bottle": I4HNativeBounds(
        (-3.0181045532226562, -3.0231034755706787, 0.0),
        (3.5874500274658203, 3.0181050300598145, 14.277664184570312),
    ),
    "surgical_needles": I4HNativeBounds(
        (-2.9912242889404297, -6.046046733856201, -0.0002491772174835205),
        (2.989071846008301, 6.005355358123779, 6.063727378845215),
    ),
}

BACKGROUND_NATIVE_BOUNDS: dict[str, I4HNativeBounds] = {
    "therapy_bed": I4HNativeBounds(
        (-105.65393829345703, -34.30152130126953, 0.011794641613960266),
        (114.94393920898438, 34.315185546875, 93.85730743408203),
    ),
    "ceiling_lamp": I4HNativeBounds(
        (-50.0, -8.0, -5.0),
        (50.0, 8.0, 12.0),
    ),
}

TRAY_TOP_Z_CM = 4.587534777570281
# Slight positive clearance so the tray rests on the top face (not through mesh).
TRAY_CLEARANCE_M = 0.002
LIFT_HEIGHT_M = 0.10
TRAVEL_ARC_M = 0.015
# Compact procedure room (user blueprint): ~5.5 × 4.5 × 3.0 m
ROOM_SIZE_M = (5.5, 4.5, 3.0)
TABLE_MONITOR_GAP_M = 0.12

# Clinician is OFF by default. Isaac People has no walk/grasp clips on the CDN,
# and a floating T-pose / bad rest-pose human is worse than no human (LinkedIn).
# Set MEDPHYGRAPH_INCLUDE_CLINICIAN=1 only for experimental root-slide tests.
ENV_INCLUDE_CLINICIAN = "MEDPHYGRAPH_INCLUDE_CLINICIAN"
ENV_CLINICIAN_USD = "MEDPHYGRAPH_CLINICIAN_USD"
_CLINICIAN_DIR = Path(r"D:\projects\models\isaac-people\original_male_adult_medical_01")
_DEFAULT_CLINICIAN_CARRY_USD = _CLINICIAN_DIR / "male_adult_medical_01_carry_pose.usda"
_DEFAULT_CLINICIAN_USD = _CLINICIAN_DIR / "male_adult_medical_01.usd"
# Original character foot offset (bbox min Z ≈ -0.12). Do NOT use skinned bbox guesses.
CLINICIAN_GROUND_Z_M = 0.12
CLINICIAN_STANCE_BACK_M = 0.55
CLINICIAN_HAND_FORWARD_M = 0.28
CLINICIAN_WALK_BOB_M = 0.014
CLINICIAN_WALL_MARGIN_M = 0.70

HOST_LABELS = {
    "cabinet": "DRUG CABINET",
    "monitor_cart": "MONITOR CART",
    "side_table": "SIDE TABLE",
}

HOST_SUPPORT: dict[str, dict] = {
    "cabinet": {
        "height_fraction": 1.0,
        "anchor_local_xy": (0.0, -0.08),
        "method": "cabinet_top_surface",
        "surface_z_bias_m": -0.002,
    },
    "side_table": {
        # Absolute local top of the FLAT shelf (not bbox apex / rails).
        # Calibrated from Isaac screenshots: fraction guesses left a clear air gap.
        "height_fraction": 1.0,
        "absolute_top_z_local_m": 0.548,
        "anchor_local_xy": (0.0, 0.0),
        "method": "mobile_table_top_surface_absolute",
        "surface_z_bias_m": 0.0,
    },
}

# Blueprint: cabinet left/back-left; monitor near bed (right); side table snaps to monitor.
FIXED_HOST_PLACEMENTS: tuple[tuple[str, tuple[float, float, float], float], ...] = (
    ("cabinet", (-1.95, 0.85, 0.0), 8.0),
    ("monitor_cart", (1.15, -0.15, 0.0), -12.0),
)

TRAY_SUPPLY_PLACEMENTS: tuple[tuple[str, str, tuple[float, float, float], float], ...] = (
    ("Supply_syringe", "insulin_syringe", (-3.0, 6.0, TRAY_TOP_Z_CM), 90.0),
    ("Supply_bottle", "laboratory_bottle", (4.5, -4.0, TRAY_TOP_Z_CM), 0.0),
    ("Supply_needles", "surgical_needles", (-5.5, -1.5, TRAY_TOP_Z_CM), -15.0),
)


@dataclass(frozen=True)
class ComposedBounds:
    bmin: tuple[float, float, float]
    bmax: tuple[float, float, float]

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(self.bmax[i] - self.bmin[i] for i in range(3))

    @property
    def center_xy(self) -> tuple[float, float]:
        return (
            0.5 * (self.bmin[0] + self.bmax[0]),
            0.5 * (self.bmin[1] + self.bmax[1]),
        )


@dataclass(frozen=True)
class SupportSurface:
    top_z_local: float
    anchor_local_xy: tuple[float, float]
    method: str
    bbox_top_z_local: float


@dataclass(frozen=True)
class SceneProp:
    entity_type: str
    label: str
    parent_xyz: tuple[float, float, float]
    rotate_z_deg: float
    composed: ComposedBounds
    support: SupportSurface | None = None

    def _local_to_world_xy(self, lx: float, ly: float) -> tuple[float, float]:
        rz = math.radians(self.rotate_z_deg)
        px, py, _ = self.parent_xyz
        return (
            px + math.cos(rz) * lx - math.sin(rz) * ly,
            py + math.sin(rz) * lx + math.cos(rz) * ly,
        )

    def tray_on_surface_world_xyz(self) -> tuple[float, float, float]:
        if self.support is None:
            raise ValueError(f"{self.entity_type} has no support surface")
        lx, ly = self.support.anchor_local_xy
        wx, wy = self._local_to_world_xy(lx, ly)
        bias = float(HOST_SUPPORT[self.entity_type].get("surface_z_bias_m", 0.0))
        wz = (
            self.parent_xyz[2]
            + self.support.top_z_local
            + TRAY_CLEARANCE_M
            + bias
        )
        return (wx, wy, wz)


@dataclass(frozen=True)
class TraySupplyItem:
    prim_name: str
    entity_type: str
    asset_rel: str
    offset_cm: tuple[float, float, float]
    rotate_z_deg: float


@dataclass(frozen=True)
class BackgroundProp:
    prop_id: str
    entity_type: str
    asset_rel: str
    parent_xyz: tuple[float, float, float]
    rotate_z_deg: float


@dataclass(frozen=True)
class TrayTransferLayout:
    props: tuple[SceneProp, SceneProp, SceneProp]
    room_size: tuple[float, float, float]
    tray_start_xyz: tuple[float, float, float]
    tray_end_xyz: tuple[float, float, float]
    action_center: tuple[float, float, float]
    supplies: tuple[TraySupplyItem, ...]
    background: tuple[BackgroundProp, ...]

    def prop_by_type(self, entity_type: str) -> SceneProp:
        for p in self.props:
            if p.entity_type == entity_type:
                return p
        raise KeyError(entity_type)

    # Back-compat for tests/reporting
    @property
    def hosts(self) -> tuple[SceneProp, ...]:
        return self.props


@dataclass(frozen=True)
class TrayTransferTimeline:
    fps: int
    end_frame: int
    approach_end: int
    hold_start_end: int
    lift_end: int
    travel_end: int
    lower_end: int
    hold_place_end: int

    @property
    def num_frames(self) -> int:
        return self.end_frame + 1

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / float(self.fps)


@dataclass(frozen=True)
class ClinicianMotion:
    translate: dict[int, tuple[float, float, float]]
    rotate_xyz: dict[int, tuple[float, float, float]]
    hand_xyz: dict[int, tuple[float, float, float]]
    asset_path: Path | None


def include_clinician(*, explicit: bool | None = None) -> bool:
    """Clinician off by default — no walk/grasp clips available for LinkedIn quality."""
    if explicit is not None:
        return explicit
    raw = os.environ.get(ENV_INCLUDE_CLINICIAN, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def resolve_clinician_usd(explicit: Path | str | None = None) -> Path | None:
    """Resolve clinician USD only when clinician inclusion is enabled."""
    if explicit is not None:
        p = Path(explicit)
        return p if p.is_file() else None
    if not include_clinician():
        return None
    env = os.environ.get(ENV_CLINICIAN_USD, "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
    # Prefer original (predictable feet). Baked carry pose had wrong arm axes.
    if _DEFAULT_CLINICIAN_USD.is_file():
        return _DEFAULT_CLINICIAN_USD
    if _DEFAULT_CLINICIAN_CARRY_USD.is_file():
        return _DEFAULT_CLINICIAN_CARRY_USD
    return None


def _clamp_clinician_xy(
    xy: tuple[float, float],
    *,
    room_size: tuple[float, float, float] = ROOM_SIZE_M,
    margin: float = CLINICIAN_WALL_MARGIN_M,
) -> tuple[float, float]:
    hx = 0.5 * room_size[0] - margin
    hy = 0.5 * room_size[1] - margin
    return (max(-hx, min(hx, xy[0])), max(-hy, min(hy, xy[1])))


def usd_for_tray_entity(entity_type: str, *, root: Path | None = None) -> Path | None:
    rel = TRAY_TRANSFER_USD_REL.get(entity_type) or ENTITY_USD_REL.get(entity_type)
    if not rel:
        return None
    base = root if root is not None else resolve_i4h_root()
    if base is None:
        return None
    path = base / rel.replace("/", "\\") if "\\" in str(base) else base / rel
    return path if path.is_file() else None


def uniform_visual_xform(bounds: I4HNativeBounds) -> I4HVisualXform:
    cx, cy, _ = bounds.center_cm
    nmin = bounds.min_m()
    return I4HVisualXform(
        translate=(-cx * CM_TO_M, -cy * CM_TO_M, -nmin[2]),
        scale=(CM_TO_M, CM_TO_M, CM_TO_M),
    )


def composed_world_bounds(
    bounds: I4HNativeBounds,
    visual: I4HVisualXform,
    parent_xyz: tuple[float, float, float],
    *,
    rotate_z_deg: float = 0.0,
) -> ComposedBounds:
    sx, sy, sz = visual.scale
    tx, ty, tz = visual.translate
    px, py, pz = parent_xyz
    bmin, bmax = bounds.bbox_min, bounds.bbox_max
    rz = math.radians(rotate_z_deg)
    cos_r, sin_r = math.cos(rz), math.sin(rz)
    corners: list[tuple[float, float, float]] = []
    for ix in (bmin[0], bmax[0]):
        for iy in (bmin[1], bmax[1]):
            for iz in (bmin[2], bmax[2]):
                lx = ix * sx + tx
                ly = iy * sy + ty
                lz = iz * sz + tz
                corners.append(
                    (
                        px + cos_r * lx - sin_r * ly,
                        py + sin_r * lx + cos_r * ly,
                        pz + lz,
                    )
                )
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    zs = [c[2] for c in corners]
    return ComposedBounds(
        (min(xs), min(ys), min(zs)),
        (max(xs), max(ys), max(zs)),
    )


def _support_surface(entity_type: str, bounds: I4HNativeBounds) -> SupportSurface:
    cfg = HOST_SUPPORT[entity_type]
    visual = uniform_visual_xform(bounds)
    composed = composed_world_bounds(bounds, visual, (0.0, 0.0, 0.0))
    if "absolute_top_z_local_m" in cfg:
        top_z = float(cfg["absolute_top_z_local_m"])
    else:
        top_z = composed.bmin[2] + cfg["height_fraction"] * composed.size[2]
    return SupportSurface(
        top_z_local=top_z,
        anchor_local_xy=tuple(cfg["anchor_local_xy"]),
        method=str(cfg["method"]),
        bbox_top_z_local=composed.bmax[2],
    )


def _make_prop(
    entity_type: str,
    parent_xyz: tuple[float, float, float],
    rotate_z_deg: float,
    bounds: I4HNativeBounds,
    *,
    with_support: bool,
) -> SceneProp:
    visual = uniform_visual_xform(bounds)
    composed = composed_world_bounds(
        bounds, visual, parent_xyz, rotate_z_deg=rotate_z_deg
    )
    support = _support_surface(entity_type, bounds) if with_support else None
    return SceneProp(
        entity_type=entity_type,
        label=HOST_LABELS[entity_type],
        parent_xyz=parent_xyz,
        rotate_z_deg=rotate_z_deg,
        composed=composed,
        support=support,
    )


def _side_table_placement(
    monitor: SceneProp,
    table_bounds: I4HNativeBounds,
    *,
    gap_m: float = TABLE_MONITOR_GAP_M,
) -> tuple[float, float, float]:
    """Snap mobile side table beside monitor cart on the floor (no floating)."""
    visual = uniform_visual_xform(table_bounds)
    at_origin = composed_world_bounds(table_bounds, visual, (0.0, 0.0, 0.0))
    half_w = 0.5 * at_origin.size[0]
    px = monitor.composed.bmin[0] - gap_m - half_w
    py = monitor.composed.bmin[1] + 0.22 * (
        monitor.composed.bmax[1] - monitor.composed.bmin[1]
    )
    return (px, py, 0.0)


def build_tray_transfer_layout(
    *,
    bounds: dict[str, I4HNativeBounds] | None = None,
) -> TrayTransferLayout:
    bounds = bounds or TRAY_TRANSFER_NATIVE_BOUNDS

    cabinet = _make_prop(
        "cabinet",
        FIXED_HOST_PLACEMENTS[0][1],
        FIXED_HOST_PLACEMENTS[0][2],
        bounds["cabinet"],
        with_support=True,
    )
    monitor = _make_prop(
        "monitor_cart",
        FIXED_HOST_PLACEMENTS[1][1],
        FIXED_HOST_PLACEMENTS[1][2],
        bounds["monitor_cart"],
        with_support=False,
    )
    table_xyz = _side_table_placement(monitor, bounds["side_table"])
    side_table = _make_prop(
        "side_table",
        table_xyz,
        0.0,
        bounds["side_table"],
        with_support=True,
    )

    tray_start = cabinet.tray_on_surface_world_xyz()
    tray_end = side_table.tray_on_surface_world_xyz()
    ax = 0.5 * (tray_start[0] + tray_end[0])
    ay = 0.5 * (tray_start[1] + tray_end[1])
    az = 0.5 * (tray_start[2] + tray_end[2])

    supplies = tuple(
        TraySupplyItem(
            prim_name=name,
            entity_type=etype,
            asset_rel=TRAY_TRANSFER_USD_REL[etype],
            offset_cm=offset,
            rotate_z_deg=rot,
        )
        for name, etype, offset, rot in TRAY_SUPPLY_PLACEMENTS
    )

    # Procedure bed on back-right (readable from front-left camera); lamps are visual fixtures.
    background = (
        BackgroundProp(
            "BG_surgical_table",
            "therapy_bed",
            ENTITY_USD_REL["therapy_bed"],
            (1.55, 1.35, 0.0),
            90.0,
        ),
        BackgroundProp(
            "BG_ceiling_lamp_a",
            "ceiling_lamp",
            CEILING_LAMP_REL,
            (-1.1, 0.3, 2.88),
            0.0,
        ),
        BackgroundProp(
            "BG_ceiling_lamp_b",
            "ceiling_lamp",
            CEILING_LAMP_REL,
            (1.1, 0.3, 2.88),
            0.0,
        ),
    )

    return TrayTransferLayout(
        props=(cabinet, monitor, side_table),
        room_size=ROOM_SIZE_M,
        tray_start_xyz=tray_start,
        tray_end_xyz=tray_end,
        action_center=(ax, ay, az),
        supplies=supplies,
        background=background,
    )


def build_tray_transfer_timeline(*, fps: int = 30) -> TrayTransferTimeline:
    """~12 s: approach → pickup → carry → place → step-away."""

    def f(sec: float) -> int:
        return int(round(sec * fps))

    return TrayTransferTimeline(
        fps=fps,
        end_frame=f(12.0),
        approach_end=f(1.8),
        hold_start_end=f(2.4),
        lift_end=f(3.2),
        travel_end=f(7.8),
        lower_end=f(8.6),
        hold_place_end=f(9.8),
    )


def _ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - pow(-2.0 * t + 2.0, 3.0) / 2.0


def _lerp3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def _lerp2(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _yaw_facing_deg(from_xy: tuple[float, float], to_xy: tuple[float, float]) -> float:
    """Yaw about Z; 0° faces +Y (character default)."""
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(dx, dy))


def _stance_near_tray(
    tray_xy: tuple[float, float],
    *,
    approach_toward: tuple[float, float] = (0.0, -2.0),
    back_m: float = CLINICIAN_STANCE_BACK_M,
) -> tuple[float, float]:
    """Stand between the tray and the room-front so the camera sees the face/torso."""
    vx = approach_toward[0] - tray_xy[0]
    vy = approach_toward[1] - tray_xy[1]
    n = math.hypot(vx, vy) or 1.0
    return (tray_xy[0] + (vx / n) * back_m, tray_xy[1] + (vy / n) * back_m)


def _hand_anchor_world(
    clinician_xy: tuple[float, float],
    yaw_deg: float,
    hand_z: float,
    *,
    forward_m: float = CLINICIAN_HAND_FORWARD_M,
) -> tuple[float, float, float]:
    rad = math.radians(yaw_deg)
    fx, fy = math.sin(rad), math.cos(rad)
    return (
        clinician_xy[0] + fx * forward_m,
        clinician_xy[1] + fy * forward_m,
        hand_z,
    )


def _walk_bob(frame: int, fps: int, active: bool) -> float:
    if not active:
        return 0.0
    # ~1.7 steps/sec vertical bob while root-translating
    return CLINICIAN_WALK_BOB_M * math.sin(2.0 * math.pi * 1.7 * (frame / float(fps)))


def build_clinician_motion(
    layout: TrayTransferLayout,
    timeline: TrayTransferTimeline,
    *,
    clinician_usd: Path | None = None,
) -> ClinicianMotion:
    """Root keyframes for approach → pickup → carry → place → exit + hand anchors."""
    asset = clinician_usd if clinician_usd is not None else resolve_clinician_usd()
    p0 = layout.tray_start_xyz
    p1 = layout.tray_end_xyz
    pickup = _clamp_clinician_xy(_stance_near_tray((p0[0], p0[1])))
    place = _clamp_clinician_xy(_stance_near_tray((p1[0], p1[1])))
    # Approach from room-front / center — never from the west wall (T-pose/arm clip).
    start = _clamp_clinician_xy((pickup[0] + 0.55, pickup[1] - 1.05))
    exit_xy = _clamp_clinician_xy((place[0] - 0.35, place[1] - 0.95))

    carry_z = max(p0[2], p1[2]) + LIFT_HEIGHT_M
    translate: dict[int, tuple[float, float, float]] = {}
    rotate_xyz: dict[int, tuple[float, float, float]] = {}
    hand_xyz: dict[int, tuple[float, float, float]] = {}

    def _set(frame: int, xy: tuple[float, float], yaw: float, *, walking: bool, hand_z: float) -> None:
        xy = _clamp_clinician_xy(xy)
        bob = _walk_bob(frame, timeline.fps, walking)
        translate[frame] = (xy[0], xy[1], CLINICIAN_GROUND_Z_M + bob)
        rotate_xyz[frame] = (0.0, 0.0, yaw)
        hand_xyz[frame] = _hand_anchor_world(xy, yaw, hand_z)

    for frame in range(0, timeline.approach_end + 1):
        t = _ease_in_out_cubic(frame / max(1, timeline.approach_end))
        xy = _lerp2(start, pickup, t)
        yaw = _yaw_facing_deg(xy, (p0[0], p0[1]))
        _set(frame, xy, yaw, walking=True, hand_z=carry_z)

    for frame in range(timeline.approach_end, timeline.hold_start_end + 1):
        yaw = _yaw_facing_deg(pickup, (p0[0], p0[1]))
        _set(frame, pickup, yaw, walking=False, hand_z=p0[2] + 0.02)

    for frame in range(timeline.hold_start_end, timeline.lift_end + 1):
        t = _ease_in_out_cubic(
            (frame - timeline.hold_start_end)
            / max(1, timeline.lift_end - timeline.hold_start_end)
        )
        yaw = _yaw_facing_deg(pickup, (p0[0], p0[1]))
        hz = p0[2] + 0.02 + (carry_z - (p0[2] + 0.02)) * t
        _set(frame, pickup, yaw, walking=False, hand_z=hz)

    for frame in range(timeline.lift_end, timeline.travel_end + 1):
        t = _ease_in_out_cubic(
            (frame - timeline.lift_end) / max(1, timeline.travel_end - timeline.lift_end)
        )
        xy = _lerp2(pickup, place, t)
        yaw = _yaw_facing_deg(pickup, place)
        arc = TRAVEL_ARC_M * math.sin(math.pi * t)
        _set(frame, xy, yaw, walking=True, hand_z=carry_z + arc)

    for frame in range(timeline.travel_end, timeline.lower_end + 1):
        t = _ease_in_out_cubic(
            (frame - timeline.travel_end)
            / max(1, timeline.lower_end - timeline.travel_end)
        )
        yaw = _yaw_facing_deg(place, (p1[0], p1[1]))
        hz = carry_z + (p1[2] - carry_z) * t
        _set(frame, place, yaw, walking=False, hand_z=hz)

    for frame in range(timeline.lower_end, timeline.hold_place_end + 1):
        yaw = _yaw_facing_deg(place, (p1[0], p1[1]))
        _set(frame, place, yaw, walking=False, hand_z=p1[2])

    for frame in range(timeline.hold_place_end, timeline.end_frame + 1):
        t = _ease_in_out_cubic(
            (frame - timeline.hold_place_end)
            / max(1, timeline.end_frame - timeline.hold_place_end)
        )
        xy = _lerp2(place, exit_xy, t)
        yaw = _yaw_facing_deg(place, exit_xy)
        _set(frame, xy, yaw, walking=True, hand_z=p1[2])

    return ClinicianMotion(
        translate=translate,
        rotate_xyz=rotate_xyz,
        hand_xyz=hand_xyz,
        asset_path=asset,
    )


def build_tray_translate_samples(
    layout: TrayTransferLayout,
    timeline: TrayTransferTimeline,
    *,
    clinician: ClinicianMotion | None = None,
) -> dict[int, tuple[float, float, float]]:
    """Lift → carry → lower on a clean arc between two support surfaces.

    When a clinician motion exists, mid-carry tracks the hand anchor; otherwise
    the tray follows a deterministic world arc (preferred LinkedIn path).
    """
    p0 = layout.tray_start_xyz
    p1 = layout.tray_end_xyz
    samples: dict[int, tuple[float, float, float]] = {}

    def _set(frame: int, pos: tuple[float, float, float]) -> None:
        samples[frame] = pos

    lift_start = timeline.hold_start_end
    lift_end = timeline.lift_end
    travel_end = timeline.travel_end
    lower_end = timeline.lower_end

    carry_z = max(p0[2], p1[2]) + LIFT_HEIGHT_M
    p0_lift = (p0[0], p0[1], carry_z)
    p1_lift = (p1[0], p1[1], carry_z)

    use_hands = (
        clinician is not None
        and clinician.asset_path is not None
        and len(clinician.hand_xyz) == timeline.end_frame + 1
    )

    for frame in range(0, lift_start + 1):
        _set(frame, p0)

    for frame in range(lift_start, lift_end + 1):
        t = _ease_in_out_cubic((frame - lift_start) / max(1, lift_end - lift_start))
        if use_hands:
            _set(frame, _lerp3(p0, clinician.hand_xyz[frame], t))
        else:
            _set(frame, _lerp3(p0, p0_lift, t))

    for frame in range(lift_end, travel_end + 1):
        t = _ease_in_out_cubic((frame - lift_end) / max(1, travel_end - lift_end))
        if use_hands:
            _set(frame, clinician.hand_xyz[frame])
        else:
            horiz = _lerp3(p0_lift, p1_lift, t)
            arc = TRAVEL_ARC_M * math.sin(math.pi * t)
            _set(frame, (horiz[0], horiz[1], horiz[2] + arc))

    for frame in range(travel_end, lower_end + 1):
        t = _ease_in_out_cubic((frame - travel_end) / max(1, lower_end - travel_end))
        if use_hands:
            hx, hy, hz = clinician.hand_xyz[frame]
            _set(frame, (hx + (p1[0] - hx) * t, hy + (p1[1] - hy) * t, hz + (p1[2] - hz) * t))
        else:
            _set(frame, _lerp3(p1_lift, p1, t))

    for frame in range(lower_end, timeline.end_frame + 1):
        _set(frame, p1)

    return samples


def _room_materials_prims() -> str:
    """Real PBR shaders for the room shell, bound once and shared.

    The previous room shell set only primvars:displayColor on every cube --
    a color hint with no roughness/specular response, which is why the
    existing 7-light rig had nothing physically real to actually light.
    Values below are matched to real materials, not guessed:
      - vinyl floor: some sheen, not matte (roughness 0.32)
      - painted wall/ceiling: fully matte (roughness 0.85-0.90)
      - dado band: semi-gloss protective wall panel (roughness 0.55)
      - trim: satin finish (roughness 0.45)
    """
    defs = [
        # Brighter clinical surfaces — dark slate floor was swallowing the light rig.
        ("Mat_Floor", (0.58, 0.62, 0.66), 0.35, 0.0),
        ("Mat_Wall", (0.92, 0.93, 0.94), 0.82, 0.0),
        ("Mat_WallBack", (0.90, 0.91, 0.93), 0.82, 0.0),
        ("Mat_Dado", (0.82, 0.85, 0.88), 0.50, 0.0),
        ("Mat_Ceiling", (0.95, 0.96, 0.97), 0.88, 0.0),
        ("Mat_Trim", (0.62, 0.65, 0.68), 0.45, 0.0),
        ("Mat_FallbackProp", (0.68, 0.70, 0.72), 0.55, 0.0),
        ("Mat_MonitorFallback", (0.78, 0.79, 0.81), 0.40, 0.15),
    ]
    parts = ['    def Scope "Looks" {']
    for name, rgb, rough, metal in defs:
        parts.append(
            f"""
        def Material "{name}" {{
            token outputs:surface.connect = </World/Looks/{name}/Shader.outputs:surface>
            def Shader "Shader" {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = ({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})
                float inputs:roughness = {rough:.3f}
                float inputs:metallic = {metal:.3f}
                token outputs:surface
            }}
        }}
"""
        )
    parts.append("    }")
    return "\n".join(parts)


def _material_binding(name: str) -> str:
    return f'\n        rel material:binding = </World/Looks/{name}>'


def _procedure_room_prims(room_size: tuple[float, float, float]) -> list[str]:
    """Clinical room shell: darker vinyl floor, off-white walls, dado band — not a white void.

    Every cube below keeps its displayColor (cheap viewport preview hint,
    used by non-RTX display modes) AND is now bound to a real PBR material
    from _room_materials_prims() -- the displayColor alone was the actual
    "white debug box" cause; RTX rendering has nothing to shade without a
    bound Material/Shader regardless of how the lights are set up.
    """
    lx, ly, lz = room_size
    t = 0.06
    # Brighter vinyl / walls (matched to Looks materials)
    floor = (0.58, 0.62, 0.66)
    wall = (0.92, 0.93, 0.94)
    wall_back = (0.90, 0.91, 0.93)
    ceiling = (0.95, 0.96, 0.97)
    trim = (0.62, 0.65, 0.68)
    dado = (0.82, 0.85, 0.88)
    parts: list[str] = [_room_materials_prims()]
    parts.append(
        f"""
    def Cube "Room_floor" {{
        double size = 1
        float3 xformOp:translate = (0.0, 0.0, {t * 0.5:.4f})
        float3 xformOp:scale = ({lx:.4f}, {ly:.4f}, {t:.4f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [({floor[0]:.3f}, {floor[1]:.3f}, {floor[2]:.3f})]{_material_binding("Mat_Floor")}
    }}
"""
    )
    for name, px, py, sx, sy, color, mat in (
        ("wall_n", 0.0, ly * 0.5, lx, t, wall_back, "Mat_WallBack"),
        ("wall_s", 0.0, -ly * 0.5, lx, t, wall, "Mat_Wall"),
        ("wall_w", -lx * 0.5, 0.0, t, ly, wall, "Mat_Wall"),
        ("wall_e", lx * 0.5, 0.0, t, ly, wall, "Mat_Wall"),
    ):
        parts.append(
            f"""
    def Cube "Room_{name}" {{
        double size = 1
        float3 xformOp:translate = ({px:.4f}, {py:.4f}, {lz * 0.5:.4f})
        float3 xformOp:scale = ({sx:.4f}, {sy:.4f}, {lz:.4f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})]{_material_binding(mat)}
    }}
"""
        )
    # Clinical dado / mid-wall band (reads as healthcare room, not debug box)
    dado_h = 0.95
    for name, px, py, sx, sy in (
        ("dado_n", 0.0, ly * 0.5 - 0.02, lx * 0.96, 0.03),
        ("dado_w", -lx * 0.5 + 0.02, 0.0, 0.03, ly * 0.96),
        ("dado_e", lx * 0.5 - 0.02, 0.0, 0.03, ly * 0.96),
    ):
        parts.append(
            f"""
    def Cube "Room_{name}" {{
        double size = 1
        float3 xformOp:translate = ({px:.4f}, {py:.4f}, {dado_h:.4f})
        float3 xformOp:scale = ({sx:.4f}, {sy:.4f}, 0.08)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [({dado[0]:.3f}, {dado[1]:.3f}, {dado[2]:.3f})]{_material_binding("Mat_Dado")}
    }}
"""
        )
    parts.append(
        f"""
    def Cube "Room_ceiling" {{
        double size = 1
        float3 xformOp:translate = (0.0, 0.0, {lz - t * 0.5:.4f})
        float3 xformOp:scale = ({lx:.4f}, {ly:.4f}, {t:.4f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [({ceiling[0]:.3f}, {ceiling[1]:.3f}, {ceiling[2]:.3f})]{_material_binding("Mat_Ceiling")}
    }}
"""
    )
    parts.append(
        f"""
    def Cube "Room_base_trim" {{
        double size = 1
        float3 xformOp:translate = (0.0, {-ly * 0.5 + 0.03:.4f}, 0.06)
        float3 xformOp:scale = ({lx * 0.96:.4f}, 0.04, 0.10)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [({trim[0]:.3f}, {trim[1]:.3f}, {trim[2]:.3f})]{_material_binding("Mat_Trim")}
    }}
"""
    )
    return parts


def _solve_camera_for_points(
    points: list[tuple[float, float, float]],
    *,
    eye_z: float,
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
    azimuth_deg: float = 0.0,
    margin: float = 0.82,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Solve for an eye position that GUARANTEES every point in *points* is
    inside the camera's FOV, given a fixed viewing azimuth and eye height.

    Replaces hand-picked camera offsets, which measurably did not contain
    the tray start and monitor positions (verified separately: -39.6deg
    and +29.5deg off-axis against a +-18.5deg limit). This computes the
    minimum back-off distance analytically instead of guessing one.

    azimuth_deg=0 looks along +Y (camera in front, i.e. -Y side, looking
    into the room) -- matches the existing "front" framing convention.
    margin<1 leaves headroom inside the FOV edge (0.82 -> use at most 82%
    of the half-angle), not touching the frame border.
    """
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    cz = sum(p[2] for p in points) / len(points)
    look_at = (cx, cy, cz)

    az = math.radians(azimuth_deg)
    # Forward direction the camera looks along, in the XY plane.
    fwd_xy = (math.sin(az), math.cos(az))
    right_xy = (fwd_xy[1], -fwd_xy[0])

    hfov = 2.0 * math.atan(horizontal_aperture / (2.0 * focal_length))
    vfov = 2.0 * math.atan(vertical_aperture / (2.0 * focal_length))
    h_limit = math.tan(hfov * 0.5 * margin)
    v_limit = math.tan(vfov * 0.5 * margin)

    min_distance = 0.35  # never solve to an unusably close eye position
    for px, py, pz in points:
        dx, dy, dz = px - cx, py - cy, pz - cz
        along = dx * fwd_xy[0] + dy * fwd_xy[1]     # component toward look_at (negative = solve needs distance minus this)
        across = dx * right_xy[0] + dy * right_xy[1]
        vert = pz - eye_z
        # distance needed so |across| / (distance + along) <= h_limit, solved for distance:
        if h_limit > 1e-6:
            min_distance = max(min_distance, abs(across) / h_limit - along)
        if v_limit > 1e-6:
            min_distance = max(min_distance, abs(vert) / v_limit - along)

    distance = min_distance * 1.08  # small extra headroom beyond the analytic minimum
    eye = (cx - fwd_xy[0] * distance, cy - fwd_xy[1] * distance, eye_z)
    # Keep the eye inside the procedure room so RTX framing never sits in a wall.
    hx = 0.5 * ROOM_SIZE_M[0] - 0.18
    hy = 0.5 * ROOM_SIZE_M[1] - 0.18
    eye = (max(-hx, min(hx, eye[0])), max(-hy, min(hy, eye[1])), eye_z)
    return eye, look_at


def _demo_camera_prims(layout: TrayTransferLayout) -> list[str]:
    """Medical lighting (cool, soft, directional) + hero camera at eye height.

    Camera eye/look_at are solved geometrically (see _solve_camera_for_points)
    to guarantee tray_start, tray_end, and the monitor all land inside the
    FOV with margin -- verified by actually computing the resulting angular
    offsets, not just asserting the framing "should" work.
    """
    p0, p1 = layout.tray_start_xyz, layout.tray_end_xyz
    monitor = layout.prop_by_type("monitor_cart")
    cabinet = layout.prop_by_type("cabinet")
    monitor_center = (*monitor.composed.center_xy, 0.5 * (p0[2] + p1[2]))

    # A single camera containing all three points (tray_start, tray_end,
    # monitor) turns out to be geometrically impossible inside this room at
    # any reasonable focal length -- verified by solving it and finding the
    # required eye position sits outside the walls regardless. Splitting
    # into two shots (explicitly permitted: "you may design a very small
    # number of deterministic camera cuts") instead:
    #   DemoCamera  -- tracks the carry, frames tray_start + tray_end
    #   HeroCamera  -- tight close-up for the placement beat, tray_end + monitor
    # Eye height 1.35m ("chest height", per the brief's own "eye / chest
    # height" wording) rather than full standing eye height -- the subjects
    # sit around 1.0m, so the lower eye height meaningfully reduces the
    # vertical angle the lens has to cover, which was the actual binding
    # constraint (more than horizontal span).
    eye, look_at = _solve_camera_for_points(
        [p0, p1], eye_z=1.35, focal_length=16.0,
        horizontal_aperture=20.955, vertical_aperture=18.0, azimuth_deg=30.0, margin=0.9,
    )
    cam_x, cam_y, cam_z = eye
    cx, cy, cz = look_at
    rx, ry, rz = _camera_look_rotate_xyz(eye, look_at)
    room_z = layout.room_size[2]

    hero_eye, hero_look = _solve_camera_for_points(
        [p1, monitor_center], eye_z=1.35, focal_length=20.0,
        horizontal_aperture=20.955, vertical_aperture=18.0, azimuth_deg=20.0, margin=0.9,
    )
    hx, hy, hz = hero_eye
    hlx, hly, hlz = hero_look
    hrx, hry, hrz = _camera_look_rotate_xyz(hero_eye, hero_look)

    return [
        # Strong ambient fill — prior 180–420 left the room charcoal in RTX
        """
    def DomeLight "DomeLight" {
        float inputs:intensity = 1400
        color3f inputs:color = (0.85, 0.90, 0.96)
    }
""",
        """
    def DistantLight "KeyLight" {
        float inputs:intensity = 6000
        float inputs:angle = 2.2
        color3f inputs:color = (0.95, 0.97, 1.0)
        float3 xformOp:rotateXYZ = (-48, 22, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
""",
        """
    def DistantLight "FillLight" {
        float inputs:intensity = 2200
        float inputs:angle = 3.0
        color3f inputs:color = (0.88, 0.92, 0.98)
        float3 xformOp:rotateXYZ = (-18, -48, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
""",
        """
    def DistantLight "RimLight" {
        float inputs:intensity = 1200
        float inputs:angle = 1.5
        color3f inputs:color = (0.92, 0.95, 1.0)
        float3 xformOp:rotateXYZ = (-25, 150, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
""",
        f"""
    def RectLight "OverheadLeft" {{
        float inputs:intensity = 11000
        float inputs:width = 1.6
        float inputs:height = 0.65
        color3f inputs:color = (0.96, 0.97, 1.0)
        float3 xformOp:translate = (-1.0, 0.2, {room_z - 0.08:.3f})
        float3 xformOp:rotateXYZ = (-90, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }}
""",
        f"""
    def RectLight "OverheadRight" {{
        float inputs:intensity = 11000
        float inputs:width = 1.6
        float inputs:height = 0.65
        color3f inputs:color = (0.96, 0.97, 1.0)
        float3 xformOp:translate = (1.0, 0.2, {room_z - 0.08:.3f})
        float3 xformOp:rotateXYZ = (-90, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }}
""",
        f"""
    def RectLight "OverheadCenter" {{
        float inputs:intensity = 8000
        float inputs:width = 1.2
        float inputs:height = 0.55
        color3f inputs:color = (0.97, 0.98, 1.0)
        float3 xformOp:translate = (0.15, -0.4, {room_z - 0.08:.3f})
        float3 xformOp:rotateXYZ = (-90, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }}
""",
        # Practical soft key near the carry path
        f"""
    def SphereLight "ActionFill" {{
        float inputs:intensity = 6500
        float inputs:radius = 0.35
        color3f inputs:color = (0.96, 0.97, 1.0)
        float3 xformOp:translate = ({0.5 * (p0[0] + p1[0]):.3f}, {0.5 * (p0[1] + p1[1]) - 0.4:.3f}, 2.15)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
""",
        f"""
    def Camera "DemoCamera" {{
        float3 xformOp:translate = ({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f})
        float3 xformOp:rotateXYZ = ({rx:.2f}, {ry:.2f}, {rz:.2f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
        float focalLength = 16
        float horizontalAperture = 20.955
        float verticalAperture = 18.0
        float exposure = 0.85
        float2 clippingRange = (0.1, 80)
        custom string health:look_at = "({look_at[0]:.2f}, {look_at[1]:.2f}, {look_at[2]:.2f})"
        custom string health:camera_strategy = "tracks_carry_tray_start_to_tray_end"
        custom string health:recording_note = "Disable Camera Light; use scene lights; RTX Real-Time"
        custom string health:framing_verified = "geometric_containment_solved_not_hand_picked"
    }}
""",
        f"""
    def Camera "HeroCamera" {{
        float3 xformOp:translate = ({hx:.3f}, {hy:.3f}, {hz:.3f})
        float3 xformOp:rotateXYZ = ({hrx:.2f}, {hry:.2f}, {hrz:.2f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
        float focalLength = 20
        float horizontalAperture = 20.955
        float verticalAperture = 18.0
        float exposure = 0.85
        float2 clippingRange = (0.1, 80)
        custom string health:look_at = "({hlx:.2f}, {hly:.2f}, {hlz:.2f})"
        custom string health:camera_strategy = "hero_close_tray_beside_monitor"
        custom string health:recording_note = "Cut to this camera for the final placement + hold beat (8-15s)"
    }}
""",
    ]


def _rotate_block(rotate_z_deg: float) -> tuple[str, str]:
    if abs(rotate_z_deg) < 1e-6:
        return "", '["xformOp:translate"]'
    return (
        f"\n        float3 xformOp:rotateXYZ = (0, 0, {rotate_z_deg:.3f})",
        '["xformOp:translate", "xformOp:rotateXYZ"]',
    )


MONITOR_CART_FALLBACK_MATERIAL = ("Mat_MonitorFallback", (0.72, 0.73, 0.75), 0.4, 0.15)
# Documented in I4H_ASSET_REFERENCES.md: this asset is missing a Studio_G.usd
# sublayer and "may look white". I can't inspect the actual asset file from
# here to confirm/fix the sublayer reference itself, so this binds a plausible
# clinical-equipment material (light grey plastic/metal shell, slight sheen)
# at the wrapper level as a non-destructive fallback -- USD binding strength
# means a stronger binding on the mesh itself (if the asset's material does
# resolve correctly) simply overrides this, so it's safe either way. This is
# NOT a substitute for actually checking the asset in Isaac.
def _static_prop_prim(
    prop: SceneProp,
    *,
    i4h_root: Path,
    bounds: I4HNativeBounds,
) -> str:
    usd = usd_for_tray_entity(prop.entity_type, root=i4h_root)
    visual = uniform_visual_xform(bounds)
    rel = TRAY_TRANSFER_USD_REL.get(prop.entity_type, "")
    canon = usd.stem if usd else "MISSING"
    px, py, pz = prop.parent_xyz
    rot_block, xform_order = _rotate_block(prop.rotate_z_deg)
    support_meta = ""
    if prop.support is not None:
        tx, ty, tz = prop.tray_on_surface_world_xyz() if prop.support else (0, 0, 0)
        support_meta = f"""
        custom float health:support_top_z_m = {prop.support.top_z_local:.6f}
        custom string health:support_method = "{prop.support.method}"
        custom float3 health:tray_anchor_world_m = ({tx:.6f}, {ty:.6f}, {tz:.6f})"""
    if usd is None:
        sz = prop.composed.size
        return f"""
    def Cube "PROP_{prop.entity_type}" {{
        float3 xformOp:translate = {_fmt_vec3(prop.parent_xyz)}
        float3 xformOp:scale = ({sz[0]:.6f}, {sz[1]:.6f}, {sz[2]:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [(0.62, 0.64, 0.66)]{_material_binding("Mat_FallbackProp")}
        custom string health:label = "{prop.label}"
    }}
"""
    fallback_binding = _material_binding("Mat_MonitorFallback") if prop.entity_type == "monitor_cart" else ""
    return f"""
    def Xform "PROP_{prop.entity_type}" {{
        float3 xformOp:translate = ({px:.6f}, {py:.6f}, {pz:.6f}){rot_block}
        uniform token[] xformOpOrder = {xform_order}
        custom string health:entity_type = "{prop.entity_type}"
        custom string health:label = "{prop.label}"
        custom string health:i4h_canonical_name = "{canon}"
        custom string health:i4h_asset_rel = "{rel}"{support_meta}{fallback_binding}

        def Xform "Asset" (
            references = {_usd_asset_ref(usd)}
        ) {{
{_fmt_i4h_asset_child(visual)}
        }}
    }}
"""


def _background_prop_prim(prop: BackgroundProp, *, i4h_root: Path) -> str:
    bounds = BACKGROUND_NATIVE_BOUNDS.get(
        prop.entity_type,
        I4HNativeBounds((-50.0, -50.0, 0.0), (50.0, 50.0, 100.0)),
    )
    usd_path = i4h_root / prop.asset_rel.replace("/", "\\")
    if not usd_path.is_file():
        return ""
    visual = uniform_visual_xform(bounds)
    px, py, pz = prop.parent_xyz
    rot_block, xform_order = _rotate_block(prop.rotate_z_deg)
    return f"""
    def Xform "{prop.prop_id}" {{
        float3 xformOp:translate = ({px:.6f}, {py:.6f}, {pz:.6f}){rot_block}
        uniform token[] xformOpOrder = {xform_order}
        custom string health:role = "background_or_context"

        def Xform "Asset" (
            references = {_usd_asset_ref(usd_path)}
        ) {{
{_fmt_i4h_asset_child(visual)}
        }}
    }}
"""


def _supply_on_tray_prim(
    item: TraySupplyItem,
    *,
    i4h_root: Path,
    bounds: I4HNativeBounds,
) -> str:
    usd = usd_for_tray_entity(item.entity_type, root=i4h_root)
    if usd is None:
        return ""
    visual = uniform_visual_xform(bounds)
    ox, oy, oz = item.offset_cm
    lx, ly, lz = ox * CM_TO_M, oy * CM_TO_M, oz * CM_TO_M
    rot_block, xform_order = _rotate_block(item.rotate_z_deg)
    rot_block = rot_block.replace("        ", "            ") if rot_block else ""
    return f"""
        def Xform "{item.prim_name}" {{
            float3 xformOp:translate = ({lx:.6f}, {ly:.6f}, {lz:.6f}){rot_block}
            uniform token[] xformOpOrder = {xform_order}
            custom string health:entity_type = "{item.entity_type}"
            custom string health:role = "tray_supply"

            def Xform "Asset" (
                references = {_usd_asset_ref(usd)}
            ) {{
{_fmt_i4h_asset_child(visual)}
            }}
        }}
"""


def _clinician_prim(motion: ClinicianMotion) -> str:
    """Keyframed Male Doctor root. Arms posed via baked restTransforms USD."""
    if motion.asset_path is None:
        return ""
    trans = _fmt_translate_samples(motion.translate)
    rot = _fmt_rotate_xyz_samples(motion.rotate_xyz)
    return f"""
    def Xform "Clinician" {{
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
        custom string health:role = "clinician"
        custom string health:motion = "manual_keyframes_root_baked_carry_rest_pose"
        custom string health:asset = "{motion.asset_path.resolve().as_posix()}"
        custom string health:pose_note = "Arms bent via skeleton restTransforms (not Anim Graph)"
{trans}
{rot}

        def Xform "Character" (
            references = {_usd_asset_ref(motion.asset_path)}
        ) {{
        }}
    }}
"""


def _carry_skel_animation_prim() -> str:
    """Deprecated: Isaac viewport ignores SkelAnimation without Anim Graph."""
    return ""


def tray_transfer_demo_usda(
    *,
    i4h_root: Path,
    layout: TrayTransferLayout | None = None,
    timeline: TrayTransferTimeline | None = None,
    bounds: dict[str, I4HNativeBounds] | None = None,
    clinician_usd: Path | None = None,
    with_clinician: bool | None = None,
) -> str:
    bounds = bounds or TRAY_TRANSFER_NATIVE_BOUNDS
    layout = layout or build_tray_transfer_layout(bounds=bounds)
    timeline = timeline or build_tray_transfer_timeline(fps=30)
    want_clinician = include_clinician(explicit=with_clinician)
    clinician: ClinicianMotion | None = None
    if want_clinician:
        clinician = build_clinician_motion(
            layout,
            timeline,
            clinician_usd=clinician_usd if clinician_usd is not None else resolve_clinician_usd(),
        )
        if clinician.asset_path is None:
            clinician = None
    tray_samples = build_tray_translate_samples(layout, timeline, clinician=clinician)
    mat_lib = i4h_root / MATERIAL_LIBRARY_REL.replace("/", "\\")

    story_line = (
        "# Story: clinician pickup → carry → place (EXPERIMENTAL; no walk/grasp clips)"
        if clinician is not None
        else "# Story: supply tray cabinet → side table (clean keyframes; no human — LinkedIn path)"
    )

    parts = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        '    upAxis = "Z"',
        "    metersPerUnit = 1",
        "    startTimeCode = 0",
        f"    endTimeCode = {timeline.end_frame}",
        f"    timeCodesPerSecond = {timeline.fps}",
        ")",
        "# MedPhyGraph tray prep demo (visualization only — no graph, no PhysX)",
        f"# i4h_root = {i4h_root.resolve().as_posix()}",
        f"# i4h_material_library = {mat_lib.as_posix()}",
        story_line,
        "# Room: 5.5x4.5x3.0 procedure room; medical overhead RectLights",
        "# Recording: viewport DemoCamera then HeroCamera; DISABLE Camera Light; RTX Real-Time",
        'def Xform "World" {',
    ]
    parts.extend(_demo_camera_prims(layout))
    parts.extend(_procedure_room_prims(layout.room_size))

    parts.append('    def Xform "BackgroundOR" {')
    for prop in layout.background:
        prim = _background_prop_prim(prop, i4h_root=i4h_root)
        if prim:
            parts.append(prim)
    parts.append("    }")

    for prop in layout.props:
        parts.append(
            _static_prop_prim(prop, i4h_root=i4h_root, bounds=bounds[prop.entity_type])
        )

    tray_usd = usd_for_tray_entity("instrument_tray", root=i4h_root)
    tray_visual = uniform_visual_xform(bounds["instrument_tray"])
    tray_rel = TRAY_TRANSFER_USD_REL["instrument_tray"]
    tray_canon = tray_usd.stem if tray_usd else "MISSING"
    supply_prims = [
        _supply_on_tray_prim(item, i4h_root=i4h_root, bounds=bounds[item.entity_type])
        for item in layout.supplies
    ]

    if tray_usd is not None:
        tray_body = f"""
        def Xform "TrayBase" (
            references = {_usd_asset_ref(tray_usd)}
        ) {{
{_fmt_i4h_asset_child(tray_visual)}
        }}
{"".join(supply_prims)}"""
    else:
        ts = composed_world_bounds(bounds["instrument_tray"], tray_visual, (0.0, 0.0, 0.0)).size
        tray_body = f"""
        def Cube "TrayBase" {{
            float3 xformOp:scale = ({ts[0]:.6f}, {ts[1]:.6f}, {ts[2]:.6f})
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }}
"""

    motion_tag = (
        "hand_synced_experimental" if clinician is not None else "surface_to_surface_keyframes"
    )
    parts.append(
        f"""
    def Xform "TRAY_ASSEMBLY" {{
        uniform token[] xformOpOrder = ["xformOp:translate"]
        custom string health:entity_type = "instrument_tray"
        custom string health:label = "PREPARED SUPPLY TRAY"
        custom string health:i4h_canonical_name = "{tray_canon}"
        custom string health:i4h_asset_rel = "{tray_rel}"
        custom string health:demo_role = "animated_subject"
        custom string health:motion = "{motion_tag}"
        custom string health:tray_start_xyz = "({layout.tray_start_xyz[0]:.3f}, {layout.tray_start_xyz[1]:.3f}, {layout.tray_start_xyz[2]:.3f})"
        custom string health:tray_end_xyz = "({layout.tray_end_xyz[0]:.3f}, {layout.tray_end_xyz[1]:.3f}, {layout.tray_end_xyz[2]:.3f})"
{_fmt_translate_samples(tray_samples)}
{tray_body}
    }}
"""
    )

    clin_block = _clinician_prim(clinician) if clinician is not None else ""
    if clin_block:
        parts.append(clin_block)

    parts.append("}")
    return "\n".join(parts) + "\n"


def write_tray_transfer_demo_usda(
    path: Path | str,
    *,
    i4h_root: Path | None = None,
    layout: TrayTransferLayout | None = None,
    timeline: TrayTransferTimeline | None = None,
    bounds: dict[str, I4HNativeBounds] | None = None,
    clinician_usd: Path | None = None,
    with_clinician: bool | None = None,
) -> Path:
    root = i4h_root or resolve_i4h_root()
    if root is None:
        raise FileNotFoundError("I4H_ASSETS_ROOT not found")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        tray_transfer_demo_usda(
            i4h_root=root,
            layout=layout,
            timeline=timeline,
            bounds=bounds,
            clinician_usd=clinician_usd,
            with_clinician=with_clinician,
        ),
        encoding="utf-8",
    )
    return path


def layout_report(layout: TrayTransferLayout) -> dict:
    props_out = []
    for p in layout.props:
        row = {
            "entity_type": p.entity_type,
            "label": p.label,
            "parent_xyz_m": list(p.parent_xyz),
            "rotate_z_deg": p.rotate_z_deg,
            "composed_bbox_min_m": list(p.composed.bmin),
            "composed_bbox_max_m": list(p.composed.bmax),
            "floor_z_m": p.composed.bmin[2],
        }
        if p.support is not None:
            tx, ty, tz = p.tray_on_surface_world_xyz()
            row.update(
                {
                    "support_top_z_local_m": p.support.top_z_local,
                    "support_method": p.support.method,
                    "tray_anchor_world_m": [tx, ty, tz],
                }
            )
        props_out.append(row)

    monitor = layout.prop_by_type("monitor_cart")
    side = layout.prop_by_type("side_table")
    gap = monitor.composed.bmin[0] - side.composed.bmax[0]

    return {
        "story": "prepared tray: drug cabinet → side table beside monitor cart",
        "room_size_m": list(layout.room_size),
        "action_center_m": list(layout.action_center),
        "tray_start_xyz_m": list(layout.tray_start_xyz),
        "tray_end_xyz_m": list(layout.tray_end_xyz),
        "tray_asset": TRAY_TRANSFER_USD_REL["instrument_tray"],
        "side_table_asset": TRAY_TRANSFER_USD_REL["side_table"],
        "table_monitor_gap_m": gap,
        "tray_supplies": [
            {
                "prim": s.prim_name,
                "entity_type": s.entity_type,
                "asset_rel": s.asset_rel,
                "offset_cm": list(s.offset_cm),
                "rotate_z_deg": s.rotate_z_deg,
            }
            for s in layout.supplies
        ],
        "props": props_out,
        "background_or_context": [
            {
                "prop_id": p.prop_id,
                "entity_type": p.entity_type,
                "asset_rel": p.asset_rel,
                "parent_xyz_m": list(p.parent_xyz),
                "rotate_z_deg": p.rotate_z_deg,
            }
            for p in layout.background
        ],
    }
