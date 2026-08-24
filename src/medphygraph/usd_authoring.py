"""Write OpenUSD (.usda) proxy stages for healthcare rehab scenes (AABB cubes)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from medphygraph.gui_assets import room_shell_parts
from medphygraph.i4h_alignment import (
    I4HVisualXform,
    compute_visual_xform,
    logical_center,
    logical_floor_anchor,
)
from medphygraph.schema import HealthScene, STRUCTURAL_TYPES

try:
    from medphygraph.i4h_visuals import ENTITY_USD_REL, resolve_i4h_root, usd_for_entity_type
except ImportError:  # pragma: no cover - optional during minimal installs
    ENTITY_USD_REL = {}
    resolve_i4h_root = None  # type: ignore[assignment]
    usd_for_entity_type = None  # type: ignore[assignment]

PAPER_FOCUS_ENTITY_IDS = frozenset(
    {"bed", "walker", "wheelchair", "monitor", "bench", "cabinet", "iv_pole"}
)

# Distinct proxy colors for Isaac GUI (not photoreal medical assets).
_TYPE_COLOR: dict[str, tuple[float, float, float]] = {
    "floor": (0.72, 0.74, 0.76),
    "wall": (0.55, 0.62, 0.70),
    "ceiling": (0.88, 0.90, 0.92),
    "therapy_bed": (0.20, 0.55, 0.58),
    "walker": (0.90, 0.45, 0.12),
    "wheelchair": (0.18, 0.42, 0.72),
    "monitor_cart": (0.28, 0.30, 0.32),
    "iv_pole": (0.70, 0.72, 0.74),
    "cabinet": (0.55, 0.38, 0.22),
    "therapy_bench": (0.30, 0.58, 0.35),
    "wall_rail": (0.85, 0.70, 0.15),
    "patient_lift": (0.75, 0.22, 0.18),
    "equipment_cart": (0.35, 0.48, 0.42),
    "other_furniture": (0.50, 0.48, 0.45),
}

_TYPE_OPACITY: dict[str, float] = {
    "wall": 0.35,
    "ceiling": 0.18,
}


def _color_for(entity_type: str) -> tuple[float, float, float]:
    return _TYPE_COLOR.get(entity_type, (0.55, 0.55, 0.55))


def _cube_prim(
    entity_id: str,
    pose,
    size,
    *,
    kinematic: bool,
    entity_type: str = "other_furniture",
    colored: bool = False,
) -> str:
    # USD Cube default edge=2; scale half-extents relative to unit cube size=1 → use size as scale
    sx, sy, sz = size
    px, py, pz = pose
    purpose = "kinematic" if kinematic else "dynamic"
    extra = ""
    if colored:
        r, g, b = _color_for(entity_type)
        op = _TYPE_OPACITY.get(entity_type, 1.0)
        extra = f"""
        color3f[] primvars:displayColor = [({r:.3f}, {g:.3f}, {b:.3f})]
        float[] primvars:displayOpacity = [{op:.3f}]"""
    return f"""
    def Cube "{entity_id}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]
    ) {{
        bool physics:kinematicEnabled = {'true' if kinematic else 'false'}
        bool physics:collisionEnabled = true
        double size = 1
        float3 xformOp:translate = ({px:.6f}, {py:.6f}, {pz:.6f})
        float3 xformOp:scale = ({sx:.6f}, {sy:.6f}, {sz:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        custom string health:entity_id = "{entity_id}"
        custom string health:entity_type = "{entity_type}"
        custom string health:role = "{purpose}"{extra}
    }}
"""


def _gui_lights_and_camera(scene: HealthScene) -> list[str]:
    """Room-centered distant light + overview camera (Z-up)."""
    xs = [e.pose_xyz[0] for e in scene.entities if e.entity_type != "zone"]
    ys = [e.pose_xyz[1] for e in scene.entities if e.entity_type != "zone"]
    cx = 0.5 * (min(xs) + max(xs)) if xs else 0.0
    cy = 0.5 * (min(ys) + max(ys)) if ys else 0.0
    # Look from southeast-ish over the room
    cam_x, cam_y, cam_z = cx + 4.5, cy - 5.0, 3.8
    return [
        """
    def DistantLight "KeyLight" {
        float inputs:intensity = 3000
        float inputs:angle = 1.0
        color3f inputs:color = (1.0, 0.98, 0.95)
        float3 xformOp:rotateXYZ = (-35, 25, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
""",
        f"""
    def Camera "DemoCamera" {{
        float3 xformOp:translate = ({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f})
        float3 xformOp:rotateXYZ = (55, 0, 40)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
        float focalLength = 18
        float horizontalAperture = 20.955
        float2 clippingRange = (0.1, 100)
        custom string health:look_at = "({cx:.2f}, {cy:.2f}, 1.0)"
    }}
""",
    ]


def scene_to_usda(scene: HealthScene, *, gui: bool = False) -> str:
    """ASCII USDA with cube proxies. Structural / fixed → kinematic.

    gui=True adds display colors, translucent walls/ceiling, light, and DemoCamera.
    """
    parts = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        '    upAxis = "Z"',
        "    metersPerUnit = 1",
        ")",
        f"# DyPhyGraph-Health scene {scene.scene_id} seed={scene.seed}"
        + (" (GUI demo)" if gui else ""),
        'def Xform "World" {',
        '    def PhysicsScene "PhysicsScene" {',
        "        vector3f physics:gravityDirection = (0, 0, -1)",
        "        float physics:gravityMagnitude = 9.81",
        "    }",
    ]
    if gui:
        parts.extend(_gui_lights_and_camera(scene))
    for e in scene.entities:
        if e.entity_type == "zone":
            continue
        kinematic = (e.entity_type in STRUCTURAL_TYPES) or (not e.movable)
        parts.append(
            _cube_prim(
                e.entity_id,
                e.pose_xyz,
                e.size_xyz,
                kinematic=kinematic,
                entity_type=e.entity_type,
                colored=gui,
            )
        )
    parts.append("}")
    return "\n".join(parts) + "\n"


def write_scene_usda(
    scene: HealthScene,
    path: Path | str,
    *,
    gui: bool = False,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scene_to_usda(scene, gui=gui), encoding="utf-8")
    return path


def _lerp(a: tuple[float, float, float], b: tuple[float, float, float], t: float):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def _fmt_vec3(v: tuple[float, float, float]) -> str:
    return f"({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})"


def _fmt_translate_samples(samples: dict[int, tuple[float, float, float]]) -> str:
    lines = ["        double3 xformOp:translate.timeSamples = {"]
    for frame, pos in sorted(samples.items()):
        lines.append(f"            {frame}: {_fmt_vec3(pos)},")
    lines.append("        }")
    return "\n".join(lines)


def _fmt_rotate_xyz_samples(samples: dict[int, tuple[float, float, float]]) -> str:
    lines = ["        float3 xformOp:rotateXYZ.timeSamples = {"]
    for frame, rot in sorted(samples.items()):
        lines.append(f"            {frame}: {_fmt_vec3(rot)},")
    lines.append("        }")
    return "\n".join(lines)


def _fmt_visibility_samples(samples: dict[int, str]) -> str:
    lines = ["        token visibility.timeSamples = {"]
    for frame, vis in sorted(samples.items()):
        lines.append(f'            {frame}: "{vis}",')
    lines.append("        }")
    return "\n".join(lines)


def _usd_asset_ref(path: Path) -> str:
    return "@" + path.resolve().as_posix() + "@"


def _i4h_place_xyz(pos: tuple[float, float, float], size: tuple[float, float, float], etype: str):
    """Parent prim floor anchor (meters). Scientific ``pos`` is unchanged."""
    return logical_floor_anchor(pos, size, etype)


def _beam_endpoint(ent: dict) -> tuple[float, float, float]:
    """3D beam attachment point: logical center for props, structural pose for hosts."""
    pos = tuple(ent["pos"])
    if ent.get("type") in STRUCTURAL_TYPES:
        return (float(pos[0]), float(pos[1]), float(pos[2]))
    return logical_center(pos)


def _fmt_i4h_asset_child(visual: I4HVisualXform) -> str:
    tx, ty, tz = visual.translate
    sx, sy, sz = visual.scale
    if abs(visual.rotate_z_deg) > 1e-6:
        return f"""
            float3 xformOp:translate = ({tx:.6f}, {ty:.6f}, {tz:.6f})
            float3 xformOp:rotateXYZ = (0, 0, {visual.rotate_z_deg:.3f})
            float3 xformOp:scale = ({sx:.6f}, {sy:.6f}, {sz:.6f})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
            custom string health:visual_align = "i4h_native_bounds"
"""
    return f"""
            float3 xformOp:translate = ({tx:.6f}, {ty:.6f}, {tz:.6f})
            float3 xformOp:scale = ({sx:.6f}, {sy:.6f}, {sz:.6f})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            custom string health:visual_align = "i4h_native_bounds"
"""


@dataclass(frozen=True)
class TransitionTimeline:
    """Shared timing for USDA animation and the 2D graph renderer.

    **Inclusive frame semantics:** integer time codes run from ``0`` through
    ``end_frame`` (both inclusive). At 30 fps with the default 3-state demo this
    yields 421 samples (0..420) spanning ~14.03 s.

    Each settled state *s* holds on
    ``[state_start_frames[s], state_start_frames[s] + hold_frames]``.
    The transition into state *s+1* runs on the half-open interval
    ``(state_start_frames[s] + hold_frames, state_start_frames[s+1])`` — i.e. it
    starts immediately after state *s* hold ends and ends when state *s+1* begins.
    """

    fps: int
    hold_frames: int
    tween_frames: int
    state_start_frames: tuple[int, ...]
    end_frame: int

    @property
    def num_states(self) -> int:
        return len(self.state_start_frames)

    @property
    def num_frames(self) -> int:
        return self.end_frame + 1

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / float(self.fps)

    def tween_range(self, from_state: int) -> tuple[int, int]:
        """Half-open ``[tween_start, tween_end)`` for transition *from_state → from_state+1*."""
        if from_state < 0 or from_state >= self.num_states - 1:
            raise IndexError(from_state)
        return (
            self.state_start_frames[from_state] + self.hold_frames,
            self.state_start_frames[from_state + 1],
        )

    def hold_range(self, state: int) -> tuple[int, int]:
        """Inclusive ``[hold_start, hold_end]`` for settled state *state*."""
        if state < 0 or state >= self.num_states:
            raise IndexError(state)
        return self.state_start_frames[state], self.state_start_frames[state] + self.hold_frames

    def phase_at_frame(self, frame: int) -> tuple[str, int, int | None]:
        """Return ``(phase, settled_state, next_state_or_None)`` with ``phase`` in ``hold``/``tween``."""
        for s in range(self.num_states - 1):
            tween_start, tween_end = self.tween_range(s)
            if tween_start <= frame < tween_end:
                return "tween", s, s + 1
        for s in range(self.num_states):
            hold_start, hold_end = self.hold_range(s)
            if hold_start <= frame <= hold_end:
                return "hold", s, None
        return "hold", self.num_states - 1, None


def build_transition_timeline(
    num_states: int,
    *,
    fps: int = 30,
    hold_seconds: float = 2.0,
    seconds_per_state: float = 4.0,
) -> TransitionTimeline:
    """Build the canonical paper-demo timeline (default: 0..420 @ 30 fps)."""
    if num_states < 1:
        raise ValueError("num_states must be >= 1")
    hold_f = max(1, int(round(hold_seconds * fps)))
    tween_f = max(1, int(round(seconds_per_state * fps)))
    t_state: list[int] = []
    cursor = 0
    for idx in range(num_states):
        if idx > 0:
            cursor += tween_f
        t_state.append(cursor)
        cursor += hold_f
    return TransitionTimeline(
        fps=fps,
        hold_frames=hold_f,
        tween_frames=tween_f,
        state_start_frames=tuple(t_state),
        end_frame=cursor,
    )


def graph_visual_at_frame(
    frames: list[dict],
    timeline: TransitionTimeline,
    frame: int,
    *,
    hide_structural_hosts: bool = False,
) -> dict:
    """Edge sets for one graph-render frame (present / added / removed + labels)."""
    phase, settled, nxt = timeline.phase_at_frame(frame)
    frame_data = frames[settled]

    def _filter(edges: list) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for subject, host in edges:
            if hide_structural_hosts:
                host_ent = frame_data["entities"].get(host, {})
                if host_ent.get("type") in STRUCTURAL_TYPES and subject not in PAPER_FOCUS_ENTITY_IDS:
                    continue
            out.add((subject, host))
        return out

    if phase == "hold":
        return {
            "phase": "hold",
            "state_index": settled,
            "operation": frame_data["operation"],
            "present": _filter(frame_data.get("edges_present", [])),
            "added": set(),
            "removed": set(),
        }

    assert nxt is not None
    next_data = frames[nxt]
    added = _filter(next_data.get("added_edges", []))
    removed = _filter(next_data.get("removed_edges", []))
    present = _filter(frame_data.get("edges_present", [])) - removed - added
    return {
        "phase": "tween",
        "state_index": settled,
        "next_state_index": nxt,
        "operation": f"{frame_data['operation']} → {next_data['operation']}",
        "present": present,
        "added": added,
        "removed": removed,
    }


def _build_transition_timeline(
    frames: list[dict],
    *,
    fps: int,
    hold_seconds: float,
    seconds_per_state: float,
) -> tuple[int, list[int], int, int]:
    """Backward-compatible tuple unpack used inside USDA writers."""
    tl = build_transition_timeline(
        len(frames),
        fps=fps,
        hold_seconds=hold_seconds,
        seconds_per_state=seconds_per_state,
    )
    return tl.end_frame, list(tl.state_start_frames), tl.hold_frames, tl.tween_frames


def _entity_animation_samples(
    eid: str,
    frames: list[dict],
    t_state: list[int],
    hold_f: int,
    *,
    place_fn=None,
) -> tuple[dict[int, tuple[float, float, float]], dict[int, str]]:
    place_fn = place_fn or (lambda pos, _size, _etype: tuple(pos))
    translate_samples: dict[int, tuple[float, float, float]] = {}
    visibility_samples: dict[int, str] = {}

    ent0 = next(frame["entities"][eid] for frame in frames if eid in frame["entities"])
    size = tuple(ent0.get("size") or (0.5, 0.5, 0.5))
    etype = ent0["type"]

    for state_idx, frame in enumerate(frames):
        if eid not in frame["entities"]:
            continue
        pos = place_fn(tuple(frame["entities"][eid]["pos"]), size, etype)
        t_hold_start = t_state[state_idx]
        t_hold_end = t_hold_start + hold_f
        translate_samples[t_hold_start] = pos
        translate_samples[t_hold_end] = pos
        visibility_samples[t_hold_start] = "inherited"
        visibility_samples[t_hold_end] = "inherited"

        if state_idx < len(frames) - 1:
            next_frame = frames[state_idx + 1]
            t_tween_end = t_state[state_idx + 1]
            if eid in next_frame["entities"]:
                next_pos = place_fn(
                    tuple(next_frame["entities"][eid]["pos"]),
                    tuple(next_frame["entities"][eid].get("size") or size),
                    next_frame["entities"][eid]["type"],
                )
                translate_samples[t_tween_end] = next_pos
                visibility_samples[t_tween_end] = "inherited"
            else:
                translate_samples[t_tween_end] = pos
                visibility_samples[t_tween_end] = "invisible"

    for state_idx in range(len(frames) - 1):
        if eid not in frames[state_idx]["entities"]:
            continue
        t0 = t_state[state_idx] + hold_f
        t1 = t_state[state_idx + 1]
        if t1 <= t0:
            continue
        p0 = translate_samples.get(t_state[state_idx] + hold_f, translate_samples.get(t_state[state_idx]))
        p1 = translate_samples.get(t1)
        if p0 is None or p1 is None:
            continue
        steps = max(2, min(8, t1 - t0))
        for step in range(1, steps):
            frac = step / steps
            translate_samples[t0 + int((t1 - t0) * frac)] = _lerp(p0, p1, frac)

    return translate_samples, visibility_samples


def _beam_between(p_a, p_b) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    ax, ay, az = p_a
    bx, by, bz = p_b
    mid = ((ax + bx) / 2.0, (ay + by) / 2.0, (az + bz) / 2.0)
    dx, dy, dz = bx - ax, by - ay, bz - az
    length = max((dx * dx + dy * dy + dz * dz) ** 0.5, 1e-6)
    scale = (0.02, 0.02, length)
    return mid, scale


def _room_shell_usda_prims(zone_size: tuple[float, float, float] = (6.0, 5.0, 3.0)) -> list[str]:
    """Static room shell (no PhysX) from gui_assets.room_shell_parts — not ATLAS_OR."""
    parts: list[str] = []
    for name, x, y, z, sx, sy, sz, rgb in room_shell_parts(zone_size):
        r, g, b = rgb
        parts.append(
            f"""
    def Cube "RoomShell_{name}" {{
        double size = 1
        float3 xformOp:translate = ({x:.4f}, {y:.4f}, {z:.4f})
        float3 xformOp:scale = ({sx:.4f}, {sy:.4f}, {sz:.4f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [({r:.3f}, {g:.3f}, {b:.3f})]
    }}
"""
        )
    return parts


def _camera_look_rotate_xyz(
    cam_pos: tuple[float, float, float],
    look_at: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Approximate USD Camera rotateXYZ (Z-up) to look from *cam_pos* toward *look_at*."""
    dx = look_at[0] - cam_pos[0]
    dy = look_at[1] - cam_pos[1]
    dz = look_at[2] - cam_pos[2]
    horiz = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = -math.degrees(math.atan2(dz, horiz))
    return (pitch + 90.0, 0.0, yaw - 90.0)


def _demo_camera_for_transition(frames: list[dict]) -> list[str]:
    """Action-focused DemoCamera from monitor path + bench (not whole-room overview)."""
    monitor_start = monitor_end = bench_pos = None
    for frame in frames:
        ents = frame["entities"]
        if "monitor" in ents:
            ent = ents["monitor"]
            size = tuple(ent.get("size") or (0.5, 0.5, 1.2))
            placed = _i4h_place_xyz(tuple(ent["pos"]), size, ent["type"])
            if monitor_start is None:
                monitor_start = placed
            monitor_end = placed
        if "bench" in ents:
            ent = ents["bench"]
            size = tuple(ent.get("size") or (1.0, 0.5, 0.5))
            bench_pos = _i4h_place_xyz(tuple(ent["pos"]), size, ent["type"])

    if monitor_start is None:
        monitor_start = (0.0, 0.0, 0.8)
    if monitor_end is None:
        monitor_end = monitor_start
    if bench_pos is None:
        bench_pos = monitor_end

    action_pts = [monitor_start, monitor_end, bench_pos]
    xs = [p[0] for p in action_pts]
    ys = [p[1] for p in action_pts]
    zs = [p[2] for p in action_pts]
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    cz = 0.5 * (min(zs) + max(zs)) + 0.35

    span_xy = max(max(xs) - min(xs), max(ys) - min(ys), 1.2)
    span_z = max(max(zs) - min(zs), 0.4)
    cam_dist = max(span_xy * 1.15, 2.0)
    cam_x = cx + cam_dist * 0.62
    cam_y = cy - cam_dist * 0.78
    cam_z = cz + span_z * 1.6 + 0.65
    cam_pos = (cam_x, cam_y, cam_z)
    look_at = (cx, cy, cz)
    rx, ry, rz = _camera_look_rotate_xyz(cam_pos, look_at)

    return [
        """
    def DistantLight "KeyLight" {
        float inputs:intensity = 3200
        float inputs:angle = 0.9
        color3f inputs:color = (1.0, 0.98, 0.95)
        float3 xformOp:rotateXYZ = (-32, 22, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
""",
        f"""
    def Camera "DemoCamera" {{
        float3 xformOp:translate = ({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f})
        float3 xformOp:rotateXYZ = ({rx:.2f}, {ry:.2f}, {rz:.2f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
        float focalLength = 26
        float horizontalAperture = 20.955
        float2 clippingRange = (0.1, 100)
        custom string health:look_at = "({look_at[0]:.2f}, {look_at[1]:.2f}, {look_at[2]:.2f})"
        custom string health:camera_strategy = "monitor_path_and_bench"
    }}
""",
    ]


def _demo_lights_and_camera_from_positions(
    positions: list[tuple[float, float, float]],
) -> list[str]:
    if not positions:
        positions = [(0.0, 0.0, 0.0)]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    cam_x, cam_y, cam_z = cx + 4.5, cy - 5.0, 3.8
    return [
        """
    def DistantLight "KeyLight" {
        float inputs:intensity = 3000
        float inputs:angle = 1.0
        color3f inputs:color = (1.0, 0.98, 0.95)
        float3 xformOp:rotateXYZ = (-35, 25, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
""",
        f"""
    def Camera "DemoCamera" {{
        float3 xformOp:translate = ({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f})
        float3 xformOp:rotateXYZ = (55, 0, 40)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
        float focalLength = 18
        float horizontalAperture = 20.955
        float2 clippingRange = (0.1, 100)
        custom string health:look_at = "({cx:.2f}, {cy:.2f}, 1.0)"
    }}
""",
    ]


def transition_log_to_usda(
    log: dict,
    *,
    fps: int = 30,
    hold_seconds: float = 2.0,
    seconds_per_state: float = 4.0,
) -> str:
    """Build an animated ASCII USDA stage from a transition_demo_log.json payload."""
    frames = log["frames"]
    if not frames:
        raise ValueError("transition log has no frames")

    end_frame, t_state, hold_f, _tween_f = _build_transition_timeline(
        frames, fps=fps, hold_seconds=hold_seconds, seconds_per_state=seconds_per_state
    )

    all_ids = {eid for frame in frames for eid in frame["entities"]}
    first_entities = frames[0]["entities"]
    positions = [tuple(ent["pos"]) for ent in first_entities.values()]

    parts = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        '    upAxis = "Z"',
        "    metersPerUnit = 1",
        "    startTimeCode = 0",
        f"    endTimeCode = {end_frame}",
        f"    timeCodesPerSecond = {fps}",
        ")",
        f"# MedPhyGraph transition demo ({log.get('scene_id_base', 'scene')})",
        'def Xform "World" {',
        '    def PhysicsScene "PhysicsScene" {',
        "        vector3f physics:gravityDirection = (0, 0, -1)",
        "        float physics:gravityMagnitude = 9.81",
        "    }",
    ]
    parts.extend(_demo_lights_and_camera_from_positions(positions))

    for eid in sorted(all_ids):
        ent0 = next(frame["entities"][eid] for frame in frames if eid in frame["entities"])
        etype = ent0["type"]
        size = tuple(ent0.get("size") or (0.5, 0.5, 0.5))
        r, g, b = _color_for(etype)
        op = _TYPE_OPACITY.get(etype, 1.0)
        translate_samples, visibility_samples = _entity_animation_samples(eid, frames, t_state, hold_f)
        sx, sy, sz = size
        parts.append(
            f"""
    def Cube "{eid}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]
    ) {{
        bool physics:kinematicEnabled = true
        bool physics:collisionEnabled = true
        double size = 1
        float3 xformOp:scale = ({sx:.6f}, {sy:.6f}, {sz:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        custom string health:entity_id = "{eid}"
        custom string health:entity_type = "{etype}"
        custom string health:operation = "transition_demo"
        color3f[] primvars:displayColor = [({r:.3f}, {g:.3f}, {b:.3f})]
        float[] primvars:displayOpacity = [{op:.3f}]
{_fmt_translate_samples(translate_samples)}
{_fmt_visibility_samples(visibility_samples)}
    }}
"""
        )

    parts.append("}")
    return "\n".join(parts) + "\n"


def transition_log_to_i4h_usda(
    log: dict,
    *,
    i4h_root: Path,
    fps: int = 30,
    hold_seconds: float = 2.0,
    seconds_per_state: float = 4.0,
    focus: bool = True,
) -> str:
    """Animated USDA with Isaac for Healthcare prop references (paper-style demo)."""
    if usd_for_entity_type is None:
        raise RuntimeError("i4h visuals unavailable")

    frames = log["frames"]
    if not frames:
        raise ValueError("transition log has no frames")

    end_frame, t_state, hold_f, _tween_f = _build_transition_timeline(
        frames, fps=fps, hold_seconds=hold_seconds, seconds_per_state=seconds_per_state
    )

    all_ids = {
        eid
        for frame in frames
        for eid, ent in frame["entities"].items()
        if ent.get("type") not in STRUCTURAL_TYPES
        and (not focus or eid in PAPER_FOCUS_ENTITY_IDS)
    }

    parts = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        '    upAxis = "Z"',
        "    metersPerUnit = 1",
        "    startTimeCode = 0",
        f"    endTimeCode = {end_frame}",
        f"    timeCodesPerSecond = {fps}",
        ")",
        f"# MedPhyGraph i4h transition demo ({log.get('scene_id_base', 'scene')})",
        f"# i4h_root = {i4h_root.resolve().as_posix()}",
        f"# focus_entities = {sorted(all_ids)}",
        'def Xform "World" {',
        """
    def DomeLight "DomeLight" {
        float inputs:intensity = 900
        color3f inputs:color = (0.93, 0.95, 0.98)
    }
""",
    ]
    parts.extend(_room_shell_usda_prims((6.0, 5.0, 3.0)))
    parts.extend(_demo_camera_for_transition(frames))

    entity_positions_at_state: dict[int, dict[str, tuple[float, float, float]]] = {}
    for state_idx, frame in enumerate(frames):
        entity_positions_at_state[state_idx] = {}
        for eid, ent in frame["entities"].items():
            entity_positions_at_state[state_idx][eid] = _beam_endpoint(ent)

    for eid in sorted(all_ids):
        ent0 = next(frame["entities"][eid] for frame in frames if eid in frame["entities"])
        etype = ent0["type"]
        size0 = tuple(ent0.get("size") or (0.5, 0.5, 0.5))
        usd = usd_for_entity_type(etype, root=i4h_root)
        visual_xform = compute_visual_xform(etype, size0)
        translate_samples, visibility_samples = _entity_animation_samples(
            eid, frames, t_state, hold_f, place_fn=_i4h_place_xyz
        )
        if usd is not None:
            parts.append(
                f"""
    def Xform "{eid}" {{
        uniform token[] xformOpOrder = ["xformOp:translate"]
        custom string health:entity_id = "{eid}"
        custom string health:entity_type = "{etype}"
        custom string health:visual = "i4h"
        custom string health:logical_center = "({ent0['pos'][0]:.3f}, {ent0['pos'][1]:.3f}, {ent0['pos'][2]:.3f})"
{_fmt_translate_samples(translate_samples)}
{_fmt_visibility_samples(visibility_samples)}

        def Xform "Asset" (
            references = {_usd_asset_ref(usd)}
        ) {{
{_fmt_i4h_asset_child(visual_xform)}
        }}
    }}
"""
            )
        else:
            size = tuple(ent0.get("size") or (0.5, 0.5, 0.5))
            sx, sy, sz = size
            r, g, b = _color_for(etype)
            parts.append(
                f"""
    def Cube "{eid}" {{
        double size = 1
        float3 xformOp:scale = ({sx:.6f}, {sy:.6f}, {sz:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        custom string health:entity_id = "{eid}"
        custom string health:entity_type = "{etype}"
        custom string health:visual = "proxy_fallback"
        color3f[] primvars:displayColor = [({r:.3f}, {g:.3f}, {b:.3f})]
{_fmt_translate_samples(translate_samples)}
{_fmt_visibility_samples(visibility_samples)}
    }}
"""
            )

    structural_hosts = STRUCTURAL_TYPES
    timeline = build_transition_timeline(
        len(frames),
        fps=fps,
        hold_seconds=hold_seconds,
        seconds_per_state=seconds_per_state,
    )

    def _append_beam_timed(
        name: str,
        subject: str,
        host: str,
        pos_state: int,
        vis: dict[int, str],
        rgb: tuple[float, float, float],
    ) -> None:
        if subject not in entity_positions_at_state.get(pos_state, {}):
            return
        if host not in entity_positions_at_state.get(pos_state, {}):
            return
        p_a = entity_positions_at_state[pos_state][subject]
        p_b = entity_positions_at_state[pos_state][host]
        mid, scale = _beam_between(p_a, p_b)
        parts.append(
            f"""
        def Cube "{name}" {{
            double size = 1
            float3 xformOp:translate = {_fmt_vec3(mid)}
            float3 xformOp:scale = ({scale[0]:.4f}, {scale[1]:.4f}, {scale[2]:.4f})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            color3f[] primvars:displayColor = [({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})]
{_fmt_visibility_samples(vis).replace('        ', '            ')}
        }}
"""
        )

    def _hold_visibility(state_idx: int) -> dict[int, str]:
        t0, t1 = timeline.hold_range(state_idx)
        return {t0 - 1: "invisible", t0: "inherited", t1: "inherited", t1 + 1: "invisible"}

    def _tween_visibility(from_state: int) -> dict[int, str]:
        tween_start, tween_end = timeline.tween_range(from_state)
        return {tween_start - 1: "invisible", tween_start: "inherited", tween_end: "invisible"}

    def _demo_edge_visible(subject: str, host: str, entities: dict) -> bool:
        if subject in PAPER_FOCUS_ENTITY_IDS:
            return True
        host_ent = entities.get(host, {})
        return host_ent.get("type") not in structural_hosts

    parts.append('    def Xform "Graph" {')
    for state_idx, frame in enumerate(frames):
        for subject, host in frame.get("edges_present", []):
            if not _demo_edge_visible(subject, host, frame["entities"]):
                continue
            _append_beam_timed(
                f"edge_s{state_idx}_{subject}_{host}",
                subject,
                host,
                state_idx,
                _hold_visibility(state_idx),
                (0.18, 0.55, 0.24),
            )

    for state_idx, frame in enumerate(frames):
        if state_idx == 0:
            continue
        tween_vis = _tween_visibility(state_idx - 1)
        pos_state = state_idx
        for subject, host in frame.get("added_edges", []):
            if not _demo_edge_visible(subject, host, frame["entities"]):
                continue
            _append_beam_timed(
                f"added_s{state_idx}_{subject}_{host}",
                subject,
                host,
                pos_state,
                tween_vis,
                (0.25, 0.85, 0.35),
            )
        for subject, host in frame.get("removed_edges", []):
            if not _demo_edge_visible(subject, host, frames[state_idx - 1]["entities"]):
                continue
            _append_beam_timed(
                f"removed_s{state_idx}_{subject}_{host}",
                subject,
                host,
                state_idx - 1,
                tween_vis,
                (0.75, 0.20, 0.16),
            )

    parts.append("    }")
    parts.append("}")
    return "\n".join(parts) + "\n"


# Grid slots for static asset-identity diagnostic (entity_id, label, grid_x, grid_y).
I4H_IDENTITY_GRID: tuple[tuple[str, str, float, float], ...] = (
    ("bed", "BED", 0.0, 0.0),
    ("bench", "BENCH", 4.0, 0.0),
    ("monitor", "MONITOR", 8.0, 0.0),
    ("cabinet", "CABINET", 12.0, 0.0),
    ("walker", "WALKER", 0.0, 4.0),
    ("wheelchair", "WHEELCHAIR", 4.0, 4.0),
    ("iv_pole", "IV POLE", 8.0, 4.0),
)

_LABEL_COLORS: dict[str, tuple[float, float, float]] = {
    "BED": (0.95, 0.55, 0.15),
    "BENCH": (0.25, 0.75, 0.35),
    "MONITOR": (0.35, 0.55, 0.95),
    "CABINET": (0.65, 0.40, 0.22),
    "WALKER": (0.90, 0.45, 0.12),
    "WHEELCHAIR": (0.18, 0.42, 0.72),
    "IV POLE": (0.70, 0.72, 0.74),
}


def _canonical_asset_name(usd_path: Path) -> str:
    return usd_path.stem


def _fmt_label_plaque(label: str, z_height: float, rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    width = max(1.0, 0.22 * len(label))
    return f"""
        def Xform "LABEL_{label.replace(' ', '_')}" {{
            float3 xformOp:translate = (0.000000, 0.000000, {z_height:.3f})
            uniform token[] xformOpOrder = ["xformOp:translate"]
            custom string health:label = "{label}"

            def Cube "Plaque" {{
                double size = 1
                float3 xformOp:scale = ({width:.3f}, 0.35, 0.05)
                uniform token[] xformOpOrder = ["xformOp:scale"]
                color3f[] primvars:displayColor = [({r:.3f}, {g:.3f}, {b:.3f})]
            }}
        }}
"""


def i4h_asset_identity_check_usda(
    log: dict,
    *,
    i4h_root: Path,
    grid_spacing: float = 4.0,
) -> str:
    """Static diagnostic stage: 7 paper-focus i4h props on a grid with labels.

    No animation, beams, or PhysX. Uses the same child alignment as the transition demo.
    """
    if usd_for_entity_type is None:
        raise RuntimeError("i4h visuals unavailable")

    frames = log.get("frames") or []
    if not frames:
        raise ValueError("transition log has no frames")
    ent_map = frames[0]["entities"]

    parts = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        '    upAxis = "Z"',
        "    metersPerUnit = 1",
        ")",
        "# MedPhyGraph i4h ASSET IDENTITY CHECK (static — no animation)",
        f"# i4h_root = {i4h_root.resolve().as_posix()}",
        "# Open in Isaac Sim; read health:label on each ENTITY_* prim and colored LABEL_* plaque.",
        'def Xform "World" {',
        """
    def DomeLight "DomeLight" {
        float inputs:intensity = 850
        color3f inputs:color = (0.93, 0.95, 0.98)
    }
""",
    ]
    parts.extend(_room_shell_usda_prims((16.0, 8.0, 3.0)))

    grid_xs: list[float] = []
    grid_ys: list[float] = []
    max_top_z = 0.0

    for eid, label, gx, gy in I4H_IDENTITY_GRID:
        if eid not in ent_map:
            continue
        ent = ent_map[eid]
        etype = ent["type"]
        size = tuple(ent.get("size") or (0.5, 0.5, 0.5))
        orig_pos = tuple(ent["pos"])
        world_x = gx * grid_spacing
        world_y = gy * grid_spacing
        logical_pos = (world_x, world_y, float(orig_pos[2]))
        grid_xs.append(world_x)
        grid_ys.append(world_y)

        usd = usd_for_entity_type(etype, root=i4h_root)
        visual_xform = compute_visual_xform(etype, size)
        anchor = logical_floor_anchor(logical_pos, size, etype)
        label_z = float(size[2]) + 0.45
        max_top_z = max(max_top_z, anchor[2] + label_z)
        rel = ENTITY_USD_REL.get(etype, "")
        canon = _canonical_asset_name(usd) if usd else "MISSING"
        label_rgb = _LABEL_COLORS.get(label, (0.9, 0.9, 0.2))

        if usd is not None:
            parts.append(
                f"""
    def Xform "ENTITY_{eid}" {{
        float3 xformOp:translate = {_fmt_vec3(anchor)}
        uniform token[] xformOpOrder = ["xformOp:translate"]
        custom string health:entity_id = "{eid}"
        custom string health:entity_type = "{etype}"
        custom string health:label = "{label}"
        custom string health:logical_center = "({logical_pos[0]:.3f}, {logical_pos[1]:.3f}, {logical_pos[2]:.3f})"
        custom string health:logical_size = "({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f})"
        custom string health:i4h_asset_rel = "{rel}"
        custom string health:i4h_canonical_name = "{canon}"

        def Xform "Asset" (
            references = {_usd_asset_ref(usd)}
        ) {{
{_fmt_i4h_asset_child(visual_xform)}
        }}
{_fmt_label_plaque(label, label_z, label_rgb)}
    }}
"""
            )
        else:
            sx, sy, sz = size
            r, g, b = _color_for(etype)
            parts.append(
                f"""
    def Cube "ENTITY_{eid}" {{
        float3 xformOp:translate = {_fmt_vec3(anchor)}
        float3 xformOp:scale = ({sx:.6f}, {sy:.6f}, {sz:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        custom string health:entity_id = "{eid}"
        custom string health:label = "{label}"
        color3f[] primvars:displayColor = [({r:.3f}, {g:.3f}, {b:.3f})]
{_fmt_label_plaque(label, label_z, label_rgb)}
    }}
"""
            )

    cx = 0.5 * (min(grid_xs) + max(grid_xs)) if grid_xs else 0.0
    cy = 0.5 * (min(grid_ys) + max(grid_ys)) if grid_ys else 0.0
    span = max(
        (max(grid_xs) - min(grid_xs)) if len(grid_xs) > 1 else grid_spacing,
        (max(grid_ys) - min(grid_ys)) if len(grid_ys) > 1 else grid_spacing,
        grid_spacing,
    )
    cam_x, cam_y = cx, cy - span * 2.2
    cam_z = max_top_z + span * 0.85
    look_at = (cx, cy, max_top_z * 0.35)
    rx, ry, rz = _camera_look_rotate_xyz((cam_x, cam_y, cam_z), look_at)

    parts.append(
        f"""
    def Camera "IdentityCheckCamera" {{
        float3 xformOp:translate = ({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f})
        float3 xformOp:rotateXYZ = ({rx:.2f}, {ry:.2f}, {rz:.2f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
        float focalLength = 18
        float horizontalAperture = 20.955
        float2 clippingRange = (0.1, 200)
        custom string health:camera_strategy = "wide_identity_grid"
        custom string health:look_at = "({look_at[0]:.2f}, {look_at[1]:.2f}, {look_at[2]:.2f})"
    }}
"""
    )
    parts.append("}")
    return "\n".join(parts) + "\n"


def write_i4h_asset_identity_check_usda(
    log: dict,
    path: Path | str,
    *,
    i4h_root: Path,
    grid_spacing: float = 4.0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        i4h_asset_identity_check_usda(log, i4h_root=i4h_root, grid_spacing=grid_spacing),
        encoding="utf-8",
    )
    return path


def write_transition_log_usda(
    log: dict,
    path: Path | str,
    *,
    fps: int = 30,
    hold_seconds: float = 2.0,
    seconds_per_state: float = 4.0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        transition_log_to_usda(
            log,
            fps=fps,
            hold_seconds=hold_seconds,
            seconds_per_state=seconds_per_state,
        ),
        encoding="utf-8",
    )
    return path


def write_transition_log_i4h_usda(
    log: dict,
    path: Path | str,
    *,
    i4h_root: Path,
    fps: int = 30,
    hold_seconds: float = 2.0,
    seconds_per_state: float = 4.0,
    focus: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        transition_log_to_i4h_usda(
            log,
            i4h_root=i4h_root,
            fps=fps,
            hold_seconds=hold_seconds,
            seconds_per_state=seconds_per_state,
            focus=focus,
        ),
        encoding="utf-8",
    )
    return path
