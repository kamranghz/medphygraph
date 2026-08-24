#!/usr/bin/env python3
"""Isaac Sim: animate a transition_log.json scenario and draw the live
SupportedBy graph as colored 3D beams in the viewport.

This is gui_viewer.py's spawn/material pattern (same Cube + material
helpers, same room/lighting setup) extended two ways:
  1. Multiple states instead of one -- objects tween smoothly between
     consecutive frames of the log instead of a single static layout.
  2. Each present support edge is drawn as a thin colored 3D "beam"
     between subject and host (green = kept, bright/emissive green =
     just added, red = just removed, shown for one transition then
     cleared) -- built from the same Cube-and-material primitives as
     the rest of the scene.

Run via Isaac Sim's python.bat, same as gui_viewer.py:
  <ISAAC>/python.bat scripts/isaac/graph_update_viewer.py \\
      --transition-log runs/transition_demo/transition_log.json

Or use the launcher:
  python scripts/isaac/open_transition_viewer.py

Generate --transition-log first with (regular medphygraph env, needs the
checkpoint): python scripts/isaac/transition_demo_log.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


def _beam_transform(p_a, p_b):
    """Return (midpoint_xyz, length, unit_direction) for a beam from a to b."""
    ax, ay, az = p_a
    bx, by, bz = p_b
    dx, dy, dz = bx - ax, by - ay, bz - az
    length = math.sqrt(dx * dx + dy * dy + dz * dz) or 1e-6
    mid = ((ax + bx) / 2.0, (ay + by) / 2.0, (az + bz) / 2.0)
    return mid, length, (dx / length, dy / length, dz / length)


def _beam_rotation_quat(direction):
    """Align a unit cube's local +Z axis to ``direction`` (world space)."""
    from pxr import Gf

    target = Gf.Vec3d(float(direction[0]), float(direction[1]), float(direction[2]))
    if target.GetLength() < 1e-9:
        return Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    target.Normalize()
    rot = Gf.Rotation()
    rot.SetRotateInto(Gf.Vec3d(0.0, 0.0, 1.0), target)
    return Gf.Quatf(rot.GetQuat())


STRUCTURAL_SKIP_TYPES = frozenset({"zone", "floor", "wall", "ceiling"})


def _is_structural_entity(ent: dict) -> bool:
    return ent.get("type") in STRUCTURAL_SKIP_TYPES


def _filter_edges_for_display(edges, entities: dict) -> list[tuple[str, str]]:
    """Hide floor/wall/ceiling host edges during steady states (keeps the viewport readable)."""
    filtered: list[tuple[str, str]] = []
    for subject_id, host_id in edges:
        host = entities.get(host_id, {})
        if host.get("type") in STRUCTURAL_SKIP_TYPES:
            continue
        filtered.append((subject_id, host_id))
    return filtered


def _pump_app(simulation_app, seconds: float) -> bool:
    """Pump Isaac frames for ``seconds``. Returns False if the app stopped."""
    t0 = time.time()
    while simulation_app.is_running() and time.time() - t0 < seconds:
        simulation_app.update()
    return simulation_app.is_running()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transition-log", type=Path, required=True)
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--i4h-root", type=Path, default=None)
    p.add_argument("--no-i4h", action="store_true")
    p.add_argument("--no-room-env", action="store_true")
    p.add_argument(
        "--room-env",
        action="store_true",
        help="Force Organs OR shell even when ATLAS_OR textures look incomplete",
    )
    p.add_argument("--seconds-per-state", type=float, default=3.0, help="tween duration per transition")
    p.add_argument("--hold-seconds", type=float, default=1.5, help="pause at each settled state")
    p.add_argument(
        "--startup-delay",
        type=float,
        default=5.0,
        help="seconds to wait after scene load before Python-driven animation begins",
    )
    p.add_argument("--loop", action="store_true", help="replay the sequence forever (Ctrl+C to stop)")
    p.add_argument(
        "--show-structural-edges",
        action="store_true",
        help="also draw support edges to floor/wall/ceiling hosts during steady states",
    )
    p.add_argument("--debug-positions", action="store_true", help="print live world positions each tween step")
    args = p.parse_args()

    log = json.loads(args.transition_log.read_text(encoding="utf-8"))
    frames = log["frames"]
    if len(frames) < 1:
        raise SystemExit(f"No frames in {args.transition_log}")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})

    try:
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
        from isaacsim.core.experimental.prims import GeomPrim
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

        repo = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo / "src"))
        from medphygraph.gui_assets import iter_entity_world_parts, room_shell_parts
        from medphygraph.i4h_visuals import (
            ROOM_ENV_TEXTURE_REL,
            ceiling_lamp_usd,
            inventory,
            material_library_dir,
            register_mdl_search_paths,
            resolve_i4h_root,
            room_environment_ready,
            room_environment_usd,
            usd_for_entity_type,
        )

        i4h_root = None if args.no_i4h else resolve_i4h_root(args.i4h_root)
        inv = inventory(i4h_root)

        # ---- material / spawn helpers: identical to gui_viewer.py ----
        def _set_scale(path: str, size_xyz) -> None:
            stage = stage_utils.get_current_stage()
            xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
            ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
            vec = Gf.Vec3f(float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2]))
            if "xformOp:scale" in ops:
                ops["xformOp:scale"].Set(vec)
            else:
                xform.AddScaleOp().Set(vec)

        def _material(path: str, rgb, *, roughness=0.55, metallic=0.0, opacity=1.0, emissive=None) -> None:
            stage = stage_utils.get_current_stage()
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                return
            mat_path = f"{path}/Looks/Preview"
            mat = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in rgb]))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
            if opacity < 0.999:
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
            if emissive is not None:
                shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(*[float(c) for c in emissive])
                )
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI(prim).Bind(mat)
            gprim = UsdGeom.Gprim(prim)
            attr = gprim.GetDisplayColorAttr() or gprim.CreateDisplayColorAttr()
            attr.Set([Gf.Vec3f(*[float(c) for c in rgb])])

        def spawn_box(path: str, xyz, size_xyz, rgb, **kw) -> None:
            Cube(paths=path, positions=[float(xyz[0]), float(xyz[1]), float(xyz[2])], sizes=1.0)
            _set_scale(path, size_xyz)
            GeomPrim(paths=path, apply_collision_apis=True)
            _material(path, rgb, **kw)

        def spawn_i4h(path: str, usd: Path, xyz, *, scale: float | None = None) -> bool:
            stage = stage_utils.get_current_stage()
            prim = stage.DefinePrim(path, "Xform")
            if not prim.IsValid():
                return False
            prim.GetReferences().AddReference(usd.resolve().as_posix())
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
            if scale is not None and abs(scale - 1.0) > 1e-3:
                xform.AddScaleOp().Set(Gf.Vec3f(float(scale), float(scale), float(scale)))
            return True

        def move_prim(path: str, xyz) -> None:
            stage = stage_utils.get_current_stage()
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                return
            xform = UsdGeom.Xformable(prim)
            ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
            vec = Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2]))
            if "xformOp:translate" in ops:
                ops["xformOp:translate"].Set(vec)
            else:
                xform.AddTranslateOp().Set(vec)

        def set_prim_visible(path: str, visible: bool) -> None:
            stage = stage_utils.get_current_stage()
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                imageable = UsdGeom.Imageable(prim)
                imageable.MakeVisible() if visible else imageable.MakeInvisible()

        def world_translation(path: str) -> tuple[float, float, float]:
            stage = stage_utils.get_current_stage()
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                return (0.0, 0.0, 0.0)
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
            t = xf.ExtractTranslation()
            return (float(t[0]), float(t[1]), float(t[2]))

        BEAM_THICKNESS = 0.02
        active_beam_paths: set[str] = set()

        def clear_graph_beams() -> None:
            stage = stage_utils.get_current_stage()
            for path in list(active_beam_paths):
                if stage.GetPrimAtPath(path).IsValid():
                    stage.RemovePrim(path)
            active_beam_paths.clear()

        def spawn_or_update_beam(
            path: str, p_a, p_b, rgb, *, thickness=BEAM_THICKNESS, emissive=None, opacity=1.0
        ) -> None:
            mid, length, direction = _beam_transform(p_a, p_b)
            stage = stage_utils.get_current_stage()
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                Cube(paths=path, positions=[mid[0], mid[1], mid[2]], sizes=1.0)
                GeomPrim(paths=path, apply_collision_apis=False)
                prim = stage.GetPrimAtPath(path)
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(mid[0], mid[1], mid[2]))
            xform.AddOrientOp().Set(_beam_rotation_quat(direction))
            xform.AddScaleOp().Set(Gf.Vec3f(thickness, thickness, float(length)))
            _material(path, rgb, roughness=0.2, metallic=0.1, opacity=opacity, emissive=emissive)
            active_beam_paths.add(path)

        def caption(text: str) -> None:
            stage = stage_utils.get_current_stage()
            root = stage.GetPrimAtPath("/World/Meta") or stage.DefinePrim("/World/Meta", "Xform")
            root.CreateAttribute("health:caption", Sdf.ValueTypeNames.String).Set(text)
            print(f"[graph_update_viewer] {text}", flush=True)

        def require_running(phase: str) -> bool:
            if simulation_app.is_running():
                return True
            print(f"[graph_update_viewer] Isaac stopped during: {phase}", flush=True)
            return False

        # ---- build the room + lighting once (identical to gui_viewer.py) ----
        first_entities = frames[0]["entities"]
        zone_size = (6.0, 5.0, 3.0)

        stage_utils.create_new_stage()
        GroundPlane("/World/GroundPlane", positions=[0, 0, -0.02])
        stage = stage_utils.get_current_stage()
        DistantLight("/World/KeyLight", positions=[-2, -3, 4]).set_intensities(2200)
        try:
            dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
            dome.CreateIntensityAttr(550)
            dome.CreateColorAttr(Gf.Vec3f(0.95, 0.96, 0.98))
        except Exception:
            DistantLight("/World/FillLight", positions=[3, 2, 3]).set_intensities(700)

        ml = material_library_dir(root=i4h_root) if i4h_root else None
        if ml is not None:
            try:
                import carb

                register_mdl_search_paths(carb.settings.get_settings(), ml, ml.parent)
            except Exception as exc:
                print(json.dumps({"mdl_path_warn": str(exc)}))

        room_loaded = False
        env_usd = None
        if not (args.no_i4h or args.no_room_env):
            env_usd = room_environment_usd(root=i4h_root)
            if (
                env_usd is not None
                and not args.room_env
                and not room_environment_ready(root=i4h_root)
            ):
                print(
                    json.dumps(
                        {
                            "room_env_warn": (
                                "Skipping Organs OR shell: ATLAS_OR textures are missing from "
                                "this i4h install. Props still load; using stylized room shell."
                            ),
                            "missing_texture": ROOM_ENV_TEXTURE_REL,
                        }
                    )
                )
                env_usd = None
        if env_usd is not None and spawn_i4h("/World/Env/OperatingRoom", env_usd, (0.0, 0.0, 0.0)):
            room_loaded = True
        lamp = None if args.no_i4h else ceiling_lamp_usd(root=i4h_root)
        if lamp is not None:
            spawn_i4h("/World/Env/CeilingLamp", lamp, (0.0, 0.0, 2.85))
        if room_loaded:
            spawn_box(
                "/World/Room/floor_pad",
                (0, 0, 0.01),
                (zone_size[0] * 0.95, zone_size[1] * 0.95, 0.02),
                (0.78, 0.80, 0.82),
                roughness=0.75,
            )
        else:
            for name, x, y, z, sx, sy, sz, rgb in room_shell_parts(zone_size):
                spawn_box(f"/World/Room/{name}", (x, y, z), (sx, sy, sz), rgb, roughness=0.8)

        entity_paths: dict[str, str] = {}
        entity_part_paths: dict[str, list[str]] = {}
        entity_anchors: dict[str, tuple[float, float, float]] = {}

        def spawn_entity(eid: str, ent: dict) -> None:
            pos = tuple(ent["pos"])
            entity_anchors[eid] = pos
            if _is_structural_entity(ent):
                return
            size = tuple(ent.get("size") or (0.5, 0.5, 0.5))
            etype = ent["type"]
            usd = usd_for_entity_type(etype, root=i4h_root) if i4h_root else None
            floor_z = float(pos[2] - 0.5 * size[2])
            if etype in ("patient_lift",):
                floor_z = float(max(pos[2] - 0.2, 1.8))
            if usd is not None:
                path = f"/World/I4H/{eid}"
                if spawn_i4h(path, usd, (pos[0], pos[1], max(0.0, floor_z))):
                    entity_paths[eid] = path
                    entity_part_paths[eid] = [path]
                    return
            parts: list[str] = []
            for name, x, y, z, sx, sy, sz, rgb in iter_entity_world_parts(eid, etype, pos, size):
                path = f"/World/Props/{name}"
                metal = 0.65 if etype in ("walker", "iv_pole", "patient_lift") else 0.05
                rough = 0.25 if metal > 0.5 else 0.55
                spawn_box(path, (x, y, z), (sx, sy, sz), rgb, roughness=rough, metallic=metal)
                parts.append(path)
            entity_paths[eid] = parts[0] if parts else f"/World/Props/{eid}"
            entity_part_paths[eid] = parts or [entity_paths[eid]]

        all_entity_ids = {eid for frame in frames for eid in frame["entities"]}
        for eid in sorted(all_entity_ids):
            ent = next(frame["entities"][eid] for frame in frames if eid in frame["entities"])
            spawn_entity(eid, ent)

        root_meta = stage.DefinePrim("/World/Meta", "Xform")
        root_meta.CreateAttribute("health:scene_id", Sdf.ValueTypeNames.String).Set(log.get("scene_id_base", ""))
        stage.DefinePrim("/World/Graph", "Xform")
        print(
            json.dumps(
                {
                    "ok": True,
                    "n_frames": len(frames),
                    "n_entities": len(entity_paths),
                    "n_anchors": len(entity_anchors),
                    "i4h_root": inv.get("root"),
                },
                indent=2,
            )
        )

        def lerp(a, b, t):
            return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))

        def edge_endpoint(eid: str) -> tuple[float, float, float] | None:
            if eid in entity_paths:
                return world_translation(entity_paths[eid])
            if eid in entity_anchors:
                return entity_anchors[eid]
            return None

        def move_entity(eid: str, pos) -> None:
            entity_anchors[eid] = tuple(pos)
            if eid not in entity_part_paths:
                return
            ent = {"pos": list(pos), "type": "equipment_cart", "size": [0.5, 0.5, 0.5]}
            for frame in frames:
                if eid in frame["entities"]:
                    ent = frame["entities"][eid]
                    break
            size = tuple(ent.get("size") or (0.5, 0.5, 0.5))
            etype = ent["type"]
            parts = iter_entity_world_parts(eid, etype, tuple(pos), size)
            for path, (name, x, y, z, *_rest) in zip(entity_part_paths[eid], parts):
                move_prim(path, (x, y, z))
            if len(entity_part_paths[eid]) == 1 and entity_part_paths[eid][0].startswith("/World/I4H/"):
                floor_z = float(pos[2] - 0.5 * size[2])
                move_prim(entity_paths[eid], (pos[0], pos[1], max(0.0, floor_z)))

        def render_edges(
            edges,
            rgb,
            *,
            entities: dict | None = None,
            emissive=None,
            opacity=1.0,
            prefix="edge",
            highlight: bool = False,
        ) -> None:
            clear_graph_beams()
            entities = entities or {}
            display_edges = (
                list(edges)
                if highlight or args.show_structural_edges
                else _filter_edges_for_display(edges, entities)
            )
            for i, (s, h) in enumerate(display_edges):
                p_a = edge_endpoint(s)
                p_b = edge_endpoint(h)
                if p_a is None or p_b is None:
                    continue
                path = f"/World/Graph/{prefix}_{i}_{s}_{h}"
                spawn_or_update_beam(path, p_a, p_b, rgb, emissive=emissive, opacity=opacity)
            if args.debug_positions and edges:
                print(
                    json.dumps(
                        {
                            "edge_sample": edges[0],
                            "p_a": p_a if edges else None,
                            "p_b": p_b if edges else None,
                        }
                    )
                )

        def settle(frame: dict, hold_s: float) -> None:
            edges_present = [tuple(e) for e in frame.get("edges_present", [])]
            render_edges(
                edges_present,
                (0.18, 0.55, 0.24),
                entities=frame.get("entities", {}),
            )
            if not _pump_app(simulation_app, hold_s):
                return

        def sync_visibility(frame: dict) -> None:
            present = set(frame["entities"])
            for eid, paths in entity_part_paths.items():
                vis = eid in present
                for path in paths:
                    set_prim_visible(path, vis)

        caption("Preview mode: Python-driven animation (not USDA timeline). Press F to frame.")
        if args.startup_delay > 0 and not _pump_app(simulation_app, args.startup_delay):
            print("[graph_update_viewer] Isaac closed during startup delay.", flush=True)
            return 1
        if not require_running("startup"):
            return 1

        prev_frame = frames[0]
        for eid in prev_frame["entities"]:
            move_entity(eid, prev_frame["entities"][eid]["pos"])
        sync_visibility(prev_frame)
        caption(f"state {prev_frame['state_index']}: {prev_frame['operation']}")
        settle(prev_frame, args.hold_seconds)

        def run_sequence():
            nonlocal prev_frame
            for frame in frames[1:]:
                if not require_running(f"before {frame['operation']}"):
                    return
                caption(f"{prev_frame['operation']} -> {frame['operation']}")
                removed = [tuple(e) for e in frame.get("removed_edges", [])]
                if removed:
                    render_edges(
                        removed,
                        (0.75, 0.20, 0.16),
                        entities=frame.get("entities", {}),
                        prefix="removed",
                        highlight=True,
                    )
                steps = max(1, int(args.seconds_per_state * 30))
                for step in range(steps + 1):
                    if not simulation_app.is_running():
                        require_running(f"tween {frame['operation']}")
                        return
                    t = step / steps
                    for eid in set(prev_frame["entities"]) | set(frame["entities"]):
                        if eid in prev_frame["entities"] and eid in frame["entities"]:
                            move_entity(
                                eid,
                                lerp(prev_frame["entities"][eid]["pos"], frame["entities"][eid]["pos"], t),
                            )
                        elif eid in frame["entities"]:
                            move_entity(eid, frame["entities"][eid]["pos"])
                    if removed and t > 0.5:
                        clear_graph_beams()
                    simulation_app.update()
                sync_visibility(frame)
                added = [tuple(e) for e in frame.get("added_edges", [])]
                if added:
                    render_edges(
                        added,
                        (0.25, 0.85, 0.35),
                        entities=frame.get("entities", {}),
                        emissive=(0.15, 0.5, 0.2),
                        prefix="added",
                        highlight=True,
                    )
                    if not _pump_app(simulation_app, min(0.75, args.hold_seconds * 0.5)):
                        return
                settle(frame, args.hold_seconds)
                prev_frame = frame

        run_sequence()
        if not simulation_app.is_running():
            print("[graph_update_viewer] Isaac window closed.", flush=True)
            return 0
        caption("Animation complete. Close the Isaac window to exit (or relaunch with --loop).")
        if args.loop:
            while simulation_app.is_running():
                prev_frame = frames[0]
                for eid in prev_frame["entities"]:
                    move_entity(eid, prev_frame["entities"][eid]["pos"])
                sync_visibility(prev_frame)
                run_sequence()
        else:
            while simulation_app.is_running():
                simulation_app.update()
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
