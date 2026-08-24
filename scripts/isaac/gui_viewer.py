#!/usr/bin/env python3
"""Isaac Sim GUI — unified i4h hospital/rehab visual twin (display only).

Loads Organs OR room shell + shared_OR / Rheo Sim-Ready props that share the
i4h Material Library (fixes pink/missing MDL materials).

  python scripts/isaac/open_scene.py
  <ISAAC>/python.bat scripts/isaac/gui_viewer.py --scene-json .../scene.json

CF / training still use AABB proxies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-json", type=Path, required=True)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--show-ceiling", action="store_true", default=False)
    parser.add_argument("--i4h-root", type=Path, default=None)
    parser.add_argument("--no-i4h", action="store_true")
    parser.add_argument(
        "--no-room-env",
        action="store_true",
        help="Skip Organs operating-room shell (cube walls only)",
    )
    args = parser.parse_args()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})

    try:
        import carb
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
        from isaacsim.core.experimental.prims import GeomPrim
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

        repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo / "src"))
        from medphygraph.gui_assets import (
            iter_entity_world_parts,
            room_shell_parts,
        )
        from medphygraph.i4h_visuals import (
            ceiling_lamp_usd,
            inventory,
            material_library_dir,
            register_mdl_search_paths,
            resolve_i4h_root,
            room_environment_ready,
            room_environment_usd,
            usd_for_entity_type,
        )
        from medphygraph.schema import HealthScene

        scene = HealthScene.from_dict(json.loads(args.scene_json.read_text(encoding="utf-8")))
        zone = next((e for e in scene.entities if e.entity_type == "zone"), None)
        room = zone.size_xyz if zone else (6.0, 5.0, 3.0)
        i4h_root = None if args.no_i4h else resolve_i4h_root(args.i4h_root)
        inv = inventory(i4h_root)

        # Register Material Library so shared_OR MDL refs resolve
        ml = material_library_dir(root=i4h_root) if i4h_root else None
        if ml is not None:
            try:
                register_mdl_search_paths(carb.settings.get_settings(), ml, ml.parent)
            except Exception as exc:
                print(json.dumps({"mdl_path_warn": str(exc)}))

        def _set_scale(path: str, size_xyz) -> None:
            stage = stage_utils.get_current_stage()
            prim = stage.GetPrimAtPath(path)
            xform = UsdGeom.Xformable(prim)
            ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
            vec = Gf.Vec3f(float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2]))
            if "xformOp:scale" in ops:
                ops["xformOp:scale"].Set(vec)
            else:
                xform.AddScaleOp().Set(vec)

        def _preview_material(path: str, rgb, *, roughness: float = 0.55, metallic: float = 0.0, opacity: float = 1.0) -> None:
            stage = stage_utils.get_current_stage()
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                return
            mat_path = f"{path}/Looks/Preview"
            mat = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
            if opacity < 0.999:
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI(prim).Bind(mat)
            gprim = UsdGeom.Gprim(prim)
            attr = gprim.GetDisplayColorAttr() or gprim.CreateDisplayColorAttr()
            attr.Set([Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))])

        def spawn_box(path: str, xyz, size_xyz, rgb, *, roughness=0.55, metallic=0.0, opacity=1.0) -> None:
            Cube(paths=path, positions=[float(xyz[0]), float(xyz[1]), float(xyz[2])], sizes=1.0)
            _set_scale(path, size_xyz)
            GeomPrim(paths=path, apply_collision_apis=True)
            _preview_material(path, rgb, roughness=roughness, metallic=metallic, opacity=opacity)

        def spawn_i4h(path: str, usd: Path, xyz, *, scale: float | None = None) -> bool:
            stage = stage_utils.get_current_stage()
            prim = stage.DefinePrim(path, "Xform")
            if not prim.IsValid():
                return False
            uri = usd.resolve().as_posix()
            prim.GetReferences().AddReference(uri)
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
            if scale is not None and abs(scale - 1.0) > 1e-3:
                xform.AddScaleOp().Set(Gf.Vec3f(float(scale), float(scale), float(scale)))
            return True

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

        spawned: list[str] = []
        used_i4h: list[dict] = []
        room_loaded = False

        # --- Unified OR / hospital room shell ---
        env_usd = None if (args.no_i4h or args.no_room_env) else room_environment_usd(root=i4h_root)
        if env_usd is not None and not room_environment_ready(root=i4h_root):
            print(
                json.dumps(
                    {
                        "room_env_warn": (
                            "Skipping Organs OR shell: ATLAS_OR textures are missing from "
                            "this i4h install. Props still load; using stylized room shell."
                        )
                    }
                )
            )
            env_usd = None
        if env_usd is not None:
            # Center room; slight scale so it wraps the 6x5 rehab footprint
            if spawn_i4h("/World/Env/OperatingRoom", env_usd, (0.0, 0.0, 0.0), scale=1.0):
                spawned.append("/World/Env/OperatingRoom")
                used_i4h.append({"entity_id": "room_env", "type": "environment", "usd": str(env_usd)})
                room_loaded = True

        lamp = None if args.no_i4h else ceiling_lamp_usd(root=i4h_root)
        if lamp is not None:
            if spawn_i4h("/World/Env/CeilingLamp", lamp, (0.0, 0.0, 2.85)):
                spawned.append("/World/Env/CeilingLamp")
                used_i4h.append({"entity_id": "ceiling_lamp", "type": "lamp", "usd": str(lamp)})

        # Minimal floor pad when room env present; full cube shell otherwise
        if room_loaded:
            spawn_box(
                "/World/Room/floor_pad",
                (0.0, 0.0, 0.01),
                (room[0] * 0.95, room[1] * 0.95, 0.02),
                (0.78, 0.80, 0.82),
                roughness=0.75,
            )
            spawned.append("/World/Room/floor_pad")
        else:
            for name, x, y, z, sx, sy, sz, rgb in room_shell_parts(room):
                path = f"/World/Room/{name}"
                rough = 0.7 if "floor" in name else 0.85
                spawn_box(path, (x, y, z), (sx, sy, sz), rgb, roughness=rough)
                spawned.append(path)

        if args.show_ceiling and not room_loaded:
            L, W, H = room
            spawn_box(
                "/World/Room/ceiling",
                (0.0, 0.0, H),
                (L, W, 0.06),
                (0.94, 0.95, 0.96),
                roughness=0.9,
                opacity=0.35,
            )

        skip = {"zone", "floor", "wall", "ceiling"}
        for e in scene.entities:
            if e.entity_type in skip:
                continue
            usd = usd_for_entity_type(e.entity_type, root=i4h_root) if i4h_root else None
            floor_z = float(e.pose_xyz[2] - 0.5 * e.size_xyz[2])
            if e.entity_type in ("patient_lift",):
                floor_z = float(max(e.pose_xyz[2] - 0.2, 1.8))
            if usd is not None:
                path = f"/World/I4H/{e.entity_id}"
                if spawn_i4h(path, usd, (e.pose_xyz[0], e.pose_xyz[1], max(0.0, floor_z))):
                    spawned.append(path)
                    used_i4h.append({"entity_id": e.entity_id, "type": e.entity_type, "usd": str(usd)})
                    continue
            for name, x, y, z, sx, sy, sz, rgb in iter_entity_world_parts(
                e.entity_id, e.entity_type, e.pose_xyz, e.size_xyz
            ):
                path = f"/World/Props/{name}"
                metal = 0.65 if e.entity_type in ("walker", "iv_pole", "patient_lift") else 0.05
                rough = 0.25 if metal > 0.5 else 0.55
                spawn_box(path, (x, y, z), (sx, sy, sz), rgb, roughness=rough, metallic=metal)
                spawned.append(path)

        root = stage.DefinePrim("/World/Meta", "Xform")
        root.CreateAttribute("health:scene_id", Sdf.ValueTypeNames.String).Set(scene.scene_id)
        root.CreateAttribute("health:visual_mode", Sdf.ValueTypeNames.String).Set(
            "unified_i4h" if room_loaded else ("i4h_props" if used_i4h else "stylized")
        )

        print(
            json.dumps(
                {
                    "ok": True,
                    "scene_id": scene.scene_id,
                    "n_prims": len(spawned),
                    "mode": "unified_i4h" if room_loaded else ("i4h+stylized" if used_i4h else "stylized_gui"),
                    "i4h_root": inv.get("root"),
                    "material_library": inv.get("material_library"),
                    "room_env": inv.get("room_env"),
                    "i4h_refs": used_i4h,
                    "hint": "Press F to Frame All. Unified hospital look; CF metrics still AABB.",
                },
                indent=2,
            )
        )

        while simulation_app.is_running():
            simulation_app.update()
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
