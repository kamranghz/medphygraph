"""Isaac Sim worker: spawn rehab AABB proxies and run one counterfactual pair.

IMPORTANT: Cube size must match entity size_xyz via scale (not max-edge), and
objects must settle before labeling. Unstable factual rollouts are rejected.

Run via Isaac Sim python.bat only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--settle", type=int, default=45)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})

    try:
        import numpy as np
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
        from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Gf, Usd, UsdGeom

        repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo / "src"))
        from medphygraph.labeling import DT, label_from_rollout
        from medphygraph.schema import HealthScene, STRUCTURAL_TYPES

        scene = HealthScene.from_dict(json.loads(args.scene.read_text(encoding="utf-8")))
        nodes = scene.entity_map()
        if args.subject not in nodes or args.host not in nodes:
            raise KeyError("subject/host not in scene")

        def _rest_z(e) -> float:
            """Place COM so AABB bottom sits just above GroundPlane z=0."""
            return float(0.5 * e.size_xyz[2] + 0.002)

        def _set_scale(path: str, size_xyz) -> None:
            stage = stage_utils.get_current_stage()
            prim = stage.GetPrimAtPath(path)
            xform = UsdGeom.Xformable(prim)
            # Cube default size=1 → scale = size_xyz
            ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
            if "xformOp:scale" in ops:
                ops["xformOp:scale"].Set(Gf.Vec3f(float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])))
            else:
                xform.AddScaleOp().Set(Gf.Vec3f(float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])))

        def build_stage(*, disable_host: bool) -> None:
            stage_utils.create_new_stage()
            GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
            DistantLight("/DistantLight").set_intensities(300)
            for e in scene.entities:
                if e.entity_type == "zone":
                    continue
                if disable_host and e.entity_id == args.host:
                    continue
                # Don't duplicate ground with a falling "floor" cube; use GroundPlane.
                if e.entity_type == "floor":
                    continue
                path = f"/World/{e.entity_id}"
                z = _rest_z(e) if e.movable else float(e.pose_xyz[2])
                # unit cube then scale to AABB
                Cube(paths=path, positions=[float(e.pose_xyz[0]), float(e.pose_xyz[1]), z], sizes=1.0)
                _set_scale(path, e.size_xyz)
                kinematic = e.entity_type in STRUCTURAL_TYPES or (not e.movable)
                if not kinematic:
                    RigidPrim(paths=path)
                GeomPrim(paths=path, apply_collision_apis=True)

        def read_pos(entity_id: str) -> list[float]:
            stage = stage_utils.get_current_stage()
            prim = stage.GetPrimAtPath(f"/World/{entity_id}")
            if not prim or not prim.IsValid():
                return list(nodes[entity_id].pose_xyz)
            xform = UsdGeom.Xformable(prim)
            t = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            return [float(t[0]), float(t[1]), float(t[2])]

        def rollout(disable_host: bool) -> tuple[list[list[float]], list[float], dict]:
            build_stage(disable_host=disable_host)
            SimulationManager.set_physics_dt(DT)
            app_utils.play()
            simulation_app.update()
            # settle
            for _ in range(args.settle):
                SimulationManager.step()
                simulation_app.update()
            traj = []
            contact = []
            for _ in range(args.frames):
                SimulationManager.step()
                simulation_app.update()
                traj.append(read_pos(args.subject))
                contact.append(0.0 if disable_host else 1.0)
            app_utils.stop()
            z = np.asarray(traj)[:, 2]
            stable = bool(np.std(z) < 0.05 and (z[0] - z[-1]) < 0.05)
            return traj, contact, {"stable_factual_window": stable if not disable_host else None}

        fact_pos, fact_c, fact_meta = rollout(False)
        cf_pos, cf_c, _ = rollout(True)
        zf = np.asarray(fact_pos)[:, 2]
        zc = np.asarray(cf_pos)[:, 2]
        factual_stable = bool(np.std(zf) < 0.05 and (zf[0] - zf[-1]) < 0.08)

        # Relative support loss: CF must fall *more* than factual residual motion
        rel_drop = float(max(0.0, (zf[-1] - zc[-1])))
        decision = label_from_rollout(
            subject_id=args.subject,
            host_id=args.host,
            z_factual=zf,
            z_counterfactual=zc,
            contact_factual=np.asarray(fact_c),
            contact_counterfactual=np.asarray(cf_c),
            structural_support_remaining=False,
        )
        # Guard: if factual itself is unstable, do not trust a positive label
        if decision.positive and not factual_stable:
            decision.positive = False
            decision.reasons = list(decision.reasons) + ["rejected_unstable_factual"]
        # Guard: require relative drop vs factual end state
        if decision.positive and rel_drop < 0.08:
            decision.positive = False
            decision.reasons = list(decision.reasons) + ["rejected_small_relative_drop"]

        out = {
            "backend": "isaac_sim",
            "scene_id": scene.scene_id,
            "subject_id": args.subject,
            "host_id": args.host,
            "factual_positions": fact_pos,
            "counterfactual_positions": cf_pos,
            "factual_stable": factual_stable,
            "relative_drop_m": rel_drop,
            "label": decision.to_dict(),
            "meta": fact_meta,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("wrote", args.out, "positive=", decision.positive, "factual_stable=", factual_stable)
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
