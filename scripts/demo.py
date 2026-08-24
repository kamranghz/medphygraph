#!/usr/bin/env python3
"""Score one scene with CF-SupportNet and write a State-Consistency twin.

Works with the public Hugging Face layout:
  - Prefer precomputed features from dataset_hard when scene_id matches
  - Otherwise recompute analytic AABB counterfactual features from the scene
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medphygraph.paths import checkpoint_seed0, dataset_hard_json, procedural_scenes_root
from medphygraph import CFSupportNet, ModelConfig, PhysicalSceneGraph, count_violations, write_twin
from medphygraph.analytic_physics import run_counterfactual_pair
from medphygraph.candidates import _xy_sep
from medphygraph.features import trajectory_features
from medphygraph.schema import HealthScene


def _default_scene_dir() -> Path | None:
    root = procedural_scenes_root()
    if not root.is_dir():
        return None
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        if (path / "graph_initial.json").is_file() and (path / "scene.json").is_file():
            return path
    return None


def _recompute_candidate_features(scene: HealthScene, subject_id: str, host_id: str) -> np.ndarray | None:
    nodes = scene.entity_map()
    if subject_id not in nodes or host_id not in nodes:
        return None
    result = run_counterfactual_pair(scene, subject_id=subject_id, host_id=host_id, n_frames=60)
    cf_pos = np.asarray(result["counterfactual"]["positions"][subject_id], dtype=float)
    contact = np.asarray(result["counterfactual"]["contact_subject_host"], dtype=float)
    subject, host = nodes[subject_id], nodes[host_id]
    return trajectory_features(
        positions_subject=cf_pos,
        contact=contact,
        host_removed=True,
        geom_xy_sep=_xy_sep(subject, host),
        geom_vertical_gap=float(subject.aabb_min()[2] - host.aabb_max()[2]),
    )


def main() -> int:
    default_scene = _default_scene_dir()
    p = argparse.ArgumentParser(
        description="Score one scene with CF-SupportNet, apply State Consistency, and write the twin update."
    )
    p.add_argument(
        "--scene-dir",
        type=Path,
        default=default_scene,
        help="Scene directory containing graph_initial.json + scene.json "
        "(default: first procedural scene after download.py)",
    )
    p.add_argument("--checkpoint", type=Path, default=checkpoint_seed0())
    p.add_argument("--dataset", type=Path, default=dataset_hard_json())
    p.add_argument("--ratio", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--thr", type=float, default=0.5)
    args = p.parse_args()

    if args.scene_dir is None:
        raise SystemExit(
            "No --scene-dir provided and no procedural scenes found. "
            "Run scripts/download.py first, or pass --scene-dir explicitly."
        )
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}. Run scripts/download.py first.")

    scene_dir = args.scene_dir
    if not (scene_dir / "graph_initial.json").is_file():
        raise SystemExit(f"Missing graph_initial.json under {scene_dir}")
    if not (scene_dir / "scene.json").is_file():
        raise SystemExit(f"Missing scene.json under {scene_dir}")

    out = args.out or (scene_dir / "twin")
    g = PhysicalSceneGraph.load(scene_dir / "graph_initial.json")
    scene = HealthScene.from_dict(json.loads((scene_dir / "scene.json").read_text(encoding="utf-8")))

    feature_source = "recompute_analytic_cf"
    samples: list[dict] = []
    if args.dataset.is_file():
        raw = json.loads(args.dataset.read_text(encoding="utf-8"))
        samples = [s for s in raw.get("samples", []) if s.get("scene_id") == g.scene.scene_id]
        if samples:
            feature_source = "dataset_hard"

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CFSupportNet(ModelConfig(hidden=int(ckpt.get("config", {}).get("hidden", 64))))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    probs: dict[tuple[str, str], float] = {}
    with torch.no_grad():
        if feature_source == "dataset_hard":
            for sample in samples:
                key = str(args.ratio)
                x = np.asarray(sample["features_partial"][key], dtype=np.float32)
                _, prob = model(torch.from_numpy(x).unsqueeze(0))
                probs[(sample["subject_id"], sample["host_id"])] = float(prob.item())
        else:
            for subject_id, host_id in list(g.edges.keys()):
                feats = _recompute_candidate_features(scene, subject_id, host_id)
                if feats is None:
                    continue
                _, prob = model(torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0))
                probs[(subject_id, host_id)] = float(prob.item())

    if not probs:
        raise SystemExit(
            f"No scorable candidates for scene {g.scene.scene_id} "
            f"(feature_source={feature_source})"
        )

    for (sid, hid), pr in probs.items():
        if (sid, hid) not in g.edges:
            g.add_candidate(sid, hid)
        g.edges[(sid, hid)].score = pr
        g.edges[(sid, hid)].confidence = pr
        g.edges[(sid, hid)].present = pr >= args.thr
        g.edges[(sid, hid)].evidence_source = "health_dyphygraph"

    viol_before = count_violations(g)
    result = write_twin(g, out_dir=out, evidence_source="health_dyphygraph", apply_graph_consistency=True)
    viol_after = count_violations(g)

    summary = {
        "scene_id": g.scene.scene_id,
        "scene_dir": str(scene_dir),
        "feature_source": feature_source,
        "n_candidates_scored": len(probs),
        "violations_before_consistency": viol_before,
        "violations_after_write_back": viol_after,
        "write_back": result.to_dict(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "update_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
