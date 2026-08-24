#!/usr/bin/env python3
"""Generate counterfactual dataset for DyPhyGraph-Health (P0-6..P0-9).

Default backend: analytic geometry physics (reproducible, no Isaac required).
Optional: --backend isaac launches Isaac Sim python.bat worker per pair (slow).

Also writes .usda proxy stages for each scene (Isaac-loadable).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_setup_spec = importlib.util.spec_from_file_location(
  "eval._path_setup", Path(__file__).with_name("_path_setup.py")
)
assert _setup_spec and _setup_spec.loader
_setup_mod = importlib.util.module_from_spec(_setup_spec)
_setup_spec.loader.exec_module(_setup_mod)

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from medphygraph.analytic_physics import run_counterfactual_pair
from medphygraph.candidates import _xy_sep, generate_candidates
from medphygraph.features import pack_sample, trajectory_features
from medphygraph.isaac_sim import resolve_health_isaac_config
from medphygraph.labeling import PROTOCOL_DOC
from medphygraph.schema import HealthScene
from medphygraph.scene_graph import PhysicalSceneGraph
from medphygraph.usd_authoring import write_scene_usda

ROOT = Path(__file__).resolve().parents[2]


def _geom_feats(scene: HealthScene, subject_id: str, host_id: str) -> tuple[float, float]:
    nodes = scene.entity_map()
    s, h = nodes[subject_id], nodes[host_id]
    sep = _xy_sep(s, h)
    gap = float(s.aabb_min()[2] - h.aabb_max()[2])
    return sep, gap


def process_scene_analytic(scene_dir: Path, *, n_frames: int) -> dict:
    scene = HealthScene.from_dict(json.loads((scene_dir / "scene.json").read_text(encoding="utf-8")))
    write_scene_usda(scene, scene_dir / "scene.usda")
    cand_g, stats = generate_candidates(scene)

    samples = []
    gt = PhysicalSceneGraph(scene=scene)
    for edge in cand_g.edges.values():
        result = run_counterfactual_pair(
            scene,
            subject_id=edge.subject_id,
            host_id=edge.host_id,
            n_frames=n_frames,
        )
        label = int(result["label"]["positive"])
        gt.add_candidate(edge.subject_id, edge.host_id)
        gt.set_gt(edge.subject_id, edge.host_id, positive=bool(label))

        cf_pos = np.asarray(result["counterfactual"]["positions"][edge.subject_id], dtype=float)
        contact = np.asarray(result["counterfactual"]["contact_subject_host"], dtype=float)
        sep, gap = _geom_feats(scene, edge.subject_id, edge.host_id)
        feats = trajectory_features(
            positions_subject=cf_pos,
            contact=contact,
            host_removed=True,
            geom_xy_sep=sep,
            geom_vertical_gap=gap,
        )
        sample = pack_sample(
            scene_id=scene.scene_id,
            subject_id=edge.subject_id,
            host_id=edge.host_id,
            label=label,
            feats_full=feats,
            seed=hash((scene.scene_id, edge.subject_id, edge.host_id)) % (2**31),
        )
        sample["label_decision"] = result["label"]
        sample["backend"] = "analytic"
        samples.append(sample)

        # save raw pair rollout
        pair_dir = scene_dir / "rollouts" / f"{edge.subject_id}__{edge.host_id}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        (pair_dir / "pair.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    for e in gt.edges.values():
        e.present = bool(e.is_gt)
        e.score = 1.0 if e.is_gt else 0.0
    gt.save(scene_dir / "graph_gt.json")

    # refresh initial candidates without labels present
    init = PhysicalSceneGraph(scene=scene)
    for e in cand_g.edges.values():
        init.add_candidate(e.subject_id, e.host_id)
        if e.key in gt.edges:
            init.edges[e.key].is_gt = gt.edges[e.key].is_gt
    init.save(scene_dir / "graph_initial.json")

    return {
        "scene_id": scene.scene_id,
        "n_candidates": len(samples),
        "n_pos": sum(s["label"] for s in samples),
        "n_neg": sum(1 - s["label"] for s in samples),
        "candidate_stats": {
            "n_candidates": stats.n_candidates,
            "n_contact": stats.n_contact,
            "n_vertical_align": stats.n_vertical_align,
            "n_proximity": stats.n_proximity,
        },
        "samples": samples,
    }


def process_scene_isaac(scene_dir: Path, *, n_frames: int, max_pairs: int | None) -> dict:
    """Invoke Isaac worker for each candidate (expensive). Falls back pair-wise on failure."""
    isaac = resolve_health_isaac_config()
    if not isaac.available or isaac.python_bat is None:
        raise RuntimeError("Isaac Sim not available")

    scene = HealthScene.from_dict(json.loads((scene_dir / "scene.json").read_text(encoding="utf-8")))
    write_scene_usda(scene, scene_dir / "scene.usda")
    cand_g, stats = generate_candidates(scene)
    edges = list(cand_g.edges.values())
    if max_pairs is not None:
        edges = edges[:max_pairs]

    samples = []
    gt = PhysicalSceneGraph(scene=scene)
    worker = ROOT / "scripts" / "isaac" / "cf_worker.py"
    for edge in edges:
        out_json = scene_dir / "rollouts" / f"{edge.subject_id}__{edge.host_id}" / "isaac_pair.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(isaac.python_bat),
            str(worker),
            "--scene",
            str(scene_dir / "scene.json"),
            "--subject",
            edge.subject_id,
            "--host",
            edge.host_id,
            "--out",
            str(out_json),
            "--frames",
            str(n_frames),
            "--headless",
        ]
        try:
            subprocess.run(cmd, check=True, timeout=600)
            result = json.loads(out_json.read_text(encoding="utf-8"))
            label = int(result["label"]["positive"])
            backend = "isaac_sim"
            cf_pos = np.asarray(result["counterfactual_positions"], dtype=float)
            contact = np.zeros(len(cf_pos))
        except Exception as exc:
            # fallback analytic for this pair
            result = run_counterfactual_pair(
                scene, subject_id=edge.subject_id, host_id=edge.host_id, n_frames=n_frames
            )
            label = int(result["label"]["positive"])
            backend = f"analytic_fallback:{exc!r}"
            cf_pos = np.asarray(result["counterfactual"]["positions"][edge.subject_id], dtype=float)
            contact = np.asarray(result["counterfactual"]["contact_subject_host"], dtype=float)
            (out_json.parent / "pair.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        gt.add_candidate(edge.subject_id, edge.host_id)
        gt.set_gt(edge.subject_id, edge.host_id, positive=bool(label))
        sep, gap = _geom_feats(scene, edge.subject_id, edge.host_id)
        feats = trajectory_features(
            positions_subject=cf_pos,
            contact=contact,
            host_removed=True,
            geom_xy_sep=sep,
            geom_vertical_gap=gap,
        )
        sample = pack_sample(
            scene_id=scene.scene_id,
            subject_id=edge.subject_id,
            host_id=edge.host_id,
            label=label,
            feats_full=feats,
            seed=hash((scene.scene_id, edge.subject_id, edge.host_id)) % (2**31),
        )
        sample["backend"] = backend
        samples.append(sample)

    for e in gt.edges.values():
        e.present = bool(e.is_gt)
        e.score = 1.0 if e.is_gt else 0.0
    gt.save(scene_dir / "graph_gt.json")
    return {
        "scene_id": scene.scene_id,
        "n_candidates": len(samples),
        "n_pos": sum(s["label"] for s in samples),
        "n_neg": sum(1 - s["label"] for s in samples),
        "candidate_stats": {
            "n_candidates": stats.n_candidates,
            "n_contact": stats.n_contact,
            "n_vertical_align": stats.n_vertical_align,
            "n_proximity": stats.n_proximity,
        },
        "samples": samples,
    }


def make_scene_split(scene_ids: list[str], *, seed: int = 20260730) -> dict:
    """Scene-level split. Handles small-n smoke runs without emptying train."""
    rng = np.random.default_rng(seed)
    ids = list(scene_ids)
    rng.shuffle(ids)
    n = len(ids)
    if n == 0:
        return {"train": [], "val": [], "test": [], "seed": seed, "rule": "scene_level_only"}
    if n == 1:
        # Smoke / debug: put the only scene in train; do not claim a held-out test.
        return {
            "train": ids,
            "val": [],
            "test": [],
            "seed": seed,
            "rule": "scene_level_only_smoke_n1_all_train",
        }
    if n < 5:
        # Keep at least one train scene; put one in test; rest val if any.
        test = ids[:1]
        val = ids[1:2] if n >= 3 else []
        train = ids[len(test) + len(val) :]
        if not train:
            train, test = test, []
        return {"train": train, "val": val, "test": test, "seed": seed, "rule": "scene_level_only_small_n"}
    n_test = max(1, int(round(0.2 * n)))
    n_val = max(1, int(round(0.2 * n)))
    # Ensure train gets the majority
    while n_test + n_val >= n:
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    test = ids[:n_test]
    val = ids[n_test : n_test + n_val]
    train = ids[n_test + n_val :]
    return {"train": train, "val": val, "test": test, "seed": seed, "rule": "scene_level_only"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", type=Path, default=ROOT / "outputs" / "dyphygraph_health" / "scenes")
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "dyphygraph_health" / "dataset")
    p.add_argument("--backend", choices=("analytic", "isaac"), default="analytic")
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--max-scenes", type=int, default=0)
    p.add_argument("--isaac-max-pairs", type=int, default=0, help="Limit pairs per scene for Isaac smoke")
    args = p.parse_args()

    scene_dirs = sorted([d for d in args.scenes.iterdir() if d.is_dir() and (d / "scene.json").exists()])
    if args.max_scenes > 0:
        scene_dirs = scene_dirs[: args.max_scenes]
    if not scene_dirs:
        raise SystemExit(f"No scenes in {args.scenes}; run generate_healthcare_scenes.py first")

    args.out.mkdir(parents=True, exist_ok=True)
    all_samples = []
    index = []
    for sd in scene_dirs:
        if args.backend == "analytic":
            report = process_scene_analytic(sd, n_frames=args.frames)
        else:
            max_pairs = args.isaac_max_pairs or None
            report = process_scene_isaac(sd, n_frames=args.frames, max_pairs=max_pairs)
        all_samples.extend(report["samples"])
        index.append({k: report[k] for k in ("scene_id", "n_candidates", "n_pos", "n_neg", "candidate_stats")})
        print(f"{report['scene_id']}: cand={report['n_candidates']} pos={report['n_pos']} neg={report['n_neg']}")

    split = make_scene_split([r["scene_id"] for r in index])
    # assign split to samples
    sid_to_split = {}
    for part, sids in split.items():
        if part == "seed" or part == "rule":
            continue
        for s in sids:
            sid_to_split[s] = part
    for s in all_samples:
        s["split"] = sid_to_split.get(s["scene_id"], "train")

    dataset = {
        "name": "medphygraph_cf",
        "backend": args.backend,
        "gt_protocol": PROTOCOL_DOC,
        "n_samples": len(all_samples),
        "n_pos": sum(s["label"] for s in all_samples),
        "n_neg": sum(1 - s["label"] for s in all_samples),
        "split": split,
        "scenes": index,
        "samples": all_samples,
    }
    (args.out / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")
    (args.out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (args.out / "gt_protocol.json").write_text(json.dumps(PROTOCOL_DOC, indent=2), encoding="utf-8")
    # lightweight index without huge feature arrays
    slim = {k: dataset[k] for k in dataset if k != "samples"}
    slim["n_samples"] = dataset["n_samples"]
    (args.out / "dataset_index.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print("wrote", args.out / "dataset.json")
    print("split", split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
