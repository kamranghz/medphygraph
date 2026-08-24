#!/usr/bin/env python3
"""Generate a per-frame transition log for a demo scenario, for pairing
with an Isaac Sim recording of the same scene.

Default scenario is the paper's canonical monitor: floor -> bench transfer
(the same template shown in Fig. 1 / the pipeline figure) — 3 states:
  s0  identity              (starting graph)
  s1  transfer_support      (monitor moved near the bench)
  s2  remove_object         (monitor removed, or the closest available
                             fallback op — see dynamic_states.build_state_sequence)

Requires the downloaded seed-0 checkpoint (run scripts/download.py --verify
first). This script has no Isaac Sim dependency itself — it only produces
JSON. scripts/isaac/render_transition_graph.py (no torch needed) consumes
that JSON to draw the matching graph animation.

Usage:
    python scripts/isaac/transition_demo_log.py
    python scripts/isaac/transition_demo_log.py --prefer-add   # add/occlude variant instead
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from medphygraph import CFSupportNet, ModelConfig, apply_consistency
from medphygraph.analytic_physics import run_counterfactual_pair
from medphygraph.candidates import _xy_sep, generate_candidates
from medphygraph.dynamic_states import build_state_sequence, canonical_transition_delta
from medphygraph.features import trajectory_features
from medphygraph.paths import checkpoint_seed0
from medphygraph.schema import example_rehab_scene


def _score_graph(g, scene, model) -> None:
    """Score every candidate edge in g with CF-SupportNet, in place.

    Mirrors the scoring pattern in scripts/demo.py: recompute the analytic
    AABB counterfactual rollout per candidate edge and run it through the
    model. No dataset_hard shortcut here since this scene is synthetic.
    """
    nodes = scene.entity_map()
    with torch.no_grad():
        for subject_id, host_id in list(g.edges.keys()):
            if subject_id not in nodes or host_id not in nodes:
                continue
            result = run_counterfactual_pair(scene, subject_id=subject_id, host_id=host_id, n_frames=60)
            cf_pos = np.asarray(result["counterfactual"]["positions"][subject_id], dtype=float)
            contact = np.asarray(result["counterfactual"]["contact_subject_host"], dtype=float)
            subj_ent, host_ent = nodes[subject_id], nodes[host_id]
            feats = trajectory_features(
                positions_subject=cf_pos,
                contact=contact,
                host_removed=True,
                geom_xy_sep=_xy_sep(subj_ent, host_ent),
                geom_vertical_gap=float(subj_ent.aabb_min()[2] - host_ent.aabb_max()[2]),
            )
            _, prob = model(torch.from_numpy(np.asarray(feats, dtype=np.float32)).unsqueeze(0))
            pr = float(prob.item())
            e = g.edges[(subject_id, host_id)]
            e.score = pr
            e.confidence = pr
            e.present = pr >= 0.5
            e.evidence_source = "health_dyphygraph"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=checkpoint_seed0())
    p.add_argument(
        "--out", type=Path, default=ROOT / "runs" / "transition_demo" / "transition_log.json"
    )
    p.add_argument(
        "--prefer-add",
        action="store_true",
        help="use the add-container/occlude-IV variant instead of the monitor->bench transfer",
    )
    args = p.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}. Run scripts/download.py first.")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CFSupportNet(ModelConfig(hidden=int(ckpt.get("config", {}).get("hidden", 64))))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    base = example_rehab_scene()
    sequence = build_state_sequence(base, prefer_add=args.prefer_add)

    frames = []
    prev_edges: set[tuple[str, str]] = set()
    for step in sequence:
        scene = step["scene"]
        g, _stats = generate_candidates(scene)
        _score_graph(g, scene, model)
        apply_consistency(g)

        curr_edges = {k for k, e in g.edges.items() if e.present}
        delta = canonical_transition_delta(prev_edges, curr_edges)

        frames.append(
            {
                "state_index": step["state_index"],
                "scene_id": scene.scene_id,
                "operation": step["operation"],
                "op_meta": step["op_meta"],
                "entities": {
                    e.entity_id: {
                        "pos": list(e.pose_xyz),
                        "size": list(e.size_xyz),
                        "type": e.entity_type,
                        "movable": e.movable,
                    }
                    for e in scene.entities
                    if e.entity_type != "zone"
                },
                "edges_present": sorted(list(t) for t in curr_edges),
                "added_edges": sorted(list(t) for t in delta["added_edges"]),
                "removed_edges": sorted(list(t) for t in delta["removed_edges"]),
            }
        )
        prev_edges = curr_edges

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"scene_id_base": base.scene_id, "frames": frames}, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(frames)} frames -> {args.out}")
    for f in frames:
        print(f"  state {f['state_index']} ({f['operation']}): +{f['added_edges']} -{f['removed_edges']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
