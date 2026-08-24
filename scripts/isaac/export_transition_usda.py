#!/usr/bin/env python3
"""Export transition_demo_log.json to an animated .usda for Isaac Sim Movie Capture.

Canonical recording workflow (USDA + graph.mp4 + ffmpeg composite) — not the live viewer.

    python scripts/download.py --verify
    python scripts/isaac/transition_demo_log.py
    python scripts/isaac/export_transition_usda.py \\
        --fps 30 --hold-seconds 2 --seconds-per-state 4
    python scripts/isaac/render_transition_graph.py \\
        --log runs/transition_demo/transition_log.json \\
        --mp4 runs/transition_demo/graph.mp4 \\
        --fps 30 --hold-seconds 2 --seconds-per-state 4

Then open ``runs/transition_demo/transition_demo_i4h.usda`` in Isaac Sim,
select DemoCamera, Movie Capture frames 0..420 @ 30 fps, and compose with
``scripts/isaac/compose_demo_video.py``.

Set I4H_ASSETS_ROOT if your assets are not at the default path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from medphygraph.i4h_visuals import resolve_i4h_root
from medphygraph.isaac_sim import resolve_health_isaac_config
from medphygraph.paths import runs_root
from medphygraph.usd_authoring import write_transition_log_i4h_usda, write_transition_log_usda


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--log",
        type=Path,
        default=runs_root() / "transition_demo" / "transition_log.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output .usda (default: transition_demo_i4h.usda or transition_demo.usda)",
    )
    p.add_argument("--i4h-root", type=Path, default=None)
    p.add_argument("--proxies", action="store_true", help="export colored cube proxies instead of i4h props")
    p.add_argument("--all-entities", action="store_true", help="include lift/rail/etc. (default: paper focus set only)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--hold-seconds", type=float, default=2.0)
    p.add_argument("--seconds-per-state", type=float, default=4.0)
    args = p.parse_args()

    log_path = args.log if args.log.is_absolute() else ROOT / args.log
    if not log_path.is_file():
        raise SystemExit(f"Missing {log_path}. Run: python scripts/isaac/transition_demo_log.py")

    log = json.loads(log_path.read_text(encoding="utf-8"))
    if args.proxies:
        out_path = args.out or (runs_root() / "transition_demo" / "transition_demo.usda")
        out_path = out_path if out_path.is_absolute() else ROOT / out_path
        write_transition_log_usda(
            log,
            out_path,
            fps=args.fps,
            hold_seconds=args.hold_seconds,
            seconds_per_state=args.seconds_per_state,
        )
        mode = "proxy_cubes"
        i4h_root = None
    else:
        i4h_root = resolve_i4h_root(args.i4h_root)
        if i4h_root is None:
            raise SystemExit(
                "i4h assets not found. Set I4H_ASSETS_ROOT or pass --i4h-root, "
                "or use --proxies for cube fallback."
            )
        out_path = args.out or (runs_root() / "transition_demo" / "transition_demo_i4h.usda")
        out_path = out_path if out_path.is_absolute() else ROOT / out_path
        write_transition_log_i4h_usda(
            log,
            out_path,
            i4h_root=i4h_root,
            fps=args.fps,
            hold_seconds=args.hold_seconds,
            seconds_per_state=args.seconds_per_state,
            focus=not args.all_entities,
        )
        mode = "i4h_props"

    isaac = resolve_health_isaac_config()
    print(
        json.dumps(
            {
                "ok": True,
                "mode": mode,
                "log": str(log_path.resolve()),
                "usda": str(out_path.resolve()),
                "i4h_root": str(i4h_root.resolve()) if i4h_root else None,
                "frames": len(log.get("frames", [])),
                "isaac_sim_bat": str(isaac.isaac_sim_bat) if isaac.isaac_sim_bat else None,
                "steps": [
                    "1. python scripts/isaac/render_transition_graph.py (graph.mp4, same timing).",
                    "2. Start Isaac Sim: isaac-sim.bat",
                    f"3. File -> Open -> {out_path.resolve()}",
                    "4. Select DemoCamera; Movie Capture frames 0..endTimeCode @ fps.",
                    "5. python scripts/isaac/compose_demo_video.py --isaac-video ... --graph-video ...",
                ],
                "note": (
                    "Canonical path is baked USDA + Movie Capture (not open_transition_viewer.py). "
                    "Visualization only (no PhysX). Shared timeline with graph.mp4."
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
