#!/usr/bin/env python3
"""Launch Isaac Sim graph_update_viewer.py for a transition demo log.

Generate the log first (regular medphygraph conda env + checkpoint):

    python scripts/isaac/transition_demo_log.py

Then launch the Isaac viewer (new console, same pattern as open_scene.py):

    python scripts/isaac/open_transition_viewer.py
    python scripts/isaac/open_transition_viewer.py --transition-log runs/transition_demo/transition_log.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from medphygraph.i4h_visuals import DEFAULT_I4H_ROOT, inventory, resolve_i4h_root, room_environment_ready
from medphygraph.isaac_sim import resolve_health_isaac_config
from medphygraph.paths import runs_root


def main() -> int:
    p = argparse.ArgumentParser(description="Open MedPhyGraph transition viewer in Isaac Sim")
    p.add_argument(
        "--transition-log",
        type=Path,
        default=runs_root() / "transition_demo" / "transition_log.json",
        help="JSON from scripts/isaac/transition_demo_log.py",
    )
    p.add_argument("--i4h-root", type=Path, default=None)
    p.add_argument("--no-i4h", action="store_true")
    p.add_argument(
        "--no-room-env",
        action="store_true",
        help="Skip Organs OR room shell (auto-enabled when ATLAS_OR textures are missing)",
    )
    p.add_argument(
        "--room-env",
        action="store_true",
        help="Force Organs OR room shell even when ATLAS_OR textures look incomplete",
    )
    p.add_argument("--headless", action="store_true")
    p.add_argument("--seconds-per-state", type=float, default=None)
    p.add_argument("--hold-seconds", type=float, default=None)
    p.add_argument(
        "--startup-delay",
        type=float,
        default=None,
        help="seconds to wait after scene load before Python-driven animation begins",
    )
    p.add_argument("--loop", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    log_path = args.transition_log if args.transition_log.is_absolute() else REPO_ROOT / args.transition_log
    log_path = log_path.resolve()
    if not log_path.is_file():
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"missing {log_path}",
                    "hint": "Run: python scripts/isaac/transition_demo_log.py",
                },
                indent=2,
            )
        )
        return 1

    isaac = resolve_health_isaac_config()
    if not isaac.available or isaac.python_bat is None:
        print(json.dumps({"ok": False, "error": isaac.notes}, indent=2))
        return 1

    i4h_root = None if args.no_i4h else resolve_i4h_root(args.i4h_root)
    inv = inventory(i4h_root)

    skip_room_env = args.no_room_env
    room_env_note = None
    if (
        not args.no_i4h
        and i4h_root is not None
        and not args.room_env
        and not room_environment_ready(root=i4h_root)
    ):
        skip_room_env = True
        room_env_note = (
            "Auto-skipping Organs OR shell: ATLAS_OR textures are missing from this i4h install. "
            "Props still load; stylized room shell is used instead."
        )

    viewer = REPO_ROOT / "scripts" / "isaac" / "graph_update_viewer.py"
    cmd = [str(isaac.python_bat), str(viewer), "--transition-log", str(log_path)]
    if args.no_i4h:
        cmd.append("--no-i4h")
    elif i4h_root is not None:
        cmd.extend(["--i4h-root", str(i4h_root)])
    if skip_room_env:
        cmd.append("--no-room-env")
    elif args.room_env:
        cmd.append("--room-env")
    if args.headless:
        cmd.append("--headless")
    if args.seconds_per_state is not None:
        cmd.extend(["--seconds-per-state", str(args.seconds_per_state)])
    if args.hold_seconds is not None:
        cmd.extend(["--hold-seconds", str(args.hold_seconds)])
    if args.startup_delay is not None:
        cmd.extend(["--startup-delay", str(args.startup_delay)])
    if args.loop:
        cmd.append("--loop")

    meta = {
        "ok": True,
        "transition_log": str(log_path),
        "isaac_root": str(isaac.root),
        "i4h": inv,
        "room_env_note": room_env_note,
        "command": cmd,
        "dry_run": bool(args.dry_run),
        "hint": "Preview/debug only. For recording use export_transition_usda.py + Movie Capture. Press F to frame.",
    }
    print(json.dumps(meta, indent=2))

    if args.dry_run:
        return 0

    env = os.environ.copy()
    if isaac.root is not None:
        env.setdefault("ISAAC_SIM_ROOT", str(isaac.root))
    if i4h_root is not None:
        env["I4H_ASSETS_ROOT"] = str(i4h_root)
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    subprocess.Popen(cmd, cwd=str(isaac.root), env=env, creationflags=creationflags)
    print("launched Isaac graph_update_viewer (new console)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
