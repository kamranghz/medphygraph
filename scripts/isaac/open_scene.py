#!/usr/bin/env python3
"""Open a DyPhyGraph-Health rehab scene in Isaac Sim GUI.

Launches Isaac's python.bat with isaac_health_gui_viewer.py.
When I4H assets are present, the viewer references Sim-Ready USDs for
bed/wheelchair/carts/cabinet (figures only; CF metrics unchanged).

Examples:
  python scripts/isaac/open_scene.py
  python scripts/isaac/open_scene.py --no-i4h
  python scripts/isaac/open_scene.py --i4h-root D:/projects/models/i4h-assets/724f82e
  python scripts/isaac/open_scene.py --dry-run
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

from medphygraph.paths import procedural_scenes_root, twinworld_phase2_scenes

from medphygraph.i4h_visuals import DEFAULT_I4H_ROOT, inventory, resolve_i4h_root, room_environment_ready
from medphygraph.isaac_sim import resolve_health_isaac_config
from medphygraph.schema import HealthScene
from medphygraph.usd_authoring import write_scene_usda


def main() -> int:
    p = argparse.ArgumentParser(description="Open Health rehab scene in Isaac Sim GUI")
    p.add_argument(
        "--scene-dir",
        type=Path,
        default=None,
        help="Scene directory (default: first procedural twinworld scene if present)",
    )
    p.add_argument(
        "--i4h-root",
        type=Path,
        default=None,
        help=f"i4h versioned asset root (default: {DEFAULT_I4H_ROOT})",
    )
    p.add_argument("--no-i4h", action="store_true", help="Stylized cubes only")
    p.add_argument("--no-room-env", action="store_true", help="Skip Organs OR room shell")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write GUI usda + print launch command without starting Isaac",
    )
    args = p.parse_args()

    scene_dir = args.scene_dir
    if scene_dir is None:
        procedural_root = procedural_scenes_root()
        if procedural_root.is_dir():
            scene_dirs = sorted(path for path in procedural_root.iterdir() if path.is_dir())
            scene_dir = scene_dirs[0] if scene_dirs else procedural_root / "rehab_000"
        else:
            scene_dir = twinworld_phase2_scenes()
    scene_dir = scene_dir if scene_dir.is_absolute() else Path(__file__).resolve().parents[2] / scene_dir
    scene_dir = scene_dir.resolve()
    scene_json = scene_dir / "scene.json"
    if not scene_json.is_file():
        print(json.dumps({"ok": False, "error": f"missing {scene_json}"}, indent=2))
        return 1

    isaac = resolve_health_isaac_config()
    if not isaac.available or isaac.python_bat is None:
        print(json.dumps({"ok": False, "error": isaac.notes}, indent=2))
        return 1

    scene = HealthScene.from_dict(json.loads(scene_json.read_text(encoding="utf-8")))
    usda = write_scene_usda(scene, scene_dir / "scene_gui.usda", gui=True)

    i4h_root = None if args.no_i4h else resolve_i4h_root(args.i4h_root)
    inv = inventory(i4h_root)

    skip_room_env = args.no_room_env
    room_env_note = None
    if not args.no_i4h and i4h_root is not None and not room_environment_ready(root=i4h_root):
        skip_room_env = True
        room_env_note = (
            "Auto-skipping Organs OR shell: ATLAS_OR textures are missing from this i4h install."
        )

    viewer = REPO_ROOT / "scripts" / "isaac" / "gui_viewer.py"
    cmd = [str(isaac.python_bat), str(viewer), "--scene-json", str(scene_json)]
    if args.no_i4h:
        cmd.append("--no-i4h")
    elif i4h_root is not None:
        cmd.extend(["--i4h-root", str(i4h_root)])
    if skip_room_env:
        cmd.append("--no-room-env")

    meta = {
        "ok": True,
        "scene_dir": str(scene_dir),
        "scene_json": str(scene_json),
        "usda_gui_fallback": str(usda),
        "isaac_root": str(isaac.root),
        "i4h": inv,
        "room_env_note": room_env_note,
        "command": cmd,
        "dry_run": bool(args.dry_run),
        "hint": (
            "Unified i4h look: OR room + shared_OR props + Material Library. "
            "Press F to Frame All. Metrics unchanged (AABB CF)."
        ),
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
    print("launched Isaac GUI viewer (new console)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
