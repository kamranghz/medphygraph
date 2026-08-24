#!/usr/bin/env python3
"""Frame-accurate 2D support-graph video synchronized with the USDA timeline.

Uses the same ``build_transition_timeline()`` as ``usd_authoring.py`` — do not
reimplement timing here.

Usage::

    python scripts/isaac/render_transition_graph.py \\
        --log runs/transition_demo/transition_log.json \\
        --mp4 runs/transition_demo/graph.mp4 \\
        --fps 30 --hold-seconds 2 --seconds-per-state 4
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from medphygraph.usd_authoring import build_transition_timeline, graph_visual_at_frame

COLOR_SUBJECT = "#7fb3e8"
COLOR_HOST = "#f2b880"
COLOR_EDGE_PRESENT = "#2e8b3d"
COLOR_EDGE_ADDED = "#3ddc63"
COLOR_EDGE_REMOVED = "#c0392b"
COLOR_STRUCTURAL = "#c9c9c9"
STRUCTURAL_TYPES = {"floor", "wall", "ceiling", "zone"}


def _role_color(entity_id: str, entity_type: str, movable: bool) -> str:
    if entity_type in STRUCTURAL_TYPES:
        return COLOR_STRUCTURAL
    return COLOR_SUBJECT if movable else COLOR_HOST


def _fixed_node_positions(frames: list[dict]) -> dict[str, tuple[float, float]]:
    """Stable floor-plan layout across the whole video (no per-frame jitter)."""
    pos: dict[str, tuple[float, float]] = {}
    for frame in frames:
        for eid, ent in frame["entities"].items():
            if eid not in pos:
                pos[eid] = (float(ent["pos"][0]), float(ent["pos"][1]))
    return pos


def _label_offset(x1, y1, x2, y2, frac=0.5, perp=0.12):
    mx, my = x1 + (x2 - x1) * frac, y1 + (y2 - y1) * frac
    dx, dy = x2 - x1, y2 - y1
    length = max((dx**2 + dy**2) ** 0.5, 1e-6)
    return mx - dy / length * perp, my + dx / length * perp


def render_graph_frame(
    frames: list[dict],
    frame_idx: int,
    timeline,
    pos: dict[str, tuple[float, float]],
    ax,
) -> None:
    ax.clear()
    ax.set_facecolor("#f7f8fa")
    visual = graph_visual_at_frame(frames, timeline, frame_idx)

    state_label = visual["state_index"]
    if visual["phase"] == "tween":
        title = f"s{state_label} → s{visual['next_state_index']}  ·  {visual['operation']}"
    else:
        title = f"s{state_label}  ·  {visual['operation']}"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.text(
        0.02,
        0.98,
        f"frame {frame_idx}/{timeline.end_frame}",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        color="#555555",
    )
    ax.set_aspect("equal")
    ax.axis("off")

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad_x = max(0.8, (max(xs) - min(xs)) * 0.28)
    pad_y = max(0.8, (max(ys) - min(ys)) * 0.28)
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    g = nx.DiGraph()
    for eid in pos:
        ent = None
        for frame in frames:
            if eid in frame["entities"]:
                ent = frame["entities"][eid]
                break
        if ent is None:
            continue
        g.add_node(eid, **ent)

    present = visual["present"]
    added = visual["added"]
    removed = visual["removed"]
    for s, h in present | added:
        g.add_edge(s, h)

    node_colors = [_role_color(n, g.nodes[n]["type"], g.nodes[n]["movable"]) for n in g.nodes]
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=1100,
        edgecolors="#333333",
        linewidths=1.2,
    )
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=9, font_weight="bold")

    kept = [(s, h) for (s, h) in present if (s, h) not in added]
    if kept:
        nx.draw_networkx_edges(
            g,
            pos,
            ax=ax,
            edgelist=kept,
            edge_color=COLOR_EDGE_PRESENT,
            width=2.2,
            arrows=True,
            arrowsize=16,
            connectionstyle="arc3,rad=0.06",
        )
    if added:
        nx.draw_networkx_edges(
            g,
            pos,
            ax=ax,
            edgelist=list(added),
            edge_color=COLOR_EDGE_ADDED,
            width=3.8,
            arrows=True,
            arrowsize=18,
            connectionstyle="arc3,rad=0.06",
        )
        for s, h in added:
            if s in pos and h in pos:
                lx, ly = _label_offset(*pos[s], *pos[h])
                ax.text(lx, ly, "+", color=COLOR_EDGE_ADDED, fontsize=11, fontweight="bold", ha="center")

    for s, h in removed:
        if s not in pos or h not in pos:
            continue
        x1, y1, x2, y2 = *pos[s], *pos[h]
        ax.plot([x1, x2], [y1, y2], color=COLOR_EDGE_REMOVED, linewidth=3.0, linestyle="--")
        lx, ly = _label_offset(x1, y1, x2, y2, frac=0.35)
        ax.text(lx, ly, "−", color=COLOR_EDGE_REMOVED, fontsize=11, fontweight="bold", ha="center")


def _encode_mp4_ffmpeg(frame_dir: Path, mp4_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH — install ffmpeg to write --mp4 output")
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(frame_dir / "frame_%05d.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
        )


def build_graph_mp4(
    log: dict,
    mp4_path: Path,
    *,
    fps: int = 30,
    hold_seconds: float = 2.0,
    seconds_per_state: float = 4.0,
    keep_frames: Path | None = None,
) -> dict:
    frames = log["frames"]
    if not frames:
        raise ValueError("transition log has no frames")
    timeline = build_transition_timeline(
        len(frames),
        fps=fps,
        hold_seconds=hold_seconds,
        seconds_per_state=seconds_per_state,
    )
    pos = _fixed_node_positions(frames)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    tmp_ctx = tempfile.TemporaryDirectory(prefix="medphygraph_graph_")
    frame_dir = keep_frames if keep_frames is not None else Path(tmp_ctx.name)
    frame_dir.mkdir(parents=True, exist_ok=True)

    try:
        for frame_idx in range(timeline.num_frames):
            render_graph_frame(frames, frame_idx, timeline, pos, ax)
            out_png = frame_dir / f"frame_{frame_idx:05d}.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        _encode_mp4_ffmpeg(frame_dir, mp4_path, fps)
    finally:
        if keep_frames is None:
            tmp_ctx.cleanup()

    return {
        "mp4": str(mp4_path.resolve()),
        "num_frames": timeline.num_frames,
        "end_frame": timeline.end_frame,
        "fps": fps,
        "duration_seconds": timeline.duration_seconds,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--mp4", type=Path, required=True, help="output H.264 mp4 (requires ffmpeg)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--hold-seconds", type=float, default=2.0)
    p.add_argument("--seconds-per-state", type=float, default=4.0)
    p.add_argument("--keep-frames", type=Path, default=None, help="optional directory to retain PNG sequence")
    args = p.parse_args()

    log_path = args.log if args.log.is_absolute() else ROOT / args.log
    mp4_path = args.mp4 if args.mp4.is_absolute() else ROOT / args.mp4
    log = json.loads(log_path.read_text(encoding="utf-8"))

    meta = build_graph_mp4(
        log,
        mp4_path,
        fps=args.fps,
        hold_seconds=args.hold_seconds,
        seconds_per_state=args.seconds_per_state,
        keep_frames=args.keep_frames,
    )
    print(json.dumps({"ok": True, "log": str(log_path.resolve()), **meta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
