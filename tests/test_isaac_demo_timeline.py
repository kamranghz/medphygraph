"""Timeline, graph renderer, USDA, and compose tests for the Isaac paper demo."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medphygraph.usd_authoring import (
    build_transition_timeline,
    graph_visual_at_frame,
    transition_log_to_i4h_usda,
    transition_log_to_usda,
)

COMPOSE = ROOT / "scripts" / "isaac" / "compose_demo_video.py"
RENDER = ROOT / "scripts" / "isaac" / "render_transition_graph.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _sample_log() -> dict:
    return {
        "scene_id_base": "demo",
        "frames": [
            {
                "state_index": 0,
                "operation": "identity",
                "entities": {
                    "floor": {"pos": [0, 0, 0], "size": [6, 5, 0.05], "type": "floor", "movable": False},
                    "bench": {"pos": [0.0, -2.0, 0.5], "size": [1.0, 0.5, 0.5], "type": "therapy_bench", "movable": False},
                    "monitor": {"pos": [1.0, 1.0, 0.7], "size": [0.5, 0.5, 1.2], "type": "monitor_cart", "movable": True},
                },
                "edges_present": [["monitor", "floor"]],
                "added_edges": [],
                "removed_edges": [],
            },
            {
                "state_index": 1,
                "operation": "transfer_support",
                "entities": {
                    "floor": {"pos": [0, 0, 0], "size": [6, 5, 0.05], "type": "floor", "movable": False},
                    "bench": {"pos": [0.0, -2.0, 0.5], "size": [1.0, 0.5, 0.5], "type": "therapy_bench", "movable": False},
                    "monitor": {"pos": [0.2, -1.8, 0.7], "size": [0.5, 0.5, 1.2], "type": "monitor_cart", "movable": True},
                },
                "edges_present": [["monitor", "bench"]],
                "added_edges": [["monitor", "bench"]],
                "removed_edges": [["monitor", "floor"]],
            },
            {
                "state_index": 2,
                "operation": "remove_object",
                "entities": {
                    "floor": {"pos": [0, 0, 0], "size": [6, 5, 0.05], "type": "floor", "movable": False},
                    "bench": {"pos": [0.0, -2.0, 0.5], "size": [1.0, 0.5, 0.5], "type": "therapy_bench", "movable": False},
                },
                "edges_present": [],
                "added_edges": [],
                "removed_edges": [["monitor", "bench"]],
            },
        ],
    }


def test_timeline_frame_count_and_duration():
    tl = build_transition_timeline(3, fps=30, hold_seconds=2.0, seconds_per_state=4.0)
    assert tl.end_frame == 420
    assert tl.num_frames == 421
    assert tl.state_start_frames == (0, 180, 360)
    assert tl.hold_frames == 60
    assert tl.tween_frames == 120
    assert abs(tl.duration_seconds - (421 / 30.0)) < 1e-6


def test_timeline_phase_boundaries():
    tl = build_transition_timeline(3, fps=30, hold_seconds=2.0, seconds_per_state=4.0)
    assert tl.phase_at_frame(30) == ("hold", 0, None)
    assert tl.phase_at_frame(60) == ("tween", 0, 1)
    assert tl.phase_at_frame(179) == ("tween", 0, 1)
    assert tl.phase_at_frame(180) == ("hold", 1, None)
    assert tl.phase_at_frame(240) == ("tween", 1, 2)
    assert tl.phase_at_frame(360) == ("hold", 2, None)


def test_usda_uses_shared_timing():
    usda = transition_log_to_usda(_sample_log(), fps=30, hold_seconds=2.0, seconds_per_state=4.0)
    assert "startTimeCode = 0" in usda
    assert "endTimeCode = 420" in usda
    assert "timeCodesPerSecond = 30" in usda


def test_graph_renderer_uses_shared_timing():
    render_mod = _load_module(RENDER, "render_transition_graph")
    frames = _sample_log()["frames"]
    tl = build_transition_timeline(len(frames), fps=30, hold_seconds=2.0, seconds_per_state=4.0)
    assert tl.num_frames == 421
    pos = render_mod._fixed_node_positions(frames)
    assert "monitor" in pos and "bench" in pos
    assert pos["monitor"] != pos["bench"]


def test_stable_graph_node_layout():
    render_mod = _load_module(RENDER, "render_transition_graph")
    frames = _sample_log()["frames"]
    pos_a = render_mod._fixed_node_positions(frames)
    pos_b = render_mod._fixed_node_positions(frames)
    assert pos_a == pos_b


def test_graph_events_at_key_frames():
    frames = _sample_log()["frames"]
    tl = build_transition_timeline(3, fps=30, hold_seconds=2.0, seconds_per_state=4.0)

    v0 = graph_visual_at_frame(frames, tl, 30)
    assert v0["phase"] == "hold"
    assert ("monitor", "floor") in v0["present"]

    v_tween1 = graph_visual_at_frame(frames, tl, 100)
    assert v_tween1["phase"] == "tween"
    assert ("monitor", "floor") in v_tween1["removed"]
    assert ("monitor", "bench") in v_tween1["added"]

    v1 = graph_visual_at_frame(frames, tl, 200)
    assert v1["phase"] == "hold"
    assert ("monitor", "bench") in v1["present"]

    v_tween2 = graph_visual_at_frame(frames, tl, 300)
    assert ("monitor", "bench") in v_tween2["removed"]


def test_i4h_usda_no_physx_and_has_room_shell(tmp_path: Path):
    from medphygraph import i4h_visuals

    i4h = tmp_path / "i4h"
    rel = "Props/shared_OR/sm_monitor.usd"
    usd = i4h / rel.replace("/", "\\")
    usd.parent.mkdir(parents=True)
    usd.write_text("#usda 1.0", encoding="utf-8")

    old = dict(i4h_visuals.ENTITY_USD_REL)
    try:
        i4h_visuals.ENTITY_USD_REL = {"monitor_cart": rel, "therapy_bench": rel}
        text = transition_log_to_i4h_usda(_sample_log(), i4h_root=i4h)
    finally:
        i4h_visuals.ENTITY_USD_REL = old

    assert "PhysicsRigidBodyAPI" not in text
    assert "PhysicsCollisionAPI" not in text
    assert "PhysicsScene" not in text
    assert 'def Cube "RoomShell_floor"' in text
    assert 'def Camera "DemoCamera"' in text
    assert "health:camera_strategy" in text
    assert "xformOp:scale" in text
    assert "health:visual_align" in text


def test_monitor_transform_and_visibility_samples():
    usda = transition_log_to_usda(_sample_log(), fps=30, hold_seconds=2.0, seconds_per_state=4.0)
    assert 'def Cube "monitor"' in usda
    assert "xformOp:translate.timeSamples" in usda
    assert "visibility.timeSamples" in usda
    assert '"invisible"' in usda
    assert "180:" in usda or "180 :" in usda


def test_support_beam_timing_markers(tmp_path: Path):
    from medphygraph import i4h_visuals

    i4h = tmp_path / "i4h"
    rel = "Props/shared_OR/sm_monitor.usd"
    usd = i4h / rel.replace("/", "\\")
    usd.parent.mkdir(parents=True)
    usd.write_text("#usda 1.0", encoding="utf-8")
    old = dict(i4h_visuals.ENTITY_USD_REL)
    try:
        i4h_visuals.ENTITY_USD_REL = {"monitor_cart": rel, "therapy_bench": rel}
        text = transition_log_to_i4h_usda(_sample_log(), i4h_root=i4h)
    finally:
        i4h_visuals.ENTITY_USD_REL = old

    assert "added_s1_monitor_bench" in text
    assert "removed_s1_monitor_floor" in text
    assert "removed_s2_monitor_bench" in text
    assert "60: \"inherited\"" in text or '60: "inherited"' in text


def test_compose_ffmpeg_command_construction(tmp_path: Path):
    compose_mod = _load_module(COMPOSE, "compose_demo_video")
    isaac = tmp_path / "isaac.mp4"
    graph = tmp_path / "graph.mp4"
    out = tmp_path / "final.mp4"
    isaac.write_bytes(b"x")
    graph.write_bytes(b"x")

    cmd = compose_mod.build_ffmpeg_hstack_cmd(isaac, graph, out, height=720, crf=18)
    assert cmd[0]
    assert "ffmpeg" in Path(cmd[0]).name.lower() or cmd[0].endswith("ffmpeg")
    assert "-filter_complex" in cmd
    assert "hstack=inputs=2" in cmd[cmd.index("-filter_complex") + 1]
    assert "-crf" in cmd and "18" in cmd
    assert "-pix_fmt" in cmd and "yuv420p" in cmd

    meta = compose_mod.compose_demo_video(isaac, graph, out, dry_run=True)
    assert meta["layout"] == "isaac_left_graph_right"


def test_compose_missing_ffmpeg_error(tmp_path: Path, monkeypatch):
    compose_mod = _load_module(COMPOSE, "compose_demo_video")
    monkeypatch.setattr(compose_mod.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        compose_mod.build_ffmpeg_hstack_cmd(tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4")
