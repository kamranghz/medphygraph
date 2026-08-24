"""Tests for Isaac transition demo helpers (no Isaac Sim required)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "scripts" / "isaac" / "graph_update_viewer.py"


def _load_viewer_module():
    spec = importlib.util.spec_from_file_location("graph_update_viewer", VIEWER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["graph_update_viewer"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_beam_transform_unit_length_and_midpoint():
    mod = _load_viewer_module()
    mid, length, direction = mod._beam_transform((0.0, 0.0, 0.0), (0.0, 0.0, 2.0))
    assert mid == (0.0, 0.0, 1.0)
    assert math.isclose(length, 2.0)
    assert math.isclose(math.sqrt(sum(d * d for d in direction)), 1.0, rel_tol=1e-6)


def test_transition_demo_log_imports():
    path = ROOT / "scripts" / "isaac" / "transition_demo_log.py"
    spec = importlib.util.spec_from_file_location("transition_demo_log", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert callable(mod.main)


def test_filter_edges_hides_structural_hosts():
    mod = _load_viewer_module()
    entities = {
        "monitor": {"type": "monitor_cart", "movable": True},
        "bench": {"type": "therapy_bench", "movable": False},
        "floor": {"type": "floor", "movable": False},
    }
    edges = [("monitor", "floor"), ("monitor", "bench")]
    filtered = mod._filter_edges_for_display(edges, entities)
    assert filtered == [("monitor", "bench")]


def test_is_structural_entity():
    mod = _load_viewer_module()
    assert mod._is_structural_entity({"type": "wall"})
    assert not mod._is_structural_entity({"type": "monitor_cart"})


def test_pump_app_stops_when_not_running():
    mod = _load_viewer_module()

    class _App:
        def __init__(self, running: bool) -> None:
            self._running = running
            self.updates = 0

        def is_running(self) -> bool:
            return self._running

        def update(self) -> None:
            self.updates += 1

    app = _App(True)
    assert mod._pump_app(app, 0.0) is True
    app._running = False
    assert mod._pump_app(app, 1.0) is False
