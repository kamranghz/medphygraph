from __future__ import annotations

import json
from pathlib import Path

from medphygraph.usd_authoring import transition_log_to_usda, write_transition_log_usda


def _sample_log() -> dict:
    return {
        "scene_id_base": "demo",
        "frames": [
            {
                "state_index": 0,
                "operation": "identity",
                "entities": {
                    "bench": {"pos": [0.0, -2.0, 0.5], "size": [1.0, 0.5, 0.5], "type": "therapy_bench", "movable": False},
                    "monitor": {"pos": [1.0, 1.0, 0.7], "size": [0.5, 0.5, 1.2], "type": "monitor_cart", "movable": True},
                },
                "edges_present": [],
                "added_edges": [],
                "removed_edges": [],
            },
            {
                "state_index": 1,
                "operation": "transfer_support",
                "entities": {
                    "bench": {"pos": [0.0, -2.0, 0.5], "size": [1.0, 0.5, 0.5], "type": "therapy_bench", "movable": False},
                    "monitor": {"pos": [0.2, -1.8, 0.7], "size": [0.5, 0.5, 1.2], "type": "monitor_cart", "movable": True},
                },
                "edges_present": [],
                "added_edges": [],
                "removed_edges": [],
            },
            {
                "state_index": 2,
                "operation": "remove_object",
                "entities": {
                    "bench": {"pos": [0.0, -2.0, 0.5], "size": [1.0, 0.5, 0.5], "type": "therapy_bench", "movable": False},
                },
                "edges_present": [],
                "added_edges": [],
                "removed_edges": [],
            },
        ],
    }


def test_transition_log_to_usda_has_animation_samples():
    usda = transition_log_to_usda(_sample_log(), fps=30, hold_seconds=1.0, seconds_per_state=2.0)
    assert "xformOp:translate.timeSamples" in usda
    assert 'def Cube "monitor"' in usda
    assert "visibility.timeSamples" in usda
    assert '"invisible"' in usda
    assert "float2 clippingRange" in usda
    assert "endTimeCode" in usda


def test_transition_log_to_i4h_usda_references_assets(tmp_path: Path):
    from medphygraph.usd_authoring import transition_log_to_i4h_usda

    i4h = tmp_path / "i4h"
    rel = "Props/shared_OR/sm_monitor.usd"
    usd = i4h / rel.replace("/", "\\")
    usd.parent.mkdir(parents=True)
    usd.write_text("#usda 1.0", encoding="utf-8")

    # Monkeypatch map for test only via direct entity type in sample - use therapy_bench and monitor_cart
    # Instead test structure with a minimal custom log and patch ENTITY_USD_REL in module - too heavy.
    # Just verify function runs when usd paths exist under a fake root by writing one mapped file.
    from medphygraph import i4h_visuals

    old = dict(i4h_visuals.ENTITY_USD_REL)
    try:
        i4h_visuals.ENTITY_USD_REL = {
            "monitor_cart": rel,
            "therapy_bench": rel,
        }
        text = transition_log_to_i4h_usda(_sample_log(), i4h_root=i4h)
    finally:
        i4h_visuals.ENTITY_USD_REL = old

    assert "references = @" in text
    assert "def Xform \"monitor\"" in text
    assert "added_s1_monitor_bench" in text or "Graph" in text
    assert "float2 clippingRange" in text


def test_write_transition_log_usda(tmp_path: Path):
    out = tmp_path / "demo.usda"
    write_transition_log_usda(_sample_log(), out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("#usda 1.0")
    assert out.is_file()
