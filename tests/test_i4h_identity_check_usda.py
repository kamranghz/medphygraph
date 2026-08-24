"""Static i4h asset identity check USDA."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medphygraph.i4h_visuals import ENTITY_USD_REL
from medphygraph.usd_authoring import (
    I4H_IDENTITY_GRID,
    PAPER_FOCUS_ENTITY_IDS,
    i4h_asset_identity_check_usda,
    write_i4h_asset_identity_check_usda,
)


def _sample_log() -> dict:
    return {
        "scene_id_base": "demo",
        "frames": [
            {
                "state_index": 0,
                "entities": {
                    eid: {
                        "pos": [0.0, 0.0, 0.5],
                        "size": [1.0, 0.5, 0.5],
                        "type": {
                            "bed": "therapy_bed",
                            "bench": "therapy_bench",
                            "monitor": "monitor_cart",
                            "cabinet": "cabinet",
                            "walker": "walker",
                            "wheelchair": "wheelchair",
                            "iv_pole": "iv_pole",
                        }[eid],
                        "movable": eid not in ("bed", "bench", "cabinet"),
                    }
                    for eid in PAPER_FOCUS_ENTITY_IDS
                },
            }
        ],
    }


def test_identity_check_usda_static_no_physics(tmp_path: Path):
    from medphygraph import i4h_visuals

    i4h = tmp_path / "i4h"
    rel = "Props/shared_OR/sm_monitor.usd"
    usd = i4h / rel.replace("/", "\\")
    usd.parent.mkdir(parents=True)
    usd.write_text("#usda 1.0", encoding="utf-8")
    old = dict(i4h_visuals.ENTITY_USD_REL)
    try:
        for et in set(ent["type"] for ent in _sample_log()["frames"][0]["entities"].values()):
            i4h_visuals.ENTITY_USD_REL[et] = rel
        text = i4h_asset_identity_check_usda(_sample_log(), i4h_root=i4h)
    finally:
        i4h_visuals.ENTITY_USD_REL = old

    assert "PhysicsScene" not in text
    assert "PhysicsRigidBodyAPI" not in text
    assert "timeSamples" not in text
    assert 'def Camera "IdentityCheckCamera"' in text
    assert "def Xform \"Graph\"" not in text
    for _eid, label, _gx, _gy in I4H_IDENTITY_GRID:
        assert f'health:label = "{label}"' in text
        assert f"LABEL_{label.replace(' ', '_')}" in text


def test_identity_check_usda_has_all_focus_entities(tmp_path: Path):
    from medphygraph import i4h_visuals

    i4h = tmp_path / "i4h"
    rel = "Props/x.usd"
    usd = i4h / rel.replace("/", "\\")
    usd.parent.mkdir(parents=True)
    usd.write_text("#usda 1.0", encoding="utf-8")
    old = dict(i4h_visuals.ENTITY_USD_REL)
    try:
        for et in ENTITY_USD_REL:
            i4h_visuals.ENTITY_USD_REL[et] = rel
        text = i4h_asset_identity_check_usda(_sample_log(), i4h_root=i4h)
    finally:
        i4h_visuals.ENTITY_USD_REL = old

    for eid in PAPER_FOCUS_ENTITY_IDS:
        assert f'def Xform "ENTITY_{eid}"' in text
    assert "health:visual_align" in text
    assert "health:i4h_canonical_name" in text


def test_write_identity_check_usda(tmp_path: Path):
    from medphygraph import i4h_visuals

    i4h = tmp_path / "i4h"
    rel = "Props/x.usd"
    usd = i4h / rel.replace("/", "\\")
    usd.parent.mkdir(parents=True)
    usd.write_text("#usda 1.0", encoding="utf-8")
    old = dict(i4h_visuals.ENTITY_USD_REL)
    try:
        for et in ENTITY_USD_REL:
            i4h_visuals.ENTITY_USD_REL[et] = rel
        out = tmp_path / "check.usda"
        write_i4h_asset_identity_check_usda(_sample_log(), out, i4h_root=i4h)
        assert out.is_file()
        assert out.read_text(encoding="utf-8").startswith("#usda 1.0")
    finally:
        i4h_visuals.ENTITY_USD_REL = old
