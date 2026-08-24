"""Lightweight tests for the public evaluation package."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "evaluation"

PUBLIC_DRIVERS = [
  "core.py",
  "multiseed.py",
  "component_analysis.py",
  "candidate_dropout.py",
  "expanded_transfer.py",
  "build_counterfactual_dataset.py",
]


def test_evaluate_package_imports():
  import evaluation as eval
  from evaluation import PKG, REPO_ROOT
  from evaluation._config import SCORERS, TRANSFER
  from evaluation._core_scoring import evaluate_all, fit_scorers, pack_dynamic

  assert eval.PKG == PKG
  assert REPO_ROOT == ROOT
  assert "health_dyphygraph" in SCORERS
  assert TRANSFER == "transfer_support"
  assert callable(evaluate_all)
  assert callable(fit_scorers)
  assert callable(pack_dynamic)


def test_no_circular_import_between_core_modules():
  importlib.import_module("evaluation.core")
  importlib.import_module("evaluation._core_scoring")
  importlib.import_module("evaluation._core_finalize")


@pytest.mark.parametrize("script_name", PUBLIC_DRIVERS)
def test_public_eval_script_help(script_name: str):
  path = SCRIPTS / script_name
  result = subprocess.run(
    [sys.executable, str(path), "--help"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=False,
  )
  assert result.returncode == 0, result.stderr


def test_prf_empty_empty_convention():
  from evaluation._benchmark_utils import prf

  out = prf(0.0, 0.0, 0.0)
  assert out["f1"] == 1.0
  assert out["empty_empty"] is True


def test_transition_aware_consistency_smoke_path():
  from medphygraph.consistency import apply_transition_aware_consistency_v2
  from medphygraph.schema import example_rehab_scene
  from medphygraph.scene_graph import PhysicalSceneGraph

  scene = example_rehab_scene()
  graph_prev = PhysicalSceneGraph(scene=scene)
  graph_curr = PhysicalSceneGraph(scene=scene)
  graph_prev.add_candidate("bed", "walker")
  graph_curr.add_candidate("bed", "walker")
  graph_prev.edges[("bed", "walker")].present = True
  graph_prev.edges[("bed", "walker")].confidence = 0.9
  graph_curr.edges[("bed", "walker")].present = True
  graph_curr.edges[("bed", "walker")].confidence = 0.85
  report = apply_transition_aware_consistency_v2(
    graph_curr,
    prev_confidence={("bed", "walker"): 0.9},
    curr_confidence={("bed", "walker"): 0.85},
    prev_refined=graph_prev,
  )
  assert report is not None


def test_preservation_and_retention_transfer_eligible():
  from evaluation._core_scoring import _preservation_and_retention
  from evaluation._config import TRANSFER

  gt_prev = {("x", "h_old"), ("a", "b")}
  gt_curr = {("x", "h_new"), ("a", "b")}
  gt_rem = {("x", "h_old")}  # not used for the transfer-eligible override

  pred_prev = {("x", "h_old"), ("a", "b")}
  pred_curr = {("x", "h_new"), ("a", "b")}

  target = {
    "intervention_subject": "x",
    "previous_direct_host": "h_old",
    "new_direct_host": "h_new",
  }

  preserv, u_cond = _preservation_and_retention(
    op=TRANSFER,
    eligible=True,
    target=target,
    gt_prev=gt_prev,
    gt_curr=gt_curr,
    gt_rem=gt_rem,
    pred_prev=pred_prev,
    pred_curr=pred_curr,
  )

  assert preserv == 1.0
  assert u_cond == 1.0


def test_resolve_non_transfer_target_add_entity_missing_curr():
  from evaluation.core import _resolve_non_transfer_target

  inconsistencies: list[dict] = []
  frozen_add = [("s_missing", "h_missing")]
  frozen_rem: list[tuple[str, str]] = []
  ents_a = {"s_prev", "h_prev"}
  ents_b = {"some_other_entity"}

  out = _resolve_non_transfer_target(
    corpus="procedural",
    a="a0",
    b="b0",
    op="other_op",
    frozen_add=frozen_add,
    frozen_rem=frozen_rem,
    ents_a=ents_a,
    ents_b=ents_b,
    inconsistencies=inconsistencies,
  )

  assert out["primary_eligible"] is True
  assert any(x.get("issue") == "add_entity_missing_curr" for x in inconsistencies)


def test_resolve_non_transfer_target_add_object_empty_delta_flags():
  from evaluation.core import _resolve_non_transfer_target

  inconsistencies: list[dict] = []
  frozen_add: list[tuple[str, str]] = []
  frozen_rem: list[tuple[str, str]] = []

  ents_a = {"old_entity"}
  ents_b = {"old_entity", "new_entity"}  # appeared={"new_entity"}

  out = _resolve_non_transfer_target(
    corpus="procedural",
    a="a0",
    b="b0",
    op="add_object",
    frozen_add=frozen_add,
    frozen_rem=frozen_rem,
    ents_a=ents_a,
    ents_b=ents_b,
    inconsistencies=inconsistencies,
  )

  # Empty add/remove delta should be countable but flagged.
  assert out["gt_add"] == []
  assert out["gt_rem"] == []
  assert any(x.get("issue") == "add_object_appeared_but_empty_edge_delta" for x in inconsistencies)
  assert any(x.get("issue") == "empty_add_remove_target" for x in inconsistencies)
