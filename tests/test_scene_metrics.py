"""Unit tests for scene-level graph maintenance metrics."""

from __future__ import annotations

from medphygraph.metrics import (
    false_edge_removal_rate,
    graph_churn,
    scene_graph_edit_f1,
    true_edge_retention_rate,
)


def test_scene_graph_edit_f1_perfect() -> None:
    e = {("a", "floor"), ("b", "floor")}
    m = scene_graph_edit_f1(e, e)
    assert m["graph_edit_f1"] == 1.0


def test_ferr_removes_false_keeps_count() -> None:
    gt = {("a", "floor")}
    initial = {("a", "floor"), ("a", "bed")}  # bed false
    final = {("a", "floor")}
    m = false_edge_removal_rate(initial_present=initial, final_present=final, gt_edges=gt)
    assert m["ferr"] == 1.0
    assert m["n_false_removed"] == 1.0


def test_terr_retention() -> None:
    gt = {("a", "floor"), ("b", "wall")}
    final = {("a", "floor")}
    m = true_edge_retention_rate(final_present=final, gt_edges=gt)
    assert abs(m["terr"] - 0.5) < 1e-9


def test_graph_churn() -> None:
    before: set = set()
    after = {("a", "floor"), ("b", "floor")}
    m = graph_churn(before=before, after=after, n_candidates=10)
    assert m["n_add"] == 2.0
    assert m["churn_rate"] == 0.2
