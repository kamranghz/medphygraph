"""Evaluation-only dynamic graph-update metrics. No model changes."""

from __future__ import annotations

from typing import Any, Iterable


def _prf(tp: float, fp: float, fn: float) -> dict[str, float]:
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def set_delta_metrics(
    *,
    gt_prev: set[tuple[str, str]],
    gt_curr: set[tuple[str, str]],
    pred_prev: set[tuple[str, str]],
    pred_curr: set[tuple[str, str]],
) -> dict[str, Any]:
    """Compare predicted edge-set dynamics to GT dynamics between two states."""
    gt_added = gt_curr - gt_prev
    gt_removed = gt_prev - gt_curr
    gt_unchanged = gt_prev & gt_curr

    pred_added = pred_curr - pred_prev
    pred_removed = pred_prev - pred_curr
    pred_unchanged = pred_prev & pred_curr

    add_tp = float(len(pred_added & gt_added))
    add_fp = float(len(pred_added - gt_added))
    add_fn = float(len(gt_added - pred_added))

    rem_tp = float(len(pred_removed & gt_removed))
    rem_fp = float(len(pred_removed - gt_removed))
    rem_fn = float(len(gt_removed - pred_removed))

    # Unchanged preservation: among GT-unchanged edges, fraction still present in pred_curr
    # (and ideally were present in pred_prev — report both).
    if gt_unchanged:
        preserved = float(len(gt_unchanged & pred_curr) / len(gt_unchanged))
        retained_if_prev = (
            float(len((gt_unchanged & pred_prev) & pred_curr) / max(len(gt_unchanged & pred_prev), 1))
            if (gt_unchanged & pred_prev)
            else float("nan")
        )
    else:
        preserved = 1.0
        retained_if_prev = 1.0

    # Overall dynamic update F1: micro over ADD∪REMOVE labels
    dyn_tp = add_tp + rem_tp
    dyn_fp = add_fp + rem_fp
    dyn_fn = add_fn + rem_fn
    overall = _prf(dyn_tp, dyn_fp, dyn_fn)

    # Consistency: pred_curr should include GT unchanged edges that pred_prev correctly had
    consistent_keep = float(len((gt_unchanged & pred_prev & pred_curr)) / max(len(gt_unchanged & pred_prev), 1.0)) if (
        gt_unchanged & pred_prev
    ) else 1.0

    return {
        "added": _prf(add_tp, add_fp, add_fn),
        "removed": _prf(rem_tp, rem_fp, rem_fn),
        "unchanged_preservation_rate": preserved,
        "unchanged_retention_given_prev_pred": retained_if_prev,
        "dynamic_graph_update_f1": overall["f1"],
        "dynamic_graph_update": overall,
        "graph_consistency_across_states": consistent_keep,
        "counts": {
            "n_gt_added": len(gt_added),
            "n_gt_removed": len(gt_removed),
            "n_gt_unchanged": len(gt_unchanged),
            "n_pred_added": len(pred_added),
            "n_pred_removed": len(pred_removed),
        },
    }


def mean_dynamic_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = [
        "dynamic_graph_update_f1",
        "unchanged_preservation_rate",
        "graph_consistency_across_states",
        "added_f1",
        "removed_f1",
        "added_precision",
        "added_recall",
        "removed_precision",
        "removed_recall",
    ]
    acc = {k: 0.0 for k in keys}
    n = 0
    for r in rows:
        n += 1
        acc["dynamic_graph_update_f1"] += float(r["dynamic_graph_update_f1"])
        acc["unchanged_preservation_rate"] += float(r["unchanged_preservation_rate"])
        acc["graph_consistency_across_states"] += float(r["graph_consistency_across_states"])
        acc["added_f1"] += float(r["added"]["f1"])
        acc["removed_f1"] += float(r["removed"]["f1"])
        acc["added_precision"] += float(r["added"]["precision"])
        acc["added_recall"] += float(r["added"]["recall"])
        acc["removed_precision"] += float(r["removed"]["precision"])
        acc["removed_recall"] += float(r["removed"]["recall"])
    return {k: round(acc[k] / max(n, 1), 3) for k in keys}
