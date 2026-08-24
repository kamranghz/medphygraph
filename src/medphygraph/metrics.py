"""Shared metrics for DyPhyGraph-Health evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def edge_metrics(y_true: np.ndarray, y_prob: np.ndarray, *, thr: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    pred = (p >= thr).astype(np.float64)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    fsr = fp / max(fp + tn, 1.0)
    # AUROC / AUPRC (simple, no sklearn required)
    auroc = _auroc(y, p)
    auprc = _auprc(y, p)
    ece = _ece(y, p)
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fsr": fsr,
        "auroc": auroc,
        "auprc": auprc,
        "ece": ece,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    y_s = y[order]
    n_pos = y_s.sum()
    n_neg = len(y_s) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tps = np.cumsum(y_s)
    fps = np.cumsum(1 - y_s)
    tpr = tps / n_pos
    fpr = fps / n_neg
    return float(np.trapz(tpr, fpr))


def _auprc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p)
    y_s = y[order]
    if y_s.sum() == 0:
        return float("nan")
    tps = np.cumsum(y_s)
    fps = np.cumsum(1 - y_s)
    prec = tps / np.maximum(tps + fps, 1)
    rec = tps / y_s.sum()
    return float(np.trapz(prec, rec))


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if not np.any(m):
            continue
        ece += abs(y[m].mean() - p[m].mean()) * (m.mean())
    return float(ece)


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    key: str = "f1",
    *,
    n_boot: int = 400,
    seed: int = 0,
    thr: float = 0.5,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(edge_metrics(y[idx], p[idx], thr=thr)[key])
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.nanmean(arr)),
        "ci95_low": float(np.nanpercentile(arr, 2.5)),
        "ci95_high": float(np.nanpercentile(arr, 97.5)),
    }


def scene_graph_edit_f1(
    pred_edges: set[tuple[str, str]],
    gt_edges: set[tuple[str, str]],
) -> dict[str, float]:
    """Present-edge sets as graphs; edit F1 = F1 over edge presence."""
    tp = len(pred_edges & gt_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": prec, "recall": rec, "graph_edit_f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def false_edge_removal_rate(
    *,
    initial_present: set[tuple[str, str]],
    final_present: set[tuple[str, str]],
    gt_edges: set[tuple[str, str]],
) -> dict[str, float]:
    """Fraction of false edges in the initial graph that are absent after maintenance.

    False = present initially but not in GT. FERR = removed_false / n_false_initial.
    """
    false_init = initial_present - gt_edges
    if not false_init:
        return {"ferr": 1.0, "n_false_initial": 0.0, "n_false_removed": 0.0}
    removed = false_init - final_present
    return {
        "ferr": len(removed) / len(false_init),
        "n_false_initial": float(len(false_init)),
        "n_false_removed": float(len(removed)),
    }


def true_edge_retention_rate(
    *,
    final_present: set[tuple[str, str]],
    gt_edges: set[tuple[str, str]],
) -> dict[str, float]:
    """Fraction of GT-positive edges retained as present after maintenance."""
    if not gt_edges:
        return {"terr": 1.0, "n_gt": 0.0, "n_retained": 0.0}
    retained = gt_edges & final_present
    return {
        "terr": len(retained) / len(gt_edges),
        "n_gt": float(len(gt_edges)),
        "n_retained": float(len(retained)),
    }


def graph_churn(
    *,
    before: set[tuple[str, str]],
    after: set[tuple[str, str]],
    n_candidates: int,
) -> dict[str, float]:
    """ADD+REMOVE count normalized by candidate count."""
    added = after - before
    removed = before - after
    n_ops = len(added) + len(removed)
    denom = max(int(n_candidates), 1)
    return {
        "n_add": float(len(added)),
        "n_remove": float(len(removed)),
        "n_churn_ops": float(n_ops),
        "churn_rate": n_ops / denom,
    }


def aggregate_scene_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean of numeric scene-level metrics across scenes."""
    if not rows:
        return {}
    keys = [
        "graph_edit_f1",
        "precision",
        "recall",
        "ferr",
        "terr",
        "churn_rate",
        "violations_before",
        "violations_after",
        "n_present",
    ]
    out: dict[str, float] = {"n_scenes": float(len(rows))}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        if vals:
            out[k] = float(np.mean(vals))
    return out


def round_metrics(m: dict[str, Any], nd: int = 3) -> dict[str, Any]:
    out = {}
    for k, v in m.items():
        if isinstance(v, float):
            out[k] = round(v, nd) if np.isfinite(v) else None
        else:
            out[k] = v
    return out
