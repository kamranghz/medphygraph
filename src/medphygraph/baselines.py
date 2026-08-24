"""Baselines B-1..B-8 for DyPhyGraph-Health."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from medphygraph.model import IDX_CONTACT, IDX_CUM_Z, IDX_DISP_Z, IDX_GEOM_GAP, IDX_GEOM_SEP


def _x_at_ratio(sample: dict[str, Any], ratio: float) -> np.ndarray:
    key = str(ratio)
    if "features_partial" in sample and key in sample["features_partial"]:
        return np.asarray(sample["features_partial"][key], dtype=np.float64)
    return np.asarray(sample["features_full"], dtype=np.float64)


def pooled_vector(x: np.ndarray) -> np.ndarray:
    """Mean over observed frames (mask>0.5)."""
    mask = x[:, -1] > 0.5
    if not np.any(mask):
        mask[:] = True
    return x[mask].mean(axis=0)


def contact_rule_prob(sample: dict[str, Any], *, ratio: float = 1.0) -> float:
    """B-1: support from contact only (mean contact on available frames)."""
    x = _x_at_ratio(sample, ratio)
    mask = x[:, -1] > 0.5
    c = x[mask, IDX_CONTACT] if np.any(mask) else x[:, IDX_CONTACT]
    return float(np.clip(c.mean(), 0.0, 1.0))


def geometry_rule_prob(sample: dict[str, Any], *, ratio: float = 1.0) -> float:
    """B-2: contact + vertical alignment + proximity/overlap proxy."""
    x = _x_at_ratio(sample, ratio)
    v = pooled_vector(x)
    contact = v[IDX_CONTACT]
    sep = v[IDX_GEOM_SEP]
    gap = v[IDX_GEOM_GAP]
    stacked = (sep <= 0.35) and (-0.05 <= gap <= 0.25)
    near = sep <= 0.45
    score = 0.0
    if contact > 0.5:
        score += 0.4
    if stacked:
        score += 0.5
    elif near:
        score += 0.2
    return float(np.clip(score, 0.0, 1.0))


def displacement_score(sample: dict[str, Any], *, ratio: float = 1.0) -> float:
    """Scalar drop / cum displacement for B-3 thresholding."""
    x = _x_at_ratio(sample, ratio)
    mask = x[:, -1] > 0.5
    xx = x[mask] if np.any(mask) else x
    drop = float(max(0.0, -xx[-1, IDX_DISP_Z]))
    cum = float(xx[-1, IDX_CUM_Z])
    return max(drop, cum)


def tune_displacement_threshold(
    val_samples: list[dict[str, Any]],
    *,
    ratio: float = 1.0,
) -> float:
    """B-3: choose threshold on validation to maximize F1."""
    from medphygraph.metrics import edge_metrics

    scores = np.asarray([displacement_score(s, ratio=ratio) for s in val_samples], dtype=float)
    y = np.asarray([s["label"] for s in val_samples], dtype=float)
    if len(scores) == 0:
        return 0.08
    cands = np.unique(np.concatenate([scores, np.quantile(scores, np.linspace(0, 1, 40))]))
    best_thr, best_f1 = float(np.median(scores)), -1.0
    for thr in cands:
        m = edge_metrics(y, scores, thr=float(thr))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = float(thr)
    return best_thr


def displacement_rule_prob(sample: dict[str, Any], *, thr: float, ratio: float = 1.0) -> float:
    s = displacement_score(sample, ratio=ratio)
    # soft score for ECE; hard thr used in metrics
    return float(1.0 / (1.0 + np.exp(-(s - thr) * 20.0)))


def _design_matrix(samples: list[dict[str, Any]], *, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([pooled_vector(_x_at_ratio(s, ratio)) for s in samples])
    # drop mask col for classical models
    X = X[:, :-1]
    y = np.asarray([s["label"] for s in samples], dtype=np.int64)
    return X, y


def fit_logistic(train: list[dict], *, ratio: float = 1.0) -> LogisticRegression:
    X, y = _design_matrix(train, ratio=ratio)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    return clf


def fit_rf(train: list[dict], *, ratio: float = 1.0, seed: int = 0) -> RandomForestClassifier:
    X, y = _design_matrix(train, ratio=ratio)
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def fit_mlp(train: list[dict], *, ratio: float = 1.0, seed: int = 0) -> MLPClassifier:
    X, y = _design_matrix(train, ratio=ratio)
    # Balanced sample weights: sklearn MLP has no class_weight=
    n_pos = max(int(y.sum()), 1)
    n_neg = max(len(y) - int(y.sum()), 1)
    sw = np.where(y == 1, len(y) / (2.0 * n_pos), len(y) / (2.0 * n_neg))
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        max_iter=800,
        random_state=seed,
        early_stopping=False,
        learning_rate_init=1e-3,
    )
    clf.fit(X, y, sample_weight=sw)
    return clf


def fit_shallow_tree(train: list[dict], *, ratio: float = 1.0, seed: int = 0) -> DecisionTreeClassifier:
    """B-9: shallow Decision Tree TASK-SIMPLICITY / FEATURE-DEGENERACY CONTROL
    probe.

    This is a *diagnostic control*, not a novel model and not a claim of
    model superiority: an unreasonably strong shallow-tree result is
    evidence about task/feature degeneracy, not about a good model.

    Hyperparameters are recovered EXACTLY (not re-derived or re-tuned) from
    the historical pre-training integrity diagnostic that first exposed this
    finding:
    ``repair_sr1_support_semantics/pretraining_integrity/07_simple_baseline_diagnostic.py``
    -> ``DecisionTreeClassifier(max_depth=4, random_state=0)``. On that
    diagnostic's own feature views (recorded in
    ``repair_sr1_support_semantics/pretraining_integrity/simple_baseline_diagnostic.json``),
    the fitted tree reached actual depth 3 with 4 leaves and 100% accuracy
    under BOTH the legacy split and the layout-held-out split. ``max_depth``
    and ``random_state`` are the only hyperparameters that diagnostic set
    explicitly; every other ``DecisionTreeClassifier`` parameter is left at
    its sklearn default here, exactly as it was there.

    Uses the same shared classical-baseline feature pipeline
    (``_design_matrix`` / ``pooled_vector``) as ``fit_logistic`` /
    ``fit_rf`` / ``fit_mlp``, so it can be dropped into identical TRAIN/VAL
    wiring; it accepts explicit TRAIN samples only, exactly like those
    functions, and never reads ``sample["split"]``.
    """
    X, y = _design_matrix(train, ratio=ratio)
    clf = DecisionTreeClassifier(max_depth=4, random_state=seed)
    clf.fit(X, y)
    return clf


def predict_sklearn(clf: Any, samples: list[dict], *, ratio: float = 1.0) -> np.ndarray:
    X, _ = _design_matrix(samples, ratio=ratio)
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.predict(X).astype(float)


BASELINE_NAMES = (
    "contact_rule",
    "geometry_rule",
    "displacement_threshold",
    "logistic",
    "random_forest",
    "mlp",
    "shallow_tree",
    "gru_dyphygraph",
    "dyphygraph_independent",
)
