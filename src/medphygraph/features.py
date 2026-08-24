"""P0-7 / P0-8: Fixed feature schema + controlled incompleteness masks."""

from __future__ import annotations

from typing import Any

import numpy as np

OBSERVATION_RATIOS = (1.0, 0.75, 0.5, 0.25, 0.1)

FEATURE_NAMES: tuple[str, ...] = (
    "disp_xyz",
    "disp_z",
    "vel_z",
    "cum_disp_z",
    "contact",
    "host_removed_flag",
    "geom_xy_sep",
    "geom_vertical_gap",
    "mask",
)


def trajectory_features(
    *,
    positions_subject: np.ndarray,
    contact: np.ndarray,
    host_removed: bool,
    geom_xy_sep: float,
    geom_vertical_gap: float,
    dt: float = 1.0 / 30.0,
) -> np.ndarray:
    """Per-frame feature matrix [T, F]."""
    p = np.asarray(positions_subject, dtype=np.float64)
    t = p.shape[0]
    contact = np.asarray(contact, dtype=np.float64).reshape(t)
    disp = p - p[0]
    disp_z = disp[:, 2]
    vel_z = np.zeros(t)
    if t > 1:
        vel_z[1:] = np.diff(p[:, 2]) / dt
    cum = np.cumsum(np.abs(np.diff(p[:, 2], prepend=p[0, 2])))
    host = np.full(t, 1.0 if host_removed else 0.0)
    geom_s = np.full(t, geom_xy_sep)
    geom_g = np.full(t, geom_vertical_gap)
    mask = np.ones(t)
    # columns: disp_x,y,z already in disp; pack as requested
    feats = np.stack(
        [
            np.linalg.norm(disp, axis=1),
            disp_z,
            vel_z,
            cum,
            contact,
            host,
            geom_s,
            geom_g,
            mask,
        ],
        axis=1,
    )
    return feats.astype(np.float32)


def apply_incompleteness(
    feats: np.ndarray,
    *,
    ratio: float,
    seed: int,
) -> np.ndarray:
    """Keep a temporal subset; forward-fill. ratio in {1,0.75,0.5,0.25,0.1}."""
    out = feats.copy()
    t = out.shape[0]
    keep_n = max(1, int(round(t * float(ratio))))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(t, size=keep_n, replace=False))
    mask = np.zeros(t, dtype=np.float32)
    mask[idx] = 1.0
    # forward fill unobserved frames from last keep
    last = out[idx[0]].copy()
    for i in range(t):
        if mask[i] > 0.5:
            last = out[i].copy()
        else:
            out[i] = last
    out[:, -1] = mask
    return out


def pack_sample(
    *,
    scene_id: str,
    subject_id: str,
    host_id: str,
    label: int,
    feats_full: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    partial = {str(r): apply_incompleteness(feats_full, ratio=r, seed=seed + int(r * 1000)).tolist() for r in OBSERVATION_RATIOS}
    return {
        "scene_id": scene_id,
        "subject_id": subject_id,
        "host_id": host_id,
        "relation": "SUPPORTED_BY",
        "label": int(label),
        "feature_names": list(FEATURE_NAMES),
        "features_full": feats_full.tolist(),
        "features_partial": partial,
        "observation_ratios": list(OBSERVATION_RATIOS),
    }
