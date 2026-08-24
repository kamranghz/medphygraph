"""P0-9: Ground-truth labeling protocol for load-bearing SUPPORT_BY.

A candidate (subject, host) is positive ONLY if removing/disabling the host causes
physically meaningful loss of support under the FULL observation rollout.

Criteria (ALL evaluated on full T; NOT identical to baseline τ):
  1. Vertical drop of subject COM >= drop_m, OR
  2. Mean downward speed over a sustained window >= v_down_m_s, OR
  3. Subject loses contact with host and does not remain supported by another
     structural element at similar height within contact_eps.

Baseline B-3 may use a single displacement threshold tuned on validation;
GT uses the multi-criteria OR above with fixed protocol constants (not val-tuned).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

# Fixed protocol constants (documented; not validation-tuned).
GT_DROP_M = 0.08
GT_V_DOWN_M_S = 0.25
GT_SUSTAIN_FRAMES = 5
GT_CONTACT_EPS_M = 0.10
GT_TILT_RAD = 0.35  # ~20 deg; optional tipping criterion for Newton CF
DT = 1.0 / 30.0


@dataclass
class LabelDecision:
    subject_id: str
    host_id: str
    positive: bool
    drop_m: float
    mean_v_down: float
    contact_lost: bool
    reasons: list[str]
    max_tilt_rad: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _down_speed(z: np.ndarray, dt: float = DT) -> np.ndarray:
    if len(z) < 2:
        return np.zeros(1)
    return np.maximum(0.0, -(np.diff(z) / dt))


def _max_tilt_rad(quaternions_wxyz: np.ndarray | None) -> float:
    """Largest tilt from upright (world +Z) given wxyz quaternions over time."""
    if quaternions_wxyz is None:
        return 0.0
    q = np.asarray(quaternions_wxyz, dtype=float)
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] < 4:
        return 0.0
    # Rotate local +Z by quaternion; upright alignment = q_z · world_z
    # wxyz → rotation of (0,0,1):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # body z-axis in world: R @ [0,0,1]
    bz_x = 2.0 * (x * z + w * y)
    bz_y = 2.0 * (y * z - w * x)
    bz_z = 1.0 - 2.0 * (x * x + y * y)
    # angle between body z and world z
    cos_t = np.clip(bz_z / np.maximum(1e-9, np.sqrt(bz_x * bz_x + bz_y * bz_y + bz_z * bz_z)), -1.0, 1.0)
    return float(np.max(np.arccos(cos_t)))


def label_from_rollout(
    *,
    subject_id: str,
    host_id: str,
    z_factual: np.ndarray,
    z_counterfactual: np.ndarray,
    contact_factual: np.ndarray | None = None,
    contact_counterfactual: np.ndarray | None = None,
    structural_support_remaining: bool = False,
    max_tilt_rad: float | None = None,
    quaternions_wxyz: np.ndarray | None = None,
    tilt_threshold_rad: float = GT_TILT_RAD,
) -> LabelDecision:
    """Compare factual vs counterfactual subject height / contact / optional tilt."""
    zf = np.asarray(z_factual, dtype=float)
    zc = np.asarray(z_counterfactual, dtype=float)
    drop = float(max(0.0, zf[-1] - zc[-1])) if len(zf) and len(zc) else 0.0
    # also absolute fall from start of CF
    drop = max(drop, float(max(0.0, (zc[0] - zc[-1]) if len(zc) else 0.0)))

    v = _down_speed(zc)
    # sustained: max mean over window
    mean_v = 0.0
    if len(v) >= GT_SUSTAIN_FRAMES:
        ker = np.ones(GT_SUSTAIN_FRAMES) / GT_SUSTAIN_FRAMES
        mean_v = float(np.max(np.convolve(v, ker, mode="valid")))
    elif len(v):
        mean_v = float(np.mean(v))

    contact_lost = False
    if contact_factual is not None and contact_counterfactual is not None:
        cf = np.asarray(contact_factual, dtype=float)
        cc = np.asarray(contact_counterfactual, dtype=float)
        contact_lost = bool(cf.mean() > 0.5 and cc.mean() < 0.5)

    tilt = float(max_tilt_rad) if max_tilt_rad is not None else _max_tilt_rad(quaternions_wxyz)

    reasons: list[str] = []
    pos = False
    if drop >= GT_DROP_M:
        pos = True
        reasons.append(f"drop>={GT_DROP_M}")
    if mean_v >= GT_V_DOWN_M_S:
        pos = True
        reasons.append(f"v_down>={GT_V_DOWN_M_S}")
    if contact_lost and not structural_support_remaining:
        pos = True
        reasons.append("contact_lost_no_structural_alt")
    if tilt >= float(tilt_threshold_rad):
        pos = True
        reasons.append(f"tilt>={tilt_threshold_rad}")
    if structural_support_remaining and drop < GT_DROP_M * 0.5:
        # still supported by structure → not attributed to this host
        if "contact_lost_no_structural_alt" in reasons and drop < GT_DROP_M:
            pos = drop >= GT_DROP_M or mean_v >= GT_V_DOWN_M_S or tilt >= float(tilt_threshold_rad)
            reasons.append("structural_alt_present")

    return LabelDecision(
        subject_id=subject_id,
        host_id=host_id,
        positive=bool(pos),
        drop_m=drop,
        mean_v_down=mean_v,
        contact_lost=contact_lost,
        reasons=reasons or ["no_support_loss"],
        max_tilt_rad=tilt,
    )


PROTOCOL_DOC = {
    "name": "medphygraph_gt",
    "drop_m": GT_DROP_M,
    "v_down_m_s": GT_V_DOWN_M_S,
    "sustain_frames": GT_SUSTAIN_FRAMES,
    "contact_eps_m": GT_CONTACT_EPS_M,
    "tilt_rad": GT_TILT_RAD,
    "dt": DT,
    "note": "Multi-criteria OR; constants fixed a priori — not the B-3 val-tuned threshold.",
}
