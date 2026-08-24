"""Analytic rigid-support physics for counterfactual rollouts (CI / fallback).

MEDPHYGRAPH DIRECT LOAD-BEARING SUPPORT SEMANTICS
(full derivation record: repair_sr1_support_semantics/CANONICAL_DEFINITION.md):

    SupportedBy(subject, host) holds iff `host` is the subject's DIRECT
    load-bearing support: removing that host, while preserving the rest of
    the scene, causes the subject to lose its IMMEDIATE support.

Consequences of the direct-immediate-host rule:
  - Furniture (bed, bench, cart, table, shelf, ...) is a valid host whenever
    it is the closest supporting surface beneath the subject. It is never
    rejected merely for not being a STRUCTURAL_TYPES entity.
  - Floor is positive only when it is itself the subject's direct support
    (i.e. nothing else is stacked directly beneath the subject). Floor does
    NOT remain positive just because the subject's XY footprint lies
    somewhere above the floor while an intermediate host (e.g. a bed)
    directly carries it.
  - Support ancestry is represented as a chain of direct edges, never
    collapsed into one edge: if tray rests on bed and bed rests on floor,
    GT is {tray->bed, bed->floor}, NOT {tray->floor}.

Support modes (physically distinct):
  - floor_rest: subject's direct host is the floor
  - wall_mount: subject anchored to a nearby wall (e.g. wall_rail)
  - ceiling_hang: subject hung from ceiling (e.g. patient_lift)
  - stack: subject's direct host is a movable/furniture entity
  - multi_support: subject requires ALL listed anchors simultaneously

Physics uses geometry / attachment modes only — never reads GT labels.
GT is assigned afterward via labeling.label_from_rollout (P0-9), gated by the
direct-host counterfactual (`will_fall`): removing a host that is not the
subject's current direct support never moves the subject, so it can never be
mislabeled positive by rollout drop/contact criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from medphygraph.candidates import _contact, _xy_overlap, _xy_sep
from medphygraph.labeling import DT, LabelDecision, label_from_rollout
from medphygraph.schema import HealthEntity, HealthScene

G = 9.81

SupportMode = Literal["floor_rest", "wall_mount", "ceiling_hang", "stack", "multi_support"]

# Direct-support geometric window (empirically validated; see
# repair_sr1_support_semantics/PROPOSED_CF_RULE.md Step 1). The small negative
# lower bound tolerates the few-millimeter AABB penetration common in
# authored poses without allowing genuinely separated objects to qualify as
# "direct".
#
# Support-mode-specific geometric tolerance (see repair_sr1_support_semantics/
# sr2a_v2/gap_tolerance_analysis.md for the full evidence trail):
#   - FLOOR_GROUNDING_GAP_HI = 0.20 — an independent audit of the 90-scene
#     corpus's ground truth found 99 positive edges with gap in [0.175, 0.20],
#     ALL host=floor, subject in {therapy_bed, equipment_cart}. These are
#     legged/wheeled proxies whose AABB bottom sits above the true floor
#     contact point; the wide tolerance is required to keep them grounded.
#   - STACK_DIRECT_GAP_HI = 0.15 — the same audit found the largest gap on any
#     real furniture-on-furniture (non-floor-host) positive edge in the whole
#     corpus is 0.1305m; nothing ever approaches 0.15-0.20m. A synthetic
#     negative control (object floating 0.16-0.18m above a bench, touching
#     nothing) was WRONGLY resolved to SupportedBy=bench under the old single
#     0.20 global tolerance, demonstrating this widened floor-only tolerance
#     is not scientifically appropriate for ordinary stacks. 0.15 matches the
#     original empirically validated value and leaves ~2cm margin over the
#     largest genuine observed stack gap for AABB proxy imprecision.
# Both apply only to the *geometric-stack* resolution path; wall_mount /
# ceiling_hang / multi_support anchors are resolved by explicit metadata and
# never consult this window at all.
DIRECT_XY_EXPAND = 0.08
DIRECT_GAP_LO = -0.05
FLOOR_GROUNDING_GAP_HI = 0.20
STACK_DIRECT_GAP_HI = 0.15
# Backward-compatible alias (equal to the more permissive of the two modes).
DIRECT_GAP_HI = FLOOR_GROUNDING_GAP_HI


@dataclass
class Rollout:
    scene_id: str
    subject_id: str
    host_id: str
    mode: str
    times: np.ndarray
    positions: dict[str, np.ndarray]
    contact_subject_host: np.ndarray
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "subject_id": self.subject_id,
            "host_id": self.host_id,
            "mode": self.mode,
            "times": self.times.tolist(),
            "positions": {k: v.tolist() for k, v in self.positions.items()},
            "contact_subject_host": self.contact_subject_host.tolist(),
            "meta": self.meta,
        }


def _direct_stack_gap(subject: HealthEntity, host: HealthEntity) -> float | None:
    """Vertical gap iff host's top surface is immediately beneath subject's bottom.

    Uses FLOOR_GROUNDING_GAP_HI when host is the floor (legged/wheeled proxy
    standoff) and the tighter STACK_DIRECT_GAP_HI for every other host type
    (furniture-on-furniture stacking never needs more; see module docstring).
    """
    if not _xy_overlap(subject, host, expand=DIRECT_XY_EXPAND):
        return None
    gap = float(subject.aabb_min()[2] - host.aabb_max()[2])
    gap_hi = FLOOR_GROUNDING_GAP_HI if host.entity_type == "floor" else STACK_DIRECT_GAP_HI
    if DIRECT_GAP_LO <= gap <= gap_hi:
        return gap
    return None


def _wall_anchor_host(scene: HealthScene, subject: HealthEntity) -> str | None:
    """Resolve the specific wall a mounted subject is anchored to."""
    nodes = scene.entity_map()
    anchor = subject.metadata.get("anchor_host")
    if anchor and anchor in nodes and nodes[anchor].entity_type == "wall":
        return str(anchor)
    # Legacy fallback (no explicit metadata): nearest contacted/adjacent wall.
    best_id: str | None = None
    best_sep = 1e9
    for host in scene.entities:
        if host.entity_type != "wall":
            continue
        if _contact(subject, host, eps=0.30) or _xy_sep(subject, host) <= 0.25:
            sep = _xy_sep(subject, host)
            if sep < best_sep:
                best_sep = sep
                best_id = host.entity_id
    return best_id


def _ceiling_anchor_host(scene: HealthScene, subject: HealthEntity) -> str | None:
    nodes = scene.entity_map()
    anchor = subject.metadata.get("anchor_host")
    if anchor and anchor in nodes and nodes[anchor].entity_type == "ceiling":
        return str(anchor)
    for host in scene.entities:
        if host.entity_type != "ceiling":
            continue
        if _xy_overlap(subject, host, expand=1.0) and float(subject.pose_xyz[2]) >= 1.6:
            return host.entity_id
    return None


def direct_immediate_host(scene: HealthScene, subject_id: str) -> str | None:
    """Return the subject's DIRECT load-bearing host under unified SupportedBy semantics.

    Resolution order:
      1. Explicit / legacy wall_mount anchor (wall_rail-style attachment).
      2. Explicit / legacy ceiling_hang anchor (patient_lift-style attachment).
      3. multi_support: highest-surface listed anchor (host-selection only;
         `host_is_direct_support` treats every listed anchor as individually
         necessary for the CF removal test).
      4. Geometric direct stack: the highest supporting surface with XY
         overlap and a small vertical gap (see DIRECT_GAP_LO and the
         mode-specific FLOOR_GROUNDING_GAP_HI / STACK_DIRECT_GAP_HI). This is
         evaluated over ALL entity types — furniture is never excluded.
    """
    nodes = scene.entity_map()
    subj = nodes.get(subject_id)
    if subj is None:
        return None

    kind = str(subj.metadata.get("support_kind") or "")
    if kind == "wall_mount" or subj.entity_type == "wall_rail":
        return _wall_anchor_host(scene, subj)
    if kind == "ceiling_hang" or subj.entity_type == "patient_lift":
        return _ceiling_anchor_host(scene, subj)
    if kind == "multi_support":
        anchors = [a for a in (subj.metadata.get("anchors") or []) if a in nodes]
        scored: list[tuple[float, float, str]] = []
        for hid in anchors:
            host = nodes[hid]
            gap = _direct_stack_gap(subj, host)
            if gap is not None:
                scored.append((host.aabb_max()[2], -abs(gap), hid))
        if scored:
            scored.sort(reverse=True)
            return scored[0][2]
        return anchors[0] if anchors else None

    best: tuple[float, float, str] | None = None
    for host in scene.entities:
        if host.entity_id == subject_id or host.entity_type == "zone":
            continue
        gap = _direct_stack_gap(subj, host)
        if gap is None:
            continue
        top = float(host.aabb_max()[2])
        cand = (top, -abs(gap), host.entity_id)
        if best is None or cand[:2] > best[:2]:
            best = cand
    return best[2] if best else None


def support_mode(scene: HealthScene, host: HealthEntity, subject: HealthEntity) -> SupportMode | None:
    """Descriptive support mode for (host, subject), IFF host is the direct host.

    Used for provenance/debug metadata and CF fall-shape (mount vs stack);
    it does not by itself gate whether removing `host` causes a fall — that
    is `host_is_direct_support`.
    """
    kind = str(subject.metadata.get("support_kind") or "")
    if kind == "multi_support":
        anchors = set(subject.metadata.get("anchors") or [])
        if host.entity_id not in anchors:
            return None
        if host.entity_type == "wall":
            return "wall_mount"
        if host.entity_type == "ceiling":
            return "ceiling_hang"
        if host.entity_type == "floor":
            return "floor_rest"
        return "stack"

    direct = direct_immediate_host(scene, subject.entity_id)
    if direct != host.entity_id:
        return None
    if kind == "wall_mount" or subject.entity_type == "wall_rail":
        return "wall_mount"
    if kind == "ceiling_hang" or subject.entity_type == "patient_lift":
        return "ceiling_hang"
    return "floor_rest" if host.entity_type == "floor" else "stack"


def host_is_direct_support(scene: HealthScene, *, subject_id: str, host_id: str) -> bool:
    """True iff host_id is (one of) the subject's necessary immediate support(s).

    For ordinary subjects this means host_id == direct_immediate_host(...).
    For multi_support subjects, every listed anchor is individually necessary
    (removing any one of them removes support), matching the existing
    multi-anchor hard-scene contract.
    """
    nodes = scene.entity_map()
    subj = nodes.get(subject_id)
    if subj is None or host_id not in nodes:
        return False
    kind = str(subj.metadata.get("support_kind") or "")
    if kind == "multi_support":
        return host_id in set(subj.metadata.get("anchors") or [])
    return host_id == direct_immediate_host(scene, subject_id)


def simulate_pair(
    scene: HealthScene,
    *,
    subject_id: str,
    host_id: str,
    n_frames: int = 60,
    remove_host: bool = False,
    dt: float = DT,
) -> Rollout:
    nodes = scene.entity_map()
    subj0 = np.array(nodes[subject_id].pose_xyz, dtype=float)
    times = np.arange(n_frames, dtype=float) * dt
    pos: dict[str, np.ndarray] = {}
    for eid, e in nodes.items():
        if e.entity_type == "zone":
            continue
        pos[eid] = np.tile(np.array(e.pose_xyz, dtype=float), (n_frames, 1))

    host = nodes[host_id]
    subj = nodes[subject_id]
    mode = support_mode(scene, host, subj)
    is_direct = host_is_direct_support(scene, subject_id=subject_id, host_id=host_id)

    contact = np.ones(n_frames, dtype=float)
    if remove_host:
        contact[:] = 0.0
        pos[host_id] = pos[host_id] + np.array([0.0, 0.0, -5.0])
        # Direct-support rule: removing a host that is not the subject's
        # current immediate support never moves the subject (ancestral
        # hosts, e.g. floor under a supporting bed, do not cause a fall).
        will_fall = bool(is_direct)

        if will_fall:
            z0 = float(subj0[2])
            half_z = 0.5 * subj.size_xyz[2]
            # Mounted assets can fall further (not clamped to standing half-height on floor).
            z_floor = half_z
            if mode in ("wall_mount", "ceiling_hang"):
                z_floor = max(0.05, 0.5 * subj.size_xyz[2])
            for t in range(n_frames):
                z = z0 - 0.5 * G * (t * dt) ** 2
                z = max(z_floor, z)
                pos[subject_id][t] = [subj0[0], subj0[1], z]
        meta = {
            "backend": "analytic",
            "removed_host": host_id,
            "support_mode": mode,
            "direct_host": direct_immediate_host(scene, subject_id),
            "host_is_direct_support": is_direct,
            "will_fall": will_fall,
        }
        mode_name = "counterfactual"
    else:
        meta = {
            "backend": "analytic",
            "removed_host": None,
            "support_mode": mode,
            "direct_host": direct_immediate_host(scene, subject_id),
        }
        mode_name = "factual"

    return Rollout(
        scene_id=scene.scene_id,
        subject_id=subject_id,
        host_id=host_id,
        mode=mode_name,
        times=times,
        positions=pos,
        contact_subject_host=contact,
        meta=meta,
    )


def run_counterfactual_pair(
    scene: HealthScene,
    *,
    subject_id: str,
    host_id: str,
    n_frames: int = 60,
) -> dict[str, Any]:
    fact = simulate_pair(scene, subject_id=subject_id, host_id=host_id, n_frames=n_frames, remove_host=False)
    cf = simulate_pair(scene, subject_id=subject_id, host_id=host_id, n_frames=n_frames, remove_host=True)

    if not cf.meta.get("will_fall"):
        # Host removal does not remove the subject's immediate support: GT
        # negative by construction, without running the rollout drop/contact
        # criteria (which would otherwise spuriously fire on contact_lost,
        # since the removed host trivially loses contact with the subject).
        decision = LabelDecision(
            subject_id=subject_id,
            host_id=host_id,
            positive=False,
            drop_m=0.0,
            mean_v_down=0.0,
            contact_lost=True,
            reasons=["no_immediate_support_loss"],
            max_tilt_rad=0.0,
        )
    else:
        zf = fact.positions[subject_id][:, 2]
        zc = cf.positions[subject_id][:, 2]
        decision = label_from_rollout(
            subject_id=subject_id,
            host_id=host_id,
            z_factual=zf,
            z_counterfactual=zc,
            contact_factual=fact.contact_subject_host,
            contact_counterfactual=cf.contact_subject_host,
            structural_support_remaining=False,
        )

    return {
        "factual": fact.to_dict(),
        "counterfactual": cf.to_dict(),
        "label": decision.to_dict(),
        "backend": "analytic",
    }
