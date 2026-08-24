"""P0-5: Geometry-based SUPPORT_BY candidate generation for healthcare scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from medphygraph.schema import HealthEntity, HealthScene, STRUCTURAL_TYPES
from medphygraph.scene_graph import PhysicalSceneGraph


@dataclass
class CandidateStats:
    n_entities: int
    n_candidates: int
    n_contact: int
    n_vertical_align: int
    n_proximity: int
    recall_vs_gt: float | None = None


def _xy_overlap(a: HealthEntity, b: HealthEntity, *, expand: float = 0.05) -> bool:
    amin, amax = a.aabb_min(), a.aabb_max()
    bmin, bmax = b.aabb_min(), b.aabb_max()
    return not (
        amax[0] + expand < bmin[0]
        or bmax[0] + expand < amin[0]
        or amax[1] + expand < bmin[1]
        or bmax[1] + expand < amin[1]
    )


def _xy_sep(a: HealthEntity, b: HealthEntity) -> float:
    amin, amax = a.aabb_min(), a.aabb_max()
    bmin, bmax = b.aabb_min(), b.aabb_max()
    dx = max(0.0, max(bmin[0] - amax[0], amin[0] - bmax[0]))
    dy = max(0.0, max(bmin[1] - amax[1], amin[1] - bmax[1]))
    return float((dx * dx + dy * dy) ** 0.5)


def _vertical_stack(subject: HealthEntity, host: HealthEntity, *, margin: float = 0.15) -> bool:
    """Subject rests above host: subject bottom near host top, with XY overlap."""
    if not _xy_overlap(subject, host, expand=0.08):
        return False
    smin, _ = subject.aabb_min(), subject.aabb_max()
    _, hmax = host.aabb_min(), host.aabb_max()
    gap = smin[2] - hmax[2]
    return -0.02 <= gap <= margin


def _contact(subject: HealthEntity, host: HealthEntity, *, eps: float = 0.08) -> bool:
    amin, amax = subject.aabb_min(), subject.aabb_max()
    bmin, bmax = host.aabb_min(), host.aabb_max()
    # expand slightly
    return not (
        amax[0] + eps < bmin[0]
        or bmax[0] + eps < amin[0]
        or amax[1] + eps < bmin[1]
        or bmax[1] + eps < amin[1]
        or amax[2] + eps < bmin[2]
        or bmax[2] + eps < amin[2]
    )


def generate_candidates(
    scene: HealthScene,
    *,
    proximity_m: float = 0.45,
    vertical_margin_m: float = 0.20,
) -> tuple[PhysicalSceneGraph, CandidateStats]:
    """Produce plausible + false SUPPORT_BY candidates from geometry cues."""
    g = PhysicalSceneGraph(scene=scene)
    ents = [e for e in scene.entities if e.entity_type != "zone"]
    n_contact = n_vert = n_prox = 0

    for subj in ents:
        if subj.entity_type in STRUCTURAL_TYPES:
            continue
        for host in ents:
            if host.entity_id == subj.entity_id:
                continue
            reasons: list[str] = []
            if _contact(subj, host):
                reasons.append("contact")
                n_contact += 1
            if _vertical_stack(subj, host, margin=vertical_margin_m):
                reasons.append("vertical_align")
                n_vert += 1
            sep = _xy_sep(subj, host)
            # proximity false-positive trap: nearby but not stacked
            if sep <= proximity_m and abs(subj.pose_xyz[2] - host.pose_xyz[2]) < 1.2:
                reasons.append("proximity")
                n_prox += 1
            # Always allow structural floor under anything with XY overlap of room
            if host.entity_type == "floor" and _xy_overlap(subj, host, expand=1.0):
                if "vertical_align" not in reasons:
                    reasons.append("vertical_align")
                    n_vert += 1
            # Attachment priors (hidden supports): keep even if AABB cues are weak.
            if host.entity_type == "wall" and subj.entity_type == "wall_rail":
                reasons.append("wall_mount_prior")
            if host.entity_type == "ceiling" and subj.entity_type == "patient_lift":
                reasons.append("ceiling_hang_prior")
            if str(subj.metadata.get("support_kind") or "") == "multi_support":
                anchors = set(subj.metadata.get("anchors") or [])
                if host.entity_id in anchors:
                    reasons.append("multi_support_prior")

            if not reasons:
                continue
            edge = g.add_candidate(subj.entity_id, host.entity_id, score=0.0)
            edge.evidence_source = ",".join(sorted(set(reasons)))

    stats = CandidateStats(
        n_entities=len(ents),
        n_candidates=len(g.edges),
        n_contact=n_contact,
        n_vertical_align=n_vert,
        n_proximity=n_prox,
    )
    return g, stats


def candidate_recall_vs_gt(pred: PhysicalSceneGraph, gt: PhysicalSceneGraph) -> float:
    gt_pos = {(e.subject_id, e.host_id) for e in gt.edges.values() if e.is_gt is True}
    if not gt_pos:
        return 1.0
    cand = set(pred.edges.keys())
    hit = len(gt_pos & cand)
    return hit / len(gt_pos)
