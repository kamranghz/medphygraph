"""P0-12: Deterministic graph consistency for SUPPORT_BY healthcare scene graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medphygraph.schema import STRUCTURAL_TYPES
from medphygraph.scene_graph import PhysicalSceneGraph, SupportEdge


@dataclass
class ConsistencyReport:
    removed: list[dict[str, Any]] = field(default_factory=list)
    retained_overrides: list[dict[str, Any]] = field(default_factory=list)
    n_cycles_broken: int = 0
    n_multi_support_resolved: int = 0
    n_disconnected_removed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "removed": self.removed,
            "retained_overrides": self.retained_overrides,
            "n_cycles_broken": self.n_cycles_broken,
            "n_multi_support_resolved": self.n_multi_support_resolved,
            "n_disconnected_removed": self.n_disconnected_removed,
            "n_modifications": len(self.removed) + len(self.retained_overrides),
        }


def _present_edges(g: PhysicalSceneGraph) -> list[SupportEdge]:
    return [e for e in g.edges.values() if e.present]


def _has_cycle(edges: list[SupportEdge]) -> list[tuple[str, str]] | None:
    """Return one cycle edge list if cycle exists (as edge keys), else None."""
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e.subject_id, []).append(e.host_id)
    visiting: set[str] = set()
    visited: set[str] = set()
    parent: dict[str, str] = {}

    def dfs(u: str) -> list[str] | None:
        visiting.add(u)
        for v in adj.get(u, []):
            if v in visiting:
                # cycle
                return [u, v]
            if v not in visited:
                parent[v] = u
                got = dfs(v)
                if got is not None:
                    return got
        visiting.discard(u)
        visited.add(u)
        return None

    for n in list(adj.keys()):
        if n not in visited:
            cyc = dfs(n)
            if cyc is not None:
                # return as edge (subject->host) for the back-edge
                return [(cyc[0], cyc[1])]
    return None


def _terminates_at_structural(g: PhysicalSceneGraph, subject: str, host: str, *, max_depth: int = 16) -> bool:
    nodes = g.nodes()
    seen = set()
    cur = host
    depth = 0
    while depth < max_depth and cur not in seen:
        seen.add(cur)
        n = nodes.get(cur)
        if n is None:
            return False
        if n.entity_type in STRUCTURAL_TYPES:
            return True
        # follow present outgoing support
        next_hosts = [e.host_id for e in g.edges.values() if e.present and e.subject_id == cur]
        if not next_hosts:
            return False
        # prefer highest confidence
        next_hosts.sort(key=lambda h: -g.edges[(cur, h)].confidence)
        cur = next_hosts[0]
        depth += 1
    return False


def _pick_primary_support(
    elist: list[SupportEdge],
    *,
    prefer_host: str | None = None,
    prev_confidence: dict[tuple[str, str], float] | None = None,
) -> tuple[SupportEdge, str]:
    """Choose one primary support edge. Default = highest confidence (legacy).

    Optional Track-B policies (only when callers pass kwargs; legacy call sites unchanged):
    - prefer_host: if that host is among present edges, keep it
    - else max Δconfidence vs prev_confidence
    - else max absolute confidence
    """
    if prefer_host is not None:
        preferred = [e for e in elist if e.host_id == prefer_host]
        if preferred:
            keep = max(preferred, key=lambda e: e.confidence)
            return keep, "primary_support_prefer_host"

    if prev_confidence is not None:
        def delta(e: SupportEdge) -> float:
            return float(e.confidence) - float(prev_confidence.get((e.subject_id, e.host_id), 0.0))

        keep = max(elist, key=lambda e: (delta(e), e.confidence))
        return keep, "primary_support_max_delta_confidence"

    keep = max(elist, key=lambda e: e.confidence)
    return keep, "primary_support"


def apply_consistency(
    g: PhysicalSceneGraph,
    *,
    allow_multi_support: bool = False,
    prev_confidence: dict[tuple[str, str], float] | None = None,
    prefer_host_by_subject: dict[str, str] | None = None,
) -> ConsistencyReport:
    """In-place deterministic consistency. Records reasons for every change.

    Default kwargs preserve legacy behavior used by frozen paper metrics.
    Optional prev_confidence / prefer_host_by_subject enable Track-B transfer-aware
    multi-support resolution without changing callers that omit them.
    """
    report = ConsistencyReport()
    nodes = g.nodes()

    # 1) Break cycles: repeatedly remove lowest-confidence edge on a cycle
    while True:
        present = _present_edges(g)
        cyc = _has_cycle(present)
        if not cyc:
            break
        # among present edges that participate in any cycle, drop lowest confidence
        # Approximate: drop the back-edge if present, else lowest confidence overall in cycle nodes
        s, h = cyc[0]
        key = (s, h)
        if key in g.edges and g.edges[key].present:
            victim = g.edges[key]
        else:
            victim = min(present, key=lambda e: e.confidence)
        victim.present = False
        report.n_cycles_broken += 1
        report.removed.append(
            {
                "subject_id": victim.subject_id,
                "host_id": victim.host_id,
                "reason": "cycle_break",
                "confidence": victim.confidence,
            }
        )

    # 2) At most one primary support per movable subject
    if not allow_multi_support:
        by_subj: dict[str, list[SupportEdge]] = {}
        for e in _present_edges(g):
            subj = nodes.get(e.subject_id)
            if subj is None or subj.entity_type in STRUCTURAL_TYPES or subj.entity_type == "zone":
                continue
            by_subj.setdefault(e.subject_id, []).append(e)
        for sid, elist in by_subj.items():
            if len(elist) <= 1:
                continue
            prefer = (prefer_host_by_subject or {}).get(sid)
            # Track-B policies apply only to subjects with an explicit prefer_host entry.
            # All other subjects keep legacy max-confidence resolution.
            if prefer is not None:
                keep, reason = _pick_primary_support(
                    elist, prefer_host=prefer, prev_confidence=prev_confidence
                )
            else:
                keep, reason = _pick_primary_support(elist)
            for e in elist:
                if e is keep:
                    continue
                e.present = False
                report.n_multi_support_resolved += 1
                report.removed.append(
                    {
                        "subject_id": e.subject_id,
                        "host_id": e.host_id,
                        "reason": (
                            "multi_support_conflict_keep_highest_confidence"
                            if reason == "primary_support"
                            else f"multi_support_conflict_{reason}"
                        ),
                        "kept_host": keep.host_id,
                        "confidence": e.confidence,
                    }
                )
            report.retained_overrides.append(
                {
                    "subject_id": sid,
                    "host_id": keep.host_id,
                    "reason": reason,
                    "confidence": keep.confidence,
                }
            )

    # 3) Load path must terminate at structural; else remove
    for e in list(_present_edges(g)):
        if not _terminates_at_structural(g, e.subject_id, e.host_id):
            e.present = False
            report.n_disconnected_removed += 1
            report.removed.append(
                {
                    "subject_id": e.subject_id,
                    "host_id": e.host_id,
                    "reason": "load_path_not_terminating_at_structural",
                    "confidence": e.confidence,
                }
            )

    # Re-check cycles after removals
    while True:
        present = _present_edges(g)
        cyc = _has_cycle(present)
        if not cyc:
            break
        s, h = cyc[0]
        key = (s, h)
        victim = g.edges[key] if key in g.edges and g.edges[key].present else min(present, key=lambda e: e.confidence)
        victim.present = False
        report.n_cycles_broken += 1
        report.removed.append(
            {
                "subject_id": victim.subject_id,
                "host_id": victim.host_id,
                "reason": "cycle_break_postpass",
                "confidence": victim.confidence,
            }
        )

    return report


def count_violations(g: PhysicalSceneGraph, *, allow_multi_support: bool = False) -> dict[str, int]:
    """Count violations without modifying the graph."""
    # Work on a shallow structural copy of edge present flags
    present_before = {k: e.present for k, e in g.edges.items()}
    # cycle
    n_cycle = 1 if _has_cycle(_present_edges(g)) else 0
    # multi
    n_multi = 0
    if not allow_multi_support:
        by_subj: dict[str, int] = {}
        for e in _present_edges(g):
            n = g.nodes().get(e.subject_id)
            if n is None or n.entity_type in STRUCTURAL_TYPES or n.entity_type == "zone":
                continue
            by_subj[e.subject_id] = by_subj.get(e.subject_id, 0) + 1
        n_multi = sum(1 for c in by_subj.values() if c > 1)
    n_disc = 0
    for e in _present_edges(g):
        if not _terminates_at_structural(g, e.subject_id, e.host_id):
            n_disc += 1
    # restore (no mutation intended)
    for k, p in present_before.items():
        g.edges[k].present = p
    return {
        "cycle_rate_flag": n_cycle,
        "multi_support_subjects": n_multi,
        "disconnected_load_paths": n_disc,
        "total": n_cycle + n_multi + n_disc,
    }


@dataclass
class DeltaTransitionConfig:
    """Initial Track-C configuration (not tuned on Isaac transfer cases)."""

    gain_threshold: float = 0.05
    switch_threshold: float = 0.10
    lambda_drop: float = 1.0
    absolute_margin: float = 0.0
    presence_threshold: float = 0.5
    missing_prev_prob: float = 0.0
    allow_below_threshold_rescue: bool = False  # must stay False for Track C v1

    def to_dict(self) -> dict[str, Any]:
        return {
            "gain_threshold": self.gain_threshold,
            "switch_threshold": self.switch_threshold,
            "lambda_drop": self.lambda_drop,
            "absolute_margin": self.absolute_margin,
            "presence_threshold": self.presence_threshold,
            "missing_prev_prob": self.missing_prev_prob,
            "allow_below_threshold_rescue": self.allow_below_threshold_rescue,
            "notes": (
                "Documented initial config for transfer_fix_delta. "
                "Not tuned on Isaac transfer test cases. "
                "Missing previous-edge probs default to missing_prev_prob (0.0)."
            ),
        }


@dataclass
class TransitionAwareReport:
    """Track-C diagnostics; wraps a ConsistencyReport for the post-switch pass."""

    consistency: ConsistencyReport = field(default_factory=ConsistencyReport)
    attempted_switches: int = 0
    accepted_switches: int = 0
    rejected_switches: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    subject_decisions: list[dict[str, Any]] = field(default_factory=list)
    selected_host_by_subject: dict[str, str] = field(default_factory=dict)
    differs_from_legacy: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "consistency": self.consistency.to_dict(),
            "attempted_switches": self.attempted_switches,
            "accepted_switches": self.accepted_switches,
            "rejected_switches": self.rejected_switches,
            "rejection_reasons": dict(self.rejection_reasons),
            "subject_decisions": self.subject_decisions,
            "selected_host_by_subject": dict(self.selected_host_by_subject),
            "differs_from_legacy": self.differs_from_legacy,
        }

    def _rej(self, reason: str) -> None:
        self.rejected_switches += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1


def _primary_host(g: PhysicalSceneGraph, subject: str) -> str | None:
    present = [
        e
        for e in g.edges.values()
        if e.present and e.subject_id == subject
    ]
    if not present:
        return None
    return max(present, key=lambda e: e.confidence).host_id


def _legacy_keep_host(elist: list[SupportEdge]) -> str:
    return max(elist, key=lambda e: e.confidence).host_id


def _would_be_valid_primary(
    g: PhysicalSceneGraph,
    subject: str,
    new_host: str,
) -> tuple[bool, str]:
    """Tentatively keep only (subject, new_host) among subject's present edges; check load-path + cycle."""
    # Snapshot present flags for this subject
    subj_keys = [k for k in g.edges if k[0] == subject]
    snap = {k: g.edges[k].present for k in subj_keys}
    try:
        for k in subj_keys:
            g.edges[k].present = k[1] == new_host and (subject, new_host) in g.edges
        key = (subject, new_host)
        if key not in g.edges:
            return False, "destination_absent_from_candidates"
        g.edges[key].present = True
        if not _terminates_at_structural(g, subject, new_host):
            return False, "load_path_not_terminating"
        if _has_cycle(_present_edges(g)) is not None:
            return False, "would_create_cycle"
        return True, "ok"
    finally:
        for k, p in snap.items():
            g.edges[k].present = p


def apply_transition_aware_consistency(
    g: PhysicalSceneGraph,
    *,
    prev_confidence: dict[tuple[str, str], float],
    prev_refined: PhysicalSceneGraph | None = None,
    prev_primary_host: dict[str, str] | None = None,
    config: DeltaTransitionConfig | None = None,
    allow_multi_support: bool = False,
) -> TransitionAwareReport:
    """Track C: optional delta-transition consistency (does not alter legacy apply_consistency).

    Uses only current/previous scores and previous refined hosts. Never reads GT or to_hint.

    Order (Track C only):
      1) previous refined host
      2) current pre-pruning present/candidate hosts
      3) delta-confidence switch selection (conservative guards)
      4) enforce one primary host
      5) load-path termination
      6) cycle removal
      7) cycle post-pass
    """
    cfg = config or DeltaTransitionConfig()
    report = TransitionAwareReport()
    nodes = g.nodes()

    # Resolve previous primary hosts
    old_host: dict[str, str] = dict(prev_primary_host or {})
    if prev_refined is not None:
        best_conf: dict[str, float] = {}
        for e in prev_refined.edges.values():
            if not e.present:
                continue
            n = prev_refined.nodes().get(e.subject_id)
            if n is None or n.entity_type in STRUCTURAL_TYPES or n.entity_type == "zone":
                continue
            conf = float(e.confidence)
            if e.subject_id not in old_host or conf >= best_conf.get(e.subject_id, float("-inf")):
                old_host[e.subject_id] = e.host_id
                best_conf[e.subject_id] = conf

    # Collect movable subjects that currently have at least one present edge
    by_subj: dict[str, list[SupportEdge]] = {}
    for e in _present_edges(g):
        subj = nodes.get(e.subject_id)
        if subj is None or subj.entity_type in STRUCTURAL_TYPES or subj.entity_type == "zone":
            continue
        by_subj.setdefault(e.subject_id, []).append(e)

    # Also consider subjects with prev host even if only one present edge (possible switch to another present)
    for sid in list(old_host.keys()):
        by_subj.setdefault(sid, [e for e in _present_edges(g) if e.subject_id == sid])

    prefer_from_delta: dict[str, str] = {}

    for sid, elist in by_subj.items():
        h_old = old_host.get(sid)
        legacy_host = _legacy_keep_host(elist) if elist else None

        # Candidate hosts from current graph edges for this subject (scores), not only present
        cand_hosts = sorted({h for (s, h) in g.edges.keys() if s == sid})
        p_t = {(sid, h): float(g.edges[(sid, h)].confidence) if (sid, h) in g.edges else 0.0 for h in cand_hosts}
        p_prev = {
            (sid, h): float(prev_confidence.get((sid, h), cfg.missing_prev_prob)) for h in cand_hosts
        }
        if h_old is not None and (sid, h_old) not in p_t:
            p_t[(sid, h_old)] = float(g.edges[(sid, h_old)].confidence) if (sid, h_old) in g.edges else 0.0
            p_prev[(sid, h_old)] = float(prev_confidence.get((sid, h_old), cfg.missing_prev_prob))

        decision: dict[str, Any] = {
            "subject_id": sid,
            "h_old": h_old,
            "legacy_host": legacy_host,
            "candidate_hosts": cand_hosts,
            "p_t": {h: p_t.get((sid, h), 0.0) for h in cand_hosts},
            "p_prev": {h: p_prev.get((sid, h), cfg.missing_prev_prob) for h in cand_hosts},
            "candidates_scored": [],
            "attempted": False,
            "accepted": False,
            "selected_host": legacy_host,
            "rejection_reason": None,
        }

        if h_old is None:
            decision["rejection_reason"] = "no_previous_refined_host"
            report.subject_decisions.append(decision)
            if legacy_host:
                prefer_from_delta[sid] = legacy_host
                report.selected_host_by_subject[sid] = legacy_host
            continue

        if not cand_hosts:
            decision["rejection_reason"] = "no_current_candidates"
            report._rej("no_current_candidates")
            report.subject_decisions.append(decision)
            continue

        drop_old = float(p_prev.get((sid, h_old), cfg.missing_prev_prob)) - float(
            p_t.get((sid, h_old), 0.0)
        )
        decision["drop_old"] = drop_old

        best: tuple[float, str, dict[str, float]] | None = None
        for h_new in cand_hosts:
            if h_new == h_old:
                continue
            gain_new = float(p_t.get((sid, h_new), 0.0)) - float(
                p_prev.get((sid, h_new), cfg.missing_prev_prob)
            )
            switch_score = gain_new + cfg.lambda_drop * max(drop_old, 0.0)
            row = {
                "h_new": h_new,
                "gain_new": gain_new,
                "drop_old": drop_old,
                "switch_score": switch_score,
                "p_t": float(p_t.get((sid, h_new), 0.0)),
                "p_prev": float(p_prev.get((sid, h_new), cfg.missing_prev_prob)),
            }
            decision["candidates_scored"].append(row)
            if best is None or switch_score > best[0] or (
                switch_score == best[0] and float(p_t.get((sid, h_new), 0.0)) > best[2]["p_t"]
            ):
                best = (switch_score, h_new, row)

        if best is None:
            decision["rejection_reason"] = "no_alternative_host"
            report.subject_decisions.append(decision)
            if legacy_host:
                prefer_from_delta[sid] = legacy_host
                report.selected_host_by_subject[sid] = legacy_host
            continue

        switch_score, h_new, best_row = best
        report.attempted_switches += 1
        decision["attempted"] = True
        decision["best_candidate"] = best_row

        # Conservative guards (no below-threshold rescue)
        p_new = float(best_row["p_t"])
        p_old_t = float(p_t.get((sid, h_old), 0.0))
        if (sid, h_new) not in g.edges:
            decision["rejection_reason"] = "destination_absent_from_candidates"
            report._rej("destination_absent_from_candidates")
        elif not cfg.allow_below_threshold_rescue and p_new < cfg.presence_threshold:
            decision["rejection_reason"] = "destination_below_presence_threshold"
            report._rej("destination_below_presence_threshold")
        elif best_row["gain_new"] < cfg.gain_threshold:
            decision["rejection_reason"] = "gain_below_threshold"
            report._rej("gain_below_threshold")
        elif switch_score < cfg.switch_threshold:
            decision["rejection_reason"] = "switch_score_below_threshold"
            report._rej("switch_score_below_threshold")
        elif cfg.absolute_margin > 0.0 and p_new + 1e-12 < p_old_t + cfg.absolute_margin:
            decision["rejection_reason"] = "absolute_margin_not_met"
            report._rej("absolute_margin_not_met")
        else:
            ok, why = _would_be_valid_primary(g, sid, h_new)
            if not ok:
                decision["rejection_reason"] = why
                report._rej(why)
            else:
                decision["accepted"] = True
                decision["selected_host"] = h_new
                decision["rejection_reason"] = None
                report.accepted_switches += 1
                prefer_from_delta[sid] = h_new
                report.selected_host_by_subject[sid] = h_new
                if legacy_host is not None and h_new != legacy_host:
                    report.differs_from_legacy.append(
                        {"subject_id": sid, "legacy_host": legacy_host, "track_c_host": h_new}
                    )

        if not decision["accepted"]:
            # Fall back exactly to legacy highest-confidence host
            if legacy_host:
                prefer_from_delta[sid] = legacy_host
                report.selected_host_by_subject[sid] = legacy_host
                decision["selected_host"] = legacy_host

        report.subject_decisions.append(decision)

    # Enforce one primary using delta-derived prefer map (hosts chosen only from scores).
    # Then load-path + cycles via the shared legacy post-steps, without Track-B to_hint.
    # Track-C order: switch select → single primary → load path → cycles → cycle post-pass.
    # Implement by: (a) force preferred present / others off for decided subjects, then
    # (b) run cycle / load-path passes matching legacy constraints.

    if not allow_multi_support:
        for sid, host in prefer_from_delta.items():
            for e in list(g.edges.values()):
                if e.subject_id != sid:
                    continue
                want = e.host_id == host
                if e.present and not want:
                    e.present = False
                    report.consistency.n_multi_support_resolved += 1
                    report.consistency.removed.append(
                        {
                            "subject_id": e.subject_id,
                            "host_id": e.host_id,
                            "reason": "multi_support_delta_transition_keep_selected",
                            "kept_host": host,
                            "confidence": e.confidence,
                        }
                    )
                elif want and (sid, host) in g.edges:
                    # Ensure selected present edge stays present if it met presence threshold
                    if float(g.edges[(sid, host)].confidence) >= cfg.presence_threshold:
                        g.edges[(sid, host)].present = True
            report.consistency.retained_overrides.append(
                {
                    "subject_id": sid,
                    "host_id": host,
                    "reason": "primary_support_delta_transition",
                    "confidence": float(g.edges[(sid, host)].confidence) if (sid, host) in g.edges else 0.0,
                }
            )

        # Subjects with multi-support not covered above: legacy max-confidence
        by_subj2: dict[str, list[SupportEdge]] = {}
        for e in _present_edges(g):
            subj = nodes.get(e.subject_id)
            if subj is None or subj.entity_type in STRUCTURAL_TYPES or subj.entity_type == "zone":
                continue
            by_subj2.setdefault(e.subject_id, []).append(e)
        for sid, elist in by_subj2.items():
            if len(elist) <= 1:
                continue
            if sid in prefer_from_delta:
                continue
            keep = max(elist, key=lambda e: e.confidence)
            for e in elist:
                if e is keep:
                    continue
                e.present = False
                report.consistency.n_multi_support_resolved += 1
                report.consistency.removed.append(
                    {
                        "subject_id": e.subject_id,
                        "host_id": e.host_id,
                        "reason": "multi_support_conflict_keep_highest_confidence",
                        "kept_host": keep.host_id,
                        "confidence": e.confidence,
                    }
                )

    # Load path
    for e in list(_present_edges(g)):
        if not _terminates_at_structural(g, e.subject_id, e.host_id):
            e.present = False
            report.consistency.n_disconnected_removed += 1
            report.consistency.removed.append(
                {
                    "subject_id": e.subject_id,
                    "host_id": e.host_id,
                    "reason": "load_path_not_terminating_at_structural",
                    "confidence": e.confidence,
                }
            )

    # Cycles + post-pass
    while True:
        present = _present_edges(g)
        cyc = _has_cycle(present)
        if not cyc:
            break
        s, h = cyc[0]
        key = (s, h)
        victim = g.edges[key] if key in g.edges and g.edges[key].present else min(present, key=lambda e: e.confidence)
        victim.present = False
        report.consistency.n_cycles_broken += 1
        report.consistency.removed.append(
            {
                "subject_id": victim.subject_id,
                "host_id": victim.host_id,
                "reason": "cycle_break",
                "confidence": victim.confidence,
            }
        )

    while True:
        present = _present_edges(g)
        cyc = _has_cycle(present)
        if not cyc:
            break
        s, h = cyc[0]
        key = (s, h)
        victim = g.edges[key] if key in g.edges and g.edges[key].present else min(present, key=lambda e: e.confidence)
        victim.present = False
        report.consistency.n_cycles_broken += 1
        report.consistency.removed.append(
            {
                "subject_id": victim.subject_id,
                "host_id": victim.host_id,
                "reason": "cycle_break_postpass",
                "confidence": victim.confidence,
            }
        )

    return report


# ---------------------------------------------------------------------------
# Track C v2: union-candidate scoring + direct-support gate
# (separate from apply_transition_aware_consistency; v1 behavior unchanged)
# ---------------------------------------------------------------------------


@dataclass
class DeltaUnionV2Config:
    """Frozen Track-C thresholds + documented direct-support gate (not Isaac-tuned)."""

    gain_threshold: float = 0.05
    switch_threshold: float = 0.10
    lambda_drop: float = 1.0
    absolute_margin: float = 0.0
    presence_threshold: float = 0.5
    allow_below_threshold_rescue: bool = False
    # Documented geometry thresholds from candidates._vertical_stack / _xy_overlap
    direct_gap_lo: float = -0.02
    direct_gap_hi: float = 0.15
    direct_xy_expand: float = 0.08
    direct_contact_eps: float = 0.08
    require_real_prev_prob: bool = True  # never fill missing p_prev with 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "delta_transition_union",
            "gain_threshold": self.gain_threshold,
            "switch_threshold": self.switch_threshold,
            "lambda_drop": self.lambda_drop,
            "absolute_margin": self.absolute_margin,
            "presence_threshold": self.presence_threshold,
            "allow_below_threshold_rescue": self.allow_below_threshold_rescue,
            "direct_gap_lo": self.direct_gap_lo,
            "direct_gap_hi": self.direct_gap_hi,
            "direct_xy_expand": self.direct_xy_expand,
            "direct_contact_eps": self.direct_contact_eps,
            "require_real_prev_prob": self.require_real_prev_prob,
            "notes": (
                "Track C v2: union-candidate real p_prev/p_curr; "
                "temporally non-comparable if subject/host missing in a state; "
                "documented direct-support gate; thresholds frozen from Track C v1 "
                "(not tuned on Isaac transfer)."
            ),
        }


def _direct_support_gate(
    subject: Any,
    host: Any,
    *,
    all_entities: list[Any],
    cfg: DeltaUnionV2Config,
) -> dict[str, Any]:
    """Documented geometry direct-support check (candidates.py thresholds)."""
    from medphygraph.candidates import (
        _contact,
        _vertical_stack,
        _xy_overlap,
    )

    if subject is None or host is None:
        return {"ok": False, "reason": "missing_entity", "closer_surfaces": []}
    gap = float(subject.aabb_min()[2] - host.aabb_max()[2])
    xy_ok = _xy_overlap(subject, host, expand=cfg.direct_xy_expand)
    below = host.aabb_max()[2] <= subject.aabb_min()[2] + 0.02
    contact_or_near = _contact(subject, host, eps=cfg.direct_contact_eps) or (
        cfg.direct_gap_lo <= gap <= cfg.direct_gap_hi
    )
    stack = _vertical_stack(subject, host, margin=cfg.direct_gap_hi)
    closer: list[str] = []
    for mid in all_entities:
        if mid.entity_id in (subject.entity_id, host.entity_id):
            continue
        if getattr(mid, "entity_type", None) == "zone":
            continue
        mtop = mid.aabb_max()[2]
        if host.aabb_max()[2] - 1e-6 < mtop < subject.aabb_min()[2] + 1e-6:
            if _xy_overlap(subject, mid, expand=cfg.direct_xy_expand) and _vertical_stack(
                subject, mid, margin=cfg.direct_gap_hi
            ):
                closer.append(mid.entity_id)
    ok = bool(xy_ok and below and contact_or_near and stack and not closer)
    reason = "ok" if ok else ("closer_surface_exists" if closer else "geometry_gate_failed")
    return {
        "ok": ok,
        "reason": reason,
        "vertical_gap": gap,
        "xy_overlap": xy_ok,
        "host_below": below,
        "contact_or_near_gap": contact_or_near,
        "vertical_stack": stack,
        "closer_surfaces": closer,
    }


def apply_transition_aware_consistency_v2(
    g: PhysicalSceneGraph,
    *,
    prev_confidence: dict[tuple[str, str], float],
    curr_confidence: dict[tuple[str, str], float] | None = None,
    prev_refined: PhysicalSceneGraph | None = None,
    prev_entity_ids: set[str] | None = None,
    curr_entity_ids: set[str] | None = None,
    curr_scene_entities: list[Any] | None = None,
    config: DeltaUnionV2Config | None = None,
    allow_multi_support: bool = False,
) -> TransitionAwareReport:
    """Track C v2: union real scores + non-comparable guard + direct-support gate.

    Does not alter apply_consistency() or apply_transition_aware_consistency() (v1).

    prev_confidence must contain *real* scores for temporally comparable edges.
    Missing keys are treated as non-comparable (never filled with 0.0).
    """
    cfg = config or DeltaUnionV2Config()
    report = TransitionAwareReport()
    nodes = g.nodes()
    prev_ids = prev_entity_ids or set()
    curr_ids = curr_entity_ids or set(nodes.keys())
    ents = curr_scene_entities or list(nodes.values())

    old_host: dict[str, str] = {}
    best_conf: dict[str, float] = {}
    if prev_refined is not None:
        for e in prev_refined.edges.values():
            if not e.present:
                continue
            n = prev_refined.nodes().get(e.subject_id)
            if n is None or n.entity_type in STRUCTURAL_TYPES or n.entity_type == "zone":
                continue
            conf = float(e.confidence)
            if e.subject_id not in old_host or conf >= best_conf.get(e.subject_id, float("-inf")):
                old_host[e.subject_id] = e.host_id
                best_conf[e.subject_id] = conf

    by_subj: dict[str, list[SupportEdge]] = {}
    for e in _present_edges(g):
        subj = nodes.get(e.subject_id)
        if subj is None or subj.entity_type in STRUCTURAL_TYPES or subj.entity_type == "zone":
            continue
        by_subj.setdefault(e.subject_id, []).append(e)
    for sid in list(old_host.keys()):
        by_subj.setdefault(sid, [e for e in _present_edges(g) if e.subject_id == sid])

    prefer_from_delta: dict[str, str] = {}

    for sid, elist in by_subj.items():
        h_old = old_host.get(sid)
        legacy_host = _legacy_keep_host(elist) if elist else None
        cand_hosts = sorted({h for (s, h) in g.edges.keys() if s == sid})
        if curr_confidence is not None:
            p_t = {
                h: float(
                    curr_confidence.get(
                        (sid, h),
                        g.edges[(sid, h)].confidence if (sid, h) in g.edges else 0.0,
                    )
                )
                for h in cand_hosts
            }
        else:
            p_t = {h: float(g.edges[(sid, h)].confidence) if (sid, h) in g.edges else 0.0 for h in cand_hosts}

        decision: dict[str, Any] = {
            "subject_id": sid,
            "h_old": h_old,
            "legacy_host": legacy_host,
            "candidate_hosts": cand_hosts,
            "p_t": dict(p_t),
            "candidates_scored": [],
            "attempted": False,
            "accepted": False,
            "selected_host": legacy_host,
            "rejection_reason": None,
            "mode": "delta_transition_union",
        }

        if h_old is None:
            decision["rejection_reason"] = "no_previous_refined_host"
            report.subject_decisions.append(decision)
            if legacy_host:
                prefer_from_delta[sid] = legacy_host
                report.selected_host_by_subject[sid] = legacy_host
            continue

        if sid not in prev_ids or sid not in curr_ids:
            decision["rejection_reason"] = "subject_not_temporally_comparable"
            report._rej("subject_not_temporally_comparable")
            report.subject_decisions.append(decision)
            if legacy_host:
                prefer_from_delta[sid] = legacy_host
                report.selected_host_by_subject[sid] = legacy_host
            continue

        if h_old not in prev_ids or h_old not in curr_ids:
            decision["rejection_reason"] = "old_host_not_temporally_comparable"
            report._rej("old_host_not_temporally_comparable")
            report.subject_decisions.append(decision)
            if legacy_host:
                prefer_from_delta[sid] = legacy_host
                report.selected_host_by_subject[sid] = legacy_host
            continue

        if (sid, h_old) not in prev_confidence:
            decision["rejection_reason"] = "missing_real_p_prev_old"
            report._rej("missing_real_p_prev_old")
            report.subject_decisions.append(decision)
            if legacy_host:
                prefer_from_delta[sid] = legacy_host
                report.selected_host_by_subject[sid] = legacy_host
            continue

        p_prev_old = float(prev_confidence[(sid, h_old)])
        p_curr_old = float(p_t.get(h_old, 0.0))
        drop_old = p_prev_old - p_curr_old
        decision["drop_old"] = drop_old
        decision["p_prev_old"] = p_prev_old

        best: tuple[float, str, dict[str, Any]] | None = None
        for h_new in cand_hosts:
            if h_new == h_old:
                continue
            row: dict[str, Any] = {"h_new": h_new, "p_t": float(p_t.get(h_new, 0.0))}
            if h_new not in prev_ids or h_new not in curr_ids:
                row["status"] = "not_temporally_comparable_entity_missing"
                row["p_prev"] = None
                row["gain_new"] = None
                row["switch_score"] = None
                decision["candidates_scored"].append(row)
                continue
            if (sid, h_new) not in prev_confidence:
                row["status"] = "not_temporally_comparable_missing_real_p_prev"
                row["p_prev"] = None
                row["gain_new"] = None
                row["switch_score"] = None
                decision["candidates_scored"].append(row)
                continue
            p_prev_new = float(prev_confidence[(sid, h_new)])
            gain_new = float(p_t.get(h_new, 0.0)) - p_prev_new
            switch_score = gain_new + cfg.lambda_drop * max(drop_old, 0.0)
            row.update(
                {
                    "status": "ok",
                    "p_prev": p_prev_new,
                    "gain_new": gain_new,
                    "drop_old": drop_old,
                    "switch_score": switch_score,
                }
            )
            decision["candidates_scored"].append(row)
            if best is None or switch_score > best[0] or (
                switch_score == best[0] and float(p_t.get(h_new, 0.0)) > best[2]["p_t"]
            ):
                best = (switch_score, h_new, row)

        if best is None:
            decision["rejection_reason"] = "no_comparable_alternative_host"
            report.subject_decisions.append(decision)
            if legacy_host:
                prefer_from_delta[sid] = legacy_host
                report.selected_host_by_subject[sid] = legacy_host
            continue

        _sw, h_new, best_row = best
        report.attempted_switches += 1
        decision["attempted"] = True
        decision["best_candidate"] = best_row

        p_new = float(best_row["p_t"])
        p_old_t = float(p_t.get(h_old, 0.0))
        subj_e = nodes.get(sid)
        host_e = nodes.get(h_new)
        gate = _direct_support_gate(subj_e, host_e, all_entities=ents, cfg=cfg)
        decision["direct_support_gate"] = gate

        if (sid, h_new) not in g.edges:
            decision["rejection_reason"] = "destination_absent_from_candidates"
            report._rej("destination_absent_from_candidates")
        elif not cfg.allow_below_threshold_rescue and p_new < cfg.presence_threshold:
            decision["rejection_reason"] = "destination_below_presence_threshold"
            report._rej("destination_below_presence_threshold")
        elif float(best_row["gain_new"]) < cfg.gain_threshold:
            decision["rejection_reason"] = "gain_below_threshold"
            report._rej("gain_below_threshold")
        elif float(best_row["switch_score"]) < cfg.switch_threshold:
            decision["rejection_reason"] = "switch_score_below_threshold"
            report._rej("switch_score_below_threshold")
        elif cfg.absolute_margin > 0.0 and p_new + 1e-12 < p_old_t + cfg.absolute_margin:
            decision["rejection_reason"] = "absolute_margin_not_met"
            report._rej("absolute_margin_not_met")
        elif not gate["ok"]:
            decision["rejection_reason"] = f"direct_support_gate_{gate['reason']}"
            report._rej(f"direct_support_gate_{gate['reason']}")
        else:
            ok, why = _would_be_valid_primary(g, sid, h_new)
            if not ok:
                decision["rejection_reason"] = why
                report._rej(why)
            else:
                decision["accepted"] = True
                decision["selected_host"] = h_new
                decision["rejection_reason"] = None
                report.accepted_switches += 1
                prefer_from_delta[sid] = h_new
                report.selected_host_by_subject[sid] = h_new
                if legacy_host is not None and h_new != legacy_host:
                    report.differs_from_legacy.append(
                        {"subject_id": sid, "legacy_host": legacy_host, "track_c_v2_host": h_new}
                    )

        if not decision["accepted"]:
            if legacy_host:
                prefer_from_delta[sid] = legacy_host
                report.selected_host_by_subject[sid] = legacy_host
                decision["selected_host"] = legacy_host

        report.subject_decisions.append(decision)

    if not allow_multi_support:
        for sid, host in prefer_from_delta.items():
            for e in list(g.edges.values()):
                if e.subject_id != sid:
                    continue
                want = e.host_id == host
                if e.present and not want:
                    e.present = False
                    report.consistency.n_multi_support_resolved += 1
                    report.consistency.removed.append(
                        {
                            "subject_id": e.subject_id,
                            "host_id": e.host_id,
                            "reason": "multi_support_delta_union_v2_keep_selected",
                            "kept_host": host,
                            "confidence": e.confidence,
                        }
                    )
                elif want and (sid, host) in g.edges:
                    if float(g.edges[(sid, host)].confidence) >= cfg.presence_threshold:
                        g.edges[(sid, host)].present = True
            report.consistency.retained_overrides.append(
                {
                    "subject_id": sid,
                    "host_id": host,
                    "reason": "primary_support_delta_union_v2",
                    "confidence": float(g.edges[(sid, host)].confidence) if (sid, host) in g.edges else 0.0,
                }
            )

        by_subj2: dict[str, list[SupportEdge]] = {}
        for e in _present_edges(g):
            subj = nodes.get(e.subject_id)
            if subj is None or subj.entity_type in STRUCTURAL_TYPES or subj.entity_type == "zone":
                continue
            by_subj2.setdefault(e.subject_id, []).append(e)
        for sid, elist in by_subj2.items():
            if len(elist) <= 1 or sid in prefer_from_delta:
                continue
            keep = max(elist, key=lambda e: e.confidence)
            for e in elist:
                if e is keep:
                    continue
                e.present = False
                report.consistency.n_multi_support_resolved += 1
                report.consistency.removed.append(
                    {
                        "subject_id": e.subject_id,
                        "host_id": e.host_id,
                        "reason": "multi_support_conflict_keep_highest_confidence",
                        "kept_host": keep.host_id,
                        "confidence": e.confidence,
                    }
                )

    for e in list(_present_edges(g)):
        if not _terminates_at_structural(g, e.subject_id, e.host_id):
            e.present = False
            report.consistency.n_disconnected_removed += 1
            report.consistency.removed.append(
                {
                    "subject_id": e.subject_id,
                    "host_id": e.host_id,
                    "reason": "load_path_not_terminating_at_structural",
                    "confidence": e.confidence,
                }
            )

    while True:
        present = _present_edges(g)
        cyc = _has_cycle(present)
        if not cyc:
            break
        s, h = cyc[0]
        key = (s, h)
        victim = g.edges[key] if key in g.edges and g.edges[key].present else min(present, key=lambda e: e.confidence)
        victim.present = False
        report.consistency.n_cycles_broken += 1
        report.consistency.removed.append(
            {
                "subject_id": victim.subject_id,
                "host_id": victim.host_id,
                "reason": "cycle_break",
                "confidence": victim.confidence,
            }
        )

    while True:
        present = _present_edges(g)
        cyc = _has_cycle(present)
        if not cyc:
            break
        s, h = cyc[0]
        key = (s, h)
        victim = g.edges[key] if key in g.edges and g.edges[key].present else min(present, key=lambda e: e.confidence)
        victim.present = False
        report.consistency.n_cycles_broken += 1
        report.consistency.removed.append(
            {
                "subject_id": victim.subject_id,
                "host_id": victim.host_id,
                "reason": "cycle_break_postpass",
                "confidence": victim.confidence,
            }
        )

    return report
