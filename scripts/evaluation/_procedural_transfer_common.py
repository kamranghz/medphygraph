"""Shared constants + generic transfer authoring for expanded-transfer coverage.

This module intentionally contains NO model-specific logic. It only:
  (a) declares the eligibility protocol (geometry / authored-metadata only), and
  (b) authors new transfer states using the same mechanism as the frozen
      hand-written ops in medphygraph_dynamic_states.py (for example
      op_transfer_monitor_near_bench, op_transfer_tray_to_bench): move the
      subject's pose so it rests on top of the destination host's AABB footprint,
      and tag metadata dynamic_op="transfer_support" / intended_surface=<dest>.

Frozen checkpoint, DeltaUnionV2Config, candidate-generation implementation, and
GT-labeling protocol (analytic_physics + labeling.py) are reused unmodified.
"""

from __future__ import annotations

from typing import Any

from evaluation._config import REPO_ROOT

ROOT = REPO_ROOT

from medphygraph.schema import HealthEntity, HealthScene  # noqa: E402
from medphygraph.dynamic_states import (  # noqa: E402
    _clone_scene,
    _get,
    _move_entity,
)

# ---------------------------------------------------------------------------
# Predeclared eligibility protocol (geometry / authored-metadata only; frozen
# BEFORE any scene is generated or any model is invoked).
# ---------------------------------------------------------------------------

PHASE1_PROCEDURAL_BASE = ROOT / "outputs" / "dyphygraph_health" / "scenes_twinworld_phase1"
I4H_STATIC_BASE = ROOT / "outputs" / "dyphygraph_health" / "scenes_twinworld_i4h_static"

BASE_CORPORA = {
    "procedural": PHASE1_PROCEDURAL_BASE,
    "isaac_hc": I4H_STATIC_BASE,
}

# Movable equipment/furniture entity_types eligible as TRANSFER subjects.
# Excludes structural types (floor/wall/ceiling), the destination-surface
# types themselves, wall_rail and patient_lift (which have dedicated
# wall/ceiling attachment semantics, not floor->surface transfer semantics),
# and therapy_bed (predeclared exclusion: a bed is not modeled as a
# transferable object in this benchmark).
MOVABLE_SUBJECT_TYPES = frozenset(
    {"equipment_cart", "monitor_cart", "iv_pole", "wheelchair", "walker", "other_furniture"}
)

# Destination-host entity_types eligible as TRANSFER destinations: the same
# furniture class already used as the canonical destination (therapy_bench),
# plus one additional pre-existing static-furniture class (cabinet) already
# present in both corpora.
SURFACE_HOST_TYPES = frozenset({"therapy_bench", "cabinet"})

# Exact (subject_id, destination_id) pairs already used by the frozen 15-case
# canonical set. Excluded here so the expansion never duplicates a canonical
# template.
CANONICAL_PAIRS = frozenset({("monitor", "bench"), ("tray", "bench")})

# Metadata support_kind values that indicate the subject is NOT ordinary
# floor-resting movable equipment (wall/ceiling mounted, or a multi-anchor
# fixture). Excluded from subject eligibility.
EXCLUDED_SUPPORT_KINDS = frozenset({"wall_mount", "ceiling_hang", "multi_support"})

# Predeclared plausibility bound: subject must be floor-level equipment, not a
# ceiling/high-mounted fixture without explicit support_kind metadata.
MAX_SUBJECT_AABB_MIN_Z = 1.0  # meters

# Predeclared geometric feasibility bound: subject footprint (XY area) must
# not exceed this multiple of the destination footprint (XY area).
MAX_SUBJECT_TO_DEST_AREA_RATIO = 1.5

CODE_PROVENANCE_FILES = [
    ROOT / "src/medphygraph/dynamic_states.py",
    ROOT / "src/medphygraph/candidates.py",
    ROOT / "src/medphygraph/analytic_physics.py",
    ROOT / "src/medphygraph/labeling.py",
    ROOT / "src/medphygraph/schema.py",
    ROOT / "src/medphygraph/scene_graph.py",
    ROOT / "scripts/evaluation/_procedural_transfer_common.py",
    ROOT / "scripts/evaluation/build_counterfactual_dataset.py",
    ROOT / "scripts/evaluation/procedural_transfer.py",
]


def xy_area(e: HealthEntity) -> float:
    return float(e.size_xyz[0]) * float(e.size_xyz[1])


def author_generic_transfer(scene: HealthScene, subject_id: str, dest_host_id: str) -> dict[str, Any]:
    """Move `subject_id` to rest on top of `dest_host_id`'s AABB footprint.

    Identical mechanism to op_transfer_monitor_near_bench / op_transfer_tray_to_bench
    in medphygraph_dynamic_states.py, generalized to an arbitrary (subject,
    destination) pair. No model predictions are consulted.
    """
    subj = _get(scene, subject_id)
    dest = _get(scene, dest_host_id)
    if subj is None or dest is None:
        return {
            "ok": False,
            "operation": "transfer_support",
            "detail": f"missing {subject_id}/{dest_host_id}",
        }
    top = float(dest.pose_xyz[2] + 0.5 * dest.size_xyz[2])
    hz = float(subj.size_xyz[2])
    _move_entity(
        scene,
        subject_id,
        (float(dest.pose_xyz[0]), float(dest.pose_xyz[1]), top + 0.5 * hz),
    )
    subj.metadata = {
        **(subj.metadata or {}),
        "dynamic_op": "transfer_support",
        "intended_surface": dest_host_id,
    }
    return {
        "ok": True,
        "operation": "transfer_support",
        "subject_id": subject_id,
        "to_hint": dest_host_id,
    }


def clone_scene(scene: HealthScene, new_id: str) -> HealthScene:
    return _clone_scene(scene, new_id)
