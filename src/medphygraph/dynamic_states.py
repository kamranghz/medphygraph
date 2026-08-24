"""MedPhyGraph dynamic-state builders (no model changes).

From each static base scene, produce 3 consecutive states with controlled ops:
  s0: base
  s1: transfer / move / add
  s2: remove / occlude / move
"""

from __future__ import annotations

from typing import Any

from medphygraph.schema import HealthEntity, HealthScene

OperationType = str


def _clone_scene(scene: HealthScene, scene_id: str) -> HealthScene:
    ents = [HealthEntity.from_dict(e.to_dict()) for e in scene.entities]
    return HealthScene(
        scene_id=scene_id,
        entities=ents,
        seed=scene.seed,
        room_name=scene.room_name,
        notes=f"MedPhyGraph dynamic state from {scene.scene_id}. Simulated only.",
    )


def _get(scene: HealthScene, eid: str) -> HealthEntity | None:
    return scene.entity_map().get(eid)


def _remove_entity(scene: HealthScene, eid: str) -> bool:
    if eid not in scene.entity_map():
        return False
    e = scene.entity_map()[eid]
    if e.entity_type in ("floor", "wall", "ceiling", "zone"):
        return False
    scene.entities = [x for x in scene.entities if x.entity_id != eid]
    return True


def _move_entity(scene: HealthScene, eid: str, xyz: tuple[float, float, float]) -> bool:
    e = _get(scene, eid)
    if e is None or e.entity_type in ("floor", "wall", "ceiling", "zone"):
        return False
    e.pose_xyz = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return True


def _add_entity(scene: HealthScene, ent: HealthEntity) -> bool:
    if ent.entity_id in scene.entity_map():
        return False
    scene.entities.append(ent)
    return True


def op_transfer_monitor_near_bench(scene: HealthScene) -> dict[str, Any]:
    """Move monitor beside/on bench footprint (support relation may change after CF)."""
    mon = _get(scene, "monitor")
    bench = _get(scene, "bench")
    if mon is None or bench is None:
        return {"ok": False, "operation": "transfer_support", "detail": "missing monitor/bench"}
    # Place monitor COM above bench top (apparent transfer to bedside table)
    top = float(bench.pose_xyz[2] + 0.5 * bench.size_xyz[2])
    hz = float(mon.size_xyz[2])
    _move_entity(
        scene,
        "monitor",
        (float(bench.pose_xyz[0]), float(bench.pose_xyz[1]), top + 0.5 * hz),
    )
    # The monitor's PRIOR metadata may carry a stale bed-relative
    # hard_case/apparent_host/true_host label (e.g. from build_hard_scene's
    # direct_furniture_stack case). After this transfer the monitor's direct
    # host is the bench, not the bed, so those keys must be replaced rather
    # than merged over — leaving them would describe a relationship the
    # object no longer has. See
    # repair_sr1_support_semantics/sr2a_final/semantic_metadata_migration.json.
    stale_keys = {"hard_case", "apparent_host", "true_host", "direct_host"}
    carried = {k: v for k, v in (mon.metadata or {}).items() if k not in stale_keys}
    mon.metadata = {
        **carried,
        "dynamic_op": "transfer_support",
        "intended_surface": "bench",
        "hard_case": "direct_furniture_stack",
        "direct_host": "bench",
    }
    return {
        "ok": True,
        "operation": "transfer_support",
        "subject_id": "monitor",
        "from_hint": "prior_location",
        "to_hint": "bench",
    }


def op_move_cart(scene: HealthScene) -> dict[str, Any]:
    cart = _get(scene, "cart")
    if cart is None:
        return {"ok": False, "operation": "move_object", "detail": "missing cart"}
    x, y, z = cart.pose_xyz
    _move_entity(scene, "cart", (float(x + 0.6), float(y - 0.4), z))
    cart.metadata = {**(cart.metadata or {}), "dynamic_op": "move_object"}
    return {"ok": True, "operation": "move_object", "subject_id": "cart"}


def op_remove_monitor(scene: HealthScene) -> dict[str, Any]:
    ok = _remove_entity(scene, "monitor")
    return {"ok": ok, "operation": "remove_object", "subject_id": "monitor"}


def op_remove_tray(scene: HealthScene) -> dict[str, Any]:
    eid = "tray" if _get(scene, "tray") else ("walker" if _get(scene, "walker") else None)
    if eid is None:
        return {"ok": False, "operation": "remove_object", "detail": "no tray/walker"}
    ok = _remove_entity(scene, eid)
    return {"ok": ok, "operation": "remove_object", "subject_id": eid}


def op_add_container(scene: HealthScene) -> dict[str, Any]:
    if _get(scene, "container_dyn") is not None:
        return {"ok": False, "operation": "add_object", "detail": "exists"}
    ent = HealthEntity(
        "container_dyn",
        "equipment_cart",
        (1.6, 1.2, 0.35),
        (0.35, 0.3, 0.3),
        parent_zone="zone_main",
        mass_kg=4.0,
        metadata={"display_label": "medical_containers", "dynamic_op": "add_object"},
    )
    ok = _add_entity(scene, ent)
    return {"ok": ok, "operation": "add_object", "subject_id": "container_dyn"}


def op_occlude_iv(scene: HealthScene) -> dict[str, Any]:
    """Move IV pole behind bed (partial occlusion proxy via pose)."""
    iv = _get(scene, "iv_pole")
    bed = _get(scene, "bed")
    if iv is None or bed is None:
        return {"ok": False, "operation": "temporary_occlusion", "detail": "missing iv/bed"}
    _move_entity(
        scene,
        "iv_pole",
        (float(bed.pose_xyz[0] - 0.15), float(bed.pose_xyz[1] + 0.15), iv.pose_xyz[2]),
    )
    iv.metadata = {**(iv.metadata or {}), "dynamic_op": "temporary_occlusion", "occluder": "bed"}
    return {"ok": True, "operation": "temporary_occlusion", "subject_id": "iv_pole", "occluder": "bed"}


def build_state_sequence(base: HealthScene, *, prefer_add: bool = False) -> list[dict[str, Any]]:
    """Return 3 states: s0 base, s1 transfer|add|move, s2 remove|occlude."""
    base_id = base.scene_id
    s0 = _clone_scene(base, f"{base_id}_s0")

    s1 = _clone_scene(base, f"{base_id}_s1")
    if prefer_add:
        op1 = op_add_container(s1)
        if not op1.get("ok"):
            op1 = op_move_cart(s1)
        if not op1.get("ok"):
            op1 = op_transfer_monitor_near_bench(s1)
    else:
        op1 = op_transfer_monitor_near_bench(s1)
        if not op1.get("ok"):
            op1 = op_move_cart(s1)
        if not op1.get("ok"):
            op1 = op_add_container(s1)

    s2 = _clone_scene(s1, f"{base_id}_s2")
    op2 = op_remove_monitor(s2)
    if not op2.get("ok"):
        op2 = op_remove_tray(s2)
    if not op2.get("ok"):
        op2 = op_occlude_iv(s2)
    if not op2.get("ok"):
        # if monitor already gone somehow, remove added container
        if _get(s2, "container_dyn") is not None:
            ok = _remove_entity(s2, "container_dyn")
            op2 = {"ok": ok, "operation": "remove_object", "subject_id": "container_dyn"}
        else:
            op2 = op_add_container(s2)

    return [
        {"state_index": 0, "scene": s0, "operation": "identity", "op_meta": {"ok": True, "operation": "identity"}},
        {"state_index": 1, "scene": s1, "operation": op1.get("operation", "unknown"), "op_meta": op1},
        {"state_index": 2, "scene": s2, "operation": op2.get("operation", "unknown"), "op_meta": op2},
    ]


def gt_edge_set(graph_dict: dict[str, Any]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for e in graph_dict.get("edges") or []:
        if e.get("is_gt") is True:
            out.add((str(e["subject_id"]), str(e["host_id"])))
    return out


def canonical_transition_delta(
    prev_gt: set[tuple[str, str]], curr_gt: set[tuple[str, str]]
) -> dict[str, set[tuple[str, str]]]:
    """Pure ADD/REMOVE/PERSIST graph difference between two GT edge sets.

    This factors out the same set-difference logic used by
    :func:`transition_annotation` below, for callers (e.g. the canonical
    Dynamic GT builder in ``medphygraph_dynamic_gt.py``) that must NOT
    require an authored ``operation``/``op_meta`` placeholder to construct a
    transition record. Reads nothing but the two GT edge sets.
    """
    return {
        "added_edges": curr_gt - prev_gt,
        "removed_edges": prev_gt - curr_gt,
        "persisted_edges": prev_gt & curr_gt,
    }


def transition_annotation(
    *,
    base_scene_id: str,
    prev_id: str,
    curr_id: str,
    operation: str,
    op_meta: dict[str, Any],
    prev_gt: set[tuple[str, str]],
    curr_gt: set[tuple[str, str]],
) -> dict[str, Any]:
    added = sorted([list(x) for x in (curr_gt - prev_gt)])
    removed = sorted([list(x) for x in (prev_gt - curr_gt)])
    unchanged = sorted([list(x) for x in (prev_gt & curr_gt)])
    return {
        "base_scene_id": base_scene_id,
        "previous_state_id": prev_id,
        "current_state_id": curr_id,
        "operation_type": operation,
        "operation_meta": op_meta,
        "added_ground_truth_edges": added,
        "removed_ground_truth_edges": removed,
        "unchanged_ground_truth_edges": unchanged,
        "n_added": len(added),
        "n_removed": len(removed),
        "n_unchanged": len(unchanged),
    }
