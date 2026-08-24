"""Graph loading, target construction, and metric helpers for core evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from medphygraph.candidates import (
  _contact,
  _vertical_stack,
  _xy_overlap,
)
from medphygraph.data import load_dataset
from medphygraph.schema import HealthScene
from medphygraph.scene_graph import PhysicalSceneGraph

from evaluation._config import CFG, DYNAMIC, TRANSFER


def present(graph: PhysicalSceneGraph) -> set[tuple[str, str]]:
  return {(edge.subject_id, edge.host_id) for edge in graph.edges.values() if edge.present}


def gt_from(samples: list[dict], graph_gt: PhysicalSceneGraph) -> set[tuple[str, str]]:
  if samples:
    return {(sample["subject_id"], sample["host_id"]) for sample in samples if int(sample["label"]) == 1}
  return {(edge.subject_id, edge.host_id) for edge in graph_gt.edges.values() if edge.is_gt is True}


def load_transitions(scenes: Path) -> list[dict]:
  index = json.loads((scenes / "transitions_index.json").read_text(encoding="utf-8"))
  return list(index["transitions"] if "transitions" in index else index)


def load_scene(scenes: Path, state_id: str) -> HealthScene:
  return HealthScene.from_dict(json.loads((scenes / state_id / "scene.json").read_text(encoding="utf-8")))


def op_meta(transition: dict, scenes: Path) -> dict:
  """Evaluation-target metadata only — never passed to inference."""
  meta_path = scenes / transition["current_state_id"] / "phase2_state_meta.json"
  merged = dict(transition.get("operation_meta") or {})
  if meta_path.exists():
    blob = json.loads(meta_path.read_text(encoding="utf-8"))
    merged = {**merged, **(blob.get("op_meta") or {})}
  return merged


def prf(tp: float, fp: float, fn: float) -> dict[str, float]:
  precision = tp / max(tp + fp, 1.0)
  recall = tp / max(tp + fn, 1.0)
  f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
  if tp == 0 and fp == 0 and fn == 0:
    return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0.0, "fp": 0.0, "fn": 0.0, "empty_empty": True}
  return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "empty_empty": False}


def edge_f1_sets(pred: set[tuple[str, str]], gt: set[tuple[str, str]]) -> dict[str, float]:
  tp = float(len(pred & gt))
  fp = float(len(pred - gt))
  fn = float(len(gt - pred))
  return prf(tp, fp, fn)


def direct_support_host(scene: HealthScene, subject_id: str, candidate_hosts: list[str]) -> str | None:
  """Operation-independent direct-support rule (geometry), no floor special-case."""
  nodes = scene.entity_map()
  subject = nodes.get(subject_id)
  if subject is None or not candidate_hosts:
    return None
  entities = list(nodes.values())
  scored: list[tuple[float, str]] = []
  for host_id in candidate_hosts:
    host = nodes.get(host_id)
    if host is None:
      continue
    gap = float(subject.aabb_min()[2] - host.aabb_max()[2])
    xy = _xy_overlap(subject, host, expand=CFG.direct_xy_expand)
    stack = _vertical_stack(subject, host, margin=CFG.direct_gap_hi)
    contact = _contact(subject, host, eps=CFG.direct_contact_eps)
    closer = False
    for middle in entities:
      if middle.entity_id in (subject_id, host_id) or middle.entity_type == "zone":
        continue
      if host.aabb_max()[2] - 1e-6 < middle.aabb_max()[2] < subject.aabb_min()[2] + 1e-6:
        if _xy_overlap(subject, middle, expand=CFG.direct_xy_expand) and _vertical_stack(
          subject, middle, margin=CFG.direct_gap_hi
        ):
          closer = True
          break
    ok = xy and stack and (contact or (CFG.direct_gap_lo <= gap <= CFG.direct_gap_hi)) and not closer
    if ok:
      scored.append((abs(gap), host_id))
  if not scored:
    return None
  scored.sort()
  return scored[0][1]


