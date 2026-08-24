"""P0-2: PhysicalSceneGraph — multi-object SUPPORT_BY graph + GraphDiff write-back helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from medphygraph.schema import HealthEntity, HealthScene, STRUCTURAL_TYPES

EdgeAction = Literal["ADD", "RETAIN", "REMOVE"]


@dataclass
class SupportEdge:
    subject_id: str
    host_id: str
    relation: str = "SUPPORTED_BY"
    present: bool = False
    score: float = 0.0
    confidence: float = 0.0
    is_candidate: bool = True
    is_gt: bool | None = None
    evidence_source: str = ""
    timestamp: float = 0.0
    version: int = 0

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject_id, self.host_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SupportEdge:
        return cls(
            subject_id=str(d["subject_id"]),
            host_id=str(d["host_id"]),
            relation=str(d.get("relation", "SUPPORTED_BY")),
            present=bool(d.get("present", False)),
            score=float(d.get("score", 0.0)),
            confidence=float(d.get("confidence", d.get("score", 0.0))),
            is_candidate=bool(d.get("is_candidate", True)),
            is_gt=d.get("is_gt", None),
            evidence_source=str(d.get("evidence_source", "")),
            timestamp=float(d.get("timestamp", 0.0)),
            version=int(d.get("version", 0)),
        )


@dataclass
class GraphDiff:
    scene_id: str
    version_from: int
    version_to: int
    operations: list[dict[str, Any]] = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PhysicalSceneGraph:
    """One graph per healthcare scene with candidate / predicted / GT SUPPORT_BY edges."""

    scene: HealthScene
    edges: dict[tuple[str, str], SupportEdge] = field(default_factory=dict)
    version: int = 0
    provenance: list[dict[str, Any]] = field(default_factory=list)

    def nodes(self) -> dict[str, HealthEntity]:
        return self.scene.entity_map()

    def add_candidate(self, subject_id: str, host_id: str, *, score: float = 0.0) -> SupportEdge:
        if subject_id not in self.nodes() or host_id not in self.nodes():
            raise KeyError(f"Unknown node in candidate ({subject_id}, {host_id})")
        key = (subject_id, host_id)
        edge = SupportEdge(
            subject_id=subject_id,
            host_id=host_id,
            present=False,
            score=score,
            confidence=score,
            is_candidate=True,
            version=self.version,
        )
        self.edges[key] = edge
        return edge

    def set_gt(self, subject_id: str, host_id: str, *, positive: bool) -> SupportEdge:
        key = (subject_id, host_id)
        if key not in self.edges:
            self.add_candidate(subject_id, host_id)
        self.edges[key].is_gt = bool(positive)
        if positive:
            self.edges[key].present = True
            self.edges[key].score = 1.0
            self.edges[key].confidence = 1.0
        return self.edges[key]

    def update_from_probability(
        self,
        subject_id: str,
        host_id: str,
        prob: float,
        *,
        thr_add: float = 0.5,
        thr_remove: float = 0.5,
        evidence_source: str = "predictor",
        timestamp: float = 0.0,
    ) -> tuple[SupportEdge, EdgeAction]:
        key = (subject_id, host_id)
        if key not in self.edges:
            self.add_candidate(subject_id, host_id)
        edge = self.edges[key]
        edge.score = float(prob)
        edge.confidence = float(prob)
        edge.evidence_source = evidence_source
        edge.timestamp = timestamp
        if (not edge.present) and prob >= thr_add:
            edge.present = True
            action: EdgeAction = "ADD"
        elif edge.present and prob < thr_remove:
            edge.present = False
            action = "REMOVE"
        else:
            action = "RETAIN"
        edge.version = self.version
        return edge, action

    def apply_probabilities(
        self,
        probs: dict[tuple[str, str], float],
        *,
        evidence_source: str = "predictor",
        timestamp: float = 0.0,
    ) -> GraphDiff:
        ops: list[dict[str, Any]] = []
        v0 = self.version
        for (s, h), p in probs.items():
            edge, action = self.update_from_probability(
                s, h, float(p), evidence_source=evidence_source, timestamp=timestamp
            )
            ops.append(
                {
                    "action": action,
                    "subject_id": s,
                    "host_id": h,
                    "prob": float(p),
                    "present": edge.present,
                    "evidence_source": evidence_source,
                }
            )
        self.version = v0 + 1
        for e in self.edges.values():
            e.version = self.version
        diff = GraphDiff(
            scene_id=self.scene.scene_id,
            version_from=v0,
            version_to=self.version,
            operations=ops,
            timestamp=timestamp,
        )
        self.provenance.append(diff.to_dict())
        return diff

    def present_edges(self) -> list[SupportEdge]:
        return [e for e in self.edges.values() if e.present]

    def candidate_edges(self) -> list[SupportEdge]:
        return [e for e in self.edges.values() if e.is_candidate]

    def gt_positive_edges(self) -> list[SupportEdge]:
        return [e for e in self.edges.values() if e.is_gt is True]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene.to_dict(),
            "version": self.version,
            "edges": [e.to_dict() for e in self.edges.values()],
            "provenance": list(self.provenance),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PhysicalSceneGraph:
        scene = HealthScene.from_dict(d["scene"])
        g = cls(scene=scene, version=int(d.get("version", 0)), provenance=list(d.get("provenance") or []))
        for ed in d.get("edges") or []:
            edge = SupportEdge.from_dict(ed)
            g.edges[edge.key] = edge
        return g

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> PhysicalSceneGraph:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def demo_graph_with_candidates(scene: HealthScene | None = None) -> PhysicalSceneGraph:
    """Build a demo graph: true bed→floor; false walker→bed (geometry-plausible)."""
    from medphygraph.schema import example_rehab_scene

    scene = scene or example_rehab_scene()
    g = PhysicalSceneGraph(scene=scene)
    # Plausible / true
    g.add_candidate("bed", "floor")
    g.set_gt("bed", "floor", positive=True)
    g.add_candidate("walker", "floor")
    g.set_gt("walker", "floor", positive=True)
    g.add_candidate("wheelchair", "floor")
    g.set_gt("wheelchair", "floor", positive=True)
    g.add_candidate("monitor", "floor")
    g.set_gt("monitor", "floor", positive=True)
    g.add_candidate("bench", "floor")
    g.set_gt("bench", "floor", positive=True)
    # Geometry-plausible false: walker near bed may look stacked/supported
    g.add_candidate("walker", "bed")
    g.set_gt("walker", "bed", positive=False)
    g.add_candidate("iv_pole", "bed")
    g.set_gt("iv_pole", "bed", positive=False)
    # Structural termination examples
    g.add_candidate("rail", "wall_n")
    g.set_gt("rail", "wall_n", positive=True)
    g.add_candidate("lift", "ceiling")
    g.set_gt("lift", "ceiling", positive=True)
    return g


def is_structural_id(graph: PhysicalSceneGraph, entity_id: str) -> bool:
    n = graph.nodes().get(entity_id)
    return bool(n and n.entity_type in STRUCTURAL_TYPES)
