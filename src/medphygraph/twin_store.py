"""P0-13: Versioned twin write-back + provenance log."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medphygraph.consistency import ConsistencyReport, apply_consistency
from medphygraph.scene_graph import GraphDiff, PhysicalSceneGraph


@dataclass
class TwinWriteBackResult:
    twin_path: Path
    provenance_path: Path
    version: int
    graph_diff: GraphDiff
    consistency: ConsistencyReport
    n_present_edges: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "twin_path": str(self.twin_path),
            "provenance_path": str(self.provenance_path),
            "version": self.version,
            "graph_diff": self.graph_diff.to_dict(),
            "consistency": self.consistency.to_dict(),
            "n_present_edges": self.n_present_edges,
        }


def write_twin(
    g: PhysicalSceneGraph,
    *,
    out_dir: Path | str,
    evidence_source: str = "medphygraph",
    timestamp: float | None = None,
    apply_graph_consistency: bool = True,
) -> TwinWriteBackResult:
    """Persist updated twin JSON + append-only provenance log."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = float(time.time() if timestamp is None else timestamp)

    consistency = ConsistencyReport()
    if apply_graph_consistency:
        consistency = apply_consistency(g)

    # bump version for write-back snapshot
    v0 = g.version
    g.version = v0 + 1
    for e in g.edges.values():
        e.version = g.version
        if e.present and not e.evidence_source:
            e.evidence_source = evidence_source
        e.timestamp = ts

    twin_doc = {
        "scene_id": g.scene.scene_id,
        "graph_version": g.version,
        "timestamp": ts,
        "evidence_source": evidence_source,
        "entities": [e.to_dict() for e in g.scene.entities],
        "edges": [
            {
                "subject_id": e.subject_id,
                "host_id": e.host_id,
                "relation": e.relation,
                "present": e.present,
                "confidence": e.confidence,
                "score": e.score,
                "evidence_source": e.evidence_source,
                "timestamp": e.timestamp,
                "version": e.version,
                "is_gt": e.is_gt,
            }
            for e in g.edges.values()
        ],
        "consistency": consistency.to_dict(),
    }
    twin_path = out_dir / f"twin_v{g.version:04d}.json"
    twin_path.write_text(json.dumps(twin_doc, indent=2), encoding="utf-8")
    # also latest pointer
    (out_dir / "twin_latest.json").write_text(json.dumps(twin_doc, indent=2), encoding="utf-8")

    diff = GraphDiff(
        scene_id=g.scene.scene_id,
        version_from=v0,
        version_to=g.version,
        operations=[
            {
                "action": "WRITE_BACK",
                "subject_id": e.subject_id,
                "host_id": e.host_id,
                "present": e.present,
                "confidence": e.confidence,
                "evidence_source": e.evidence_source,
            }
            for e in g.edges.values()
        ],
        timestamp=ts,
    )
    g.provenance.append(
        {
            "event": "twin_write_back",
            "timestamp": ts,
            "evidence_source": evidence_source,
            "version": g.version,
            "consistency": consistency.to_dict(),
            "diff": diff.to_dict(),
        }
    )
    prov_path = out_dir / "provenance.jsonl"
    with prov_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(g.provenance[-1]) + "\n")

    g.save(out_dir / f"graph_v{g.version:04d}.json")
    return TwinWriteBackResult(
        twin_path=twin_path,
        provenance_path=prov_path,
        version=g.version,
        graph_diff=diff,
        consistency=consistency,
        n_present_edges=len([e for e in g.edges.values() if e.present]),
    )
