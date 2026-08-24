"""Public MedPhyGraph API (thin facade over paper-frozen implementation)."""

from medphygraph.analytic_physics import run_counterfactual_pair
from medphygraph.candidates import generate_candidates
from medphygraph.consistency import (
    DeltaUnionV2Config,
    apply_consistency,
    apply_transition_aware_consistency_v2,
    count_violations,
)
from medphygraph.data import load_dataset, split_samples
from medphygraph.model import HealthDyPhyGraph, ModelConfig, build_model
from medphygraph.schema import (
    ENTITY_TYPES,
    STRUCTURAL_TYPES,
    EntityType,
    HealthEntity,
    HealthScene,
)
from medphygraph.scene_graph import (
    EdgeAction,
    GraphDiff,
    PhysicalSceneGraph,
    SupportEdge,
)
from medphygraph.twin_store import write_twin

CFSupportNet = HealthDyPhyGraph

__all__ = [
  "CFSupportNet",
  "DeltaUnionV2Config",
  "ENTITY_TYPES",
  "STRUCTURAL_TYPES",
  "EntityType",
  "EdgeAction",
  "GraphDiff",
  "HealthEntity",
  "HealthScene",
  "ModelConfig",
  "PhysicalSceneGraph",
  "SupportEdge",
  "apply_consistency",
  "apply_transition_aware_consistency_v2",
  "build_model",
  "count_violations",
  "generate_candidates",
  "load_dataset",
  "run_counterfactual_pair",
  "split_samples",
  "write_twin",
]
