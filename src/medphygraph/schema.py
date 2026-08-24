"""P0-1: Healthcare rehabilitation-room entity / scene schema.

Nodes represent healthcare equipment, furniture, structural elements, and room hierarchy.
No clinical claims. Simulation / digital-twin research only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EntityType = Literal[
    "floor",
    "wall",
    "ceiling",
    "therapy_bed",
    "walker",
    "wheelchair",
    "monitor_cart",
    "iv_pole",
    "cabinet",
    "therapy_bench",
    "wall_rail",
    "patient_lift",
    "equipment_cart",
    "other_furniture",
    "zone",
]

STRUCTURAL_TYPES: frozenset[str] = frozenset({"floor", "wall", "ceiling"})

ENTITY_TYPES: tuple[str, ...] = (
    "floor",
    "wall",
    "ceiling",
    "therapy_bed",
    "walker",
    "wheelchair",
    "monitor_cart",
    "iv_pole",
    "cabinet",
    "therapy_bench",
    "wall_rail",
    "patient_lift",
    "equipment_cart",
    "other_furniture",
    "zone",
)

# Inventory required by P0-4 (distribution filled when scenes are authored).
REQUIRED_OBJECT_INVENTORY: tuple[str, ...] = (
    "therapy_bed",
    "walker",
    "wheelchair",
    "monitor_cart",
    "iv_pole",
    "cabinet",
    "therapy_bench",
    "wall_rail",
    "patient_lift",
    "floor",
    "wall",
    "ceiling",
)


@dataclass
class HealthEntity:
    """One twin node in a healthcare rehab-room scene."""

    entity_id: str
    entity_type: EntityType
    pose_xyz: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    parent_zone: str
    timestamp: float = 0.0
    geometry_ref: str = ""
    movable: bool = True
    mass_kg: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.entity_type not in ENTITY_TYPES:
            raise ValueError(f"Unknown entity_type: {self.entity_type}")
        if self.entity_type in STRUCTURAL_TYPES:
            self.movable = False
        if not self.geometry_ref:
            self.geometry_ref = f"proxy://aabb/{self.entity_type}"

    @property
    def is_structural(self) -> bool:
        return self.entity_type in STRUCTURAL_TYPES

    def aabb_min(self) -> tuple[float, float, float]:
        hx, hy, hz = (0.5 * s for s in self.size_xyz)
        x, y, z = self.pose_xyz
        return (x - hx, y - hy, z - hz)

    def aabb_max(self) -> tuple[float, float, float]:
        hx, hy, hz = (0.5 * s for s in self.size_xyz)
        x, y, z = self.pose_xyz
        return (x + hx, y + hy, z + hz)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pose_xyz"] = list(self.pose_xyz)
        d["size_xyz"] = list(self.size_xyz)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HealthEntity:
        return cls(
            entity_id=str(d["entity_id"]),
            entity_type=d["entity_type"],
            pose_xyz=tuple(float(x) for x in d["pose_xyz"]),
            size_xyz=tuple(float(x) for x in d["size_xyz"]),
            parent_zone=str(d["parent_zone"]),
            timestamp=float(d.get("timestamp", 0.0)),
            geometry_ref=str(d.get("geometry_ref", "")),
            movable=bool(d.get("movable", True)),
            mass_kg=float(d.get("mass_kg", 1.0)),
            metadata=dict(d.get("metadata") or {}),
        )


@dataclass
class HealthScene:
    """One rehabilitation / assisted-care room scene."""

    scene_id: str
    entities: list[HealthEntity]
    seed: int = 0
    room_name: str = "rehab_room"
    notes: str = "Simulated healthcare built environment — not clinical validation."

    def __post_init__(self) -> None:
        ids = [e.entity_id for e in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate entity_id in scene {self.scene_id}")
        zones = {e.entity_id for e in self.entities if e.entity_type == "zone"}
        for e in self.entities:
            if e.entity_type == "zone":
                continue
            if e.parent_zone and e.parent_zone not in zones and e.parent_zone != self.room_name:
                # allow parent_zone == room_name without explicit zone node
                if not any(z.entity_id == e.parent_zone for z in self.entities):
                    pass

    def entity_map(self) -> dict[str, HealthEntity]:
        return {e.entity_id: e for e in self.entities}

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "seed": self.seed,
            "room_name": self.room_name,
            "notes": self.notes,
            "entities": [e.to_dict() for e in self.entities],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HealthScene:
        return cls(
            scene_id=str(d["scene_id"]),
            seed=int(d.get("seed", 0)),
            room_name=str(d.get("room_name", "rehab_room")),
            notes=str(d.get("notes", "")),
            entities=[HealthEntity.from_dict(e) for e in d["entities"]],
        )


def example_rehab_scene(*, scene_id: str = "rehab_000", seed: int = 0) -> HealthScene:
    """Compact demo scene for schema / graph unit tests."""
    z = "zone_main"
    ents = [
        HealthEntity("zone_main", "zone", (0.0, 0.0, 1.5), (6.0, 5.0, 3.0), parent_zone=z, movable=False),
        HealthEntity("floor", "floor", (0.0, 0.0, 0.0), (6.0, 5.0, 0.05), parent_zone=z),
        HealthEntity("wall_n", "wall", (0.0, 2.5, 1.5), (6.0, 0.1, 3.0), parent_zone=z),
        HealthEntity("wall_s", "wall", (0.0, -2.5, 1.5), (6.0, 0.1, 3.0), parent_zone=z),
        HealthEntity("wall_w", "wall", (-3.0, 0.0, 1.5), (0.1, 5.0, 3.0), parent_zone=z),
        HealthEntity("wall_e", "wall", (3.0, 0.0, 1.5), (0.1, 5.0, 3.0), parent_zone=z),
        HealthEntity("ceiling", "ceiling", (0.0, 0.0, 3.0), (6.0, 5.0, 0.05), parent_zone=z),
        HealthEntity("bed", "therapy_bed", (0.0, 0.5, 0.45), (2.0, 0.9, 0.5), parent_zone=z, mass_kg=80.0),
        HealthEntity("walker", "walker", (1.2, -0.8, 0.45), (0.6, 0.5, 0.9), parent_zone=z, mass_kg=8.0),
        HealthEntity("wheelchair", "wheelchair", (-1.5, -1.0, 0.45), (0.7, 0.7, 0.9), parent_zone=z, mass_kg=20.0),
        HealthEntity("monitor", "monitor_cart", (1.8, 0.8, 0.7), (0.5, 0.5, 1.2), parent_zone=z, mass_kg=25.0),
        HealthEntity("iv_pole", "iv_pole", (-0.8, 1.0, 0.9), (0.3, 0.3, 1.8), parent_zone=z, mass_kg=12.0),
        HealthEntity("cabinet", "cabinet", (-2.5, 1.5, 0.6), (0.8, 0.5, 1.2), parent_zone=z, movable=False, mass_kg=60.0),
        HealthEntity("bench", "therapy_bench", (0.5, -1.8, 0.25), (1.2, 0.4, 0.45), parent_zone=z, mass_kg=30.0),
        HealthEntity("rail", "wall_rail", (0.0, 2.4, 0.9), (2.0, 0.08, 0.1), parent_zone=z, movable=False, mass_kg=5.0),
        HealthEntity("lift", "patient_lift", (2.0, -1.5, 2.2), (1.0, 0.3, 0.4), parent_zone=z, movable=False, mass_kg=40.0),
    ]
    return HealthScene(scene_id=scene_id, entities=ents, seed=seed)
