"""Physically plausible hard rehab scenes for DyPhyGraph-Health.

These scenes create geometry-ambiguous SUPPORT_BY cases where geometry-only
rules are insufficient, without relying on artificial feature noise:

  - hidden_wall_support: wall_rail mounted on wall (floor must not cancel GT)
  - hidden_ceiling_support: patient_lift hung from ceiling
  - direct_furniture_stack (was "contact_without_support"): monitor rests
    directly on the bed. Under MedPhyGraph direct load-bearing support
    semantics this IS load-bearing support (monitor->bed positive,
    monitor->floor negative) — furniture is a valid direct host, and floor is
    not retained through an intermediate host. See
    repair_sr1_support_semantics/CANONICAL_DEFINITION.md.
  - proximity_without_support: object near/contacting furniture but NOT
    vertically stacked on it (no direct-stack gap) — remains floor-supported
  - multi_support: subject requires BOTH floor and wall anchors

Partial observation ratios are produced later by features.pack_sample (unchanged).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from medphygraph.schema import HealthEntity, HealthScene


def _base_structural(zone: str = "zone_main") -> list[HealthEntity]:
    return [
        HealthEntity("zone_main", "zone", (0, 0, 1.5), (6, 5, 3), parent_zone=zone, movable=False),
        HealthEntity("floor", "floor", (0, 0, 0), (6, 5, 0.05), parent_zone=zone),
        HealthEntity("wall_n", "wall", (0, 2.5, 1.5), (6, 0.1, 3), parent_zone=zone),
        HealthEntity("wall_s", "wall", (0, -2.5, 1.5), (6, 0.1, 3), parent_zone=zone),
        HealthEntity("wall_w", "wall", (-3.0, 0.0, 1.5), (0.1, 5, 3), parent_zone=zone),
        HealthEntity("wall_e", "wall", (3.0, 0.0, 1.5), (0.1, 5, 3), parent_zone=zone),
        HealthEntity("ceiling", "ceiling", (0, 0, 3), (6, 5, 0.05), parent_zone=zone),
    ]


def _core_furniture(z: str, rng: np.random.Generator) -> list[HealthEntity]:
    bed_xy = (float(rng.uniform(-0.8, 0.8)), float(rng.uniform(-0.3, 0.8)))
    return [
        HealthEntity(
            "bed",
            "therapy_bed",
            (bed_xy[0], bed_xy[1], 0.45),
            (2.0, 0.9, 0.5),
            parent_zone=z,
            mass_kg=80,
        ),
        HealthEntity(
            "wheelchair",
            "wheelchair",
            (float(rng.uniform(-2.2, -1.2)), float(rng.uniform(-1.4, 0.2)), 0.45),
            (0.7, 0.7, 0.9),
            parent_zone=z,
            mass_kg=20,
        ),
        HealthEntity(
            "cabinet",
            "cabinet",
            (-2.4, 1.4, 0.6),
            (0.8, 0.5, 1.2),
            parent_zone=z,
            movable=False,
            mass_kg=60,
        ),
        HealthEntity(
            "bench",
            "therapy_bench",
            (float(rng.uniform(-0.2, 1.2)), float(rng.uniform(-2.0, -1.5)), 0.25),
            (1.2, 0.4, 0.45),
            parent_zone=z,
            mass_kg=30,
        ),
        HealthEntity(
            "iv_pole",
            "iv_pole",
            (bed_xy[0] - 1.0, bed_xy[1] + 0.8, 0.9),
            (0.3, 0.3, 1.8),
            parent_zone=z,
            mass_kg=12,
        ),
    ]


def build_hard_scene(scene_idx: int, *, seed: int = 20260730) -> tuple[HealthScene, dict[str, Any]]:
    """Build one hard scene with tagged ambiguity cases."""
    rng = np.random.default_rng(seed + 1000 + scene_idx)
    z = "zone_main"
    ents = _base_structural(z)
    ents.extend(_core_furniture(z, rng))

    hard_tags: list[str] = []

    # --- Hidden wall support: rail mounted on wall_n (not floor-supported) ---
    ents.append(
        HealthEntity(
            "rail",
            "wall_rail",
            (float(rng.uniform(-0.5, 0.5)), 2.42, 0.95),
            (2.0, 0.08, 0.1),
            parent_zone=z,
            movable=False,
            mass_kg=5,
            metadata={"support_kind": "wall_mount", "anchor_host": "wall_n", "hard_case": "hidden_wall_support"},
        )
    )
    hard_tags.append("hidden_wall_support")

    # --- Hidden ceiling support: lift hung from ceiling ---
    ents.append(
        HealthEntity(
            "lift",
            "patient_lift",
            (float(rng.uniform(1.2, 2.2)), float(rng.uniform(-1.8, -1.0)), 2.35),
            (1.0, 0.3, 0.4),
            parent_zone=z,
            movable=False,
            mass_kg=40,
            metadata={"support_kind": "ceiling_hang", "anchor_host": "ceiling", "hard_case": "hidden_ceiling_support"},
        )
    )
    hard_tags.append("hidden_ceiling_support")

    # --- Direct furniture stack: monitor rests directly on the bed ---
    # Bed top ≈ 0.70; place monitor COM so bottom rests on mattress.
    # This IS the monitor's direct load-bearing host (bed), not the floor —
    # furniture support is valid and floor is not retained through an
    # intermediate host.
    bed = next(e for e in ents if e.entity_id == "bed")
    bed_top = float(bed.pose_xyz[2] + 0.5 * bed.size_xyz[2])
    mon_h = 0.9
    mon_z = bed_top + 0.5 * mon_h
    ents.append(
        HealthEntity(
            "monitor",
            "monitor_cart",
            (float(bed.pose_xyz[0] + rng.uniform(-0.15, 0.15)), float(bed.pose_xyz[1] + rng.uniform(-0.1, 0.1)), mon_z),
            (0.45, 0.45, mon_h),
            parent_zone=z,
            mass_kg=25,
            metadata={"hard_case": "direct_furniture_stack", "direct_host": "bed"},
        )
    )
    hard_tags.append("direct_furniture_stack")

    # --- Walker: proximity/contact to bed without true stack (classic false candidate) ---
    ents.append(
        HealthEntity(
            "walker",
            "walker",
            (
                float(bed.pose_xyz[0] + 1.05),
                float(bed.pose_xyz[1] + rng.uniform(-0.35, 0.1)),
                0.45,
            ),
            (0.6, 0.5, 0.9),
            parent_zone=z,
            mass_kg=8,
            metadata={"hard_case": "proximity_without_support", "apparent_host": "bed", "true_host": "floor"},
        )
    )
    hard_tags.append("proximity_without_support")

    # --- Multi-support: cart requires BOTH floor and wall_w (both anchors necessary) ---
    ents.append(
        HealthEntity(
            "cart",
            "equipment_cart",
            (-2.75, float(rng.uniform(-0.4, 0.4)), 0.5),
            (0.55, 0.4, 0.9),
            parent_zone=z,
            mass_kg=18,
            metadata={
                "support_kind": "multi_support",
                "anchors": ["floor", "wall_w"],
                "hard_case": "multi_support",
            },
        )
    )
    hard_tags.append("multi_support")

    scene = HealthScene(
        scene_id=f"hard_{scene_idx:03d}",
        entities=ents,
        seed=seed + 1000 + scene_idx,
        notes=(
            "Hard benchmark scene: geometry-ambiguous SUPPORT_BY cases "
            f"({', '.join(hard_tags)}). Analytic CF labels; not clinical."
        ),
    )
    meta = {
        "scene_id": scene.scene_id,
        "hard_tags": hard_tags,
        "expected_positive_edges": [
            ["rail", "wall_n"],
            ["lift", "ceiling"],
            ["monitor", "bed"],  # direct furniture stack IS load-bearing
            ["walker", "floor"],
            ["cart", "floor"],
            ["cart", "wall_w"],
            ["bed", "floor"],
            ["wheelchair", "floor"],
            ["cabinet", "floor"],
            ["bench", "floor"],
            ["iv_pole", "floor"],
        ],
        "expected_hard_negative_edges": [
            ["monitor", "floor"],  # ancestral only (bed is the direct host)
            ["walker", "bed"],  # proximity / contact without direct-stack gap
            ["rail", "floor"],  # mounted, not floor-supported
            ["lift", "floor"],  # hung, not floor-supported
        ],
    }
    return scene, meta


def build_hard_suite(*, n: int = 12, seed: int = 20260730) -> list[tuple[HealthScene, dict[str, Any]]]:
    return [build_hard_scene(i, seed=seed) for i in range(n)]
