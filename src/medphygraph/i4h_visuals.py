"""Resolve Isaac for Healthcare (i4h) visual assets for DyPhyGraph-Health GUI.

Display-only. Counterfactual / training still use AABB proxies.

Default root: D:\\projects\\models\\i4h-assets\\724f82e (v0.7.0),
overridable via I4H_ASSETS_ROOT.

Unified mode prefers one visual family (shared_OR + Rheo table + Organs room)
so materials/MDL resolve from the same Library tree.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_I4H_ROOT = Path(r"D:\projects\models\i4h-assets\724f82e")
ENV_I4H_ASSETS_ROOT = "I4H_ASSETS_ROOT"

_OR = (
    "Props/shared_OR_without_Mark/Collected_surgery_room_movie_with_heart_adjusted"
)

# Cohesive hospital/rehab visual map (same material library family).
ENTITY_USD_REL: dict[str, str] = {
    "therapy_bed": "Props/Rheo/Visuals/SurgicalTable_A/sm_surgicaltable_a01_01.usd",
    "wheelchair": f"{_OR}/Assets/Patient Room/Wheelchair_B/sm_wheelchair_b03_01.usd",
    "monitor_cart": (
        f"{_OR}/Assets/Patient Room/DiagnosticEquipmentCart_A/"
        "sm_diagnosticequipmentcart_a01_01.usd"
    ),
    "cabinet": f"{_OR}/Assets/Surgery Room/DrugCabinets_A/sm_drugcabinet_a01_01.usd",
    "equipment_cart": (
        f"{_OR}/Assets/Surgery Room/MedicalUtilityCart_A/sm_medicalutilitycart_a03_01.usd"
    ),
    "therapy_bench": f"{_OR}/Assets/Patient Room/SoftFurniture_B/sm_softfurniture_b03_01.usd",
    "patient_lift": (
        f"{_OR}/Assets/Surgery Room/ArticulatedSupportArm_A/"
        "sm_articulatedsupportarm_a01_01.usd"
    ),
    # Closest trolley stand-in (no dedicated walker/IV in catalog)
    "walker": f"{_OR}/Assets/Surgery Room/InstrumentTrolley_B/sm_instrumenttrolley_b01_01.usd",
    "iv_pole": f"{_OR}/Assets/Surgery Room/EmergencyTrolley_A/sm_emergencytrolley_a01_01.usd",
    "wall_rail": f"{_OR}/Assets/Patient Room/HospitalScreen_A/sm_hospitalscreen_a01_01.usd",
}

ROOM_ENV_REL = "Props/Organs/models/operating_room/room/Over_GRP_Room_Additions_merged.usd"
# The Organs OR shell references companion ATLAS_OR textures via relative USD paths.
# Partial i4h installs often ship Props/ without this tree; loading the shell then
# spams missing-texture errors and can destabilize Isaac Sim.
ROOM_ENV_TEXTURE_REL = (
    "Props/Organs/models/operating_room/ATLAS_OR/Assets/GRP_Room/textures/T_Products_D.jpg"
)
CEILING_LAMP_REL = f"{_OR}/Assets/Lamps/CeilingLamp_A/sm_ceilinglamp_a01_1m_straight_01.usd"
MATERIAL_LIBRARY_REL = f"{_OR}/Library/Material Library"


def resolve_i4h_root(explicit: Path | str | None = None) -> Path | None:
    if explicit is not None:
        p = Path(explicit)
        return p if p.is_dir() else None
    env = os.environ.get(ENV_I4H_ASSETS_ROOT, "").strip()
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    if DEFAULT_I4H_ROOT.is_dir():
        return DEFAULT_I4H_ROOT
    return None


def _join(root: Path, rel: str) -> Path:
    return root / rel.replace("/", os.sep)


def usd_for_entity_type(entity_type: str, *, root: Path | None = None) -> Path | None:
    rel = ENTITY_USD_REL.get(entity_type)
    if not rel:
        return None
    base = root if root is not None else resolve_i4h_root()
    if base is None:
        return None
    path = _join(base, rel)
    return path if path.is_file() else None


def room_environment_usd(*, root: Path | None = None) -> Path | None:
    base = root if root is not None else resolve_i4h_root()
    if base is None:
        return None
    path = _join(base, ROOM_ENV_REL)
    return path if path.is_file() else None


def room_environment_ready(*, root: Path | None = None) -> bool:
    """True when the Organs OR shell and its ATLAS_OR texture deps are present."""
    base = root if root is not None else resolve_i4h_root()
    if base is None:
        return False
    room = room_environment_usd(root=base)
    texture = _join(base, ROOM_ENV_TEXTURE_REL)
    return room is not None and texture.is_file()


def ceiling_lamp_usd(*, root: Path | None = None) -> Path | None:
    base = root if root is not None else resolve_i4h_root()
    if base is None:
        return None
    path = _join(base, CEILING_LAMP_REL)
    return path if path.is_file() else None


def material_library_dir(*, root: Path | None = None) -> Path | None:
    base = root if root is not None else resolve_i4h_root()
    if base is None:
        return None
    path = _join(base, MATERIAL_LIBRARY_REL)
    return path if path.is_dir() else None


def register_mdl_search_paths(settings, *paths: Path | str) -> None:
    """Append i4h material-library folders to Isaac's MDL search path setting."""
    key = "/renderer/mdl/searchPaths"
    to_add = [str(Path(p).resolve()) for p in paths if p]

    def _read_paths() -> list[str]:
        raw = settings.get(key)
        if isinstance(raw, str):
            return [part.strip() for part in raw.split(";") if part.strip()]
        if isinstance(raw, (list, tuple)):
            return [str(part) for part in raw if str(part).strip()]
        return []

    merged = _read_paths()
    for add in to_add:
        if add.lower() not in [part.lower() for part in merged]:
            merged.insert(0, add)
    if not merged:
        return

    raw = settings.get(key)
    if isinstance(raw, str) or raw is None:
        settings.set_string(key, ";".join(merged))
    else:
        settings.set(key, merged)


def inventory(root: Path | None = None) -> dict[str, str | None]:
    base = root if root is not None else resolve_i4h_root()
    out: dict[str, str | None] = {
        "root": str(base) if base else None,
        "room_env": None,
        "room_env_ready": None,
        "material_library": None,
        "ceiling_lamp": None,
    }
    if base is None:
        return out
    re = room_environment_usd(root=base)
    ml = material_library_dir(root=base)
    cl = ceiling_lamp_usd(root=base)
    out["room_env"] = str(re) if re else None
    out["room_env_ready"] = str(room_environment_ready(root=base))
    out["material_library"] = str(ml) if ml else None
    out["ceiling_lamp"] = str(cl) if cl else None
    for et in ENTITY_USD_REL:
        p = usd_for_entity_type(et, root=base)
        out[et] = str(p) if p else None
    return out
