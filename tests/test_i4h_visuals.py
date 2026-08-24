from __future__ import annotations

from pathlib import Path

from medphygraph.i4h_visuals import ROOM_ENV_TEXTURE_REL, room_environment_ready


def test_room_environment_ready_false_when_texture_missing(tmp_path: Path) -> None:
    root = tmp_path / "i4h"
    room_dir = root / "Props/Organs/models/operating_room/room"
    room_dir.mkdir(parents=True)
    (room_dir / "Over_GRP_Room_Additions_merged.usd").write_text("#usda 1.0", encoding="utf-8")
    assert room_environment_ready(root=root) is False


def test_room_environment_ready_true_when_texture_present(tmp_path: Path) -> None:
    root = tmp_path / "i4h"
    room_dir = root / "Props/Organs/models/operating_room/room"
    room_dir.mkdir(parents=True)
    (room_dir / "Over_GRP_Room_Additions_merged.usd").write_text("#usda 1.0", encoding="utf-8")
    tex = root / ROOM_ENV_TEXTURE_REL.replace("/", "\\")
    tex.parent.mkdir(parents=True, exist_ok=True)
    tex.write_bytes(b"jpg")
    assert room_environment_ready(root=root) is True


class _FakeSettings:
    def __init__(self) -> None:
        self._values: dict[str, object] = {"/renderer/mdl/searchPaths": ["C:/existing"]}

    def get(self, key: str):
        return self._values.get(key)

    def set(self, key: str, value) -> None:
        self._values[key] = value

    def set_string(self, key: str, value: str) -> None:
        self._values[key] = value


def test_register_mdl_search_paths_preserves_array_setting() -> None:
    from medphygraph.i4h_visuals import register_mdl_search_paths

    settings = _FakeSettings()
    register_mdl_search_paths(settings, "D:/libs/Material Library", "D:/libs")
    paths = settings.get("/renderer/mdl/searchPaths")
    assert isinstance(paths, list)
    lowered = [str(p).lower() for p in paths]
    assert any(p.endswith("material library") for p in lowered)
    assert any(p.endswith("libs") for p in lowered)
