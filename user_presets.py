# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
# ForestOptiLM is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
Пользовательские пресеты подключения: сохранить текущий сервер+ключ+модели под
именем и переключаться между ними (в отличие от встроенных шаблонов провайдеров
в connection_presets.py — те лишь подставляют Base URL/режим).

Хранятся в `.local/connection_presets.json` рядом с приложением (как ui_runtime),
gitignored — там реальные ключи, в гит НЕ коммитим. Файл-путь инъектируется в
функции, чтобы тесты не трогали реальный.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("nocturne")

PRESETS_FILE = ".local/connection_presets.json"
_SCHEMA_VERSION = 1
# Поля, которые входят в пресет (настройки «нейронки и API»).
_FIELDS = (
    "base_url", "api_key", "api_mode",
    "llm_model", "embedding_model", "vision_model",
)


@dataclass(slots=True)
class ConnectionPreset:
    name: str
    base_url: str = ""
    api_key: str = ""
    api_mode: str = "native"
    llm_model: str = ""
    embedding_model: str = ""
    vision_model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ConnectionPreset":
        name = str(d.get("name") or "").strip()
        kw = {k: str(d.get(k) or "") for k in _FIELDS}
        kw["api_mode"] = kw.get("api_mode") or "native"
        return cls(name=name, **kw)


def presets_path() -> Path:
    """`.local/connection_presets.json` в едином каталоге конфига рядом с exe
    (не внутри `_internal`) — как lmstudio.json/ui_runtime.json."""
    try:
        from lmstudio_config import app_config_dir
        return app_config_dir() / "connection_presets.json"
    except Exception:
        return Path(__file__).resolve().parent / PRESETS_FILE


def _legacy_presets_path() -> Path:
    """Старое расположение (относительно модуля → _internal/.local в exe)."""
    return Path(__file__).resolve().parent / PRESETS_FILE


def load_presets(path: Path | None = None) -> list[ConnectionPreset]:
    """Прочитать пресеты. Битый/отсутствующий файл → пустой список (не падаем)."""
    p = path or presets_path()
    # миграция: если в новом каталоге пусто, читаем из старого _internal/.local
    if path is None and not p.is_file():
        legacy = _legacy_presets_path()
        if legacy.is_file() and legacy != p:
            p = legacy
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("user_presets: файл %s не читается (%s) — игнорирую", p, exc)
        return []
    items = raw.get("presets") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    out: list[ConnectionPreset] = []
    seen: set[str] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        pr = ConnectionPreset.from_dict(it)
        key = pr.name.lower()
        if not pr.name or key in seen:
            continue
        seen.add(key)
        out.append(pr)
    return out


def save_presets(presets: list[ConnectionPreset], path: Path | None = None) -> None:
    """Атомарная запись (tmp+replace): сбой на середине не бьёт существующий файл."""
    p = path or presets_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": _SCHEMA_VERSION,
                   "presets": [pr.to_dict() for pr in presets]}
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception as exc:  # noqa: BLE001
        logger.warning("user_presets: не удалось сохранить %s — %s", p, exc)


def upsert_preset(preset: ConnectionPreset, path: Path | None = None) -> list[ConnectionPreset]:
    """Добавить/перезаписать пресет по имени (регистронезависимо). Возвращает список."""
    if not preset.name.strip():
        raise ValueError("имя пресета не должно быть пустым")
    presets = load_presets(path)
    key = preset.name.strip().lower()
    presets = [p for p in presets if p.name.lower() != key]
    presets.append(preset)
    presets.sort(key=lambda p: p.name.lower())
    save_presets(presets, path)
    return presets


def delete_preset(name: str, path: Path | None = None) -> list[ConnectionPreset]:
    key = (name or "").strip().lower()
    presets = [p for p in load_presets(path) if p.name.lower() != key]
    save_presets(presets, path)
    return presets


def get_preset(name: str, path: Path | None = None) -> ConnectionPreset | None:
    key = (name or "").strip().lower()
    for p in load_presets(path):
        if p.name.lower() == key:
            return p
    return None
