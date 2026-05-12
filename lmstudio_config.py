# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
# ForestOptiLM is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ForestOptiLM is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public
# License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ForestOptiLM. If not, see <https://www.gnu.org/licenses/>.
"""
Локальная конфигурация подключения к LM Studio (OpenAI-совместимый API).

Приоритет загрузки:
1. Путь из переменной окружения NOCTURNE_LMSTUDIO_CONFIG
2. Файл .local/lmstudio.json рядом с приложением

Не коммитьте реальные ключи. Используйте config/lmstudio.example.json как шаблон.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Final

logger = logging.getLogger("nocturne")

# Значения по умолчанию (если файл отсутствует или поля пустые)
DEFAULT_BASE_URL: Final = "http://localhost:1234/v1"
DEFAULT_API_KEY: Final = "forest"
DEFAULT_TIMEOUT_SECONDS: Final = 600.0
RUNTIME_UI_FILE: Final = ".local/ui_runtime.json"

_cached: dict[str, object] | None = None


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    env_p = os.getenv("NOCTURNE_LMSTUDIO_CONFIG", "").strip()
    if env_p:
        paths.append(Path(env_p).expanduser())
    root = Path(__file__).resolve().parent
    paths.append(root / ".local" / "lmstudio.json")
    return paths


def _runtime_ui_path() -> Path:
    return Path(__file__).resolve().parent / RUNTIME_UI_FILE


def load_lmstudio_config_file() -> dict[str, object]:
    """Прочитать первый доступный JSON-конфиг."""
    for path in _candidate_paths():
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                logger.info("LM Studio config file: %s", path)
                return data
        except Exception as exc:
            logger.warning("LM Studio config unreadable %s: %s", path, exc)
    return {}


def _normalize_base_url(bu: str) -> str:
    """Убедиться, что base URL заканчивается на /v1 (OpenAI-совместимый API)."""
    base_url = bu.strip().rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    if "/v1" not in base_url:
        base_url = base_url + "/v1"
    return base_url


def _to_float_timeout(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if f > 0 else None
    if isinstance(value, str):
        s = value.strip().replace(",", ".")
        if not s:
            return None
        try:
            f = float(s)
            return f if f > 0 else None
        except ValueError:
            return None
    return None


def _to_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "on", "y"}:
            return True
        if s in {"0", "false", "no", "off", "n"}:
            return False
    return default


def _to_int_in_range(value: object, default: int, lo: int, hi: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        iv = int(value)
        return max(lo, min(hi, iv))
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            iv = int(s)
            return max(lo, min(hi, iv))
    return default


def _load_all() -> dict[str, object]:
    global _cached
    if _cached is not None:
        return _cached

    base_url = DEFAULT_BASE_URL
    api_key = DEFAULT_API_KEY
    source = "default"
    timeout_s = DEFAULT_TIMEOUT_SECONDS
    default_model: str | None = None
    config_path: str | None = None

    data = load_lmstudio_config_file()
    if data:
        for path in _candidate_paths():
            if path.is_file():
                config_path = str(path)
                break
        bu = data.get("base_url")
        ak = data.get("api_key")
        if isinstance(bu, str) and bu.strip():
            base_url = _normalize_base_url(bu)
        if isinstance(ak, str) and ak.strip():
            api_key = ak.strip()
        to = _to_float_timeout(data.get("timeout"))
        if to is not None:
            timeout_s = to
        dm = data.get("default_model")
        if isinstance(dm, str) and dm.strip():
            default_model = dm.strip()
        source = f"file:{config_path}" if config_path else "file"

    _cached = {
        "base_url": base_url,
        "api_key": api_key,
        "source": source,
        "timeout_sec": timeout_s,
        "default_model": default_model,
    }
    return _cached


def get_connection_defaults() -> tuple[str, str, str]:
    """
    Возвращает (base_url, api_key, source_tag).
    source_tag: default | file:<path>
    """
    d = _load_all()
    return (str(d["base_url"]), str(d["api_key"]), str(d["source"]))


def get_timeout_seconds() -> float:
    """Таймаут HTTP read для LLM-запросов (секунды)."""
    d = _load_all()
    return float(d["timeout_sec"])


def get_default_model_optional() -> str | None:
    """Опциональная модель по умолчанию из конфига."""
    d = _load_all()
    dm = d.get("default_model")
    return str(dm) if dm else None


def load_ui_runtime_state() -> dict[str, object]:
    """
    Локальные runtime-настройки GUI.
    Хранятся в .local/ui_runtime.json рядом с приложением.
    """
    defaults: dict[str, object] = {
        "selected_model": "",
        "selected_vision_model": "",
        "selected_composer_model": "",
        "selected_embedding_model": "",
        "composer_enabled": False,
        "workers": 3,
        "api_mode": "native",  # native|openai
        "low_vram_mode": True,
        "dual_instance_mode": True,
        "base_url": "",
        "max_reduce_input_tokens": 24000,
        "max_chunk_tokens": 6000,
        "rag_index_dir": ".nocturne_index",
        "rag_top_k": 8,
    }
    path = _runtime_ui_path()
    if not path.is_file():
        return defaults
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return defaults
        out = dict(defaults)
        out["selected_model"] = str(data.get("selected_model") or "").strip()
        out["selected_vision_model"] = str(data.get("selected_vision_model") or "").strip()
        out["selected_composer_model"] = str(data.get("selected_composer_model") or "").strip()
        out["selected_embedding_model"] = str(data.get("selected_embedding_model") or "").strip()
        out["composer_enabled"] = _to_bool(data.get("composer_enabled"), False)
        out["workers"] = _to_int_in_range(data.get("workers"), 3, 1, 4)
        api_mode = str(data.get("api_mode") or "native").strip().lower()
        out["api_mode"] = "openai" if api_mode == "openai" else "native"
        out["low_vram_mode"] = _to_bool(data.get("low_vram_mode"), True)
        out["dual_instance_mode"] = _to_bool(data.get("dual_instance_mode"), True)
        out["base_url"] = str(data.get("base_url") or "").strip()
        out["max_reduce_input_tokens"] = _to_int_in_range(data.get("max_reduce_input_tokens"), 24000, 1000, 200000)
        out["max_chunk_tokens"] = _to_int_in_range(data.get("max_chunk_tokens"), 6000, 500, 50000)
        out["rag_index_dir"] = str(data.get("rag_index_dir") or ".nocturne_index").strip() or ".nocturne_index"
        out["rag_top_k"] = _to_int_in_range(data.get("rag_top_k"), 8, 1, 100)
        return out
    except Exception as exc:
        logger.warning("Runtime UI state unreadable %s: %s", path, exc)
        return defaults


def save_ui_runtime_state(state: dict[str, object]) -> None:
    """Сохранить локальные runtime-настройки GUI в .local/ui_runtime.json."""
    path = _runtime_ui_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "selected_model": str(state.get("selected_model") or "").strip(),
        "selected_vision_model": str(state.get("selected_vision_model") or "").strip(),
        "selected_composer_model": str(state.get("selected_composer_model") or "").strip(),
        "selected_embedding_model": str(state.get("selected_embedding_model") or "").strip(),
        "composer_enabled": bool(state.get("composer_enabled", False)),
        "workers": _to_int_in_range(state.get("workers"), 3, 1, 4),
        "api_mode": "openai" if str(state.get("api_mode") or "").strip().lower() == "openai" else "native",
        "low_vram_mode": bool(state.get("low_vram_mode", True)),
        "dual_instance_mode": bool(state.get("dual_instance_mode", True)),
        "base_url": str(state.get("base_url") or "").strip(),
        "max_reduce_input_tokens": _to_int_in_range(state.get("max_reduce_input_tokens"), 24000, 1000, 200000),
        "max_chunk_tokens": _to_int_in_range(state.get("max_chunk_tokens"), 6000, 500, 50000),
        "rag_index_dir": str(state.get("rag_index_dir") or ".nocturne_index").strip() or ".nocturne_index",
        "rag_top_k": _to_int_in_range(state.get("rag_top_k"), 8, 1, 100),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mask_secret(value: str | None, head: int = 4, tail: int = 2) -> str:
    """Короткая маска для логов (не для криптографии)."""
    if not value:
        return "(empty)"
    s = str(value)
    if len(s) <= head + tail + 3:
        return "***"
    return s[:head] + "…" + s[-tail:]


_BEARER_RE = re.compile(r"(?i)(Bearer\s+)([A-Za-z0-9._\-+/=]{8,})")
_API_KEY_PARAM_RE = re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)(\S+)")
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b")


def sanitize_for_log(text: str) -> str:
    """Убрать/замаскировать секреты в произвольной строке для UI/логов."""
    if not text:
        return text
    out = str(text)
    out = _BEARER_RE.sub(r"\1***", out)
    out = _API_KEY_PARAM_RE.sub(r"\1***", out)
    out = _SK_RE.sub("sk-***", out)
    return out


def invalidate_cache() -> None:
    """Для тестов: сбросить кэш после смены файла."""
    global _cached
    _cached = None
