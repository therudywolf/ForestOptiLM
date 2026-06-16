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
Пресеты подключения к локальным LLM-серверам.

Один клик заполняет Base URL и режим API (native/openai) под конкретный
бэкенд, чтобы не вспоминать порты и пути:

- **LM Studio (REST v1)** — нативный ``/api/v1`` (reasoning:off, load/unload,
  контекст из метаданных).
- **LM Studio (OpenAI)** — тот же сервер, но OpenAI-совместимый ``/v1``.
- **Ollama** — OpenAI-совместимый ``/v1`` на ``:11434`` (chat + embeddings).
- **OpenAI-совместимый** — vLLM / llama.cpp server / LocalAI и т.п.
- **Вручную** — ничего не подставляем.

Слой чисто декларативный (никаких сетевых вызовов), легко тестируется и
переиспользуется и в GUI, и в CLI.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    key: str
    label: str
    base_url: str          # пустая строка → не подставлять (пользователь вводит сам)
    api_mode: str          # "native" | "openai"
    needs_api_key: bool
    hint: str

    @property
    def autofills_url(self) -> bool:
        return bool(self.base_url)


# Порядок = порядок в выпадающем списке. LM Studio первым (дефолт проекта).
PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset(
        key="lmstudio",
        label="LM Studio (REST v1)",
        base_url="http://127.0.0.1:1234",
        api_mode="native",
        needs_api_key=False,
        hint="LM Studio 0.3+. Нативный REST /api/v1: reasoning:off для thinking-моделей, "
             "авто-load/unload, контекст из метаданных. Запустите Local Server в LM Studio.",
    ),
    ProviderPreset(
        key="lmstudio_openai",
        label="LM Studio (OpenAI API)",
        base_url="http://127.0.0.1:1234",
        api_mode="openai",
        needs_api_key=False,
        hint="Тот же LM Studio, но через OpenAI-совместимый /v1. Берите, если нативный "
             "режим капризничает на вашей версии.",
    ),
    ProviderPreset(
        key="ollama",
        label="Ollama",
        base_url="http://127.0.0.1:11434",
        api_mode="openai",
        needs_api_key=False,
        hint="Ollama через OpenAI-совместимый /v1 на :11434. Модели: `ollama pull qwen2.5` "
             "и т.п.; эмбеддинги — `ollama pull nomic-embed-text`. API Key не нужен.",
    ),
    ProviderPreset(
        key="openai_compatible",
        label="OpenAI-совместимый",
        base_url="",
        api_mode="openai",
        needs_api_key=True,
        hint="Любой сервер с /v1/chat/completions: vLLM, llama.cpp server, LocalAI, "
             "text-generation-webui. Укажите Base URL и (при необходимости) API Key.",
    ),
    ProviderPreset(
        key="custom",
        label="Вручную",
        base_url="",
        api_mode="native",
        needs_api_key=False,
        hint="Ручная настройка — Base URL и режим API задаёте сами.",
    ),
)

_BY_KEY = {p.key: p for p in PRESETS}
_BY_LABEL = {p.label: p for p in PRESETS}

DEFAULT_PRESET_KEY = "lmstudio"


def all_presets() -> tuple[ProviderPreset, ...]:
    return PRESETS


def preset_labels() -> list[str]:
    return [p.label for p in PRESETS]


def get_preset(key: str) -> ProviderPreset | None:
    return _BY_KEY.get(key)


def preset_by_label(label: str) -> ProviderPreset | None:
    return _BY_LABEL.get(label)


def _parsed(base_url: str):
    try:
        return urlparse(base_url if "://" in base_url else "http://" + base_url)
    except Exception:
        return None


def _is_local(host: str | None) -> bool:
    return (host or "").lower() in {"", "localhost", "127.0.0.1", "::1", "0.0.0.0"}


def detect_preset(base_url: str, api_mode: str) -> str:
    """Угадать пресет по текущим Base URL и режиму API (для преселекта в GUI).

    Возвращает ключ пресета; «вручную» если ничего характерного не распознано.
    Порт-эвристика применяется только для локального хоста — иначе vLLM/llama.cpp
    на :1234 не записываем в LM Studio.
    """
    mode = (api_mode or "").strip().lower()
    mode = "openai" if mode == "openai" else "native"
    parsed = _parsed(base_url or "")
    port = parsed.port if parsed else None
    local = _is_local(parsed.hostname if parsed else None)

    if port == 11434 and local:
        return "ollama"
    if port == 1234 and local:
        return "lmstudio_openai" if mode == "openai" else "lmstudio"
    if not (base_url or "").strip():
        # Пустой URL + дефолтный нативный режим → дефолтный LM Studio пресет.
        return DEFAULT_PRESET_KEY if mode == "native" else "openai_compatible"
    if mode == "openai":
        return "openai_compatible"
    return "custom"
