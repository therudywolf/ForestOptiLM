# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
"""Описание изображений vision-моделью на этапе индексации.

Картинки (схемы, диаграммы, скриншоты, фото таблиц) иначе попадают в индекс
только по имени файла и не находятся поиском по содержимому. Здесь мы один раз
прогоняем картинку через vision-модель и кладём её текстовое описание в чанк —
тогда «что на этой схеме / в этой таблице» становится отвечаемым.

Функции синхронные (индексация идёт в ThreadPoolExecutor, не в asyncio).
"""
from __future__ import annotations

import base64
import hashlib
import logging
import mimetypes
from collections.abc import Callable
from pathlib import Path

import httpx

from lm_client import chat_endpoint, extract_chat_response_content
from lmstudio_config import sanitize_for_log

logger = logging.getLogger("nocturne")

_DESCRIBE_PROMPT = (
    "Подробно и по делу опиши по-русски, что изображено на картинке: весь видимый "
    "текст и надписи, таблицы (с их содержимым), схемы и диаграммы (узлы, связи, "
    "подписи), числа, имена систем/сервисов/серверов/виртуальных машин. Передай всю "
    "фактическую информацию, которую видно. Не выдумывай того, чего на картинке нет."
)


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    raw = path.read_bytes()
    return f"data:{mime};base64," + base64.standard_b64encode(raw).decode("ascii")


def describe_image(
    path: Path,
    model: str,
    base_url: str,
    api_key: str,
    api_mode: str = "native",
    timeout: float = 120.0,
) -> str:
    """Вернуть текстовое описание картинки (или '' при неудаче). Native + OpenAI-fallback."""
    data_url = _data_url(Path(path))
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    native_payload = {
        "model": model,
        "input": [
            {"type": "text", "content": _DESCRIBE_PROMPT},
            {"type": "image", "data_url": data_url},
        ],
        "max_output_tokens": 700, "temperature": 0, "store": False,
    }
    openai_payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": _DESCRIBE_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        "max_tokens": 700, "temperature": 0,
    }
    order = [True, False] if api_mode.strip().lower() != "openai" else [False, True]
    with httpx.Client(timeout=timeout) as client:
        for use_native in order:
            url = chat_endpoint(base_url, native=use_native)
            payload = native_payload if use_native else openai_payload
            try:
                r = client.post(url, json=payload, headers=headers)
                if r.status_code >= 400:
                    continue  # пробуем другой транспорт
                content, _reasoning = extract_chat_response_content(r.json())
                content = (content or "").strip()
                if content:
                    return content
            except Exception as exc:  # noqa: BLE001
                logger.info("vision describe failed: %s", sanitize_for_log(str(exc)))
    return ""


def make_image_describer(
    model: str,
    base_url: str,
    api_key: str,
    api_mode: str = "native",
    cache: dict[str, str] | None = None,
) -> Callable[[Path], str] | None:
    """callable(image_path)->описание, с кешем по хешу содержимого. None если vision-модель не задана."""
    model = (model or "").strip()
    if not model or model.startswith("("):
        return None
    _cache: dict[str, str] = cache if cache is not None else {}

    def describer(path: Path) -> str:
        p = Path(path)
        try:
            key = hashlib.sha256(p.read_bytes()).hexdigest()
        except Exception:
            key = str(p)
        if key in _cache:
            return _cache[key]
        desc = describe_image(p, model, base_url, api_key, api_mode)
        _cache[key] = desc
        return desc

    return describer
