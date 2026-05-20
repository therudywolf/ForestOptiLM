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
Nocturne Data Forge — вызовы LLM, Map-Reduce, batching, backoff, адаптивный семафор, кэш, иерархический Reduce.
"""
from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import mimetypes
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Union

import httpx
import pandas as pd

from cache import build_job_id, get_cached_response, set_cached_response
from lmstudio_config import (
    get_connection_defaults,
    get_timeout_seconds,
    lmstudio_root_url,
    normalize_lmstudio_base_url,
    sanitize_for_log,
)

logger = logging.getLogger("nocturne")

API_BASE, API_KEY, _LM_CONFIG_SOURCE = get_connection_defaults()

DEFAULT_TIMEOUT = get_timeout_seconds()

SYSTEM_PROMPT = (
    "YOU MUST FOLLOW THESE RULES WITHOUT EXCEPTION:\n"
    "1. DO NOT write ANY thinking process, reasoning steps, or internal monologue.\n"
    "2. DO NOT write phrases like 'The user is asking', 'Let me analyze', 'Step 1', 'First I need to'.\n"
    "3. Start your response with <results> immediately.\n"
    "4. Your ENTIRE response must be: <results>YOUR ANSWER HERE</results>\n"
    "5. Answer is based ONLY on the provided text.\n"
    "6. No text outside the <results> tags."
)

SYSTEM_PROMPT_TABLE = (
    "YOU MUST FOLLOW THESE RULES WITHOUT EXCEPTION:\n"
    "1. DO NOT write ANY thinking process or reasoning.\n"
    "2. Start your response with <results> immediately.\n"
    "3. Your ENTIRE response must be: <results>JSON_ARRAY_HERE</results>\n"
    "4. Reply with a JSON array of objects, one per row.\n"
    "5. No text outside the <results> tags."
)

# MAP: структурированный JSON для последующего REDUCE без потери источников
SYSTEM_PROMPT_MAP = (
    "YOU MUST FOLLOW THESE RULES WITHOUT EXCEPTION:\n"
    "1. DO NOT write thinking, reasoning, or text outside <results>.\n"
    "2. Your ENTIRE response must be wrapped in <results> and </results> tags.\n"
    "3. Inside <results> write ONE valid JSON object (no markdown fences).\n"
    "4. Base your analysis ONLY on the fragment and metadata in the user message.\n"
    "5. If the fragment is not relevant to the user query, set no_relevant_data to true "
    "and use empty arrays for findings/evidence/recommendations.\n"
    "6. Every finding in findings[] MUST include at least one entry in evidence_refs:\n"
    "   - file: MUST equal the [FILE_PATH:...] tag from the chunk header EXACTLY.\n"
    "     Example: if header is [FILE_PATH: reports/sonar/scan.json], "
    "then file = \"reports/sonar/scan.json\"\n"
    "   - DO NOT use IDs, keys, or values from the file content as the \"file\" field.\n"
    "   - chunk: CHUNK_INDEX value from the chunk header (as string)\n"
    "   - quote: a short verbatim substring from the fragment (max 80 chars)\n"
    "7. The outer JSON 'file' field must also equal the FILE_PATH from the chunk header.\n"
    "8. Do NOT invent vulnerabilities or issues not supported by the fragment text.\n"
    "9. If the chunk header contains [FILE_TITLE:], use that title to identify the component/service "
    "in finding explanations (e.g. 'В сервисе <FILE_TITLE> обнаружена...').\n"
    "10. If the chunk header contains [FILE_LABELS:], include those labels as context in query_alignment "
    "to identify which subsystem/module this fragment belongs to.\n"
    "JSON schema:\n"
    '{ "chunk_index": number, "file": "FILE_PATH string", "query_alignment": string, '
    '"no_relevant_data": boolean, '
    '"findings": [ { "severity": "critical"|"high"|"medium"|"low"|"info", '
    '"type": string, "explanation": string, '
    '"evidence_refs": [ { "file": "FILE_PATH string", "chunk": "CHUNK_INDEX string", "quote": string } ] } ], '
    '"recommendations": [ string ] }\n'
)

# SCOUT: быстрый проход релевантности перед тяжёлым MAP (большие корпуса)
SYSTEM_PROMPT_SCOUT = (
    "YOU MUST FOLLOW THESE RULES WITHOUT EXCEPTION:\n"
    "1. DO NOT write thinking or text outside <results>.\n"
    "2. Your ENTIRE response must be wrapped in <results> and </results>.\n"
    "3. Inside <results> write ONE valid JSON object (no markdown fences).\n"
    "4. Judge ONLY whether the fragment may help answer the user query.\n"
    "JSON schema:\n"
    '{ "relevance_score": number between 0 and 1, "relevant": boolean, '
    '"topics": [ string ], "one_line_summary": string, "no_relevant_data": boolean }\n'
    "5. If clearly irrelevant, set no_relevant_data=true, relevant=false, relevance_score<=0.2.\n"
)

# META-PLANNER: генерация оптимального промта перед MAP-фазой (composer model)
SYSTEM_PROMPT_META_PLANNER = (
    "YOU MUST FOLLOW THESE RULES WITHOUT EXCEPTION:\n"
    "1. DO NOT write thinking, reasoning, or text outside <results>.\n"
    "2. Wrap your entire response in <results> and </results> tags.\n"
    "3. Inside <results> write a SINGLE precise analysis directive (plain text, no JSON, no markdown headers).\n"
    "4. The directive will be used as the user query for analysing EACH individual document fragment.\n"
    "5. The directive MUST:\n"
    "   a) Name the specific data entities, fields, and patterns to extract from THIS type of data.\n"
    "   b) List what constitutes Critical / High / Medium severity for THIS data type.\n"
    "   c) State what identifying information (service names, file paths, issue keys, IDs) must be preserved.\n"
    "   d) Instruct the model to use FILE_TITLE and FILE_LABELS from the chunk header to identify the component.\n"
    "   e) Be written in the SAME language as the user goal.\n"
    "6. Keep the directive under 400 words. No preamble, no meta-commentary.\n"
)

# REDUCE: итоговый отчёт с обязательными разделами и evidence
SYSTEM_PROMPT_REDUCE = (
    "YOU MUST FOLLOW THESE RULES WITHOUT EXCEPTION:\n"
    "1. DO NOT write thinking or reasoning text.\n"
    "2. Wrap your entire response in <results> and </results> tags.\n"
    "3. Inside <results> write a Markdown report with EXACTLY these sections in order:\n"
    "   ## Executive Summary\n"
    "   ## Comprehensive Findings\n"
    "   ## Evidence Matrix\n"
    "   ## Action Plan\n"
    "4. In Evidence Matrix list the full file path (from evidence_refs.file), chunk id, and a short quote for every important finding.\n"
    "5. Merge all input data; do not drop findings.\n"
    "6. Address the user's original query directly.\n"
    "7. Write in the same language as the user's query.\n"
    "8. Do NOT list Critical/High items in Comprehensive Findings unless they appear in Evidence Matrix.\n"
)

# Финальный синтез: объединяет готовые markdown-секции по файлам в единый отчёт
SYSTEM_PROMPT_SYNTHESIZE = (
    "YOU MUST FOLLOW THESE RULES WITHOUT EXCEPTION:\n"
    "1. DO NOT write thinking or reasoning text.\n"
    "2. Wrap your entire response in <results> and </results> tags.\n"
    "3. Inside <results> write a unified Markdown report with EXACTLY these sections in order:\n"
    "   ## Executive Summary\n"
    "   ## Comprehensive Findings\n"
    "   ## Evidence Matrix\n"
    "   ## Action Plan\n"
    "4. The input contains separate per-file/per-service report sections.\n"
    "5. Merge ALL findings and evidence from ALL input sections — do NOT drop any.\n"
    "6. In Comprehensive Findings: group by severity (Critical > High > Medium > Low > Info).\n"
    "7. In Evidence Matrix: include ALL evidence entries from every section as a Markdown table.\n"
    "8. In Action Plan: consolidate recommendations without duplication, ordered by priority.\n"
    "9. Write in the same language as the user's query.\n"
    "10. Do NOT invent findings not present in the input sections.\n"
)

# Минимальный PNG 1×1 для проверки vision
_MIN_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

# DEFAULT_TIMEOUT задан выше из lmstudio_config (локальный JSON или дефолт)
MAX_RETRIES = 3               # меньше повторов, но более высокий таймаут
SUCCESS_STREAK_TO_INCREASE = 10
MIN_WORKERS = 1
CONTEXT_FALLBACK = 8096
# Hard cap на объём данных, передаваемых в merge/reduce (в токенах).
# Даже если модель заявляет 128k контекст, больше этого порога не отправляем.
MAX_REDUCE_INPUT_TOKENS = int(os.getenv("NOCTURNE_MAX_REDUCE_INPUT_TOKENS", "24000"))


def _context_safety_margin(claimed_max_context: int) -> int:
    """
    Запас под расхождение tiktoken и llama.cpp.

    По умолчанию — 15% от заявленного контекста (512..8192), а не фиксированные 20480,
    чтобы модели 8k–32k не обрезались до пола 4096.
    Явное значение: NOCTURNE_CONTEXT_SAFETY_MARGIN (целое число токенов).
    """
    env_raw = os.getenv("NOCTURNE_CONTEXT_SAFETY_MARGIN", "").strip()
    if env_raw:
        try:
            return max(0, int(env_raw))
        except ValueError:
            pass
    adaptive = int(claimed_max_context * 0.15)
    return max(512, min(8192, adaptive))


def _server_safe_context_limit(claimed_max_context: int) -> int:
    """
    Консервативный бюджет для подсчёта размера промпта (MAP / REDUCE / синтез).

    В LM Studio (llama.cpp) токены часто не совпадают с tiktoken (parser.count_tokens).
    Кроме того, в API может быть max_context_length выше реального n_ctx загрузки модели.
    Без запаса возможна ошибка вида: n_keep > n_ctx.
    """
    if claimed_max_context <= 0:
        return CONTEXT_FALLBACK
    margin = _context_safety_margin(claimed_max_context)
    util_raw = os.getenv("NOCTURNE_CONTEXT_UTILIZATION", "").strip()
    util = float(util_raw) if util_raw else 0.82
    util = max(0.5, min(0.99, util))
    floor = 4096 if claimed_max_context >= 8192 else max(1024, claimed_max_context // 4)
    by_margin = max(floor, claimed_max_context - margin)
    by_util = max(floor, int(claimed_max_context * util))
    return min(by_margin, by_util)
ENABLE_CONTEXT_PROBE = os.getenv("NOCTURNE_ENABLE_CONTEXT_PROBE", "").strip() == "1"
MAX_CONTEXT_PROBES_PER_REFRESH = 2
USE_ADAPTIVE_SEMAPHORE = os.getenv("NOCTURNE_ADAPTIVE_SEMAPHORE", "").strip() == "1"
SERVER_5XX_CIRCUIT_BREAKER_THRESHOLD = 5
SERVER_5XX_CIRCUIT_BREAKER_PAUSE_SECONDS = 3.0
# По умолчанию работаем в low-VRAM режиме: не переключаем разные модели одновременно.
LOW_VRAM_SEQUENTIAL_MODE = os.getenv("NOCTURNE_LOW_VRAM_SEQUENTIAL_MODE", "1").strip() != "0"
# По умолчанию используем native LM Studio REST API (/api/v1/*).
USE_LMSTUDIO_NATIVE_API = os.getenv("NOCTURNE_LMSTUDIO_NATIVE_API", "1").strip() != "0"
DUAL_INSTANCE_5PLUS_MODE = os.getenv("NOCTURNE_DUAL_INSTANCE_5PLUS_MODE", "0").strip() != "0"
DUAL_MAP_RESOLVE = os.getenv("NOCTURNE_DUAL_MAP_RESOLVE", "0").strip() == "1"

_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "executive summary": ("executive summary", "исполнительное резюме", "краткое резюме"),
    "comprehensive findings": ("comprehensive findings", "ключевые находки", "выявленные проблемы"),
    "evidence matrix": ("evidence matrix", "матрица доказательств", "доказательная база"),
    "action plan": ("action plan", "план действий", "рекомендации"),
}


def set_runtime_modes(
    *,
    api_mode: str | None = None,
    low_vram_mode: bool | None = None,
    dual_instance_mode: bool | None = None,
) -> dict[str, object]:
    """Применить runtime-режимы без рестарта процесса."""
    global USE_LMSTUDIO_NATIVE_API, LOW_VRAM_SEQUENTIAL_MODE, DUAL_INSTANCE_5PLUS_MODE
    if api_mode is not None:
        USE_LMSTUDIO_NATIVE_API = str(api_mode).strip().lower() != "openai"
    if low_vram_mode is not None:
        LOW_VRAM_SEQUENTIAL_MODE = bool(low_vram_mode)
    if dual_instance_mode is not None:
        DUAL_INSTANCE_5PLUS_MODE = bool(dual_instance_mode)
    return {
        "api_mode": "native" if USE_LMSTUDIO_NATIVE_API else "openai",
        "low_vram_mode": LOW_VRAM_SEQUENTIAL_MODE,
        "dual_instance_mode": DUAL_INSTANCE_5PLUS_MODE,
    }


def set_runtime_limits(
    *,
    max_reduce_input_tokens: int | None = None,
    max_chunk_tokens: int | None = None,
) -> dict[str, int]:
    """Применить runtime-лимиты токенов (используются без перезапуска)."""
    global MAX_REDUCE_INPUT_TOKENS
    if max_reduce_input_tokens is not None:
        MAX_REDUCE_INPUT_TOKENS = max(1000, int(max_reduce_input_tokens))
        os.environ["NOCTURNE_MAX_REDUCE_INPUT_TOKENS"] = str(MAX_REDUCE_INPUT_TOKENS)
    if max_chunk_tokens is not None:
        os.environ["NOCTURNE_MAX_CHUNK_TOKENS"] = str(max(500, int(max_chunk_tokens)))
    return {
        "max_reduce_input_tokens": MAX_REDUCE_INPUT_TOKENS,
        "max_chunk_tokens": int(os.getenv("NOCTURNE_MAX_CHUNK_TOKENS", "6000")),
    }


def _lmstudio_root(base_url: str) -> str:
    return lmstudio_root_url(base_url)


def _openai_base(base_url: str) -> str:
    return normalize_lmstudio_base_url(base_url)


def _auth_headers(api_key: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _chat_endpoint(base_url: str, native: bool | None = None) -> str:
    from lm_client import chat_endpoint

    return chat_endpoint(base_url, native=USE_LMSTUDIO_NATIVE_API if native is None else native)


def _models_endpoint(base_url: str, native: bool | None = None) -> str:
    from lm_client import models_endpoint

    return models_endpoint(base_url, native=USE_LMSTUDIO_NATIVE_API if native is None else native)


def _native_input_from_messages(messages: list[dict[str, Any]]) -> tuple[str | None, str | list[dict[str, Any]]]:
    system_prompt: str | None = None
    input_items: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role") or "").strip().lower()
        content = msg.get("content")
        if role == "system":
            system_prompt = str(content or "")
            continue
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                typ = str(item.get("type") or "").strip().lower()
                if typ == "text":
                    input_items.append({"type": "text", "content": str(item.get("text") or "")})
                elif typ == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        data_url = str(image_url.get("url") or "")
                        if data_url:
                            input_items.append({"type": "image", "data_url": data_url})
        else:
            text = str(content or "")
            if text:
                # LM Studio native chat ожидает message без role-поля в input item.
                input_items.append({"type": "message", "content": text})
    if len(input_items) == 1 and input_items[0].get("type") in ("message", "text"):
        return system_prompt, str(input_items[0].get("content") or "")
    return system_prompt, input_items


def _extract_chat_response_content(data: dict[str, Any]) -> tuple[str, str]:
    from lm_client import extract_chat_response_content

    return extract_chat_response_content(data)


def _prefer_russian(query: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", query or ""))


def _extract_ids_from_models_payload(data: Any) -> list[str]:
    """Поддержать /api/v1/models (models[]) и /v1/models (data[])."""
    items: list[Any] = []
    if isinstance(data, dict):
        raw = data.get("data")
        if isinstance(raw, list):
            items = raw
        else:
            raw = data.get("models")
            if isinstance(raw, list):
                items = raw
    elif isinstance(data, list):
        items = data
    ids: list[str] = []
    for item in items:
        if isinstance(item, str):
            ids.append(item)
            continue
        if not isinstance(item, dict):
            continue
        mid = item.get("id") or item.get("key")
        if mid:
            ids.append(str(mid))
    return ids


class AdaptiveSemaphore:
    """Семафор с динамическим лимитом: при ошибках — throttle down, при серии успехов — восстановление."""

    def __init__(self, max_workers: int) -> None:
        self._max = max(1, max_workers)
        # Стартуем с запрошенного пользователем уровня параллелизма.
        self._limit = self._max
        self._in_use = 0
        self._success_streak = 0
        self._lock = asyncio.Lock()
        self._waiters: list[asyncio.Future[None]] = []

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                if self._in_use < self._limit:
                    self._in_use += 1
                    return
                fut: asyncio.Future[None] = asyncio.get_event_loop().create_future()
                self._waiters.append(fut)
            await fut

    async def release(self) -> None:
        to_wake: asyncio.Future[None] | None = None
        async with self._lock:
            self._in_use -= 1
            if self._waiters and self._in_use < self._limit:
                to_wake = self._waiters.pop(0)
        if to_wake is not None and not to_wake.done():
            to_wake.set_result(None)

    async def record_success(self) -> None:
        async with self._lock:
            self._success_streak += 1
            if self._success_streak >= SUCCESS_STREAK_TO_INCREASE:
                self._limit = min(self._max, self._limit + 1)
                self._success_streak = 0
                logger.debug("AdaptiveSemaphore: limit increased to %s", self._limit)

    async def record_failure(self) -> None:
        async with self._lock:
            self._limit = max(MIN_WORKERS, self._limit - 1)
            self._success_streak = 0
            logger.debug("AdaptiveSemaphore: limit decreased to %s", self._limit)


def fetch_models(base_url: str, api_key: str) -> list[str]:
    """Синхронно получить список ID моделей с API."""
    ids, _, _ = fetch_models_info(base_url, api_key)
    return ids


def categorize_models(base_url: str, api_key: str) -> dict[str, list[str]]:
    """Разделить модели на chat / vision / embedding / reasoning по каталогу LM Studio."""
    from reasoning_models import model_has_reasoning_capability, refresh_model_catalog_cache

    catalog = fetch_models_catalog(base_url, api_key)
    refresh_model_catalog_cache(catalog)
    out: dict[str, list[str]] = {"chat": [], "vision": [], "embedding": [], "reasoning": []}
    for m in catalog:
        key = str(m.get("key") or m.get("id") or "").strip()
        if not key:
            continue
        mtype = str(m.get("type") or "llm").strip().lower()
        caps = m.get("capabilities")
        if model_has_reasoning_capability(key, caps):
            out["reasoning"].append(key)
        if mtype == "embedding" or "embed" in key.lower():
            out["embedding"].append(key)
        elif isinstance(caps, dict) and caps.get("vision"):
            out["vision"].append(key)
            out["chat"].append(key)
        else:
            out["chat"].append(key)
    if not out["chat"] and catalog:
        out["chat"] = [str(m.get("key") or m.get("id") or "") for m in catalog if m.get("key") or m.get("id")]
    return out


def fetch_models_catalog(base_url: str, api_key: str) -> list[dict[str, Any]]:
    """Получить raw-каталог моделей (native /api/v1/models или fallback /v1/models)."""
    headers = _auth_headers(api_key)
    candidates = [_models_endpoint(base_url, native=True), _models_endpoint(base_url, native=False)]
    for url in candidates:
        try:
            with httpx.Client(timeout=15.0) as client:
                r = client.get(url, headers=headers)
                r.raise_for_status()
                data = r.json()
            if isinstance(data, dict):
                if isinstance(data.get("models"), list):
                    return [m for m in data["models"] if isinstance(m, dict)]
                if isinstance(data.get("data"), list):
                    return [m for m in data["data"] if isinstance(m, dict)]
        except Exception:
            continue
    return []


def summarize_model_tokens_by_category(base_url: str, api_key: str) -> dict[str, int]:
    """Суммарный context/token budget по категориям моделей (llm/embedding/vision/tool)."""
    catalog = fetch_models_catalog(base_url, api_key)
    out = {"llm": 0, "embedding": 0, "vision": 0, "tool": 0}
    for m in catalog:
        mtype = str(m.get("type") or "llm").strip().lower()
        ctx = _to_positive_int(m.get("max_context_length")) or _extract_context_length(m) or 0
        if mtype == "embedding":
            out["embedding"] += ctx
            continue
        caps = m.get("capabilities")
        is_vision = isinstance(caps, dict) and bool(caps.get("vision"))
        is_tool = isinstance(caps, dict) and bool(caps.get("trained_for_tool_use"))
        out["llm"] += ctx
        if is_vision:
            out["vision"] += ctx
        if is_tool:
            out["tool"] += ctx
    return out


def _extract_loaded_instance_ids(base_url: str, api_key: str, model_key: str) -> list[str]:
    catalog = fetch_models_catalog(base_url, api_key)
    for m in catalog:
        mid = str(m.get("key") or m.get("id") or "").strip()
        if mid != model_key:
            continue
        loaded = m.get("loaded_instances")
        if isinstance(loaded, list):
            ids: list[str] = []
            for inst in loaded:
                if not isinstance(inst, dict):
                    continue
                iid = str(
                    inst.get("id")
                    or inst.get("model_instance_id")
                    or inst.get("instance_id")
                    or ""
                ).strip()
                if iid:
                    ids.append(iid)
            return ids
    return []


def _loaded_instances_snapshot(base_url: str, api_key: str) -> dict[str, int]:
    catalog = fetch_models_catalog(base_url, api_key)
    out: dict[str, int] = {}
    for m in catalog:
        key = str(m.get("key") or m.get("id") or "").strip()
        if not key:
            continue
        loaded = m.get("loaded_instances")
        if isinstance(loaded, list):
            out[key] = len([x for x in loaded if isinstance(x, dict)])
    return out


def _try_unload_model(base_url: str, api_key: str, model_key: str) -> bool:
    """
    Best-effort unload всех loaded instances модели.
    Для LM Studio /api/v1/models/unload требуется payload с instance_id.
    """
    root = _lmstudio_root(base_url)
    headers = {"Content-Type": "application/json", **_auth_headers(api_key)}
    from lm_studio_api import V1_MODELS_UNLOAD, v1_url

    unload_url = v1_url(root, V1_MODELS_UNLOAD)
    instance_ids = _extract_loaded_instance_ids(base_url, api_key, model_key)
    if not instance_ids:
        return True
    ok = True
    with httpx.Client(timeout=20.0) as client:
        for iid in instance_ids:
            try:
                r = client.post(unload_url, json={"instance_id": iid}, headers=headers)
                if r.status_code >= 400:
                    ok = False
                    logger.warning(
                        "Native unload failed: model=%s instance_id=%s http=%s body=%s",
                        model_key,
                        iid,
                        r.status_code,
                        sanitize_for_log(r.text[:220]),
                    )
                else:
                    logger.info("Native unload OK: model=%s instance_id=%s", model_key, iid)
            except Exception as exc:
                ok = False
                logger.warning(
                    "Native unload exception: model=%s instance_id=%s err=%s",
                    model_key,
                    iid,
                    sanitize_for_log(str(exc)),
                )
    return ok


def _classify_http_400(text: str) -> str:
    low = (text or "").lower()
    if any(s in low for s in ("unsupported", "unknown field", "invalid field", "schema")):
        return "payload_mismatch"
    if any(s in low for s in ("context", "token", "max_output_tokens", "length", "too long", "exceed")):
        return "context_limit"
    if any(s in low for s in ("thinking", "reasoning", "tool", "image", "vision", "model")):
        return "unsupported_option"
    return "unknown"


def _to_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        iv = int(value)
        return iv if iv > 0 else None
    if isinstance(value, str):
        s = value.strip().replace(" ", "").replace(",", "")
        if not s:
            return None
        if s.isdigit():
            iv = int(s)
            return iv if iv > 0 else None
    return None


def _extract_loaded_context_length(m: dict[str, Any]) -> int | None:
    loaded = m.get("loaded_instances")
    if not isinstance(loaded, list):
        return None
    values: list[int] = []
    for inst in loaded:
        if not isinstance(inst, dict):
            continue
        ctx = _to_positive_int(inst.get("loaded_context_length"))
        if not ctx:
            cfg = inst.get("config")
            if isinstance(cfg, dict):
                ctx = _to_positive_int(cfg.get("context_length"))
        if ctx:
            values.append(ctx)
    return max(values) if values else None


def _extract_context_length(m: dict[str, Any]) -> int | None:
    """Извлечь context_length из разных вариантов структуры /models."""
    loaded_ctx = _extract_loaded_context_length(m)
    if loaded_ctx:
        return loaded_ctx

    direct_keys = (
        "loaded_context_length",
        "context_length",
        "max_context_length",
        "n_ctx",
        "num_ctx",
        "ctx_size",
        "context",
        "max_seq_len",
        "max_sequence_length",
    )
    for k in direct_keys:
        v = _to_positive_int(m.get(k))
        if v:
            return v

    nested_keys = (
        "model_info",
        "metadata",
        "limits",
        "capabilities",
        "architecture",
        "config",
        "parameters",
    )
    nested_ctx_keys = (
        "context_length",
        "max_context_length",
        "n_ctx",
        "num_ctx",
        "ctx_size",
        "max_seq_len",
        "max_sequence_length",
        "llm.context_length",
    )
    for nk in nested_keys:
        obj = m.get(nk)
        if not isinstance(obj, dict):
            continue
        for ck in nested_ctx_keys:
            if "." in ck:
                parts = ck.split(".", 1)
                inner = obj.get(parts[0])
                v = _to_positive_int(inner.get(parts[1]) if isinstance(inner, dict) else obj.get(ck))
            else:
                v = _to_positive_int(obj.get(ck))
            if v:
                return v
    return None


def _probe_context_length(base_url: str, api_key: str, model: str) -> int | None:
    """
    Эвристический probe: отправляем короткие запросы с разными размерами контекста.
    Берём максимальный размер, который сервер принимает без ошибки длины.
    """
    use_native = USE_LMSTUDIO_NATIVE_API
    url = _chat_endpoint(base_url, native=use_native)
    headers = {"Content-Type": "application/json", **_auth_headers(api_key)}

    # быстрый набор кандидатов: от безопасного к более крупному
    # начинаем с fallback-значения, далее пробуем более высокие уровни
    candidates = [CONTEXT_FALLBACK, 16384, 32768, 65536, 131072]
    ok_limit: int | None = None
    with httpx.Client(timeout=8.0) as client:
        for candidate in candidates:
            # Консервативный probe: ограничиваем нагрузку, чтобы не дестабилизировать LM Studio.
            approx_tokens = max(128, min(3000, candidate // 16))
            big_user = ("token " * approx_tokens).strip()
            if use_native:
                payload = {
                    "model": model,
                    "input": big_user,
                    "max_output_tokens": 1,
                    "temperature": 0,
                    "store": False,
                }
            else:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": big_user}],
                    "max_tokens": 1,
                    "temperature": 0,
                }
            try:
                r = client.post(url, json=payload, headers=headers)
                if r.status_code >= 400:
                    body = ""
                    try:
                        body = r.text.lower()
                    except Exception:
                        pass
                    # если ошибка явно про длину контекста — текущий candidate не проходит
                    if any(
                        x in body
                        for x in ("context", "token", "length", "too long", "exceed", "maximum")
                    ):
                        continue
                    # иные ошибки probe не должны ломать обновление моделей
                    continue
                ok_limit = candidate
            except Exception:
                continue
            time.sleep(0.05)
    return ok_limit


def fetch_models_info(
    base_url: str, api_key: str
) -> tuple[list[str], dict[str, int], dict[str, str]]:
    """
    Получить список моделей и их context_length.
    Возвращает:
    - list[model_id]
    - {model_id: context_length}
    - {model_id: source}, где source: metadata|probe|fallback
    LM Studio возвращает context_length в поле model_info.context_length
    или напрямую в поле context_length объекта модели.
    """
    root = _lmstudio_root(base_url)
    headers = _auth_headers(api_key)
    primary_url = _models_endpoint(base_url)
    data: dict[str, Any] | None = None
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(primary_url, headers=headers)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        logger.warning(
            "Primary models endpoint failed (%s): %s",
            primary_url,
            sanitize_for_log(str(exc)),
        )

    # LM Studio often exposes rich metadata at /api/v0/models.
    if not _extract_ids_from_models_payload(data):
        alt_url = root + "/api/v0/models"
        try:
            with httpx.Client(timeout=15.0) as client:
                r = client.get(alt_url, headers=headers)
                r.raise_for_status()
                data = r.json()
                logger.info("Using LM Studio metadata endpoint: %s", alt_url)
        except Exception as exc:
            logger.exception(
                "fetch_models_info failed on both endpoints: %s",
                sanitize_for_log(str(exc)),
            )
            raise

    ids: list[str] = []
    ctx_lengths: dict[str, int] = {}
    ctx_sources: dict[str, str] = {}

    items: list[Any] = []
    if isinstance(data, dict):
        raw_data = data.get("data")
        if isinstance(raw_data, list):
            items = raw_data
        else:
            raw_models = data.get("models")
            if isinstance(raw_models, list):
                items = raw_models

    for m in items:
        if isinstance(m, str):
            ids.append(m)
            continue
        if not isinstance(m, dict):
            continue
        mid = m.get("id") or m.get("key")
        if not mid:
            continue
        ids.append(str(mid))
        ctx = _extract_context_length(m)
        if not ctx:
            ctx = _to_positive_int(m.get("max_context_length"))
        if ctx:
            mid_s = str(mid)
            ctx_lengths[mid_s] = ctx
            ctx_sources[mid_s] = "metadata"

    # Если /v1/models вернул только базовые поля (id/object/owned_by),
    # дополнительно подтягиваем metadata из актуального /api/v1/models и legacy /api/v0/models.
    if ids and not ctx_lengths:
        for alt_url in (root + "/api/v1/models", root + "/api/v0/models"):
            try:
                with httpx.Client(timeout=15.0) as client:
                    alt_r = client.get(alt_url, headers=headers)
                    alt_r.raise_for_status()
                    alt_data = alt_r.json()
                by_id: dict[str, dict[str, Any]] = {}
                if isinstance(alt_data, dict):
                    raw_items = alt_data.get("models")
                    if not isinstance(raw_items, list):
                        raw_items = alt_data.get("data")
                    if isinstance(raw_items, list):
                        for item in raw_items:
                            if not isinstance(item, dict):
                                continue
                            mid = item.get("id") or item.get("key")
                            if mid:
                                by_id[str(mid)] = item
                for mid in ids:
                    m = by_id.get(mid)
                    if not m:
                        continue
                    ctx = _extract_context_length(m)
                    if ctx:
                        ctx_lengths[mid] = ctx
                        ctx_sources[mid] = "metadata"
                if ctx_lengths:
                    logger.info(
                        "Enriched model context from LM Studio metadata endpoint %s: %s models",
                        alt_url,
                        len(ctx_lengths),
                    )
                    break
            except Exception as exc:
                logger.warning(
                    "Cannot enrich context from %s: %s",
                    alt_url,
                    sanitize_for_log(str(exc)),
                )

    # Дополнительно идем в per-model endpoint для КАЖДОЙ модели:
    # это повышает шанс получить корректный context на runtime metadata.
    # Делаем параллельно, чтобы не тратить N последовательных RTT.
    if ids:
        def _fetch_one(mid: str) -> tuple[str, int | None]:
            try:
                detail_url = root + f"/api/v0/models/{mid}"
                with httpx.Client(timeout=10.0) as client:
                    rr = client.get(detail_url, headers=headers)
                    rr.raise_for_status()
                    meta = rr.json()
                if isinstance(meta, dict):
                    return mid, _extract_context_length(meta)
            except Exception:
                return mid, None
            return mid, None

        pool_workers = max(1, min(8, len(ids)))
        with ThreadPoolExecutor(max_workers=pool_workers) as pool:
            futures = {pool.submit(_fetch_one, mid): mid for mid in ids}
            for fut in as_completed(futures):
                mid, ctx = fut.result()
                if ctx:
                    ctx_lengths[mid] = ctx
                    ctx_sources[mid] = "metadata"

    # fallback: probe only models without metadata context
    without_ctx = [mid for mid in ids if mid not in ctx_lengths]
    probed_ok = 0
    probe_budget = MAX_CONTEXT_PROBES_PER_REFRESH if ENABLE_CONTEXT_PROBE else 0
    for mid in without_ctx:
        probed: int | None = None
        if probe_budget > 0:
            probed = _probe_context_length(base_url, api_key, mid)
            probe_budget -= 1
        if probed:
            ctx_lengths[mid] = probed
            ctx_sources[mid] = "probe"
            probed_ok += 1
        else:
            ctx_lengths[mid] = CONTEXT_FALLBACK
            ctx_sources[mid] = "fallback"

    logger.info(
        "Model context detection: total=%s metadata=%s probe=%s fallback=%s",
        len(ids),
        sum(1 for m in ids if ctx_sources.get(m) == "metadata"),
        probed_ok,
        sum(1 for m in ids if ctx_sources.get(m) == "fallback"),
    )
    if without_ctx:
        logger.warning(
            "Models without context in /models: %s",
            ", ".join(without_ctx[:10]) + ("..." if len(without_ctx) > 10 else ""),
        )
        if not ENABLE_CONTEXT_PROBE:
            logger.info(
                "Context probe is disabled; using fallback=%s. "
                "Set NOCTURNE_ENABLE_CONTEXT_PROBE=1 to enable limited probing.",
                CONTEXT_FALLBACK,
            )

    return ids, ctx_lengths, ctx_sources


def resolve_runtime_model_context(
    base_url: str,
    api_key: str,
    model: str,
    *,
    wait_for_loaded: bool = True,
    max_wait_seconds: float = 180.0,
    poll_interval_seconds: float = 1.0,
    trigger_load_request: bool = True,
) -> tuple[int | None, str, str]:
    """
    Получить runtime-контекст выбранной модели через LM Studio /api/v0/models/{id}.
    Возвращает (context, source, state):
      - source=runtime_loaded, если есть loaded_context_length у загруженной модели;
      - source=metadata_not_loaded, если модель не загружена, но metadata доступна;
      - source=unavailable, если данные получить не удалось.
    """
    root = _lmstudio_root(base_url)
    openai_base = _openai_base(base_url)
    headers = _auth_headers(api_key)
    native_models_url = root + "/api/v1/models"
    detail_url = root + f"/api/v0/models/{model}"
    deadline = time.monotonic() + max_wait_seconds
    load_triggered = False

    def _trigger_model_load() -> None:
        nonlocal load_triggered
        if load_triggered or not trigger_load_request:
            return
        try:
            with httpx.Client(timeout=20.0) as client:
                if USE_LMSTUDIO_NATIVE_API:
                    load_url = _lmstudio_root(base_url) + "/api/v1/models/load"
                    client.post(load_url, json={"model": model}, headers=headers)
                else:
                    chat_url = openai_base + "/chat/completions"
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1,
                        "temperature": 0,
                    }
                    client.post(chat_url, json=payload, headers=headers)
            load_triggered = True
        except Exception:
            # Даже если trigger не удался, продолжаем polling metadata.
            load_triggered = True

    last_ctx: int | None = None
    last_state = ""
    while True:
        handled_native_catalog = False
        try:
            with httpx.Client(timeout=10.0) as client:
                nr = client.get(native_models_url, headers=headers)
                nr.raise_for_status()
                native_data = nr.json()
            if isinstance(native_data, dict):
                raw_models = native_data.get("models")
                if isinstance(raw_models, list):
                    for item in raw_models:
                        if not isinstance(item, dict):
                            continue
                        mid = str(item.get("key") or item.get("id") or "").strip()
                        if mid != model:
                            continue
                        handled_native_catalog = True
                        loaded_ctx = _extract_loaded_context_length(item)
                        any_ctx = _extract_context_length(item)
                        loaded = bool(item.get("loaded_instances") or [])
                        last_state = "loaded" if loaded else "not-loaded"
                        if loaded_ctx:
                            return loaded_ctx, "runtime_loaded", last_state
                        if any_ctx:
                            last_ctx = any_ctx
                        if wait_for_loaded and not loaded:
                            _trigger_model_load()
                        break
        except Exception:
            handled_native_catalog = False

        if handled_native_catalog:
            if not wait_for_loaded or last_state == "loaded":
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(max(0.2, poll_interval_seconds))
            continue

        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.get(detail_url, headers=headers)
                r.raise_for_status()
                meta = r.json()
            if isinstance(meta, dict):
                state = str(meta.get("state") or "").strip().lower()
                last_state = state or last_state
                loaded_ctx = _to_positive_int(meta.get("loaded_context_length"))
                if loaded_ctx and state == "loaded":
                    return loaded_ctx, "runtime_loaded", state
                any_ctx = _extract_context_length(meta)
                if any_ctx:
                    last_ctx = any_ctx
                if wait_for_loaded and state != "loaded":
                    _trigger_model_load()
                if not wait_for_loaded:
                    break
                if state == "loaded":
                    # Загружена, но loaded_context_length нет — используем лучшее доступное значение.
                    break
        except Exception:
            if wait_for_loaded:
                _trigger_model_load()
            if not wait_for_loaded:
                break
        if time.monotonic() >= deadline:
            break
        time.sleep(max(0.2, poll_interval_seconds))

    if last_ctx:
        return last_ctx, "metadata_not_loaded", last_state or "unknown"
    return None, "unavailable", last_state or "unknown"


def check_lmstudio_connection(
    base_url: str,
    api_key: str,
    embedding_model: str | None = None,
    *,
    full_smoke: bool = False,
    chat_model: str | None = None,
) -> tuple[bool, str]:
    """Проверить LM Studio: список моделей; при full_smoke — chat + embedding."""
    if full_smoke:
        return run_lmstudio_smoke_test(
            base_url,
            api_key,
            chat_model=chat_model,
            embedding_model=embedding_model,
        )
    try:
        models = fetch_models(base_url, api_key)
        if not models:
            return (False, "LM Studio доступен, но список моделей пуст")
        return (True, f"OK — моделей: {len(models)}: {', '.join(models[:3])}{'…' if len(models) > 3 else ''}")
    except Exception as exc:
        logger.exception("LM Studio connection test failed: %s", sanitize_for_log(str(exc)))
        return (False, sanitize_for_log(str(exc)))


# Backward-compatible alias for GUI and external callers.
test_lmstudio_connection = check_lmstudio_connection


def run_lmstudio_smoke_test(
    base_url: str,
    api_key: str,
    *,
    chat_model: str | None = None,
    embedding_model: str | None = None,
) -> tuple[bool, str]:
    """Smoke: GET /models + короткий chat + embedding (если модель найдена)."""
    steps: list[str] = []
    try:
        models = fetch_models(base_url, api_key)
        if not models:
            return False, "Список моделей пуст"
        steps.append(f"models={len(models)}")
        chat = chat_model or next((m for m in models if "embed" not in m.lower()), models[0])
        embed = embedding_model or next((m for m in models if "embed" in m.lower()), "")
        headers = {"Content-Type": "application/json", **_auth_headers(api_key)}
        chat_url = _chat_endpoint(base_url, native=USE_LMSTUDIO_NATIVE_API)
        payload: dict[str, Any]
        if USE_LMSTUDIO_NATIVE_API:
            from reasoning_models import native_reasoning_payload, refresh_model_catalog_cache

            refresh_model_catalog_cache(fetch_models_catalog(base_url, api_key))
            payload = {
                "model": chat,
                "input": "Reply with exactly: OK",
                "max_output_tokens": 16,
                "temperature": 0,
                "store": False,
                "context_length": 2048,
            }
            payload.update(native_reasoning_payload(chat))
        else:
            payload = {
                "model": chat,
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                "max_tokens": 16,
                "temperature": 0,
            }
        with httpx.Client(timeout=60.0) as client:
            r = client.post(chat_url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            content, _ = _extract_chat_response_content(data)
            if not (content or "").strip():
                return False, "Chat вернул пустой ответ"
            steps.append(f"chat={chat}")
            if embed:
                emb_url = _openai_base(base_url) + "/embeddings"
                er = client.post(
                    emb_url,
                    json={"model": embed, "input": ["smoke test"]},
                    headers=headers,
                )
                er.raise_for_status()
                emb_data = er.json()
                if not emb_data.get("data"):
                    return False, "Embedding вернул пустой data"
                steps.append(f"embed={embed}")
        return True, "Smoke OK: " + ", ".join(steps)
    except Exception as exc:
        logger.exception("LM Studio smoke failed: %s", sanitize_for_log(str(exc)))
        return False, sanitize_for_log(str(exc))


async def call_llm(
    messages: list[dict[str, Any]],
    model: str,
    base_url: str,
    api_key: str,
    semaphore: Union[asyncio.Semaphore, AdaptiveSemaphore],
    max_tokens: int = 4096,
    max_retries: int = MAX_RETRIES,
    allow_empty_content: bool = False,
    on_retry: Callable[[int, int, str, float], None] | None = None,
    api_mode: str = "native",
    client: httpx.AsyncClient | None = None,
) -> str:
    """
    Один запрос к чату (native LM Studio или OpenAI-совместимый), с семафором и backoff.
    Важно: слот семафора освобождается ДО backoff sleep, чтобы другие воркеры не простаивали.
    """
    headers = {"Content-Type": "application/json", **_auth_headers(api_key)}
    use_native = api_mode.strip().lower() != "openai"
    endpoint = _chat_endpoint(base_url, native=use_native)

    def _build_payload(native: bool, include_reasoning_control: bool = True) -> dict[str, Any]:
        if native:
            from reasoning_models import native_reasoning_payload

            system_prompt, native_input = _native_input_from_messages(messages)
            payload: dict[str, Any] = {
                "model": model,
                "input": native_input,
                "max_output_tokens": max_tokens,
                "temperature": 0,
                "store": False,
            }
            if include_reasoning_control:
                payload.update(native_reasoning_payload(model))
            if system_prompt:
                payload["system_prompt"] = system_prompt
            ctx_env = os.getenv("NOCTURNE_NATIVE_CHAT_CONTEXT_LENGTH", "").strip()
            if ctx_env.isdigit():
                payload["context_length"] = int(ctx_env)
            return payload
        return {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }

    for attempt in range(max_retries):
        await semaphore.acquire()
        delay = 0.0
        try:
            payload = _build_payload(use_native)
            timeout_cfg = httpx.Timeout(
                connect=15.0,
                read=DEFAULT_TIMEOUT,
                write=30.0,
                pool=15.0,
            )
            from lm_client import is_unsupported_reasoning_response
            from reasoning_models import is_no_reasoning_param, mark_no_reasoning_param

            if client is None:
                async with httpx.AsyncClient(timeout=timeout_cfg) as local_client:
                    r = await local_client.post(endpoint, json=payload, headers=headers)
                    if (
                        r.status_code == 400
                        and use_native
                        and not is_no_reasoning_param(model)
                        and is_unsupported_reasoning_response(r)
                    ):
                        mark_no_reasoning_param(model)
                        logger.info(
                            "Model %s does not support reasoning param; retrying without it.", model
                        )
                        fallback_payload = _build_payload(use_native, include_reasoning_control=False)
                        r = await local_client.post(endpoint, json=fallback_payload, headers=headers)
            else:
                r = await client.post(endpoint, json=payload, headers=headers)
                if (
                    r.status_code == 400
                    and use_native
                    and not is_no_reasoning_param(model)
                    and is_unsupported_reasoning_response(r)
                ):
                    mark_no_reasoning_param(model)
                    logger.info(
                        "Model %s does not support reasoning param; retrying without it.", model
                    )
                    fallback_payload = _build_payload(use_native, include_reasoning_control=False)
                    r = await client.post(endpoint, json=fallback_payload, headers=headers)
            if r.status_code in (500, 502, 503):
                raise httpx.HTTPStatusError(
                    f"Server error {r.status_code}",
                    request=r.request,
                    response=r,
                )
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                raise RuntimeError("Unexpected response format from LM Studio chat endpoint")
            content, reasoning = _extract_chat_response_content(data)
            if not str(content).strip():
                reasoning_str = str(reasoning).strip()
                if reasoning_str:
                    salvaged = _extract_results_tag(reasoning_str)
                    if salvaged:
                        content = salvaged
                    elif re.search(r"<results>", reasoning_str, re.IGNORECASE):
                        content = reasoning_str
                if not str(content).strip():
                    if allow_empty_content:
                        content = ""
                    else:
                        raise RuntimeError(
                            "Model returned empty content (possibly reasoning-only output)"
                        )
            if isinstance(semaphore, AdaptiveSemaphore):
                await semaphore.record_success()
            return content.strip()
        except (httpx.TimeoutException, httpx.HTTPStatusError,
                httpx.HTTPError, RuntimeError) as exc:
            if isinstance(semaphore, AdaptiveSemaphore):
                await semaphore.record_failure()
            is_http_4xx = (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response is not None
                and 400 <= exc.response.status_code < 500
            )
            retry_after_s: float | None = None
            if (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response is not None
                and exc.response.status_code == 429
            ):
                try:
                    retry_after_raw = exc.response.headers.get("Retry-After", "").strip()
                    retry_after_s = float(retry_after_raw) if retry_after_raw else None
                except Exception:
                    retry_after_s = None
            if retry_after_s is not None:
                delay = max(0.5, retry_after_s)
            else:
                delay = 0.0 if is_http_4xx else (2 ** attempt) * (0.5 + random.uniform(0, 1))
            if (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response is not None
                and exc.response.status_code == 429
            ):
                retry_kind = "rate_limited_429"
            elif (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response is not None
                and exc.response.status_code in (500, 502, 503)
            ):
                retry_kind = "server_5xx"
            elif is_http_4xx:
                retry_kind = "client_4xx"
            elif isinstance(exc, httpx.TimeoutException):
                retry_kind = "timeout"
            elif isinstance(exc, RuntimeError):
                retry_kind = "empty_content"
            else:
                retry_kind = "http_error"
            if on_retry:
                try:
                    on_retry(attempt + 1, max_retries, retry_kind, delay)
                except Exception:
                    pass
            logger.warning(
                "call_llm attempt %s/%s failed: %s (%s); release slot, retry in %.1fs",
                attempt + 1,
                max_retries,
                type(exc).__name__,
                sanitize_for_log(str(exc)),
                delay,
            )
            if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                body_preview = sanitize_for_log(exc.response.text[:280])
                logger.warning(
                    "call_llm http_status=%s classifier=%s body=%s",
                    exc.response.status_code,
                    _classify_http_400(exc.response.text) if exc.response.status_code == 400 else "n/a",
                    body_preview,
                )
            # Для client 4xx дальнейшие ретраи обычно бесполезны, кроме 429.
            if is_http_4xx and not (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response is not None
                and exc.response.status_code == 429
            ):
                raise
            if attempt == max_retries - 1:
                raise
        finally:
            if isinstance(semaphore, AdaptiveSemaphore):
                await semaphore.release()
            else:
                semaphore.release()
        if delay > 0:
            await asyncio.sleep(delay)
    return ""


async def warm_up(
    base_url: str,
    api_key: str,
    model: str,
    semaphore: Union[asyncio.Semaphore, AdaptiveSemaphore],
    api_mode: str = "native",
    client: httpx.AsyncClient | None = None,
) -> None:
    """Прогрев модели одним минимальным запросом."""
    await call_llm(
        [{"role": "user", "content": "Hi"}],
        model=model,
        base_url=base_url,
        api_key=api_key,
        semaphore=semaphore,
        max_tokens=10,
        max_retries=2,
        allow_empty_content=True,
        api_mode=api_mode,
        client=client,
    )


_VISION_FILE_RE = re.compile(r"\[VISION_FILE:\s*(.+?)\]\s*", re.IGNORECASE)


def check_vision_capability(base_url: str, api_key: str, model: str) -> tuple[bool, str]:
    """Синхронная проверка: модель принимает image input (native + fallback OpenAI-compat)."""
    headers = {"Content-Type": "application/json", **_auth_headers(api_key)}
    native_url = _chat_endpoint(base_url, native=True)
    native_payload: dict[str, Any] = {
        "model": model,
        "input": [
            {"type": "text", "content": "Reply with exactly: OK"},
            {"type": "image", "data_url": f"data:image/png;base64,{_MIN_PNG_B64}"},
        ],
        "max_output_tokens": 16,
        "temperature": 0,
        "store": False,
    }
    openai_url = _chat_endpoint(base_url, native=False)
    openai_payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply with exactly: OK"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_MIN_PNG_B64}"},
                    },
                ],
            }
        ],
        "max_tokens": 16,
        "temperature": 0,
    }
    try:
        with httpx.Client(timeout=90.0) as client:
            r = client.post(
                native_url if USE_LMSTUDIO_NATIVE_API else openai_url,
                json=native_payload if USE_LMSTUDIO_NATIVE_API else openai_payload,
                headers=headers,
            )
            if r.status_code == 400:
                # Пробуем второй транспорт на случай несовместимости конкретной сборки LM Studio.
                r = client.post(
                    openai_url if USE_LMSTUDIO_NATIVE_API else native_url,
                    json=openai_payload if USE_LMSTUDIO_NATIVE_API else native_payload,
                    headers=headers,
                )
            if r.status_code >= 400:
                return False, sanitize_for_log(f"HTTP {r.status_code}: {r.text[:280]}")
            r.raise_for_status()
        return True, "Vision OK — модель приняла image_url"
    except Exception as exc:
        return False, sanitize_for_log(f"{type(exc).__name__}: {exc}")


def _build_vision_map_messages(
    user_query: str,
    chunk_index: int,
    file_label: str,
    image_path: Path,
    language_hint: str = "",
) -> list[dict[str, Any]]:
    mime = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    raw = image_path.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    prompt = (
        f"{user_query}\n\n"
        f"{language_hint}\n\n"
        "Проанализируй изображение. Верни JSON по схеме из system prompt.\n"
        f"Поле chunk_index в JSON = {chunk_index}. Поле file = {file_label!r}.\n"
        "В evidence_refs.quote опиши кратко, что видно на изображении (не выдумывай текста, которых нет).\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_MAP},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]


_THINKING_PATTERNS = re.compile(
    r"(?i)^(\s*)("
    r"thinking process|internal monologue|let me (think|analyze|check)|"
    r"the user (is asking|wants|asked|request)|"
    r"step \d|first,?\s+I|analyzing the|"
    r"i need to (look|check|find|extract|analyze)|"
    r"looking at the (provided|text|data)|"
    r"ok,?\s+so|wait,?\s+(let me|I)|"
    r"let's (start|begin|check|look)"
    r")"
)


_TEMPLATE_LITERALS = re.compile(
    r"^(MARKDOWN_REPORT|JSON_OBJECT|JSON_ARRAY_HERE|YOUR\s+ANSWER\s+HERE)$",
    re.IGNORECASE,
)


def _extract_results_tag(text: str) -> str:
    """Extract content from <results> tags, or detect and strip thinking-only output."""
    m = re.search(r"<results>\s*([\s\S]*?)\s*</results>", text, re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        # Guard: model outputted the literal template string from the system prompt.
        if _TEMPLATE_LITERALS.match(inner):
            logger.warning("Model returned template literal '%s' instead of content — treating as empty.", inner[:40])
            return ""
        return inner

    # Strip leading thinking/reasoning bleed-through before returning raw
    first_line = text.strip().split("\n")[0] if text.strip() else ""
    if _THINKING_PATTERNS.match(first_line):
        return ""

    # Unwrapped JSON / fenced output: many local models emit a valid JSON object
    # (or ```json fence) without the <results> wrapper. Keep it — downstream
    # parsers (_parse_map_json_payload / _parse_scout_json_payload) handle it.
    # Without this, a valid MAP result >500 chars would be silently discarded.
    stripped = text.strip()
    if stripped[:1] in ("{", "[") or stripped.startswith("```"):
        return stripped

    # If text is long and never gets to <results>, it's unformatted reasoning
    if len(text) > 500 and not re.search(r"<results>", text, re.IGNORECASE):
        # Last-resort: try to salvage if there's a clear answer-like block after "---" or heading
        salvage = re.search(
            r"(?i)(?:conclusion|summary|список уязвимост|итог|result):?\s*\n+([\s\S]{50,})",
            text,
        )
        if salvage:
            return salvage.group(1).strip()
        return ""

    return text.strip()


def _parse_map_json_payload(raw: str) -> dict[str, Any] | None:
    """Извлечь JSON объекта MAP из содержимого <results>."""
    s = raw.strip()
    if not s:
        return None
    if s.startswith("```"):
        s = re.sub(r"^```\w*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _sanitize_map_json(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Critical/High без file+quote в evidence — понижаем severity (не в финальный «критичный» блок).
    """
    findings_out: list[dict[str, Any]] = []
    for f in obj.get("findings") or []:
        if not isinstance(f, dict):
            continue
        sev = str(f.get("severity") or "").lower()
        refs = f.get("evidence_refs") or []
        valid = False
        for er in refs:
            if not isinstance(er, dict):
                continue
            if (er.get("quote") or "").strip() and (er.get("file") or "").strip():
                valid = True
                break
        if sev in ("critical", "high") and not valid:
            f = dict(f)
            f["severity"] = "medium"
            expl = str(f.get("explanation") or "")
            f["explanation"] = expl + " [severity downgraded: critical/high requires file+quote evidence]"
        findings_out.append(f)
    out = dict(obj)
    out["findings"] = findings_out
    return out


def _map_metrics_from_results(map_results: list[str]) -> dict[str, int]:
    """Подсчёт findings/evidence и релевантных чанков по MAP-JSON."""
    relevant = 0
    findings_n = 0
    evidence_n = 0
    for raw in map_results:
        inner = _extract_results_tag(raw) if "<results>" in raw.lower() else raw
        parsed = _parse_map_json_payload(inner)
        if not parsed:
            continue
        if parsed.get("no_relevant_data"):
            continue
        relevant += 1
        for f in parsed.get("findings") or []:
            if isinstance(f, dict):
                findings_n += 1
                for er in f.get("evidence_refs") or []:
                    if isinstance(er, dict) and (er.get("quote") or er.get("file")):
                        evidence_n += 1
    return {
        "relevant_chunks": relevant,
        "findings_count": findings_n,
        "evidence_refs_count": evidence_n,
    }


def _reduce_needs_refine(text: str, min_chars: int = 1200) -> bool:
    """Эвристика: итог слишком короткий или без ключевых разделов."""
    t = text.strip()
    if len(t) < min_chars:
        return True
    low = t.lower()
    for aliases in _SECTION_ALIASES.values():
        if not any(alias in low for alias in aliases):
            return True
    # Count both single-# and ##-level headers (reports may use either style).
    header_count = len(re.findall(r"(?m)^#{1,2}\s+\S", t))
    if header_count < 4:
        return True
    return False


async def _refine_final_report(
    draft: str,
    user_query: str,
    base_url: str,
    api_key: str,
    model: str,
    semaphore: Union[asyncio.Semaphore, AdaptiveSemaphore],
    max_tokens: int,
    api_mode: str = "native",
    language_hint: str = "",
    client: httpx.AsyncClient | None = None,
    max_context_tokens: int | None = None,
) -> str:
    """Второй проход: расширить отчёт без потери структуры."""
    from parser import count_tokens

    effective_limit = max_context_tokens if max_context_tokens is not None else MAX_REDUCE_INPUT_TOKENS
    header = (
        f"Исходный запрос пользователя:\n{user_query}\n\n"
        f"{language_hint}\n\n"
        "Ниже черновик отчёта. Расширь и углуби его: сохрани те же 4 раздела, "
        "добавь детали из черновика, не удаляй Evidence Matrix, добавь недостающие ссылки на источники.\n\n"
    )
    draft_budget = max(500, effective_limit - count_tokens(header) - count_tokens(SYSTEM_PROMPT_REDUCE) - max_tokens - 256)
    safe_draft = _truncate_text_to_tokens(draft, draft_budget)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_REDUCE},
        {"role": "user", "content": header + safe_draft},
    ]
    try:
        out = await call_llm(
            messages,
            model,
            base_url,
            api_key,
            semaphore,
            max_tokens=max_tokens,
            api_mode=api_mode,
            client=client,
        )
        return _extract_results_tag(out)
    except Exception as exc:
        logger.warning(
            "_refine_final_report call_llm failed (%s: %s); returning empty to skip refine.",
            type(exc).__name__,
            sanitize_for_log(str(exc)[:180]),
        )
        return ""


async def answer_with_context(
    question: str,
    contexts: list[str],
    base_url: str,
    api_key: str,
    model: str,
    workers: int = 2,
    api_mode: str = "native",
) -> str:
    """Синтез ответа по найденным контекстам (retrieval hits)."""
    from parser import count_tokens

    semaphore: Union[asyncio.Semaphore, AdaptiveSemaphore]
    if USE_ADAPTIVE_SEMAPHORE:
        semaphore = AdaptiveSemaphore(max(1, workers))
    else:
        semaphore = asyncio.Semaphore(max(1, workers))
    trimmed: list[str] = []
    budget = 10000
    used = 0
    for c in contexts:
        t = count_tokens(c)
        if used + t > budget and trimmed:
            break
        if t > 2000:
            c = c[:7000]
            t = count_tokens(c)
        trimmed.append(c)
        used += t

    combined = "\n\n".join([f"[CTX {i+1}]\n{c}" for i, c in enumerate(trimmed)])
    prompt = (
        f"Запрос: {question}\n\n"
        "Текстовые фрагменты:\n\n"
        f"{combined}\n\n"
        "Ответь на запрос строго на основе фрагментов выше. Укажи [CTX N] как источник."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    out = await call_llm(
        messages,
        model,
        base_url,
        api_key,
        semaphore,
        max_tokens=3000,
        api_mode=api_mode,
    )
    return _extract_results_tag(out)


def _dedupe_evidence_refs(refs: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for er in refs:
        if not isinstance(er, dict):
            continue
        file_ = str(er.get("file") or "").strip()
        chunk = str(er.get("chunk") or "").strip()
        quote = str(er.get("quote") or "").strip()
        key = (file_.lower(), chunk.lower(), quote[:40].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({"file": file_, "chunk": chunk, "quote": quote})
    return out


def _normalize_map_json(parsed: dict[str, Any]) -> dict[str, Any]:
    parsed = _sanitize_map_json(parsed)
    findings_out: list[dict[str, Any]] = []
    for f in parsed.get("findings") or []:
        if not isinstance(f, dict):
            continue
        ff = dict(f)
        ff["evidence_refs"] = _dedupe_evidence_refs(list(ff.get("evidence_refs") or []))
        findings_out.append(ff)
    out = dict(parsed)
    out["findings"] = findings_out
    return out


_FILE_PATH_HEADER_RE = re.compile(r"\[FILE_PATH:\s*(.+?)\]", re.IGNORECASE)


def _extract_file_path_from_chunk(chunk_text: str) -> str | None:
    """Extract the [FILE_PATH: ...] value from a chunk header line."""
    m = _FILE_PATH_HEADER_RE.search(chunk_text)
    if m:
        return m.group(1).strip()
    return None


def _fix_map_json_file_field(obj: dict[str, Any], chunk_text: str) -> dict[str, Any]:
    """
    Override the model-generated 'file' field with the authoritative FILE_PATH from the chunk
    header. This corrects the common mistake where MAP models use Sonar component IDs
    (e.g. 'digital-official:edopm:...:service-name') instead of the actual relative file path.
    Also fix evidence_refs.file for the same reason.
    """
    authoritative_path = _extract_file_path_from_chunk(chunk_text)
    if not authoritative_path:
        return obj

    out = dict(obj)
    # Fix the top-level file field
    out["file"] = authoritative_path

    # Fix evidence_refs.file in all findings
    fixed_findings: list[dict[str, Any]] = []
    for f in out.get("findings") or []:
        if not isinstance(f, dict):
            fixed_findings.append(f)
            continue
        ff = dict(f)
        new_refs: list[dict[str, Any]] = []
        for er in ff.get("evidence_refs") or []:
            if not isinstance(er, dict):
                new_refs.append(er)
                continue
            ref = dict(er)
            ref["file"] = authoritative_path
            new_refs.append(ref)
        ff["evidence_refs"] = new_refs
        fixed_findings.append(ff)
    out["findings"] = fixed_findings
    return out


def _dedupe_map_findings(obj: dict[str, Any], max_findings: int = 30) -> dict[str, Any]:
    """
    Deduplicate findings within a single MAP result by (severity, type, explanation[:80]).
    Also caps total findings to max_findings to prevent hallucinating models from flooding
    the merge phase with hundreds of identical entries.
    """
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for f in obj.get("findings") or []:
        if not isinstance(f, dict):
            continue
        sev = str(f.get("severity") or "").lower().strip()
        ftype = str(f.get("type") or "").lower().strip()
        expl = str(f.get("explanation") or "")[:80].lower().strip()
        key = (sev, ftype, expl)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)
        if len(deduped) >= max_findings:
            break
    out = dict(obj)
    out["findings"] = deduped
    return out


def _max_findings_per_chunk() -> int:
    """Лимит находок на один MAP-чанк (env NOCTURNE_MAX_FINDINGS_PER_CHUNK)."""
    raw = os.getenv("NOCTURNE_MAX_FINDINGS_PER_CHUNK", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return 60


def _to_normalized_map_json_text(
    raw: str,
    chunk_text: str = "",
    max_findings_per_chunk: int | None = None,
) -> str | None:
    if not raw or not raw.strip():
        return None
    if max_findings_per_chunk is None:
        max_findings_per_chunk = _max_findings_per_chunk()
    inner = _extract_results_tag(raw) if "<results>" in raw.lower() else raw
    parsed = _parse_map_json_payload(inner)
    if not parsed:
        return None
    normalized = _normalize_map_json(parsed)
    # Deduplicate findings within this chunk (prevents hallucinating models from
    # generating dozens of identical entries for a single CSV row).
    normalized = _dedupe_map_findings(normalized, max_findings=max_findings_per_chunk)
    # Fix file field: override model-generated component IDs with authoritative FILE_PATH.
    if chunk_text:
        normalized = _fix_map_json_file_field(normalized, chunk_text)
    return json.dumps(normalized, ensure_ascii=False)


def _fallback_merge_map_json(items: list[str]) -> str:
    merged_findings: list[dict[str, Any]] = []
    merged_recommendations: list[str] = []
    query_alignment = ""
    any_relevant = False

    for raw in items:
        normalized_txt = _to_normalized_map_json_text(raw)
        if not normalized_txt:
            continue
        parsed = json.loads(normalized_txt)
        if not bool(parsed.get("no_relevant_data")):
            any_relevant = True
        if not query_alignment:
            query_alignment = str(parsed.get("query_alignment") or "")
        for f in parsed.get("findings") or []:
            if isinstance(f, dict):
                merged_findings.append(f)
        for rec in parsed.get("recommendations") or []:
            if isinstance(rec, str) and rec.strip() and rec.strip() not in merged_recommendations:
                merged_recommendations.append(rec.strip())

    merged: dict[str, Any] = {
        "chunk_index": -1,
        "file": "MULTI",
        "query_alignment": query_alignment,
        "no_relevant_data": not any_relevant,
        "findings": merged_findings,
        "recommendations": merged_recommendations,
    }
    merged = _normalize_map_json(merged)
    return json.dumps(merged, ensure_ascii=False)


def compute_job_id(
    file_path: Path,
    user_query: str,
    *,
    file_paths: list[Path] | None = None,
    chunk_size: int | None = None,
    model: str | None = None,
    composer_model: str | None = None,
) -> str:
    """job_id для кэша: инвалидация при смене файла, запроса, состава корпуса
    или параметров прогона (размер чанка, модель, composer).

    chunk_size/model влияют на текст чанка под данным chunk_index — без них
    закэшированный MAP-ответ мог быть посчитан для другого фрагмента.
    """
    fingerprint: str | None = None
    if file_paths:
        from cache import corpus_fingerprint_from_paths

        fingerprint = corpus_fingerprint_from_paths(file_paths)
    params_bits: list[str] = []
    if chunk_size is not None:
        params_bits.append(f"cs={int(chunk_size)}")
    if model:
        params_bits.append(f"m={model}")
    if composer_model:
        params_bits.append(f"c={composer_model}")
    params = "|".join(params_bits) if params_bits else None
    return build_job_id(
        file_path, user_query, corpus_fingerprint=fingerprint, params=params,
    )


def _parse_scout_json_payload(raw: str) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return None
    inner = _extract_results_tag(raw) if "<results>" in raw.lower() else raw.strip()
    try:
        obj = json.loads(inner)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def scout_relevance_score(parsed: dict[str, Any]) -> float:
    """Оценка 0..1 из scout JSON; при ошибке парсинга вызывающий код должен трактовать как «пропустить фильтр»."""
    if bool(parsed.get("no_relevant_data")):
        return 0.0
    score = 0.0
    try:
        score = float(parsed.get("relevance_score", 0))
    except (TypeError, ValueError):
        score = 0.0
    if bool(parsed.get("relevant")):
        score = max(score, 0.45)
    return max(0.0, min(1.0, score))


def filter_indices_by_scout_scores(
    indices: list[int],
    scores: dict[int, float],
    threshold: float,
) -> tuple[list[int], list[int]]:
    """Вернуть (deep_map_indices, skipped_indices). Неизвестный score → deep MAP (безопасный fallback)."""
    deep: list[int] = []
    skipped: list[int] = []
    thr = max(0.0, min(1.0, threshold))
    for i in indices:
        if i not in scores:
            deep.append(i)
            continue
        if scores[i] >= thr:
            deep.append(i)
        else:
            skipped.append(i)
    return deep, skipped


def _available_user_tokens(max_context_tokens: int, system_prompt: str, reserve: int = 2048) -> int:
    """Сколько токенов остаётся под user-сообщение при заданном контексте и system prompt."""
    from parser import count_tokens

    # Используем реальный контекст модели без искусственного обрезания по MAX_REDUCE_INPUT_TOKENS.
    # MAX_REDUCE_INPUT_TOKENS применяется только к размеру MAP-чанков, а не к reduce/merge-вызовам.
    return max(500, max_context_tokens - count_tokens(system_prompt) - reserve)


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Обрезать строку до max_tokens (аппроксимация: 1 токен ≈ 4 символа)."""
    from parser import count_tokens
    if count_tokens(text) <= max_tokens:
        return text
    suffix = "\n... [TRUNCATED: data too large for model context] ..."
    suffix_tokens = count_tokens(suffix)
    # Reserve budget for suffix so total stays within max_tokens.
    effective_max = max(0, max_tokens - suffix_tokens)
    # Бинарный поиск по длине символов
    lo, hi = 0, len(text)
    while hi - lo > 32:
        mid = (lo + hi) // 2
        if count_tokens(text[:mid]) <= effective_max:
            lo = mid
        else:
            hi = mid
    truncated = text[:lo]
    logger.warning(
        "Input truncated to fit context: original=%s chars -> %s chars (%s tokens max)",
        len(text),
        len(truncated),
        max_tokens,
    )
    return truncated + suffix


def _merge_map_json_deterministic(map_results: list[str]) -> str:
    """
    Детерминированное слияние MAP JSON без LLM.
    Просто объединяет все findings и recommendations через _fallback_merge_map_json.
    Никаких LLM-вызовов — никакой потери данных при «дедупликации».
    """
    if not map_results:
        return ""
    if len(map_results) == 1:
        normalized_single = _to_normalized_map_json_text(map_results[0].strip())
        return normalized_single or map_results[0].strip()
    return _fallback_merge_map_json(map_results)


def _count_report_sections(text: str) -> int:
    """Подсчёт заголовков ## в итоговом отчёте."""
    return len(re.findall(r"(?m)^##\s+\S", text))


async def _reduce_json_to_markdown(
    aggregated_json: str,
    user_query: str,
    base_url: str,
    api_key: str,
    reduce_model: str,
    semaphore: Union[asyncio.Semaphore, AdaptiveSemaphore],
    max_tokens: int,
    api_mode: str = "native",
    language_hint: str = "",
    client: httpx.AsyncClient | None = None,
    max_context_tokens: int | None = None,
) -> str:
    """Финальный REDUCE: JSON → markdown с разделами."""
    from parser import count_tokens

    effective_limit = max_context_tokens if max_context_tokens is not None else MAX_REDUCE_INPUT_TOKENS
    header = (
        f"Исходный запрос пользователя:\n{user_query}\n\n"
        f"{language_hint}\n\n"
        "Ниже объединённый JSON MAP-результатов. Составь полный отчёт с разделами "
        "## Executive Summary, ## Comprehensive Findings, ## Evidence Matrix, ## Action Plan. "
        "Опирайся только на данные JSON; не придумывай факты.\n\n"
    )
    json_budget = max(500, effective_limit - count_tokens(header) - count_tokens(SYSTEM_PROMPT_REDUCE) - max_tokens - 256)
    safe_json = _truncate_text_to_tokens(aggregated_json, json_budget)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_REDUCE},
        {"role": "user", "content": header + safe_json},
    ]
    try:
        out = await call_llm(
            messages,
            reduce_model,
            base_url,
            api_key,
            semaphore,
            max_tokens=max_tokens,
            api_mode=api_mode,
            client=client,
        )
        return _extract_results_tag(out)
    except Exception as exc:
        logger.warning(
            "_reduce_json_to_markdown call_llm failed (%s: %s); returning empty to trigger fallback.",
            type(exc).__name__,
            sanitize_for_log(str(exc)[:180]),
        )
        return ""


def _validate_final_report(
    report: str,
    metrics: dict[str, int],
) -> tuple[str, list[str]]:
    """Пост-проверки: длина, разделы, согласованность с метриками MAP."""
    warnings: list[str] = []
    t = report.strip()
    if len(t) < 400:
        warnings.append("short_report")
    low = t.lower()
    for sec, aliases in _SECTION_ALIASES.items():
        if not any(alias in low for alias in aliases):
            warnings.append(f"missing_section:{sec}")
    ev = metrics.get("evidence_refs_count", 0)
    if ev >= 3 and low.count("quote") < 1 and low.count("```") < 1:
        warnings.append("low_evidence_density")
    # Duplicate heading lines (simple heuristic)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip().startswith("##")]
    if len(lines) != len(set(lines)):
        warnings.append("duplicate_sections")
    findings_n = metrics.get("findings_count", 0)
    if findings_n >= 5 and ev < max(1, findings_n // 3):
        warnings.append("low_evidence_coverage")
    footer = ""
    if warnings:
        footer = "\n\n---\n*(Валидация отчёта: " + ", ".join(warnings) + ")*\n"
    return t + footer, warnings


async def verify_report_with_evidence(
    report: str,
    evidence_index: list[dict[str, str]],
    user_query: str,
    base_url: str,
    api_key: str,
    model: str,
    semaphore: Union[asyncio.Semaphore, AdaptiveSemaphore],
    api_mode: str = "native",
    client: httpx.AsyncClient | None = None,
) -> tuple[str, list[str]]:
    """Опциональный verifier: малая модель проверяет отчёт против индекса цитат."""
    if not evidence_index or len(report) < 200:
        return report, []
    from parser import count_tokens

    rows = "\n".join(
        f"- {e.get('file','?')} chunk {e.get('chunk','?')}: {e.get('quote','')[:80]}"
        for e in evidence_index[:35]
    )
    prompt = (
        f"Запрос: {user_query}\n\nИндекс доказательств:\n{rows}\n\n"
        f"Отчёт:\n{report[:12000]}\n\n"
        "Верни в <results> краткий список проблем (или NO_ISSUES), если critical/high без цитат."
    )
    if count_tokens(prompt) > 6000:
        prompt = prompt[:24000]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        out = await call_llm(
            messages, model, base_url, api_key, semaphore,
            max_tokens=512, max_retries=1, api_mode=api_mode, client=client,
        )
        note = _extract_results_tag(out).strip()
        if note and "NO_ISSUES" not in note.upper():
            return report + f"\n\n---\n*(Verifier)*\n{note}\n", ["verifier_notes"]
    except Exception as exc:
        logger.warning("Verifier skipped: %s", sanitize_for_log(str(exc)[:120]))
    return report, []


def _build_fallback_report(
    aggregated_json: str,
    user_query: str,
    map_results: list[str],
    language_hint: str = "",
) -> str:
    """
    Запасной отчёт когда REDUCE-модель не справилась.
    Строится из MAP JSON напрямую: все findings → markdown таблица.
    """
    findings: list[dict[str, Any]] = []
    evidence_rows: list[str] = []
    recommendations: list[str] = []
    seen_findings: set[tuple[str, str, str]] = set()

    # Use aggregated_json if available (it already contains all map_results),
    # otherwise fall back to individual map_results to avoid double-counting.
    sources = [aggregated_json] if aggregated_json and aggregated_json.strip() else list(map_results)
    for raw in sources:
        inner = _extract_results_tag(raw) if "<results>" in raw.lower() else raw
        parsed = _parse_map_json_payload(inner)
        if not parsed:
            continue
        for f in parsed.get("findings") or []:
            if not isinstance(f, dict):
                continue
            dedup_key = (
                str(f.get("type") or "").lower(),
                str(f.get("severity") or "").lower(),
                str(f.get("explanation") or "")[:60].lower(),
            )
            if dedup_key in seen_findings:
                continue
            seen_findings.add(dedup_key)
            findings.append(f)
            for er in (f.get("evidence_refs") or []):
                if isinstance(er, dict):
                    file_ = er.get("file", "?")
                    chunk = er.get("chunk", "?")
                    quote = er.get("quote", "")[:80]
                    evidence_rows.append(f"| {file_} | {chunk} | {quote} |")
        for rec in parsed.get("recommendations") or []:
            if isinstance(rec, str) and rec.strip() and rec.strip() not in recommendations:
                recommendations.append(rec.strip())

    query_preview = (user_query[:160] + "…") if len(user_query) > 160 else user_query

    if not findings:
        return (
            f"## Executive Summary\n\n"
            f"Анализ завершён — значимые находки не обнаружены.\n\n"
            f"**Запрос:** {query_preview}\n\n"
            "**Возможные причины:**\n"
            "- Данные не содержат нарушений, соответствующих критериям запроса.\n"
            "- Модель отметила все фрагменты как нерелевантные (`no_relevant_data: true`).\n"
            "- Попробуйте другую модель, смягчите фильтры severity или проверьте формат входных данных.\n\n"
            "## Comprehensive Findings\n\nНет данных.\n\n"
            "## Evidence Matrix\n\nНет данных.\n\n"
            "## Action Plan\n\nПовторите анализ с другой моделью или уточните критерии запроса.\n"
        )

    findings_lines = []
    for f in findings:
        sev = f.get("severity", "?")
        ftype = f.get("type", "?")
        expl = f.get("explanation", "")[:200]
        findings_lines.append(f"- **[{sev.upper()}]** `{ftype}`: {expl}")

    rec_section = "\n".join(f"{i+1}. {r}" for i, r in enumerate(recommendations)) if recommendations else "Нет рекомендаций."

    ev_section = (
        "| Файл | Chunk | Цитата |\n|------|-------|--------|\n" + "\n".join(evidence_rows)
        if evidence_rows
        else "Нет evidence данных."
    )

    hint = f"\n\n{language_hint}" if language_hint else ""
    return (
        f"## Executive Summary{hint}\n\n"
        f"Анализ по запросу: **{query_preview}**. "
        f"Обнаружено находок: {len(findings)}. "
        f"(Отчёт сгенерирован из MAP-данных — REDUCE-модель вернула неполный результат.)\n\n"
        f"## Comprehensive Findings\n\n"
        + "\n".join(findings_lines) + "\n\n"
        f"## Evidence Matrix\n\n{ev_section}\n\n"
        f"## Action Plan\n\n{rec_section}\n"
    )


def _fallback_section_from_json(merged_json: str, file_key: str) -> str:
    """
    Минимальная markdown-секция из MAP JSON когда REDUCE-модель не справилась.
    Используется как fallback внутри _section_reduce_groups.
    """
    inner = _extract_results_tag(merged_json) if "<results>" in merged_json.lower() else merged_json
    parsed = _parse_map_json_payload(inner)
    if not parsed:
        return f"### Файл: {file_key}\n\nНет данных.\n"
    findings = parsed.get("findings") or []
    lines = [f"### Файл: {file_key}\n"]
    for f in findings:
        if not isinstance(f, dict):
            continue
        sev = str(f.get("severity", "?")).upper()
        ftype = f.get("type", "?")
        expl = f.get("explanation", "")[:200]
        lines.append(f"- **[{sev}]** `{ftype}`: {expl}")
        for er in (f.get("evidence_refs") or []):
            if isinstance(er, dict):
                quote = er.get("quote", "")[:80]
                if quote:
                    lines.append(f"  > *{quote}*")
    return "\n".join(lines) + "\n"


def _concatenate_sections_fallback(sections: list[str], user_query: str, language_hint: str = "") -> str:
    """
    Детерминированное объединение секций когда LLM-синтез не удался.
    Формирует валидный отчёт со всеми 4 разделами.
    """
    query_preview = (user_query[:160] + "…") if len(user_query) > 160 else user_query
    hint = f"\n\n{language_hint}" if language_hint else ""
    header = (
        f"## Executive Summary{hint}\n\n"
        f"Анализ по запросу: **{query_preview}**. "
        f"Обработано файловых групп: {len(sections)}. "
        "(Отчёт сгенерирован объединением посекционных результатов.)\n\n"
    )
    findings_body = "\n\n---\n\n".join(sections)
    return (
        header
        + "## Comprehensive Findings\n\n"
        + findings_body
        + "\n\n## Evidence Matrix\n\nСм. секции выше.\n\n"
        + "## Action Plan\n\nУстраните выявленные проблемы согласно секциям выше.\n"
    )


def _coalesce_file_groups(
    file_groups: dict[str, list[str]],
    max_tokens_per_group: int,
) -> dict[str, list[str]]:
    """
    Упаковать мелкие файловые группы в более крупные по бюджету токенов.

    Без этого посекционный REDUCE делает один LLM-вызов на каждый файл —
    для корпуса в тысячи файлов это тысячи вызовов REDUCE поверх MAP.
    После упаковки число REDUCE-вызовов масштабируется по объёму данных.
    """
    from parser import count_tokens

    if len(file_groups) <= 1:
        return dict(file_groups)

    sized: list[tuple[str, list[str], int]] = []
    for key, items in file_groups.items():
        toks = sum(count_tokens(s) for s in items)
        sized.append((key, items, toks))
    # По возрастанию: мелкие группы пакуются вместе, крупные уходят отдельно.
    sized.sort(key=lambda t: t[2])

    out: dict[str, list[str]] = {}
    cur_keys: list[str] = []
    cur_items: list[str] = []
    cur_toks = 0
    budget = max(2000, max_tokens_per_group)

    def _flush() -> None:
        nonlocal cur_keys, cur_items, cur_toks
        if not cur_items:
            return
        label = cur_keys[0] if len(cur_keys) == 1 else f"{cur_keys[0]} (+{len(cur_keys) - 1})"
        base = label
        n = 2
        while label in out:
            label = f"{base} #{n}"
            n += 1
        out[label] = list(cur_items)
        cur_keys = []
        cur_items = []
        cur_toks = 0

    for key, items, toks in sized:
        if cur_items and cur_toks + toks > budget:
            _flush()
        cur_keys.append(key)
        cur_items.extend(items)
        cur_toks += toks
        if cur_toks >= budget:
            _flush()
    _flush()
    return out


async def _section_reduce_groups(
    file_groups: dict[str, list[str]],
    user_query: str,
    base_url: str,
    api_key: str,
    reduce_model: str,
    semaphore: Union[asyncio.Semaphore, AdaptiveSemaphore],
    reduce_max_tokens: int,
    api_mode: str,
    language_hint: str,
    client: httpx.AsyncClient | None,
    max_context_tokens: int,
    on_progress: Callable[..., None] | None = None,
) -> list[str]:
    """
    Параллельный REDUCE каждой файловой группы → список markdown-секций.
    Каждая группа сначала детерминированно объединяется, затем сокращается до markdown.
    """
    file_keys = list(file_groups.keys())
    total = len(file_keys)

    async def _reduce_one(idx: int, file_key: str) -> str:
        results = file_groups[file_key]
        merged_json = _merge_map_json_deterministic(results)
        if on_progress:
            try:
                on_progress(idx + 1, total, "section_reduce", 0, section_file=file_key)
            except TypeError:
                pass
        section = await _reduce_json_to_markdown(
            merged_json,
            user_query,
            base_url,
            api_key,
            reduce_model,
            semaphore,
            reduce_max_tokens,
            api_mode,
            language_hint,
            client,
            max_context_tokens=max_context_tokens,
        )
        if section and len(section.strip()) > 100:
            return section
        logger.warning(
            "_section_reduce_groups: REDUCE returned empty for file=%s, using fallback section",
            file_key,
        )
        return _fallback_section_from_json(merged_json, file_key)

    tasks = [_reduce_one(i, k) for i, k in enumerate(file_keys)]
    return list(await asyncio.gather(*tasks))


async def _synthesize_final_report(
    sections: list[str],
    user_query: str,
    base_url: str,
    api_key: str,
    reduce_model: str,
    semaphore: Union[asyncio.Semaphore, AdaptiveSemaphore],
    reduce_max_tokens: int,
    api_mode: str,
    language_hint: str,
    client: httpx.AsyncClient | None,
    max_context_tokens: int,
    on_progress: Callable[..., None] | None = None,
    _depth: int = 0,
) -> str:
    """
    Иерархический финальный синтез markdown-секций в единый отчёт.
    Если все секции влезают в контекст — один LLM-вызов.
    Если не влезают — рекурсивная группировка с промежуточным синтезом.
    """
    from parser import count_tokens

    _MAX_SYNTH_DEPTH = 6

    if not sections:
        return ""
    if len(sections) == 1:
        return sections[0]
    if _depth >= _MAX_SYNTH_DEPTH:
        logger.warning("_synthesize_final_report: max depth %s reached, using fallback", _depth)
        return _concatenate_sections_fallback(sections, user_query, language_hint)

    combined = "\n\n---\n\n".join(sections)
    lang_line = f"{language_hint}\n\n" if language_hint else ""
    header = (
        f"Исходный запрос пользователя:\n{user_query}\n\n"
        f"{lang_line}"
        "Ниже готовые отчёты для отдельных файлов/сервисов. "
        "Объедини их в единый итоговый отчёт, не теряя ни одной находки.\n\n"
    )
    # Место под ответ (max_output_tokens) тоже входит в окно контекста сервера.
    synth_reserve = max(2048, reduce_max_tokens + 1024)
    available_budget = _available_user_tokens(
        max_context_tokens, SYSTEM_PROMPT_SYNTHESIZE, reserve=synth_reserve
    )
    header_tokens = count_tokens(header)
    content_budget = max(500, available_budget - header_tokens)

    if count_tokens(combined) <= content_budget:
        if on_progress and _depth == 0:
            try:
                on_progress(1, 1, "synthesize", 0)
            except TypeError:
                pass
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SYNTHESIZE},
            {"role": "user", "content": header + combined},
        ]
        try:
            out = await call_llm(
                messages,
                reduce_model,
                base_url,
                api_key,
                semaphore,
                max_tokens=reduce_max_tokens,
                api_mode=api_mode,
                client=client,
            )
            result = _extract_results_tag(out)
            if result and len(result.strip()) > 300:
                return result
        except Exception as exc:
            logger.warning(
                "_synthesize_final_report call_llm failed (%s: %s); using fallback.",
                type(exc).__name__,
                sanitize_for_log(str(exc)[:180]),
            )
        return _concatenate_sections_fallback(sections, user_query, language_hint)

    # Секции не влезают → группируем и рекурсивно синтезируем
    tokens_per_section = max(1, count_tokens(combined) // len(sections))
    group_size = max(2, min(len(sections), content_budget // tokens_per_section))
    total_groups = (len(sections) + group_size - 1) // group_size
    logger.info(
        "_synthesize_final_report depth=%s: %s sections, group_size=%s, groups=%s",
        _depth, len(sections), group_size, total_groups,
    )
    group_tasks = []
    for i in range(0, len(sections), group_size):
        group = sections[i : i + group_size]
        group_tasks.append(
            _synthesize_final_report(
                group, user_query, base_url, api_key, reduce_model, semaphore,
                reduce_max_tokens, api_mode, language_hint, client, max_context_tokens,
                on_progress=None,
                _depth=_depth + 1,
            )
        )
    intermediate = list(await asyncio.gather(*group_tasks))
    if on_progress and _depth == 0:
        try:
            on_progress(1, 1, "synthesize", 0)
        except TypeError:
            pass
    return await _synthesize_final_report(
        intermediate, user_query, base_url, api_key, reduce_model, semaphore,
        reduce_max_tokens, api_mode, language_hint, client, max_context_tokens,
        on_progress=None, _depth=_depth + 1,
    )


def _extract_chunk_headers(chunks: list[str], max_samples: int = 12) -> list[str]:
    """Extract the first header line from a sample of chunks for meta-prompt context."""
    step = max(1, len(chunks) // max_samples)
    headers: list[str] = []
    for i in range(0, len(chunks), step):
        if len(headers) >= max_samples:
            break
        chunk = chunks[i]
        # Grab first two non-empty lines (FILE_PATH/FILE_TITLE/FILE_LABELS tags)
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        snippet = " | ".join(lines[:2])
        if snippet:
            headers.append(snippet[:300])
    return headers


async def generate_meta_prompt(
    user_query: str,
    sample_headers: list[str],
    base_url: str,
    api_key: str,
    model: str,
    semaphore: Union[asyncio.Semaphore, "AdaptiveSemaphore"],
    api_mode: str = "native",
    client: "httpx.AsyncClient | None" = None,
    language_hint: str = "",
) -> str:
    """
    Ask the composer model to produce a precise domain-specific analysis directive
    that will be used as the effective user_query during the MAP phase.
    Falls back to the original user_query on any error.
    """
    from parser import count_tokens

    headers_text = "\n".join(f"  • {h}" for h in sample_headers[:12])
    lang_line = f"\n{language_hint}" if language_hint else ""
    user_msg = (
        f"User goal:{lang_line}\n{user_query}\n\n"
        f"Sample fragment headers from the corpus ({len(sample_headers)} shown):\n"
        f"{headers_text}\n\n"
        "Generate the analysis directive."
    )
    if count_tokens(user_msg) > 3000:
        # Safety truncation — meta-planner call should be cheap
        user_msg = user_msg[:12000]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_META_PLANNER},
        {"role": "user", "content": user_msg},
    ]
    try:
        out = await call_llm(
            messages,
            model,
            base_url,
            api_key,
            semaphore,
            max_tokens=600,
            max_retries=2,
            allow_empty_content=False,
            api_mode=api_mode,
            client=client,
        )
        directive = _extract_results_tag(out).strip()
        if directive and len(directive) > 30:
            logger.info(
                "Meta-prompt generated by %s (%d chars): %s…",
                model, len(directive), directive[:120],
            )
            return directive
    except Exception as exc:
        logger.warning(
            "generate_meta_prompt failed (%s: %s); using original query.",
            type(exc).__name__,
            sanitize_for_log(str(exc)[:200]),
        )
    return user_query


async def run_map_reduce(
    chunks: list[str] | Any,
    user_query: str,
    base_url: str,
    api_key: str,
    model: str,
    workers: int,
    dynamic_chunk_size: int,
    on_progress: Callable[..., None] | None = None,
    job_id: str | None = None,
    max_context_tokens: int | None = None,
    composer_model: str | None = None,
    vision_model: str | None = None,
    api_mode: str = "native",
    low_vram_mode: bool = True,
    dual_instance_mode: bool = False,
    scout_mode: bool = False,
    scout_relevance_threshold: float = 0.35,
    scout_min_chunks: int = 8,
    scout_model: str | None = None,
    source_path: str = "",
    stop_flag: Callable[[], bool] | None = None,
) -> str:
    """
    MAP с кэшем по job_id; адаптивный семафор; иерархический merge MAP JSON и финальный REDUCE.
    on_progress(current, total, phase, from_cache, **extra) — статус MAP/REDUCE.
    max_context_tokens — бюджет контекста модели для merge (по умолчанию CONTEXT_FALLBACK).
    composer_model — отдельная модель для merge/reduce/refine (иначе model).
    vision_model — модель для чанков с [VISION_FILE:...] (иначе model).
    stop_flag — кооперативная отмена: проверяется между батчами MAP. При остановке
    прогресс остаётся в кэше, job помечается paused, REDUCE не запускается.
    """
    if not chunks:
        return ""

    n_chunks = len(chunks)
    if job_id:
        try:
            from cache import save_job_state

            save_job_state(
                job_id,
                chunks_total=n_chunks,
                query_preview=user_query,
                source_path=source_path,
                status="running",
            )
        except Exception:
            pass

    metrics_row_id = 0
    if job_id:
        try:
            from metrics import record_run_start

            metrics_row_id = record_run_start(
                job_id,
                user_query,
                {
                    "map": model,
                    "scout": scout_model or model,
                    "composer": composer_model or model,
                    "vision": vision_model or model,
                },
            )
        except Exception:
            metrics_row_id = 0

    _claimed_ctx = max_context_tokens if max_context_tokens is not None else CONTEXT_FALLBACK
    max_context_tokens = _server_safe_context_limit(_claimed_ctx)
    if max_context_tokens < _claimed_ctx:
        logger.info(
            "Бюджет контекста: заявлено %s → безопасно %s ток. "
            "(tiktoken ≠ llama.cpp; см. NOCTURNE_CONTEXT_SAFETY_MARGIN, NOCTURNE_CONTEXT_UTILIZATION)",
            _claimed_ctx,
            max_context_tokens,
        )

    workers = max(1, min(4, workers))
    reduce_model = (composer_model or model).strip() or model
    vision_llm = (vision_model or model).strip() or model
    scout_llm = (scout_model or model).strip() or model
    # language_hint is derived from the ORIGINAL user_query so meta-prompt generation doesn't break it
    language_hint = "Ответь строго на русском языке." if _prefer_russian(user_query) else ""
    api_mode = api_mode.strip().lower()
    if api_mode not in {"native", "openai"}:
        api_mode = "native"
    use_native = api_mode == "native"
    try:
        from reasoning_models import list_reasoning_models, refresh_model_catalog_cache

        catalog = fetch_models_catalog(base_url, api_key)
        refresh_model_catalog_cache(catalog)
        reasoning_models = list_reasoning_models(catalog)
        if reasoning_models:
            logger.info(
                "Reasoning-capable models (%s): %s",
                len(reasoning_models),
                ", ".join(reasoning_models[:6])
                + ("…" if len(reasoning_models) > 6 else ""),
            )
    except Exception:
        pass
    # effective_query will be replaced by meta-prompt if composer is set
    effective_query = user_query

    semaphore: Union[asyncio.Semaphore, AdaptiveSemaphore]
    if USE_ADAPTIVE_SEMAPHORE:
        semaphore = AdaptiveSemaphore(workers)
    else:
        semaphore = asyncio.Semaphore(max(1, workers))
    logger.info(
        "MAP concurrency mode: %s (workers=%s, low_vram_sequential=%s, api_mode=%s)",
        "adaptive" if USE_ADAPTIVE_SEMAPHORE else "fixed",
        workers,
        low_vram_mode,
        api_mode,
    )
    started_at = time.monotonic()
    from_cache_count = 0
    failed_count = 0
    completed_count = 0
    started_count = 0
    in_flight_count = 0
    retry_count = 0
    retrying_chunks: set[int] = set()
    server_5xx_count = 0
    pause_until = 0.0
    stop_triggered = False

    def _stopped() -> bool:
        if stop_flag is None:
            return False
        try:
            return bool(stop_flag())
        except Exception:
            return False
    current_loaded_model: str | None = None
    model_instance_pool: dict[str, list[str]] = {}
    dual_instance_active = False
    touched_models: set[str] = set()

    def _chunk_debug(chunk: str) -> tuple[str, str]:
        file_name = "n/a"
        body = chunk
        m = re.search(r"\[Файл:\s*(.+?)\]", chunk)
        if m:
            file_name = m.group(1).strip()
        nl = chunk.rfind("\n")
        if nl != -1:
            body = chunk[nl + 1 :]
        preview = " ".join(body.strip().split())
        if len(preview) > 180:
            preview = preview[:180] + "..."
        return file_name, preview

    def _counts_snapshot() -> tuple[int, int, int, int]:
        active = max(0, started_count - completed_count)
        retrying = len(retrying_chunks)
        effective_workers = max(1, workers - retrying)
        return active, in_flight_count, retrying, effective_workers

    timeout_cfg = httpx.Timeout(
        connect=15.0,
        read=DEFAULT_TIMEOUT,
        write=30.0,
        pool=15.0,
    )

    async with httpx.AsyncClient(timeout=timeout_cfg) as shared_client:
        async def _ensure_model_ready(target_model: str, stage: str) -> None:
            nonlocal current_loaded_model, dual_instance_active
            if current_loaded_model == target_model and use_native:
                snapshot_same = _loaded_instances_snapshot(base_url, api_key)
                for mk, cnt in snapshot_same.items():
                    if mk != target_model and cnt > 0:
                        _try_unload_model(base_url, api_key, mk)
                        touched_models.add(mk)
                return
            if current_loaded_model == target_model:
                return
            if on_progress:
                active, in_flight, retrying, effective_workers = _counts_snapshot()
                on_progress(
                    completed_count,
                    len(chunks),
                    "model_switch",
                    from_cache_count,
                    stage=stage,
                    from_model=current_loaded_model or "",
                    to_model=target_model,
                    active=active,
                    in_flight=in_flight,
                    retrying=retrying,
                    effective_workers=effective_workers,
                )
            try:
                touched_models.add(target_model)
                target_instances = 1
                if use_native:
                    before_snapshot = _loaded_instances_snapshot(base_url, api_key)
                    if current_loaded_model and current_loaded_model != target_model:
                        _try_unload_model(base_url, api_key, current_loaded_model)
                        before_snapshot = _loaded_instances_snapshot(base_url, api_key)
                    for mk, cnt in before_snapshot.items():
                        if mk != target_model and cnt > 0:
                            _try_unload_model(base_url, api_key, mk)
                            touched_models.add(mk)
                    before_snapshot = _loaded_instances_snapshot(base_url, api_key)

                    current_count = before_snapshot.get(target_model, 0)
                    if current_count > target_instances and low_vram_mode:
                        _try_unload_model(base_url, api_key, target_model)
                        current_count = 0

                    from lm_studio_api import V1_MODELS_LOAD, v1_url

                    load_url = v1_url(_lmstudio_root(base_url), V1_MODELS_LOAD)
                    headers = {"Content-Type": "application/json", **_auth_headers(api_key)}
                    load_body: dict[str, Any] = {"model": target_model}
                    if max_context_tokens and max_context_tokens > 0:
                        load_body["context_length"] = int(max_context_tokens)
                    to_load = max(0, target_instances - current_count)
                    for _ in range(to_load):
                        rr = await shared_client.post(
                            load_url, json=load_body, headers=headers,
                        )
                        if rr.status_code >= 400:
                            logger.warning(
                                "Native load failed for %s on stage %s: HTTP %s %s",
                                target_model,
                                stage,
                                rr.status_code,
                                sanitize_for_log(rr.text[:220]),
                            )
                    after_snapshot = _loaded_instances_snapshot(base_url, api_key)
                    logger.info(
                        "Model lifecycle stage=%s target=%s loaded_before=%s loaded_after=%s",
                        stage,
                        target_model,
                        before_snapshot,
                        after_snapshot,
                    )
                await warm_up(
                    base_url,
                    api_key,
                    target_model,
                    semaphore,
                    api_mode=api_mode,
                    client=shared_client,
                )
                if use_native:
                    ids = _extract_loaded_instance_ids(base_url, api_key, target_model)
                    model_instance_pool[target_model] = ids
                    if stage == "text_map":
                        dual_instance_active = target_instances > 1 and len(ids) >= 2
                    if on_progress:
                        active, in_flight, retrying, effective_workers = _counts_snapshot()
                        on_progress(
                            completed_count,
                            len(chunks),
                            "instance_pool",
                            from_cache_count,
                            model=target_model,
                            stage=stage,
                            instance_ids=ids,
                            instances_loaded=len(ids),
                            active=active,
                            in_flight=in_flight,
                            retrying=retrying,
                            effective_workers=effective_workers,
                        )
                current_loaded_model = target_model
            except Exception as exc:
                logger.warning(
                    "Warm-up failed for %s on stage %s: %s(%s)",
                    target_model,
                    stage,
                    type(exc).__name__,
                    sanitize_for_log(str(exc)),
                )

        def _cleanup_models_after_run() -> None:
            if not use_native:
                return
            for mk in sorted(touched_models):
                try:
                    _try_unload_model(base_url, api_key, mk)
                except Exception:
                    continue

        async def process_chunk(i: int, chunk: str, *, map_llm: str, is_vision: bool) -> str:
            nonlocal from_cache_count, failed_count, completed_count, started_count
            nonlocal retry_count, server_5xx_count, pause_until, in_flight_count, dual_instance_active
            file_name, preview = _chunk_debug(chunk)
            if job_id:
                cached = get_cached_response(job_id, i)
                if cached is not None:
                    from_cache_count += 1
                    completed_count += 1
                    if on_progress:
                        done = completed_count
                        cache = from_cache_count
                        active, in_flight, retrying, effective_workers = _counts_snapshot()
                        on_progress(
                            done, len(chunks), "map", cache,
                            active=active,
                            in_flight=in_flight,
                            retrying=retrying,
                            effective_workers=effective_workers,
                            chunk_idx=i + 1,
                            file=file_name,
                            preview=preview,
                        )
                    return cached

            if on_progress:
                started_count += 1
                started = started_count
                cache = from_cache_count
                active, in_flight, retrying, effective_workers = _counts_snapshot()
                on_progress(
                    started, len(chunks), "map_started", cache,
                    active=active,
                    in_flight=in_flight,
                    retrying=retrying,
                    effective_workers=effective_workers,
                    chunk_idx=i + 1,
                    file=file_name,
                    preview=preview,
                )

            labeled = chunk
            if "[CHUNK_INDEX:" not in chunk.upper():
                labeled = f"[CHUNK_INDEX: {i + 1}]\n{chunk}"

            messages: list[dict[str, Any]]
            if is_vision:
                vision_match = _VISION_FILE_RE.search(labeled)
                if not vision_match:
                    failed_count += 1
                    if on_progress:
                        active, in_flight, retrying, effective_workers = _counts_snapshot()
                        on_progress(
                            completed_count, len(chunks), "map_failed", from_cache_count,
                            active=active,
                            in_flight=in_flight,
                            retrying=retrying,
                            effective_workers=effective_workers,
                            chunk_idx=i + 1,
                            file=file_name,
                            preview=preview,
                            error_kind="vision_marker_missing",
                            error="VISION chunk has no marker",
                        )
                    completed_count += 1
                    return ""
                vpath = Path(vision_match.group(1).strip().strip('"'))
                if not vpath.is_file():
                    failed_count += 1
                    logger.error("Vision chunk: file not found: %s", vpath)
                    if on_progress:
                        active, in_flight, retrying, effective_workers = _counts_snapshot()
                        on_progress(
                            completed_count, len(chunks), "map_failed", from_cache_count,
                            active=active,
                            in_flight=in_flight,
                            retrying=retrying,
                            effective_workers=effective_workers,
                            chunk_idx=i + 1,
                            file=file_name,
                            preview=preview,
                            error_kind="vision_file_missing",
                            error=str(vpath),
                        )
                    completed_count += 1
                    return ""
                messages = _build_vision_map_messages(
                    user_query, i + 1, file_name, vpath, language_hint=language_hint,
                )
                if on_progress:
                    active, in_flight, retrying, effective_workers = _counts_snapshot()
                    on_progress(
                        completed_count, len(chunks), "vision_map", from_cache_count,
                        active=active,
                        in_flight=in_flight,
                        retrying=retrying,
                        effective_workers=effective_workers,
                        chunk_idx=i + 1,
                        file=file_name,
                        preview=preview,
                        vision_model=map_llm,
                    )
            else:
                from parser import count_tokens as _ct
                # Extract authoritative file path for explicit injection into prompt
                _auth_fp = _extract_file_path_from_chunk(labeled) or ""
                _fp_line = (
                    f'CRITICAL: поля "file" в JSON (верхний уровень и все evidence_refs) '
                    f'ДОЛЖНЫ быть равны ровно "{_auth_fp}" — это значение из [FILE_PATH:] заголовка.\n'
                    if _auth_fp else ""
                )
                prompt_prefix = (
                    f"{effective_query}\n\n"
                    f"{language_hint}\n\n"
                    "Обработай только этот фрагмент. Верни JSON по схеме из system prompt.\n"
                    f"Поле chunk_index в JSON должно быть равно {i + 1}.\n"
                    f"{_fp_line}\n"
                )
                # Ensure the full MAP prompt fits within the model's context window.
                # Reserve: system_prompt tokens + prefix tokens + 512 overhead + 4096 output budget.
                if max_context_tokens and max_context_tokens > 0:
                    _map_output_reserve = 4096
                    overhead = _ct(SYSTEM_PROMPT_MAP) + _ct(prompt_prefix) + 512 + _map_output_reserve
                    chunk_budget = max(200, max_context_tokens - overhead)
                    if _ct(labeled) > chunk_budget:
                        labeled = _truncate_text_to_tokens(labeled, chunk_budget)
                        logger.warning(
                            "MAP chunk %d truncated to fit model context (%d tokens max)",
                            i + 1, chunk_budget,
                        )
                prompt = prompt_prefix + labeled
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_MAP},
                    {"role": "user", "content": prompt},
                ]

            now = time.monotonic()
            if now < pause_until:
                await asyncio.sleep(pause_until - now)
            route = "single"
            instance_id = ""
            call_model = map_llm
            if (
                not is_vision
                and dual_instance_mode
                and workers > 4
                and map_llm == model
                and use_native
            ):
                ids = model_instance_pool.get(map_llm) or []
                if len(ids) >= 2:
                    route = "instanceA" if (i % 2 == 0) else "instanceB"
                    instance_id = ids[0] if route == "instanceA" else ids[1]
                    call_model = instance_id
                    dual_instance_active = True

            chunk_has_backoff = False

            def on_retry_event(attempt_no: int, max_attempts: int, kind: str, delay: float) -> None:
                nonlocal retry_count, chunk_has_backoff
                retry_count += 1
                if not chunk_has_backoff:
                    chunk_has_backoff = True
                    retrying_chunks.add(i)
                if on_progress:
                    active, in_flight, retrying, effective_workers = _counts_snapshot()
                    on_progress(
                        completed_count, len(chunks), "retry_scheduled", from_cache_count,
                        active=active,
                        in_flight=in_flight,
                        retrying=retrying,
                        effective_workers=effective_workers,
                        chunk_idx=i + 1,
                        file=file_name,
                        preview=preview,
                        attempt=attempt_no,
                        max_attempts=max_attempts,
                        error_kind=kind,
                        retry_delay=delay,
                        backoff_slot_freed=True,
                        route=route,
                        instance_id=instance_id,
                    )

            in_flight_count += 1
            try:
                out = await call_llm(
                    messages,
                    call_model,
                    base_url,
                    api_key,
                    semaphore,
                    on_retry=on_retry_event,
                    api_mode=api_mode,
                    client=shared_client,
                )
                result = _extract_results_tag(out)
                parsed_map = _parse_map_json_payload(result)
                if parsed_map is not None:
                    parsed_map = _normalize_map_json(parsed_map)
                    result = json.dumps(parsed_map, ensure_ascii=False)
                if (
                    DUAL_MAP_RESOLVE
                    and result
                    and not is_vision
                    and dual_instance_mode
                    and use_native
                ):
                    pool_ids = model_instance_pool.get(map_llm) or []
                    if len(pool_ids) >= 2:
                        alt_id = pool_ids[1] if call_model == pool_ids[0] else pool_ids[0]
                        try:
                            out_b = await call_llm(
                                messages,
                                alt_id,
                                base_url,
                                api_key,
                                semaphore,
                                on_retry=on_retry_event,
                                api_mode=api_mode,
                                client=shared_client,
                            )
                            result_b = _extract_results_tag(out_b)
                            parsed_b = _parse_map_json_payload(result_b)
                            if parsed_b is not None:
                                parsed_b = _normalize_map_json(parsed_b)
                                result_b = json.dumps(parsed_b, ensure_ascii=False)
                            if result_b:
                                from conflict_resolve import pick_findings_from_dual_worker

                                result = pick_findings_from_dual_worker(result, result_b)
                        except Exception as exc:
                            logger.debug(
                                "Dual MAP resolve skipped for chunk %s: %s",
                                i + 1,
                                sanitize_for_log(str(exc)),
                            )
                server_5xx_count = 0
            except Exception as exc:
                failed_count += 1
                error_kind = (
                    "server_5xx" if isinstance(exc, httpx.HTTPStatusError)
                    and exc.response is not None
                    and exc.response.status_code in (500, 502, 503)
                    else "client_4xx" if isinstance(exc, httpx.HTTPStatusError)
                    and exc.response is not None
                    and 400 <= exc.response.status_code < 500
                    else "timeout" if isinstance(exc, httpx.TimeoutException)
                    else "empty_content" if isinstance(exc, RuntimeError)
                    else "http_error"
                )
                error_classifier = "unknown"
                if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None and exc.response.status_code == 400:
                    try:
                        error_classifier = _classify_http_400(exc.response.text)
                    except Exception:
                        error_classifier = "unknown"
                if error_kind == "server_5xx":
                    server_5xx_count += 1
                    if server_5xx_count >= SERVER_5XX_CIRCUIT_BREAKER_THRESHOLD:
                        pause_until = time.monotonic() + SERVER_5XX_CIRCUIT_BREAKER_PAUSE_SECONDS
                        server_5xx_count = 0
                        if on_progress:
                            active, in_flight, retrying, effective_workers = _counts_snapshot()
                            on_progress(
                                completed_count, len(chunks), "circuit_breaker", from_cache_count,
                                active=active,
                                in_flight=in_flight,
                                retrying=retrying,
                                effective_workers=effective_workers,
                                pause_seconds=SERVER_5XX_CIRCUIT_BREAKER_PAUSE_SECONDS,
                            )
                logger.error(
                    "Chunk %s/%s failed permanently: %s(%s); skipping",
                    i + 1,
                    len(chunks),
                    type(exc).__name__,
                    sanitize_for_log(str(exc)),
                )
                result = ""
                if on_progress:
                    on_progress(
                        completed_count, len(chunks), "map_failed", from_cache_count,
                        active=max(0, started_count - completed_count),
                        in_flight=in_flight_count,
                        retrying=len(retrying_chunks),
                        effective_workers=max(1, workers - len(retrying_chunks)),
                        chunk_idx=i + 1,
                        file=file_name,
                        preview=preview,
                        error_kind=error_kind,
                        error_classifier=error_classifier,
                        error=sanitize_for_log(str(exc)),
                        route=route,
                        instance_id=instance_id,
                    )
            finally:
                in_flight_count = max(0, in_flight_count - 1)
                if chunk_has_backoff:
                    retrying_chunks.discard(i)

            if result and job_id:
                set_cached_response(job_id, i, result)
            completed_count += 1
            if on_progress:
                done = completed_count
                cache = from_cache_count
                active, in_flight, retrying, effective_workers = _counts_snapshot()
                on_progress(
                    done, len(chunks), "map", cache,
                    active=active,
                    in_flight=in_flight,
                    retrying=retrying,
                    effective_workers=effective_workers,
                    chunk_idx=i + 1,
                    file=file_name,
                    preview=preview,
                    route=route,
                    instance_id=instance_id,
                )
            return result

        map_results_by_index: dict[int, str] = {}
        if job_id:
            for i in range(len(chunks)):
                cached = get_cached_response(job_id, i)
                if cached:
                    map_results_by_index[i] = cached
                    from_cache_count += 1
                    completed_count += 1
            if from_cache_count and on_progress:
                active, in_flight, retrying, effective_workers = _counts_snapshot()
                on_progress(
                    completed_count,
                    len(chunks),
                    "map_resume",
                    from_cache_count,
                    active=active,
                    in_flight=in_flight,
                    retrying=retrying,
                    effective_workers=effective_workers,
                )

        text_indices = [i for i, c in enumerate(chunks) if _VISION_FILE_RE.search(c) is None]
        vision_indices = [i for i, c in enumerate(chunks) if _VISION_FILE_RE.search(c) is not None]

        def _map_batch_size() -> int:
            raw = os.getenv("NOCTURNE_MAP_BATCH_SIZE", "").strip()
            if raw.isdigit():
                return max(1, int(raw))
            return max(1, workers) * 4

        async def _run_phase(indices: list[int], *, phase_name: str, phase_model: str, is_vision: bool) -> None:
            nonlocal stop_triggered
            pending = [i for i in indices if i not in map_results_by_index]
            if not pending:
                return
            if _stopped():
                stop_triggered = True
                return
            if on_progress:
                active, in_flight, retrying, effective_workers = _counts_snapshot()
                on_progress(
                    completed_count,
                    len(chunks),
                    "model_phase",
                    from_cache_count,
                    phase_name=phase_name,
                    model=phase_model,
                    chunks_count=len(pending),
                    active=active,
                    in_flight=in_flight,
                    retrying=retrying,
                    effective_workers=effective_workers,
                )
            await _ensure_model_ready(phase_model, phase_name)
            batch_sz = _map_batch_size()
            for offset in range(0, len(pending), batch_sz):
                if _stopped():
                    stop_triggered = True
                    return
                batch = pending[offset : offset + batch_sz]
                phase_results = await asyncio.gather(
                    *[
                        process_chunk(
                            i,
                            chunks[i],
                            map_llm=phase_model,
                            is_vision=is_vision,
                        )
                        for i in batch
                    ],
                    return_exceptions=False,
                )
                for idx, result in zip(batch, phase_results):
                    map_results_by_index[idx] = result
                if on_progress:
                    active, in_flight, retrying, effective_workers = _counts_snapshot()
                    on_progress(
                        completed_count,
                        len(chunks),
                        "map_batch",
                        from_cache_count,
                        phase_name=phase_name,
                        batch_done=offset + len(batch),
                        batch_total=len(pending),
                        active=active,
                        in_flight=in_flight,
                        retrying=retrying,
                        effective_workers=effective_workers,
                    )

        # ── META-PROMPT GENERATION (before MAP) ───────────────────────
        # When composer_model is set and we have enough chunks, ask the
        # composer to generate a precise analysis directive that will be
        # used as the effective query throughout MAP / merge / reduce.
        if composer_model and len(chunks) > 5:
            if on_progress:
                on_progress(0, len(chunks), "meta_plan", from_cache_count)
            try:
                await _ensure_model_ready(reduce_model, "meta_plan")
                sample_hdrs = _extract_chunk_headers(chunks)
                meta_directive = await generate_meta_prompt(
                    user_query=user_query,
                    sample_headers=sample_hdrs,
                    base_url=base_url,
                    api_key=api_key,
                    model=reduce_model,
                    semaphore=semaphore,
                    api_mode=api_mode,
                    client=shared_client,
                    language_hint=language_hint,
                )
                if meta_directive and meta_directive != user_query:
                    effective_query = meta_directive
                    logger.info(
                        "META-PLANNER: effective_query updated (%d chars)",
                        len(effective_query),
                    )
                if on_progress:
                    on_progress(
                        0, len(chunks), "meta_plan_done", from_cache_count,
                        preview=effective_query[:300],
                    )
            except Exception as _meta_exc:
                logger.warning("Meta-plan step failed: %s", _meta_exc)

        map_text_indices = list(text_indices)
        scout_skipped = 0
        if (
            scout_mode
            and text_indices
            and len(text_indices) >= max(2, scout_min_chunks)
        ):
            scout_job = f"{job_id}:scout" if job_id else None
            if on_progress:
                on_progress(
                    0, len(text_indices), "scout", from_cache_count,
                    scout_total=len(text_indices),
                )
            await _ensure_model_ready(scout_llm, "scout")

            async def _scout_one(i: int) -> tuple[int, float]:
                chunk = chunks[i]
                if scout_job:
                    cached = get_cached_response(scout_job, i)
                    if cached is not None:
                        parsed = _parse_scout_json_payload(cached)
                        if parsed is not None:
                            return i, scout_relevance_score(parsed)
                from parser import count_tokens as _ct_scout

                labeled = chunk
                if "[CHUNK_INDEX:" not in chunk.upper():
                    labeled = f"[CHUNK_INDEX: {i + 1}]\n{chunk}"
                query_line = effective_query
                if max_context_tokens and max_context_tokens > 0:
                    overhead = _ct_scout(SYSTEM_PROMPT_SCOUT) + _ct_scout(query_line) + 400
                    chunk_budget = max(150, max_context_tokens - overhead)
                    if _ct_scout(labeled) > chunk_budget:
                        labeled = _truncate_text_to_tokens(labeled, chunk_budget)
                prompt = (
                    f"{query_line}\n\n"
                    f"{language_hint}\n\n"
                    "Оцени релевантность ТОЛЬКО этого фрагмента запросу. Верни JSON по схеме scout.\n\n"
                    f"{labeled}"
                )
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_SCOUT},
                    {"role": "user", "content": prompt},
                ]
                try:
                    out = await call_llm(
                        messages,
                        scout_llm,
                        base_url,
                        api_key,
                        semaphore,
                        max_tokens=320,
                        max_retries=2,
                        allow_empty_content=False,
                        api_mode=api_mode,
                        client=shared_client,
                    )
                    result = _extract_results_tag(out)
                    parsed = _parse_scout_json_payload(result)
                    if parsed is None:
                        return i, 1.0
                    if scout_job:
                        set_cached_response(scout_job, i, json.dumps(parsed, ensure_ascii=False))
                    return i, scout_relevance_score(parsed)
                except Exception as exc:
                    logger.warning(
                        "Scout chunk %s failed: %s — will run full MAP",
                        i + 1,
                        sanitize_for_log(str(exc)[:120]),
                    )
                    return i, 1.0

            scout_results = await asyncio.gather(
                *[_scout_one(i) for i in text_indices],
                return_exceptions=False,
            )
            score_map = {idx: sc for idx, sc in scout_results}
            map_text_indices, skipped_list = filter_indices_by_scout_scores(
                text_indices, score_map, scout_relevance_threshold,
            )
            scout_skipped = len(skipped_list)
            # Пропущенные scout'ом чанки засчитываем как завершённые, иначе
            # прогресс-бар MAP никогда не дойдёт до 100% при включённом scout.
            completed_count += scout_skipped
            if on_progress:
                on_progress(
                    len(map_text_indices),
                    len(text_indices),
                    "scout_done",
                    from_cache_count,
                    scout_deep=len(map_text_indices),
                    scout_skipped=scout_skipped,
                    scout_threshold=scout_relevance_threshold,
                )
            logger.info(
                "Scout phase: total=%s deep_map=%s skipped=%s threshold=%.2f",
                len(text_indices),
                len(map_text_indices),
                scout_skipped,
                scout_relevance_threshold,
            )

        await _run_phase(map_text_indices, phase_name="text_map", phase_model=model, is_vision=False)
        await _run_phase(vision_indices, phase_name="vision_map", phase_model=vision_llm, is_vision=True)

        if stop_triggered or _stopped():
            # Кооперативная остановка: каждый завершённый чанк уже в SQLite-кэше
            # (set_cached_response). REDUCE не запускаем, job помечаем paused —
            # на «Продолжить» прогон возобновится с того же job_id.
            if job_id:
                try:
                    from cache import mark_job_paused

                    mark_job_paused(job_id)
                except Exception:
                    pass
            if on_progress:
                try:
                    on_progress(completed_count, len(chunks), "stopped", from_cache_count)
                except TypeError:
                    pass
            _cleanup_models_after_run()
            logger.info(
                "run_map_reduce stopped by user: %s/%s chunks done",
                completed_count,
                len(chunks),
            )
            return (
                f"(Остановлено пользователем. MAP-чанков обработано: "
                f"{completed_count}/{len(chunks)}. Прогресс сохранён в кэше — "
                "нажмите «Продолжить» для возобновления.)"
            )

        from map_result_store import MapResultStore, normalize_spill_threshold

        use_result_store = bool(job_id) and len(chunks) >= normalize_spill_threshold()
        result_store: MapResultStore | None = None
        file_groups_prebuilt: dict[str, list[str]] | None = None
        if use_result_store:
            result_store = MapResultStore(job_id or "anon", chunk_count=len(chunks))
        map_results_raw: list[str] = []
        for i in range(len(chunks)):
            raw_r = map_results_by_index.get(i, "")
            if not raw_r and job_id:
                raw_r = get_cached_response(job_id, i) or ""
            if not raw_r or not raw_r.strip():
                if not use_result_store:
                    map_results_raw.append(raw_r)
                continue
            chunk_text_for_fix = chunks[i] if i < len(chunks) else ""
            norm = _to_normalized_map_json_text(raw_r, chunk_text=chunk_text_for_fix) or raw_r
            if use_result_store and result_store is not None:
                inner = _extract_results_tag(norm) if "<results>" in norm.lower() else norm
                parsed_norm = _parse_map_json_payload(inner)
                result_store.add(i, norm, parsed=parsed_norm)
            else:
                map_results_raw.append(norm)

        if use_result_store and result_store is not None:
            file_groups_prebuilt = result_store.build_file_groups()
            if result_store._spilled:
                map_results = []
                map_results_count = result_store.nonempty_count
            else:
                map_results = list(result_store.iter_nonempty())
                map_results_count = len(map_results)
        else:
            map_results = [r for r in map_results_raw if r and r.strip()]
            map_results_count = len(map_results)

        if map_results_count == 0:
            if result_store is not None:
                result_store.cleanup()
            _cleanup_models_after_run()
            return (
                f"(Нет данных по запросу. "
                f"Провалено чанков: {failed_count}/{len(chunks)}. "
                f"Проверьте доступность LM Studio и таймаут.)"
            )
        if failed_count:
            logger.warning(
                "MAP finished: %s/%s chunks OK, %s failed",
                map_results_count,
                len(chunks),
                failed_count,
            )

        elapsed_s = max(0.001, time.monotonic() - started_at)
        if use_result_store and result_store is not None:
            metrics = dict(result_store.metrics)
        else:
            metrics = _map_metrics_from_results(map_results)
        if on_progress:
            on_progress(
                map_results_count, len(chunks), "summary", from_cache_count,
                retries=retry_count,
                failed=failed_count,
                ok=map_results_count,
                elapsed_s=elapsed_s,
                chunks_per_min=(len(chunks) / elapsed_s) * 60.0,
                workers=workers,
                map_model=model,
                vision_model=vision_llm,
                reduce_model=reduce_model,
                text_chunks=len(text_indices),
                text_map_chunks=len(map_text_indices),
                scout_skipped=scout_skipped,
                scout_mode=scout_mode,
                vision_chunks=len(vision_indices),
                low_vram_sequential=low_vram_mode,
                api_mode=api_mode,
                instances_loaded=max(1, len(model_instance_pool.get(model) or [])),
                dual_instance_active=dual_instance_active,
            )
            on_progress(
                map_results_count, len(chunks), "map_metrics", from_cache_count,
                relevant_chunks=metrics["relevant_chunks"],
                findings_count=metrics["findings_count"],
                evidence_refs_count=metrics["evidence_refs_count"],
            )

        if file_groups_prebuilt is not None:
            file_groups = file_groups_prebuilt
            merge_inputs = [r for group in file_groups.values() for r in group]
            relevant_count = len(merge_inputs)
        else:
            relevant_for_merge: list[str] = []
            for _r in map_results:
                _inner = _extract_results_tag(_r) if "<results>" in _r.lower() else _r
                _parsed = _parse_map_json_payload(_inner)
                if _parsed is None:
                    relevant_for_merge.append(_r)
                elif not _parsed.get("no_relevant_data") and _parsed.get("findings"):
                    relevant_for_merge.append(_r)
            merge_inputs = relevant_for_merge if relevant_for_merge else map_results
            relevant_count = len(relevant_for_merge)
            from collections import defaultdict as _defaultdict

            file_groups = _defaultdict(list)
            for _r in merge_inputs:
                _inner = _extract_results_tag(_r) if "<results>" in _r.lower() else _r
                _parsed = _parse_map_json_payload(_inner)
                _fkey = (_parsed.get("file") or "unknown") if _parsed else "unknown"
                file_groups[_fkey].append(_r)
            file_groups = dict(file_groups)

        logger.info(
            "Merge inputs: %s relevant (with findings) out of %s total MAP results",
            relevant_count,
            map_results_count,
        )
        if on_progress:
            try:
                on_progress(
                    relevant_count, map_results_count, "map_relevant",
                    from_cache_count, relevant_files=relevant_count,
                )
            except TypeError:
                pass

        ctx_limit = max_context_tokens if max_context_tokens is not None else CONTEXT_FALLBACK
        # Резерв под ответ REDUCE не должен съедать всё окно: на моделях ~8k
        # фикс 4096–8192 не оставлял места под входной JSON. Ограничиваем ~1/3
        # контекста, оставляя ~2/3 под входные данные.
        reduce_max_tokens = min(
            8192,
            max(1024, min(dynamic_chunk_size + 2048, ctx_limit // 3)),
        )
        await _ensure_model_ready(reduce_model, "reduce")
        n_file_groups = len(file_groups)
        logger.info("File groups for reduce: %s", n_file_groups)

        refine_used = False
        draft: str = ""

        use_hierarchical = os.getenv("NOCTURNE_HIERARCHICAL_MERGE", "1").strip() != "0"
        if n_file_groups <= 1:
            # ── ОДНА ГРУППА: классический путь (одиночный reduce) ────────────────
            if use_hierarchical and len(merge_inputs) > 5:
                from merge_hierarchy import hierarchical_merge_map_results

                aggregated, _merge_tree = hierarchical_merge_map_results(merge_inputs, chunks)
            else:
                aggregated = _merge_map_json_deterministic(merge_inputs)
            if on_progress:
                try:
                    on_progress(1, 1, "reduce", from_cache_count)
                except TypeError:
                    on_progress(1, 1, "reduce")
            draft = await _reduce_json_to_markdown(
                aggregated,
                effective_query,
                base_url,
                api_key,
                reduce_model,
                semaphore,
                reduce_max_tokens,
                api_mode,
                language_hint,
                shared_client,
                max_context_tokens=ctx_limit,
            )
            if _reduce_needs_refine(draft):
                refine_used = True
                if on_progress:
                    on_progress(1, 1, "reduce_refine", from_cache_count)
                refined = await _refine_final_report(
                    draft,
                    effective_query,
                    base_url,
                    api_key,
                    reduce_model,
                    semaphore,
                    reduce_max_tokens,
                    api_mode,
                    language_hint,
                    shared_client,
                    max_context_tokens=ctx_limit,
                )
                if refined and len(refined) > len(draft):
                    draft = refined
            if _reduce_needs_refine(draft):
                logger.warning(
                    "Reduce model '%s' produced incomplete output — generating fallback report.",
                    reduce_model,
                )
                draft = _build_fallback_report(
                    aggregated,
                    effective_query,
                    map_results if map_results else merge_inputs,
                    language_hint,
                )
        else:
            # ── НЕСКОЛЬКО ГРУПП: посекционный reduce + синтез ───────────────────
            # Сначала пакуем мелкие файловые группы по бюджету токенов, чтобы
            # число REDUCE-вызовов росло по объёму данных, а не по числу файлов.
            reduce_groups = _coalesce_file_groups(file_groups, max(4000, ctx_limit // 2))
            if len(reduce_groups) < n_file_groups:
                logger.info(
                    "REDUCE groups coalesced: %s files -> %s groups",
                    n_file_groups,
                    len(reduce_groups),
                )
            # Для каждой группы — независимый параллельный reduce-вызов,
            # затем финальный синтез секций в единый отчёт.
            sections = await _section_reduce_groups(
                file_groups=reduce_groups,
                user_query=effective_query,
                base_url=base_url,
                api_key=api_key,
                reduce_model=reduce_model,
                semaphore=semaphore,
                reduce_max_tokens=reduce_max_tokens,
                api_mode=api_mode,
                language_hint=language_hint,
                client=shared_client,
                max_context_tokens=ctx_limit,
                on_progress=on_progress,
            )
            draft = await _synthesize_final_report(
                sections=sections,
                user_query=effective_query,
                base_url=base_url,
                api_key=api_key,
                reduce_model=reduce_model,
                semaphore=semaphore,
                reduce_max_tokens=reduce_max_tokens,
                api_mode=api_mode,
                language_hint=language_hint,
                client=shared_client,
                max_context_tokens=ctx_limit,
                on_progress=on_progress,
            )
            if _reduce_needs_refine(draft):
                logger.warning(
                    "Synthesize produced incomplete output — using fallback concatenation.",
                )
                draft = _concatenate_sections_fallback(sections, effective_query, language_hint)

        sections_n = _count_report_sections(draft)
        final_report, val_warnings = _validate_final_report(draft, metrics)
        if os.getenv("NOCTURNE_ENABLE_VERIFIER", "").strip() == "1":
            from merge_hierarchy import hierarchical_merge_map_results, top_evidence_from_tree

            ev_index: list[dict[str, str]] = []
            if use_hierarchical and len(merge_inputs) > 5:
                try:
                    _, merge_tree = hierarchical_merge_map_results(merge_inputs, chunks)
                    ev_index = top_evidence_from_tree(merge_tree, limit=40)
                except Exception:
                    ev_index = []
            verifier_model = scout_llm
            final_report, v_warn = await verify_report_with_evidence(
                final_report,
                ev_index,
                user_query,
                base_url,
                api_key,
                verifier_model,
                semaphore,
                api_mode=api_mode,
                client=shared_client,
            )
            val_warnings = list(val_warnings) + v_warn
        if on_progress:
            on_progress(
                1,
                1,
                "quality_metrics",
                from_cache_count,
                covered_chunks=metrics["relevant_chunks"],
                evidence_count=metrics["evidence_refs_count"],
                final_sections_count=sections_n,
                refine_used=refine_used,
                validation_warnings=val_warnings,
            )
        if metrics_row_id:
            try:
                from metrics import record_run_finish

                record_run_finish(
                    metrics_row_id,
                    duration_s=time.monotonic() - started_at,
                    chunks_total=len(chunks),
                    chunks_ok=map_results_count,
                    chunks_failed=failed_count,
                    scout_skipped=scout_skipped,
                    retries=retry_count,
                    warnings=val_warnings,
                )
            except Exception:
                pass
        _cleanup_models_after_run()
        if result_store is not None:
            result_store.cleanup()
        if job_id:
            try:
                from cache import mark_job_complete

                mark_job_complete(job_id)
            except Exception:
                pass
        return final_report


def _batch_to_csv_like(batch: Union[list[dict], tuple[list[str], list[dict]]]) -> str:
    """Сериализовать батч: либо list[dict], либо (header, rows)."""
    if isinstance(batch, tuple):
        header, rows = batch
        lines = ["\t".join(str(h) for h in header)]
        for row in rows:
            lines.append("\t".join(str(row.get(k, "")) for k in header))
        return "\n".join(lines)
    if not batch:
        return ""
    keys = list(batch[0].keys())
    lines = ["\t".join(keys)]
    for row in batch:
        lines.append("\t".join(str(row.get(k, "")) for k in keys))
    return "\n".join(lines)


def _parse_table_response(text: str) -> list[dict]:
    raw = _extract_results_tag(text)
    try:
        s = raw.strip()
        if s.startswith("```"):
            s = re.sub(r"^```\w*\n?", "", s)
            s = re.sub(r"\n?```\s*$", "", s)
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return []
    headers = [p.strip() for p in lines[0].split("|")]
    rows = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == len(headers):
            rows.append(dict(zip(headers, parts)))
    return rows


# Батч: либо list[dict] (старый формат), либо tuple[list[str], list[dict]]
TableBatchType = Union[list[list[dict]], list[tuple[list[str], list[dict]]]]


async def run_batching(
    batches: TableBatchType,
    user_query: str,
    base_url: str,
    api_key: str,
    model: str,
    workers: int,
    on_progress: Callable[[int, int], None] | None = None,
    api_mode: str = "native",
) -> pd.DataFrame:
    """Обработка батчей таблицы; батч — list[dict] или (header, rows)."""
    workers = max(1, min(4, workers))
    semaphore = asyncio.Semaphore(workers)
    timeout_cfg = httpx.Timeout(
        connect=15.0,
        read=DEFAULT_TIMEOUT,
        write=30.0,
        pool=15.0,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout_cfg) as shared_client:
            await warm_up(base_url, api_key, model, semaphore, api_mode=api_mode, client=shared_client)

            async def process_batch(i: int, batch: Union[list[dict], tuple[list[str], list[dict]]]) -> list[dict]:
                table_text = _batch_to_csv_like(batch)
                prompt = f"{user_query}\n\nДанные (таблица):\n{table_text}"
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_TABLE},
                    {"role": "user", "content": prompt},
                ]
                out = await call_llm(
                    messages,
                    model,
                    base_url,
                    api_key,
                    semaphore,
                    api_mode=api_mode,
                    client=shared_client,
                )
                if on_progress:
                    on_progress(i + 1, len(batches))
                return _parse_table_response(out)

            results = await asyncio.gather(*[process_batch(i, b) for i, b in enumerate(batches)])
        all_rows: list[dict] = []
        for r in results:
            all_rows.extend(r)
        if not all_rows:
            return pd.DataFrame()
        return pd.DataFrame(all_rows)
    finally:
        if api_mode.strip().lower() == "native":
            _try_unload_model(base_url, api_key, model)
