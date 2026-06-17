# SPDX-License-Identifier: AGPL-3.0-or-later
"""
LM Studio REST API v1 endpoint helpers (0.4+).

Docs: https://lmstudio.ai/docs/api/rest-api
"""
from __future__ import annotations

from typing import Any, Final

import httpx

# Native REST v1 (LM Studio 0.4+)
V1_MODELS: Final = "/api/v1/models"
V1_CHAT: Final = "/api/v1/chat"
V1_MODELS_LOAD: Final = "/api/v1/models/load"
V1_MODELS_UNLOAD: Final = "/api/v1/models/unload"
V1_MODELS_DOWNLOAD: Final = "/api/v1/models/download"
V1_MODELS_DOWNLOAD_STATUS: Final = "/api/v1/models/download/status"

# Legacy metadata (still useful for context fields on some builds)
V0_MODELS: Final = "/api/v0/models"

# OpenAI-compatible (same server)
OPENAI_V1_MODELS: Final = "/v1/models"
OPENAI_V1_CHAT: Final = "/v1/chat/completions"
OPENAI_V1_EMBEDDINGS: Final = "/v1/embeddings"


def v1_url(root: str, path: str) -> str:
    return root.rstrip("/") + path


def is_lm_studio_api_token(api_key: str) -> bool:
    """LM Studio issued tokens (Manage Tokens in Server Settings)."""
    return api_key.strip().lower().startswith("sk-lm-")


def _headers(api_key: str) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def start_model_download(
    root: str,
    api_key: str,
    model: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Запустить загрузку модели через LM Studio (REST v1, 0.4+).

    POST /api/v1/models/download  →  {"job_id": "...", ...}. Скачивание идёт
    асинхронно; прогресс смотри через :func:`get_model_download_status`.
    Возвращает распарсенный JSON ответа (или {"error": "..."} при сбое).
    """
    url = v1_url(root, V1_MODELS_DOWNLOAD)
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json={"model": model}, headers=_headers(api_key))
            if r.status_code >= 400:
                return {"error": f"HTTP {r.status_code}", "detail": r.text[:300]}
            return r.json() if r.content else {}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def get_model_download_status(
    root: str,
    api_key: str,
    job_id: str,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Статус задачи загрузки модели: GET /api/v1/models/download/status/:job_id."""
    url = v1_url(root, V1_MODELS_DOWNLOAD_STATUS) + "/" + str(job_id).strip("/")
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url, headers=_headers(api_key))
            if r.status_code >= 400:
                return {"error": f"HTTP {r.status_code}", "detail": r.text[:300]}
            return r.json() if r.content else {}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
