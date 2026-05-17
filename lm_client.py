# SPDX-License-Identifier: AGPL-3.0-or-later
"""LM Studio / OpenAI-compatible HTTP client helpers."""
from __future__ import annotations

import os
from typing import Any

import httpx

from lmstudio_config import lmstudio_root_url, normalize_lmstudio_base_url, sanitize_for_log

USE_LMSTUDIO_NATIVE_API = os.getenv("NOCTURNE_LMSTUDIO_NATIVE_API", "1").strip() != "0"


def auth_headers(api_key: str) -> dict[str, str]:
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def chat_endpoint(base_url: str, *, native: bool | None = None) -> str:
    use_native = USE_LMSTUDIO_NATIVE_API if native is None else native
    if use_native:
        return lmstudio_root_url(base_url) + "/api/v1/chat"
    return normalize_lmstudio_base_url(base_url) + "/chat/completions"


def models_endpoint(base_url: str, *, native: bool | None = None) -> str:
    use_native = USE_LMSTUDIO_NATIVE_API if native is None else native
    if use_native:
        return lmstudio_root_url(base_url) + "/api/v1/models"
    return normalize_lmstudio_base_url(base_url) + "/models"


def embeddings_endpoint(base_url: str) -> str:
    return normalize_lmstudio_base_url(base_url) + "/embeddings"


def extract_model_ids(data: Any) -> list[str]:
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
        elif isinstance(item, dict):
            mid = item.get("id") or item.get("key")
            if mid:
                ids.append(str(mid))
    return ids


def fetch_models_http(base_url: str, api_key: str, timeout: float = 15.0) -> list[str]:
    headers = auth_headers(api_key)
    for url in (models_endpoint(base_url, native=True), models_endpoint(base_url, native=False)):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.get(url, headers=headers)
                r.raise_for_status()
                return extract_model_ids(r.json())
        except Exception:
            continue
    return []


def classify_http_400(body: str) -> str:
    low = (body or "").lower()
    if "reasoning" in low:
        return "unsupported_option"
    if "context" in low or "n_ctx" in low or "n_keep" in low:
        return "context_limit"
    if "payload" in low or "invalid" in low:
        return "payload_mismatch"
    return "unknown"
