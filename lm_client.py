# SPDX-License-Identifier: AGPL-3.0-or-later
"""LM Studio / OpenAI-compatible HTTP client helpers."""
from __future__ import annotations

import os
from typing import Any

import httpx

from lm_studio_api import V1_CHAT, V1_MODELS, v1_url
from lmstudio_config import lmstudio_root_url, normalize_lmstudio_base_url

USE_LMSTUDIO_NATIVE_API = os.getenv("NOCTURNE_LMSTUDIO_NATIVE_API", "1").strip() != "0"


def auth_headers(api_key: str) -> dict[str, str]:
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def chat_endpoint(base_url: str, *, native: bool | None = None) -> str:
    use_native = USE_LMSTUDIO_NATIVE_API if native is None else native
    if use_native:
        return v1_url(lmstudio_root_url(base_url), V1_CHAT)
    return normalize_lmstudio_base_url(base_url) + "/chat/completions"


def models_endpoint(base_url: str, *, native: bool | None = None) -> str:
    use_native = USE_LMSTUDIO_NATIVE_API if native is None else native
    if use_native:
        return v1_url(lmstudio_root_url(base_url), V1_MODELS)
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


def extract_chat_response_content(data: dict[str, Any]) -> tuple[str, str]:
    """(message_content, reasoning_content) for OpenAI-compatible and native LM Studio."""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = (choices[0] or {}).get("message", {}) or {}
        content = str(message.get("content") or "")
        reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
        return content, reasoning

    top = data.get("content")
    if isinstance(top, str) and top.strip():
        return top.strip(), ""

    output = data.get("output")
    if isinstance(output, list):
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            typ = str(item.get("type") or "").strip().lower()
            if typ == "message":
                c = str(item.get("content") or "")
                if c:
                    content_parts.append(c)
            elif typ in ("text", "output_text"):
                c = str(item.get("content") or item.get("text") or "")
                if c:
                    content_parts.append(c)
            elif typ == "reasoning":
                r = str(item.get("content") or item.get("text") or "")
                if r:
                    reasoning_parts.append(r)
        return "\n".join(content_parts).strip(), "\n".join(reasoning_parts).strip()
    return "", ""


def is_unsupported_reasoning_response(response: httpx.Response) -> bool:
    if response.status_code != 400:
        return False
    try:
        err_body = response.json()
        if isinstance(err_body, dict):
            err = err_body.get("error")
            if isinstance(err, dict):
                param = str(err.get("param") or "").lower()
                message = str(err.get("message") or "").lower()
                code = str(err.get("code") or "").lower()
                if param == "reasoning" or (
                    "reasoning" in message
                    and code in {"invalid_value", "invalid_request_error", ""}
                ):
                    return True
    except Exception:
        pass
    return "reasoning" in response.text.lower()


def classify_http_400(body: str) -> str:
    low = (body or "").lower()
    if "reasoning" in low:
        return "unsupported_option"
    if "context" in low or "n_ctx" in low or "n_keep" in low:
        return "context_limit"
    if "payload" in low or "invalid" in low:
        return "payload_mismatch"
    return "unknown"
