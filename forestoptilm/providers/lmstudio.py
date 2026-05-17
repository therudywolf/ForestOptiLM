# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from typing import Any

import httpx

from lm_client import auth_headers, chat_endpoint, fetch_models_http


class LMStudioProvider:
    def __init__(self, config: Any) -> None:
        self.config = config

    def list_models(self) -> list[str]:
        return fetch_models_http(self.config.base_url, self.config.api_key)

    def chat(self, model: str, messages: list[dict[str, Any]], max_tokens: int = 1024) -> str:
        native = self.config.api_mode != "openai"
        url = chat_endpoint(self.config.base_url, native=native)
        headers = {"Content-Type": "application/json", **auth_headers(self.config.api_key)}
        if native:
            payload = {
                "model": model,
                "input": messages[-1].get("content", "") if messages else "",
                "max_output_tokens": max_tokens,
                "temperature": 0,
            }
        else:
            payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
        with httpx.Client(timeout=120.0) as client:
            r = client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        choices = data.get("choices") or []
        if choices:
            return str((choices[0].get("message") or {}).get("content") or "")
        return str(data.get("output", ""))
