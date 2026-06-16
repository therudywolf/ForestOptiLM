# SPDX-License-Identifier: AGPL-3.0-or-later
"""Detect and control LM Studio reasoning / thinking models (REST API v1)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

# Models that returned HTTP 400 for reasoning=off|on (param unsupported).
_MODELS_NO_REASONING_PARAM: set[str] = set()
_CATALOG_CAPS: dict[str, Any] = {}

REASONING_ID_HEURISTIC: Final = re.compile(
    r"(?i)"
    r"(?:^|[/_\-])"
    r"(?:"
    r"qwen3[\.\-_]|qwen-?3[\.\-_]|"
    r"deepseek[-_]?r\d|deepseek-r|"
    r"glm[-_]?4\.?[5-9][-_]?air|glm[-_]?4\.?6|"
    r"gemma[-_]?[4-9]|"  # gemma-4+ are thinking models (gemma-2/3 are not)
    r"magistral|"
    r"gpt[-_]?oss|"
    r"reasoning|thinker|thinking|"
    r"o1[-_]|o3[-_]|o4[-_]"
    r")"
)


@dataclass(frozen=True)
class ReasoningCapability:
    allowed_options: tuple[str, ...]
    default: str

    def supports_off(self) -> bool:
        return "off" in self.allowed_options

    def supports_on(self) -> bool:
        return "on" in self.allowed_options


def parse_reasoning_capability(capabilities: Any) -> ReasoningCapability | None:
    """Parse capabilities.reasoning from GET /api/v1/models."""
    if not isinstance(capabilities, dict):
        return None
    raw = capabilities.get("reasoning")
    if raw is True:
        return ReasoningCapability(("off", "on"), "on")
    if not isinstance(raw, dict):
        return None
    opts = raw.get("allowed_options")
    if not isinstance(opts, list) or not opts:
        return None
    allowed = tuple(str(o).strip().lower() for o in opts if str(o).strip())
    if not allowed:
        return None
    default = str(raw.get("default") or allowed[0]).strip().lower()
    return ReasoningCapability(allowed, default)


def model_id_suggests_reasoning(model_id: str) -> bool:
    return bool(REASONING_ID_HEURISTIC.search(model_id or ""))


def model_has_reasoning_capability(
    model_id: str,
    capabilities: Any | None = None,
) -> bool:
    caps = capabilities if capabilities is not None else get_capabilities(model_id)
    parsed = parse_reasoning_capability(caps)
    if parsed is not None:
        return parsed.supports_on() or parsed.supports_off()
    return model_id_suggests_reasoning(model_id)


def refresh_model_catalog_cache(catalog: list[dict[str, Any]]) -> None:
    _CATALOG_CAPS.clear()
    for m in catalog:
        key = str(m.get("key") or m.get("id") or "").strip()
        if key:
            _CATALOG_CAPS[key] = m.get("capabilities")


def get_capabilities(model_id: str) -> Any:
    return _CATALOG_CAPS.get(model_id)


def mark_no_reasoning_param(model_id: str) -> None:
    _MODELS_NO_REASONING_PARAM.add(model_id)


def is_no_reasoning_param(model_id: str) -> bool:
    return model_id in _MODELS_NO_REASONING_PARAM


def clear_no_reasoning_param_cache() -> None:
    _MODELS_NO_REASONING_PARAM.clear()


def native_reasoning_payload(
    model_id: str,
    *,
    prefer_off: bool = True,
    capabilities: Any | None = None,
) -> dict[str, str]:
    """
    Extra fields for POST /api/v1/chat.

    For MAP/REDUCE we prefer reasoning=off so the model emits <results> in message,
    not chain-of-thought in a separate reasoning channel.
    """
    if is_no_reasoning_param(model_id):
        return {}
    caps = capabilities if capabilities is not None else get_capabilities(model_id)
    parsed = parse_reasoning_capability(caps)
    if parsed:
        if prefer_off and parsed.supports_off():
            return {"reasoning": "off"}
        return {}
    if prefer_off and model_id_suggests_reasoning(model_id):
        return {"reasoning": "off"}
    return {}


def list_reasoning_models(catalog: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for m in catalog:
        key = str(m.get("key") or m.get("id") or "").strip()
        if not key:
            continue
        if model_has_reasoning_capability(key, m.get("capabilities")):
            out.append(key)
    return sorted(set(out))
