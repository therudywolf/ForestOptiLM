# SPDX-License-Identifier: AGPL-3.0-or-later
"""
LM Studio REST API v1 endpoint helpers (0.4+).

Docs: https://lmstudio.ai/docs/api/rest-api
"""
from __future__ import annotations

from typing import Final

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
