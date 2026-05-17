# SPDX-License-Identifier: AGPL-3.0-or-later
"""Immutable runtime configuration for a processing run."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunConfig:
    base_url: str
    api_key: str
    chat_model: str
    vision_model: str
    composer_model: str
    scout_model: str
    embedding_model: str
    api_mode: str = "native"
    low_vram_mode: bool = True
    workers: int = 3
    context_budget: int = 8096
    max_chunk_tokens: int = 6000
    max_reduce_input_tokens: int = 24000
    scout_mode: bool = False
    scout_threshold: float = 0.35

    @classmethod
    def from_gui(
        cls,
        *,
        base_url: str,
        api_key: str,
        chat_model: str,
        vision_model: str | None,
        composer_model: str | None,
        scout_model: str | None,
        embedding_model: str,
        api_mode: str,
        low_vram: bool,
        workers: int,
        context_budget: int,
        max_chunk_tokens: int,
        max_reduce_input_tokens: int,
        scout_mode: bool,
        scout_threshold: float,
    ) -> RunConfig:
        chat = chat_model.strip()
        return cls(
            base_url=base_url.strip(),
            api_key=api_key.strip(),
            chat_model=chat,
            vision_model=(vision_model or chat).strip(),
            composer_model=(composer_model or chat).strip(),
            scout_model=(scout_model or chat).strip(),
            embedding_model=embedding_model.strip(),
            api_mode=api_mode.strip().lower(),
            low_vram_mode=bool(low_vram),
            workers=max(1, min(4, workers)),
            context_budget=max(500, context_budget),
            max_chunk_tokens=max(500, max_chunk_tokens),
            max_reduce_input_tokens=max(1000, max_reduce_input_tokens),
            scout_mode=bool(scout_mode),
            scout_threshold=max(0.0, min(1.0, scout_threshold)),
        )
