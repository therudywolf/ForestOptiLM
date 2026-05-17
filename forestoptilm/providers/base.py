# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    base_url: str
    api_key: str
    api_mode: str = "native"


class Provider(ABC):
    @abstractmethod
    def list_models(self) -> list[str]:
        ...

    @abstractmethod
    def chat(self, model: str, messages: list[dict[str, Any]], max_tokens: int = 1024) -> str:
        ...
