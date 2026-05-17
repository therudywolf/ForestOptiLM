# SPDX-License-Identifier: AGPL-3.0-or-later
"""Structured processing errors for GUI and CLI."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ErrorCode(str, Enum):
    SERVER_UNAVAILABLE = "server_unavailable"
    MODEL_MISSING = "model_missing"
    UNSUPPORTED_OPTION = "unsupported_option"
    CONTEXT_OVERFLOW = "context_overflow"
    EMBEDDING_MISMATCH = "embedding_mismatch"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ProcessingError(Exception):
    code: ErrorCode
    message: str
    detail: str = ""

    def __str__(self) -> str:
        if self.detail:
            return f"{self.code.value}: {self.message} ({self.detail})"
        return f"{self.code.value}: {self.message}"

    def user_hint(self) -> str:
        hints = {
            ErrorCode.SERVER_UNAVAILABLE: "Проверьте, что LM Studio запущен и URL верный.",
            ErrorCode.MODEL_MISSING: "Загрузите модель в LM Studio или выберите другую.",
            ErrorCode.UNSUPPORTED_OPTION: "Смените API mode (native/openai) или отключите reasoning.",
            ErrorCode.CONTEXT_OVERFLOW: "Уменьшите chunk size или выберите модель с большим контекстом.",
            ErrorCode.EMBEDDING_MISMATCH: "Выберите embedding-модель в списке.",
        }
        return hints.get(self.code, self.message)


def classify_exception(exc: BaseException) -> ProcessingError:
    import httpx

    if isinstance(exc, httpx.ConnectError):
        return ProcessingError(ErrorCode.SERVER_UNAVAILABLE, "Не удалось подключиться к серверу", str(exc))
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response is not None and exc.response.status_code == 400:
            return ProcessingError(ErrorCode.UNSUPPORTED_OPTION, "HTTP 400", exc.response.text[:200])
        if exc.response is not None and exc.response.status_code >= 500:
            return ProcessingError(ErrorCode.SERVER_UNAVAILABLE, f"HTTP {exc.response.status_code}", "")
    text = str(exc).lower()
    if "context" in text or "n_ctx" in text:
        return ProcessingError(ErrorCode.CONTEXT_OVERFLOW, str(exc))
    if "model" in text and "not found" in text:
        return ProcessingError(ErrorCode.MODEL_MISSING, str(exc))
    return ProcessingError(ErrorCode.UNKNOWN, str(exc))
