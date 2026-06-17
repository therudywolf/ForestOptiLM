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
Grounded-чат блокнота: ответы СТРОГО по найденным фрагментам с цитатами [N].

Заземление (grounding) в духе NotebookLM: модель отвечает только на основе
извлечённых из индекса блокнота фрагментов, каждую мысль помечает ссылкой [N]
на конкретный источник, а если ответа в источниках нет — честно сообщает об
этом, а не фантазирует.

Чистые функции (``select_contexts`` / ``build_chat_messages`` /
``parse_used_citations``) тестируются без сети; ``answer_question`` —
асинхронный оркестратор поверх ``processor.call_llm``.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("nocturne")

REFUSAL_TEXT = "В источниках блокнота нет ответа на этот вопрос."

CHAT_SYSTEM_PROMPT = (
    "Ты — ассистент, отвечающий СТРОГО на основе источников блокнота.\n"
    "Правила, которые нельзя нарушать:\n"
    "1. Используй только информацию из пронумерованных фрагментов в разделе "
    "[Источники]. Не добавляй внешних знаний и не домысливай.\n"
    "2. После каждого утверждения ставь ссылку на источник в виде [N] — номер "
    "фрагмента. Если факт опирается на несколько фрагментов, перечисли их: [1][3].\n"
    "3. Если в источниках нет ответа на вопрос — ответь ровно фразой: "
    f"\"{REFUSAL_TEXT}\" и ничего не выдумывай.\n"
    "4. Отвечай на языке вопроса, по делу, без воды и без описания своих "
    "размышлений."
)

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")


@dataclass(slots=True)
class ContextItem:
    n: int  # 1-based номер цитаты
    source_path: str
    display: str
    text: str
    chunk_id: str = ""
    score: float = 0.0
    page: int | None = None
    line_start: int | None = None

    def locator(self) -> str:
        """Человекочитаемая привязка: «стр. 4» / «строка 120» / ''."""
        if self.page:
            return f"стр. {self.page}"
        if self.line_start:
            return f"строка {self.line_start}"
        return ""

    def to_citation(self, quote_chars: int = 320) -> dict[str, Any]:
        quote = _strip_headers(self.text).strip()
        if len(quote) > quote_chars:
            quote = quote[:quote_chars].rstrip() + "…"
        return {
            "n": self.n,
            "source_path": self.source_path,
            "display": self.display,
            "quote": quote,
            "chunk_id": self.chunk_id,
            "score": round(float(self.score), 4),
            "page": self.page,
            "line_start": self.line_start,
            "locator": self.locator(),
        }


@dataclass(slots=True)
class ChatResult:
    answer: str
    citations: list[dict[str, Any]]  # только реально использованные [N]
    contexts: list[dict[str, Any]]   # все извлечённые фрагменты (для панели «источники»)
    refused: bool = False
    model: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _strip_headers(text: str) -> str:
    """Убрать служебные [FILE_PATH:…]/[SOURCE_URL:…] заголовки из цитаты."""
    lines = [ln for ln in text.splitlines()
             if not re.match(r"^\s*\[(FILE_PATH|FILE_TITLE|FILE_LABELS|FILE_FORMAT|"
                             r"SOURCE_URL|SOURCE_TITLE|VISION_FILE|CHUNK_INDEX|FILE_PART)\b", ln)]
    return "\n".join(lines).strip()


def _display_for_hit(hit: Any) -> str:
    meta = getattr(hit, "metadata", None) or {}
    title = str(meta.get("title") or "").strip()
    src = str(getattr(hit, "source_path", "") or "")
    base = src.replace("\\", "/").rsplit("/", 1)[-1] if src else "источник"
    if title and title.lower() not in base.lower():
        return f"{base} — {title}"[:80]
    return base[:80] or "источник"


def select_contexts(
    hits: list[Any],
    *,
    max_tokens: int = 8000,
    max_items: int = 12,
    per_item_char_cap: int = 6000,
) -> list[ContextItem]:
    """Отобрать фрагменты под бюджет токенов, пронумеровать как [1..N]."""
    try:
        from parser import count_tokens
    except Exception:  # pragma: no cover - parser всегда есть, но не падаем в тестах
        def count_tokens(t: str) -> int:  # type: ignore[misc]
            return max(1, len(t) // 4)

    out: list[ContextItem] = []
    used = 0
    for hit in hits:
        if len(out) >= max_items:
            break
        text = str(getattr(hit, "text", "") or "")
        if not text.strip():
            continue
        if len(text) > per_item_char_cap:
            text = text[:per_item_char_cap]
        tok = count_tokens(text)
        if out and used + tok > max_tokens:
            break
        meta = getattr(hit, "metadata", None) or {}

        def _as_int(v: Any) -> int | None:
            try:
                return int(v) if v is not None else None
            except Exception:
                return None

        out.append(
            ContextItem(
                n=len(out) + 1,
                source_path=str(getattr(hit, "source_path", "") or ""),
                display=_display_for_hit(hit),
                text=text,
                chunk_id=str(getattr(hit, "chunk_id", "") or ""),
                score=float(getattr(hit, "score", 0.0) or 0.0),
                page=_as_int(meta.get("page")),
                line_start=_as_int(meta.get("line_start")),
            )
        )
        used += tok
    return out


def _format_history(history: list[dict[str, Any]], max_turns: int, max_chars: int) -> str:
    if not history:
        return ""
    tail = history[-max_turns:]
    lines: list[str] = []
    for turn in tail:
        role = str(turn.get("role") or "")
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "…"
        prefix = "Пользователь" if role == "user" else "Ассистент"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def build_chat_messages(
    question: str,
    contexts: list[ContextItem],
    history: list[dict[str, Any]] | None = None,
    *,
    history_turns: int = 4,
    history_chars: int = 600,
) -> list[dict[str, str]]:
    """Собрать messages для call_llm: system + единый grounded user-промпт."""
    parts: list[str] = []
    hist = _format_history(history or [], history_turns, history_chars)
    if hist:
        parts.append("[Предыдущий диалог]\n" + hist)

    if contexts:
        ctx_block = "\n\n".join(
            f"[{c.n}] ({c.display})\n{_strip_headers(c.text)}" for c in contexts
        )
    else:
        ctx_block = "(источники не найдены)"
    parts.append("[Источники]\n" + ctx_block)
    parts.append("[Вопрос]\n" + question.strip())
    parts.append(
        "Ответь строго по источникам выше и ставь ссылки [N]. "
        f"Если ответа в источниках нет — напиши ровно: \"{REFUSAL_TEXT}\"."
    )
    user = "\n\n".join(parts)
    return [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_used_citations(
    answer: str, contexts: list[ContextItem]
) -> list[dict[str, Any]]:
    """Вернуть цитаты для тех [N], что реально встречаются в ответе (по порядку)."""
    by_n = {c.n: c for c in contexts}
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for m in _CITATION_RE.finditer(answer or ""):
        n = int(m.group(1))
        if n in seen or n not in by_n:
            continue
        seen.add(n)
        out.append(by_n[n].to_citation())
    return out


def is_refusal(answer: str) -> bool:
    norm = re.sub(r"\s+", " ", (answer or "")).strip().lower()
    return REFUSAL_TEXT.lower() in norm and len(norm) < len(REFUSAL_TEXT) + 40


async def answer_question(
    notebook: Any,
    question: str,
    *,
    base_url: str,
    api_key: str,
    chat_model: str,
    embedding_model: str = "",
    api_mode: str = "native",
    top_k: int = 8,
    history: list[dict[str, Any]] | None = None,
    max_context_tokens: int = 8000,
    max_answer_tokens: int = 1500,
    on_log: Callable[[str], None] | None = None,
) -> ChatResult:
    """Полный цикл: retrieval по блокноту → grounded-ответ с цитатами."""
    from processor import call_llm

    def _log(msg: str) -> None:
        if on_log:
            try:
                on_log(msg)
            except Exception:
                pass

    hits = notebook.query(
        question,
        base_url=base_url,
        api_key=api_key,
        embedding_model=embedding_model,
        top_k=top_k,
    )
    _log(f"retrieval: {len(hits)} фрагментов")
    contexts = select_contexts(hits, max_tokens=max_context_tokens)

    if not contexts:
        return ChatResult(
            answer=REFUSAL_TEXT,
            citations=[],
            contexts=[],
            refused=True,
            model=chat_model,
        )

    messages = build_chat_messages(question, contexts, history)
    semaphore = asyncio.Semaphore(1)
    try:
        raw = await call_llm(
            messages,
            chat_model,
            base_url,
            api_key,
            semaphore,
            max_tokens=max_answer_tokens,
            api_mode=api_mode,
        )
    except RuntimeError as exc:
        # Маленькие reasoning-модели c reasoning:off иногда отдают пустой вывод —
        # не роняем чат, а возвращаем понятное сообщение (ретраи внутри call_llm
        # уже отработали). Прочие ошибки (сеть/HTTP) пробрасываем.
        if "empty content" in str(exc).lower():
            return ChatResult(
                answer="Модель вернула пустой ответ. Попробуйте переформулировать "
                       "вопрос или выбрать модель побольше.",
                citations=[],
                contexts=[c.to_citation() for c in contexts],
                refused=False,
                model=chat_model,
                extra={"empty_output": True},
            )
        raise
    answer = (raw or "").strip()
    used = parse_used_citations(answer, contexts)
    return ChatResult(
        answer=answer,
        citations=used,
        contexts=[c.to_citation() for c in contexts],
        refused=is_refusal(answer),
        model=chat_model,
    )
