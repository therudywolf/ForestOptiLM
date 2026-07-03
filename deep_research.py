# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
# ForestOptiLM is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""W4 веб-стека: дипресёрч — поиск → чтение → grounded-синтез с веб-цитатами.

Связывает keyless-поиск (web_search) и чтение страниц (web_fetch) с локальной
LLM: находит источники, читает top-N, синтезирует ответ СТРОГО по прочитанному,
каждый факт помечает [N] → URL. Ничего не выдумывает вне источников (как чат по
блокноту, но источники — веб). Сборка промпта — чистая функция (тест без сети).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

import web_fetch
import web_search

logger = logging.getLogger("nocturne")

_RESEARCH_SYSTEM = (
    "Ты — веб-исследователь. Отвечай СТРОГО по извлечённым веб-страницам из "
    "раздела [Веб-источники]. Правила:\n"
    "1. Только факты из источников; не добавляй внешних знаний и не домысливай.\n"
    "2. После каждого утверждения ставь ссылку [N] — номер источника (можно "
    "несколько: [1][3]).\n"
    "3. Если источники противоречат — отметь это. Если ответа в них нет — честно "
    "скажи, а не выдумывай.\n"
    "4. Отвечай на языке вопроса, структурно, по делу."
)


def build_research_messages(question: str, pages: list[Any], per_source_chars: int = 4000) -> list[dict[str, str]]:
    """system + grounded-user-промпт из прочитанных страниц. Чистая функция."""
    blocks = []
    for i, p in enumerate(pages, 1):
        title = getattr(p, "title", "") or ""
        url = getattr(p, "final_url", "") or getattr(p, "url", "")
        text = (getattr(p, "text", "") or "")[:per_source_chars]
        blocks.append(f"[{i}] {title} ({url})\n{text}")
    user = (
        "[Веб-источники]\n" + "\n\n".join(blocks) + "\n\n"
        f"[Вопрос]\n{question.strip()}\n\n"
        "Ответь по источникам выше, ставя ссылку [N] к каждому факту. Сопоставляй "
        "источники между собой; если ответ следует из их совокупности — сформулируй "
        "вывод со ссылками. Не выдумывай того, чего в источниках нет."
    )
    return [{"role": "system", "content": _RESEARCH_SYSTEM},
            {"role": "user", "content": user}]


async def research(
    question: str,
    *,
    base_url: str,
    api_key: str,
    chat_model: str,
    max_sources: int = 5,
    max_hits: int = 10,
    api_mode: str = "native",
    max_answer_tokens: int = 1500,
    on_log: Callable[[str], None] | None = None,
) -> dict:
    """Провести веб-исследование: найти → прочитать top-N → синтезировать ответ
    с цитатами. Возвращает {answer, sources:[{n,url,title}], n_pages}."""
    log = on_log or (lambda _m: None)
    q = (question or "").strip()
    if not q:
        return {"answer": "Пустой запрос.", "sources": [], "n_pages": 0}

    hits = web_search.search(q, max_results=max_hits)
    if not hits:
        return {"answer": "Веб-поиск не дал результатов.", "sources": [], "n_pages": 0}
    log(f"🔎 найдено {len(hits)} ссылок — читаю страницы…")

    pages: list[Any] = []
    for h in hits:
        page = web_fetch.fetch_safe(h.url)
        if page:
            pages.append(page)
        if len(pages) >= max_sources:
            break
    if not pages:
        return {"answer": "Не удалось прочитать ни один источник.", "sources": [], "n_pages": 0}
    log(f"📄 прочитано {len(pages)} страниц — синтезирую ответ…")

    from processor import call_llm
    messages = build_research_messages(q, pages)
    sem = asyncio.Semaphore(1)
    answer = await call_llm(messages, chat_model, base_url, api_key, sem,
                            max_tokens=max_answer_tokens, api_mode=api_mode)
    sources = [{"n": i, "url": getattr(p, "final_url", "") or getattr(p, "url", ""),
                "title": getattr(p, "title", "")} for i, p in enumerate(pages, 1)]
    return {"answer": (answer or "").strip(), "sources": sources, "n_pages": len(pages)}
