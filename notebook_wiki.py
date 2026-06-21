# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
# ForestOptiLM is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
Слой скомпилированных знаний по мотивам LLM-Wiki Андрея Карпатого.

Вместо «амнезийного» RAG (каждый вопрос — поиск по сырью с нуля) знание
КОМПИЛИРУЕТСЯ один раз в набор связанных markdown-страниц (обзор, сущности,
понятия, глоссарий), которые человек может читать и править, а модель —
переиспользовать. Рядом ведутся ``index.md`` (каталог) и ``log.md``
(append-only журнал операций). Сами вики-страницы потом индексируются вместе с
сырьём (B2), так что чат достаёт уже плотное знание.

Чистые функции (``build_wiki_index`` / ``append_log`` / ``page_summary``)
тестируются без сети; ``compile_wiki`` — async-оркестратор поверх call_llm,
переиспользует дайджест корпуса из :mod:`notebook_studio`.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable

import notebook_studio as st

# Страницы вики переиспользуют форму MaterialSpec из Studio (kind/title/filename/
# instructions) — тот же grounded-конвейер генерации.
WIKI_PAGES: list[st.MaterialSpec] = [
    st.MaterialSpec(
        kind="wiki_overview", title="Обзор", filename="overview.md",
        instructions=(
            "Напиши страницу-ОБЗОР всей базы знаний. Заголовок '# Обзор'. 3–6 абзацев: "
            "о чём корпус в целом, какие крупные темы он покрывает и как они связаны. "
            "Только по содержимому фрагментов, без воды."
        ),
    ),
    st.MaterialSpec(
        kind="wiki_entities", title="Сущности", filename="entities.md",
        instructions=(
            "Составь страницу КЛЮЧЕВЫХ СУЩНОСТЕЙ (системы, подсистемы, компоненты, "
            "виртуальные машины/серверы, люди, организации, продукты). Формат Markdown: "
            "заголовок '# Сущности', затем по каждой сущности '## <имя>' и 1–3 "
            "предложения сути с опорой на источники. Связанные сущности упоминай по имени."
        ),
    ),
    st.MaterialSpec(
        kind="wiki_concepts", title="Понятия", filename="concepts.md",
        instructions=(
            "Опиши КЛЮЧЕВЫЕ ПОНЯТИЯ и процессы корпуса. Формат Markdown: заголовок "
            "'# Понятия', по каждому '## <понятие>' и объяснение СТРОГО по источникам. "
            "Не выдумывай определений, которых нет в материалах."
        ),
    ),
    st.MaterialSpec(
        kind="wiki_glossary", title="Глоссарий", filename="glossary.md",
        instructions=(
            "Составь ГЛОССАРИЙ терминов и аббревиатур из материалов. Формат Markdown: "
            "заголовок '# Глоссарий', затем список '- **ТЕРМИН** — определение' по "
            "алфавиту. Только термины, встречающиеся в корпусе."
        ),
    ),
]

WIKI_INDEX_FILE = "index.md"
WIKI_LOG_FILE = "log.md"


def today_utc() -> str:
    """Дата для журнала в формате [YYYY-MM-DD] (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def page_summary(markdown: str, max_len: int = 120) -> str:
    """Однострочное резюме страницы: первая содержательная строка (не заголовок,
    не таблица), обрезанная до max_len."""
    for line in (markdown or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("|"):
            continue
        s = s.lstrip("-*").strip()
        if s:
            return (s[:max_len].rstrip() + "…") if len(s) > max_len else s
    return ""


def build_wiki_index(entries: list[tuple[str, str, str]], notebook_name: str = "") -> str:
    """Каталог вики (index.md): по странице — ссылка + однострочное резюме."""
    head = "# Вики блокнота" + (f": {notebook_name}" if notebook_name else "")
    lines = [head, "", "Скомпилированные страницы знаний:", ""]
    for title, filename, summary in entries:
        suffix = f" — {summary}" if summary else ""
        lines.append(f"- [{title}]({filename}){suffix}")
    return "\n".join(lines) + "\n"


def append_log(existing: str, op: str, detail: str, ts: str) -> str:
    """Дописать запись в append-only журнал (log.md). Формат записи парсится
    простыми unix-инструментами: '## [<дата>] <операция> | <детали>'."""
    entry = f"## [{ts}] {op} | {detail}"
    if not (existing or "").strip():
        return f"# Журнал операций блокнота\n\n{entry}\n"
    return existing.rstrip("\n") + f"\n{entry}\n"


async def compile_wiki(
    notebook: Any,
    *,
    base_url: str,
    api_key: str,
    chat_model: str,
    api_mode: str = "native",
    max_context_tokens: int = 12000,
    ts: str | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    stop_flag: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Скомпилировать вики блокнота: сгенерировать страницы → wiki/, обновить
    index.md и дописать log.md. Возвращает {'pages': [...], 'wiki_dir': str}."""
    from processor import call_llm

    chunks = st.read_index_chunks(notebook.index_dir)
    if not chunks:
        raise RuntimeError("Индекс блокнота пуст — сначала постройте индекс.")
    digest = st.gather_corpus_digest(chunks, max_tokens=max_context_tokens)

    wiki_dir = notebook.wiki_dir
    wiki_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(1)
    entries: list[tuple[str, str, str]] = []

    for i, spec in enumerate(WIKI_PAGES):
        if stop_flag and stop_flag():
            break
        messages = st.build_material_messages(spec, digest, notebook_name=notebook.name)
        raw = await call_llm(
            messages, chat_model, base_url, api_key, semaphore,
            max_tokens=2000, api_mode=api_mode, prefer_reasoning_off=True,
        )
        content = (raw or "").strip()
        if content:
            (wiki_dir / spec.filename).write_text(content, encoding="utf-8")
            entries.append((spec.title, spec.filename, page_summary(content)))
        if on_progress:
            on_progress(i + 1, len(WIKI_PAGES), spec.title)

    (wiki_dir / WIKI_INDEX_FILE).write_text(
        build_wiki_index(entries, notebook.name), encoding="utf-8")

    stamp = ts or today_utc()
    log_path = wiki_dir / WIKI_LOG_FILE
    existing = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    log_path.write_text(
        append_log(existing, "compile", f"{len(entries)} страниц(ы)", stamp), encoding="utf-8")

    return {"pages": entries, "wiki_dir": str(wiki_dir)}
