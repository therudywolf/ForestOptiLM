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
from pathlib import Path
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
SCHEMA_MAX_CHARS = 2000  # кап схемы домена, чтобы не раздувать контекст

# Журнал (log.md) дописывается read-modify-write; сериализуем на случай гонки
# (компиляция и «подшить ответ» в разных потоках).
import threading  # noqa: E402
_LOG_LOCK = threading.Lock()


def _append_log_locked(wiki_dir: Path, op: str, detail: str, ts: str) -> None:
    with _LOG_LOCK:
        log_path = wiki_dir / WIKI_LOG_FILE
        existing = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        log_path.write_text(append_log(existing, op, detail, ts), encoding="utf-8")


def wiki_index_sources(notebook: Any) -> list[Path]:
    """Все файлы, которые должны попасть в wiki-индекс: стандартные страницы +
    подшитые ответы (wiki/answers/*.md). index.md/log.md НЕ индексируем."""
    files: list[Path] = []
    for spec in WIKI_PAGES:
        p = notebook.wiki_dir / spec.filename
        if p.is_file():
            files.append(p)
    answers = notebook.wiki_dir / "answers"
    if answers.is_dir():
        files += sorted(answers.glob("*.md"))
    return files


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


def _slug(text: str, max_len: int = 40) -> str:
    """Файло-безопасный слаг из вопроса (для имени страницы-ответа)."""
    import re
    s = re.sub(r"[^\wА-Яа-яЁё]+", "-", (text or "").strip().lower()).strip("-")
    return (s[:max_len].rstrip("-") or "answer")


def save_answer_page(
    notebook: Any,
    question: str,
    answer: str,
    *,
    citations: list[str] | None = None,
    ts: str | None = None,
    base_url: str = "",
    api_key: str = "",
    embedding_model: str = "",
) -> Path:
    """Подшить ответ чата в знания (B5): сохранить вопрос+ответ как постоянную
    вики-страницу в wiki/answers/, дописать log.md и (если задана embedding-модель
    и вики уже проиндексирована) инкрементально добавить страницу в wiki-индекс,
    чтобы подшитый ответ сразу искался. Возвращает путь.

    Имя файла = дата + слаг + хеш(вопрос+ответ): разные ответы одного дня не
    перезаписывают друг друга, идентичные Q+A → один файл (идемпотентно)."""
    import hashlib
    stamp = ts or today_utc()
    answers_dir = notebook.wiki_dir / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1((question + "\x00" + answer).encode("utf-8")).hexdigest()[:8]
    path = answers_dir / f"{stamp}-{_slug(question)}-{h}.md"
    body = [f"# {question.strip() or 'Вопрос'}", "", f"_Сохранено {stamp}_", "", answer.strip()]
    if citations:
        body += ["", "## Источники", *[f"- {c}" for c in citations]]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    _append_log_locked(notebook.wiki_dir, "answer", (question.strip()[:60] or "ответ"), stamp)

    # Инкрементально доиндексировать (передаём ПОЛНЫЙ набор вики-источников, иначе
    # add_to_index счёл бы остальные удалёнными и сделал полную пересборку).
    emb = (embedding_model or "").strip()
    if base_url and emb and not emb.startswith("(") and notebook.has_wiki_index:
        try:
            from pipeline import add_to_index
            from notebook_store import notebook_index_chunk_tokens
            add_to_index(
                input_paths=wiki_index_sources(notebook), index_dir=notebook.wiki_index_dir,
                base_url=base_url, api_key=api_key, embedding_model=emb,
                chunk_size_tokens=notebook_index_chunk_tokens())
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger("nocturne").warning("wiki answer index add failed: %s", exc)
    return path


async def compile_wiki(
    notebook: Any,
    *,
    base_url: str,
    api_key: str,
    chat_model: str,
    embedding_model: str = "",
    api_mode: str = "native",
    max_context_tokens: int = 12000,
    ts: str | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    stop_flag: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Скомпилировать вики блокнота: сгенерировать страницы → wiki/, обновить
    index.md, дописать log.md и (если задана embedding-модель) построить отдельный
    индекс по вики-страницам, чтобы чат предпочитал скомпилированное знание (B2).

    Возвращает {'pages': [...], 'wiki_dir': str, 'indexed': bool}."""
    from processor import call_llm

    chunks = st.read_index_chunks(notebook.index_dir)
    if not chunks:
        raise RuntimeError("Индекс блокнота пуст — сначала постройте индекс.")
    digest = st.gather_corpus_digest(chunks, max_tokens=max_context_tokens)
    # B4: схема домена помогает модели правильно выделять сущности/понятия.
    # Кап по длине — чтобы свободный текст схемы не раздул контекст/не выбил бюджет.
    schema = str(getattr(notebook, "schema", "") or "").strip()[:SCHEMA_MAX_CHARS]
    if schema:
        import dataclasses
        digest = dataclasses.replace(
            digest, text="[Контекст домена]\n" + schema + "\n\n" + digest.text)

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

    # Полным считаем прогон только если не остановили и сгенерировались ВСЕ страницы.
    completed = not (stop_flag and stop_flag()) and len(entries) == len(WIKI_PAGES)
    # B2: индексируем страницы + подшитые ответы (не index.md/log.md) в отдельный
    # wiki-индекс. Build полностью перестраивает его (и сбрасывает кэш-стор).
    indexed = False
    sources = wiki_index_sources(notebook)
    emb = (embedding_model or "").strip()
    if completed and sources and emb and not emb.startswith("("):
        try:
            from pipeline import build_index
            from notebook_store import notebook_index_chunk_tokens
            build_index(
                input_paths=sources, index_dir=notebook.wiki_index_dir,
                base_url=base_url, api_key=api_key, embedding_model=emb,
                chunk_size_tokens=notebook_index_chunk_tokens(),
            )
            indexed = True
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger("nocturne").warning("wiki index build failed: %s", exc)
    if not indexed and notebook.wiki_index_dir.exists():
        # Страницы пересобраны, но индекс НЕ обновлён (неполный прогон / нет embed /
        # сбой) → старый индекс теперь устарел и цитировался бы зря. Убираем его,
        # чтобы чат шёл по сырью, а не по протухшей вики.
        import shutil
        try:
            shutil.rmtree(notebook.wiki_index_dir)
            from pipeline import _evict_store
            _evict_store(notebook.wiki_index_dir)
        except Exception:
            pass

    detail = f"{len(entries)}/{len(WIKI_PAGES)} страниц" + (", проиндексировано" if indexed else "")
    _append_log_locked(wiki_dir, "compile", detail, ts or today_utc())

    return {"pages": entries, "wiki_dir": str(wiki_dir), "indexed": indexed, "completed": completed}
