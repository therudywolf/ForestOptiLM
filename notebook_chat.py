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
CANCELLED_TEXT = "⏹ Запрос остановлен."


class ChatCancelled(Exception):
    """Пользователь нажал «Стоп» — кооперативная отмена запроса чата."""


async def _await_with_stop(coro: Any, stopped: Callable[[], bool], poll: float = 0.15) -> Any:
    """Ждать coro, периодически проверяя stop-флаг. По «Стоп» отменяет задачу
    (in-flight httpx-запрос рвётся) и поднимает ChatCancelled."""
    task = asyncio.ensure_future(coro)
    while True:
        done, _ = await asyncio.wait({task}, timeout=poll)
        if task in done:
            return task.result()
        if stopped():
            task.cancel()
            try:
                await task
            except BaseException:
                pass
            raise ChatCancelled()

CHAT_SYSTEM_PROMPT = (
    "Ты — ассистент, отвечающий СТРОГО на основе источников блокнота.\n"
    "Правила, которые нельзя нарушать:\n"
    "1. Используй только информацию из пронумерованных фрагментов в разделе "
    "[Источники]. Не добавляй внешних знаний и не домысливай.\n"
    "2. После каждого утверждения ставь ссылку на источник в виде [N] — номер "
    "фрагмента. Если факт опирается на несколько фрагментов, перечисли их: [1][3].\n"
    "3. Если источники отвечают лишь ЧАСТИЧНО — всё равно ответь тем, что в них "
    "есть (со ссылками [N]), и честно отметь, чего в источниках не хватает. "
    "Не отказывайся от ответа, если хоть что-то по теме во фрагментах есть.\n"
    f"4. Только если во фрагментах СОВСЕМ нет относящейся к вопросу информации — "
    f"ответь ровно фразой: \"{REFUSAL_TEXT}\" и ничего не выдумывай.\n"
    "5. Отвечай на языке вопроса, по делу, без воды и без описания своих "
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
    parts = [p for p in src.replace("\\", "/").split("/") if p]
    base = parts[-1] if parts else "источник"
    # Родительская папка в подписи: generic-имена (messages5.html, index.html)
    # без неё неразличимы между источниками (напр. чаты Telegram-экспорта).
    if len(parts) >= 2:
        base = f"{parts[-2]}/{base}"
    if title and title.lower() not in base.lower():
        return f"{base} — {title}"[:80]
    return base[:80] or "источник"


def select_contexts(
    hits: list[Any],
    *,
    max_tokens: int = 8000,
    max_items: int = 12,
    per_item_char_cap: int = 6000,
    max_per_source: int = 5,
) -> list[ContextItem]:
    """Отобрать фрагменты под бюджет токенов, пронумеровать как [1..N].

    Полнота ответа, а не только релевантность:
    - почти-одинаковые фрагменты (перекрывающиеся чанки, дубли пересылок)
      схлопываются — бюджет не сгорает на повторах;
    - не больше ``max_per_source`` фрагментов с одного файла, ЕСЛИ есть
      кандидаты из других файлов (backfill: при недоборе отсечённые
      возвращаются) — топ не монополизируется одним источником.
    """
    try:
        from parser import count_tokens
    except Exception:  # pragma: no cover - parser всегда есть, но не падаем в тестах
        def count_tokens(t: str) -> int:  # type: ignore[misc]
            return max(1, len(t) // 4)

    def _as_int(v: Any) -> int | None:
        try:
            return int(v) if v is not None else None
        except Exception:
            return None

    out: list[ContextItem] = []
    used = 0
    seen_keys: set[str] = set()
    per_source: dict[str, int] = {}

    def _try_add(hit: Any, enforce_cap: bool) -> str:
        """Вернуть 'added' | 'skip' | 'capped' | 'stop'."""
        nonlocal used
        if len(out) >= max_items:
            return "stop"
        text = str(getattr(hit, "text", "") or "")
        if not text.strip():
            return "skip"
        if len(text) > per_item_char_cap:
            text = text[:per_item_char_cap]
        # Дедуп по нормализованному началу содержимого (без служебных заголовков):
        # перекрывающиеся чанки и повторные пересылки дают одинаковый префикс.
        norm = re.sub(r"\s+", " ", _strip_headers(text)).strip().lower()
        if not norm:
            return "skip"
        key = norm[:240]
        if key in seen_keys:
            return "skip"
        src = str(getattr(hit, "source_path", "") or "")
        if enforce_cap and max_per_source and per_source.get(src, 0) >= max_per_source:
            return "capped"
        tok = count_tokens(text)
        if out and used + tok > max_tokens:
            return "stop"
        meta = getattr(hit, "metadata", None) or {}
        out.append(
            ContextItem(
                n=len(out) + 1,
                source_path=src,
                display=_display_for_hit(hit),
                text=text,
                chunk_id=str(getattr(hit, "chunk_id", "") or ""),
                score=float(getattr(hit, "score", 0.0) or 0.0),
                page=_as_int(meta.get("page")),
                line_start=_as_int(meta.get("line_start")),
            )
        )
        seen_keys.add(key)
        per_source[src] = per_source.get(src, 0) + 1
        used += tok
        return "added"

    capped: list[Any] = []
    for hit in hits:
        status = _try_add(hit, enforce_cap=True)
        if status == "stop":
            break
        if status == "capped":
            capped.append(hit)
    # Backfill: если после диверсификации остался бюджет — вернуть лучшие из
    # отсечённых по per-source cap (для однофайловых блокнотов это весь хвост).
    for hit in capped:
        if _try_add(hit, enforce_cap=False) == "stop":
            break
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
    schema: str = "",
) -> list[dict[str, str]]:
    """Собрать messages для call_llm: system + единый grounded user-промпт.

    ``schema`` (B4) — свободное описание домена блокнота; добавляется к system-
    промпту как контекст, помогая модели правильно трактовать сущности/термины.
    """
    system = CHAT_SYSTEM_PROMPT
    s = schema.strip()[:2000]  # кап: свободный текст схемы не должен раздувать контекст
    if s:
        system += "\n\n[Контекст домена этого блокнота]\n" + s
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
        "Ответь по источникам выше, ставя ссылку [N] к каждому факту. Сопоставляй "
        "фрагменты между собой (даты, авторы, причины/следствия): если ответ "
        "следует из их совокупности — сформулируй его как вывод со ссылками на "
        "использованные фрагменты. Если ответ есть лишь частично — дай частичный "
        "ответ и отметь, чего не хватает. Напиши ровно "
        f"\"{REFUSAL_TEXT}\" только если по теме во фрагментах нет ничего."
    )
    user = "\n\n".join(parts)
    return [
        {"role": "system", "content": system},
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
    top_k: int = 16,
    history: list[dict[str, Any]] | None = None,
    max_context_tokens: int = 12000,
    max_answer_tokens: int = 1500,
    prefer_reasoning_off: bool = True,
    on_log: Callable[[str], None] | None = None,
    stop_flag: Callable[[], bool] | None = None,
    enhanced: bool = False,
    on_token: Callable[[str], None] | None = None,
    deep_mode: str = "off",
) -> ChatResult:
    """Полный цикл: retrieval по блокноту → grounded-ответ с цитатами.

    prefer_reasoning_off=True: для grounded-чата это даёт ЧИСТЫЙ ответ — проверено
    живьём на gemma-4-12b (с reasoning:on модель парротит ограничения промпта и
    «думает вслух» прямо в текст ответа). call_llm при пустом выводе сам эскалирует
    reasoning:off→on, так что маленькие reasoning-модели тоже не остаются без ответа.
    """
    from processor import call_llm

    def _log(msg: str) -> None:
        if on_log:
            try:
                on_log(msg)
            except Exception:
                pass

    def _stopped() -> bool:
        return bool(stop_flag and stop_flag())

    def _cancelled_result(ctxs: list[ContextItem]) -> ChatResult:
        return ChatResult(
            answer=CANCELLED_TEXT, citations=[],
            contexts=[c.to_citation() for c in ctxs],
            refused=False, model=chat_model, extra={"cancelled": True},
        )

    semaphore = asyncio.Semaphore(1)

    async def _llm(msgs: list[dict[str, str]], max_tokens: int) -> str:
        # Вспомогательные LLM-вызовы (expansion/rerank) — всегда reasoning:off
        # (структурный JSON-вывод, не нужно «думать вслух»), с учётом «Стоп».
        return await _await_with_stop(
            call_llm(msgs, chat_model, base_url, api_key, semaphore,
                     max_tokens=max_tokens, api_mode=api_mode, prefer_reasoning_off=True),
            _stopped,
        )

    def _retrieve(q: str, k: int | None = None) -> list[Any]:
        return notebook.query(q, base_url=base_url, api_key=api_key,
                              embedding_model=embedding_model, top_k=k or top_k)

    # --- Глубокий анализ (map-reduce над всем следом сущности/темы) ----------
    import deep_analysis as _da
    want_deep = (deep_mode == "on") or (
        deep_mode == "auto" and _da.is_analytical_question(question))
    if want_deep:
        try:
            res = await _run_deep_analysis(
                notebook, question, _llm=_llm, _retrieve=_retrieve,
                stopped=_stopped, log=_log, cancelled=_cancelled_result,
                chat_model=chat_model, on_token=on_token,
                max_context_tokens=max_context_tokens, max_answer_tokens=max_answer_tokens,
            )
            if res is not None:
                return res
        except ChatCancelled:
            return _cancelled_result([])
        except Exception as exc:  # noqa: BLE001 — глубокий режим не должен рушить чат
            _log(f"глубокий анализ недоступен, обычный режим: {exc}")

    if enhanced:
        # «Точный поиск» по мотивам qmd: query-expansion (выше recall) →
        # listwise LLM-реранк (выше precision). Любой сбой → мягкий фолбэк на базу.
        import retrieval_enhance as _re
        queries = [question]
        entities: list[str] = []
        try:
            _schema = str(getattr(notebook, "schema", "") or "")
            _raw = await _llm(_re.build_expansion_messages(question, schema=_schema), 250)
            queries = _re.parse_expansions(_raw, question)
            entities = _re.parse_entities(_raw)
        except ChatCancelled:
            return _cancelled_result([])
        except Exception as exc:  # noqa: BLE001
            _log(f"expansion пропущен: {exc}")
        _log(f"expansion: {len(queries)} запрос(ов), сущностей: {len(entities)}")
        # Агрегирующие вопросы («какой человек X», «что по теме Y»): семантика
        # запроса далека от реальных сообщений сущности — добираем их отдельным
        # лексическим поиском по имени (BM25 внутри hybrid_search вытащит их).
        hit_lists = [_retrieve(q) for q in queries]
        for ent in entities:
            hit_lists.append(_retrieve(ent))
        cap = 40 if entities else 30
        hits = _re.merge_hits(hit_lists, cap=cap)
        # Для агрегирующих вопросов держим шире окно после реранка — портрет/сводку
        # не собрать из 16 фрагментов (нужны десятки сообщений сущности).
        rerank_keep = max(top_k, 30) if entities else max(top_k, 16)
        if len(hits) > 1 and not _stopped():
            try:
                order = _re.parse_rerank_order(
                    await _llm(_re.build_rerank_messages(question, hits), 300), len(hits))
                hits = _re.apply_rerank(hits, order, top_k=rerank_keep)
                _log(f"rerank: {len(hits)} фрагмент(ов)")
            except ChatCancelled:
                return _cancelled_result([])
            except Exception as exc:  # noqa: BLE001
                _log(f"rerank пропущен: {exc}")
                hits = hits[:rerank_keep]
    else:
        hits = _retrieve(question)
    _log(f"retrieval: {len(hits)} фрагментов")
    contexts = select_contexts(hits, max_tokens=max_context_tokens, max_items=max(12, top_k))

    if not contexts:
        return ChatResult(
            answer=REFUSAL_TEXT,
            citations=[],
            contexts=[],
            refused=True,
            model=chat_model,
        )

    if _stopped():  # успели нажать «Стоп» ещё на этапе поиска
        return _cancelled_result(contexts)

    schema = str(getattr(notebook, "schema", "") or "")
    messages = build_chat_messages(question, contexts, history, schema=schema)

    raw: str | None = None
    # C4: потоковый вывод. Стрим идёт по openai-совместимому пути (LM Studio его
    # отдаёт и в native-режиме), кроме «точного поиска» (там свой много-вызовный
    # цикл). Любой сбой стрима → тихий откат на обычный call_llm.
    if on_token and not enhanced:
        from processor import call_llm_stream
        try:
            # Оборачиваем в _await_with_stop: при «Стоп» задача отменяется и рвёт
            # in-flight стрим даже если модель зависла между токенами (иначе Стоп
            # ждал бы read-timeout). stop_flag внутри тоже проверяется по-строчно.
            raw = await _await_with_stop(
                call_llm_stream(
                    messages, chat_model, base_url, api_key,
                    max_tokens=max_answer_tokens, api_mode=api_mode,
                    on_token=on_token, stop_flag=_stopped,
                ),
                _stopped,
            )
        except ChatCancelled:
            return _cancelled_result(contexts)
        except Exception as exc:  # noqa: BLE001
            _log(f"стриминг недоступен, обычный режим: {exc}")
            raw = None
        if _stopped():
            return _cancelled_result(contexts)

    if raw is not None:
        pass  # получили ответ стримингом
    else:
        try:
            raw = await _await_with_stop(
                call_llm(
                    messages,
                    chat_model,
                    base_url,
                    api_key,
                    semaphore,
                    max_tokens=max_answer_tokens,
                    api_mode=api_mode,
                    prefer_reasoning_off=prefer_reasoning_off,
                ),
                _stopped,
            )
        except ChatCancelled:
            return _cancelled_result(contexts)
        except RuntimeError as exc:
            # Маленькие reasoning-модели c reasoning:off иногда отдают пустой вывод
            # — не роняем чат, а возвращаем понятное сообщение (ретраи внутри
            # call_llm уже отработали). Прочие ошибки (сеть/HTTP) пробрасываем.
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


async def _run_deep_analysis(
    notebook: Any,
    question: str,
    *,
    _llm: Callable[[list[dict[str, str]], int], Any],
    _retrieve: Callable[..., list[Any]],
    stopped: Callable[[], bool],
    log: Callable[[str], None],
    cancelled: Callable[[list[ContextItem]], ChatResult],
    chat_model: str,
    on_token: Callable[[str], None] | None,
    max_context_tokens: int,
    max_answer_tokens: int,
    cap_units: int = 400,
    max_batch_tokens: int = 3500,
    max_batches: int = 24,
) -> ChatResult | None:
    """Оркестратор глубокого анализа: identify → gather(+соседи) → map → reduce.

    Возвращает ChatResult, либо None — если для map-reduce не набралось материала
    (тогда answer_question мягко откатывается на обычный путь)."""
    import deep_analysis as _da
    import retrieval_enhance as _re
    from retrieval import LocalFaissStore

    schema = str(getattr(notebook, "schema", "") or "")
    if on_token:
        try:
            on_token("🔬 Глубокий анализ (может занять пару минут): собираю данные "
                     "по всему корпусу, анализирую по частям…\n")
        except Exception:
            pass

    # 1) Сущности/переформулировки из вопроса (один LLM-вызов).
    entities: list[str] = []
    queries = [question]
    try:
        raw = await _llm(_re.build_expansion_messages(question, schema=schema), 250)
        entities = _re.parse_entities(raw)
        queries = _re.parse_expansions(raw, question)
    except ChatCancelled:
        raise
    except Exception as exc:  # noqa: BLE001
        log(f"deep: expansion пропущен: {exc}")

    # 2) Широкий сбор: семантика по вопросу + лексика по каждой сущности.
    hit_lists = [_retrieve(question, 150)]
    for ent in entities[:4]:
        hit_lists.append(_retrieve(ent, 250))
    hits = _re.merge_hits(hit_lists, cap=max(cap_units - 100, 200))
    log(f"deep: сущности={entities or '—'}, кандидатов={len(hits)}")
    if not hits:
        return None  # нечего анализировать → откат на обычный путь
    if stopped():
        raise ChatCancelled()

    # 3) Добор соседей того же источника (диалоговый/документный контекст).
    try:
        _, meta, _, _ = LocalFaissStore(notebook.index_dir)._load_cached_index_meta()
    except Exception:
        meta = []
    units = _da.expand_with_neighbors(hits, meta, radius=1, cap=cap_units)

    # 4) Пачки под map (сквозная нумерация [n] для цитат).
    try:
        from parser import count_tokens
    except Exception:  # pragma: no cover
        count_tokens = None
    batches = _da.batch_units(units, max_batch_tokens=max_batch_tokens, count_tokens=count_tokens)
    truncated = 0
    if len(batches) > max_batches:
        truncated = len(batches) - max_batches
        batches = batches[:max_batches]
    log(f"deep: единиц={len(units)}, пачек={len(batches)}"
        + (f" (+{truncated} пачек отсечено по лимиту)" if truncated else ""))

    # 5) MAP: выжать сигналы из каждой пачки (последовательно — один инстанс LLM).
    summaries: list[str] = []
    numbered_all: dict[int, _da.Unit] = {}
    for bi, batch in enumerate(batches, 1):
        for n, u in batch:
            numbered_all[n] = u
        if stopped():
            raise ChatCancelled()
        try:
            raw = await _llm(_da.build_map_messages(question, batch), 500)
        except ChatCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            log(f"deep: пачка {bi} пропущена: {exc}")
            continue
        s = _da.parse_map_result(raw)
        if s:
            summaries.append(s)
        log(f"deep: map {bi}/{len(batches)} — {'✓' if s else '—'}")

    if not summaries:
        return None  # ничего релевантного не выжали → откат
    if stopped():
        raise ChatCancelled()

    # 6) REDUCE: синтез цельного разбора.
    log(f"deep: reduce из {len(summaries)} выжимок")
    reduce_msgs = _da.build_reduce_messages(question, summaries, schema=schema)
    try:
        raw = await _llm(reduce_msgs, max_answer_tokens)
    except ChatCancelled:
        raise
    answer = (raw or "").strip()
    if not answer:
        return None

    # 7) Цитаты: [N] в ответе → единицы (для панели источников).
    used_ctx: list[ContextItem] = []
    seen: set[int] = set()
    for m in _CITATION_RE.finditer(answer):
        n = int(m.group(1))
        if n in seen or n not in numbered_all:
            continue
        seen.add(n)
        u = numbered_all[n]
        used_ctx.append(ContextItem(
            n=n, source_path=u.source_path, display=_display_for_hit(u),
            text=u.text, chunk_id=u.chunk_id, score=u.score,
        ))
    note = ""
    if truncated:
        note = (f"\n\n_(разобрано {len(numbered_all)} единиц; часть корпуса за лимитом — "
                f"уточните сущность/период для полного охвата)_")
    return ChatResult(
        answer=answer + note,
        citations=[c.to_citation() for c in used_ctx],
        contexts=[c.to_citation() for c in used_ctx],
        refused=is_refusal(answer),
        model=chat_model,
        extra={"deep": True, "units": len(numbered_all), "batches": len(batches)},
    )
