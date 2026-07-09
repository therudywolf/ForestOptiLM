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
    "3. По умолчанию ВСЕГДА собирай из фрагментов всё, что относится к вопросу "
    "хотя бы косвенно, и формулируй ответ из этого (со ссылками [N]). Частичный "
    "ответ лучше отказа: дай, что есть, и честно отметь, чего не хватает.\n"
    "4. Если спрашивают про конкретный ЯРЛЫК/ТЕРМИН/НАЗВАНИЕ, которого во "
    "фрагментах нет ДОСЛОВНО, но есть СВЯЗАННЫЕ по смыслу факты — изложи эти факты "
    "со ссылками [N] и отметь, что дословно такого термина в источниках нет. НЕ "
    "отказывай только из-за отсутствия точной формулировки. НО: каждое число, "
    "название параметра, порог и деталь в ответе ДОЛЖНЫ дословно присутствовать в "
    "том фрагменте, на который ты ссылаешься. НЕ добавляй параметры/цифры/детали, "
    "которых в тексте фрагмента нет; лучше меньше фактов, но 100% проверяемых.\n"
    f"5. Ровно фразой \"{REFUSAL_TEXT}\" отвечай ТОЛЬКО когда фрагменты про "
    f"совершенно другую тему и связать с вопросом реально нечего (напр. вопрос про "
    f"рецепт, а источники — про ИБ). Ничего не выдумывай.\n"
    "6. Отвечай на языке вопроса, по делу, без воды и без описания своих "
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
    rich: bool = False,
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
        "использованные фрагменты. ВАЖНО: приписывай факт ТОЛЬКО той системе/"
        "сущности, о которой он в тексте; если фрагмент про ДРУГУЮ систему — не "
        "переноси его данные на запрошенную (версии, инциденты и настройки у "
        "разных систем свои). Для вопросов «кто/почему принял решение»: если во "
        "фрагментах есть И ответственное лицо/встреча, И причина/основание — "
        "приведи их как ВЫВОД со ссылками, не пиши «нет информации» при наличии "
        "релевантных фактов (но не выдумывай, если их правда нет). "
        "Если ответ есть лишь частично — дай частичный "
        "ответ и отметь, чего не хватает. Если ТОЧНОГО термина/ярлыка из вопроса "
        "во фрагментах нет, но есть связанные по смыслу факты — изложи их со "
        "ссылками, не отказывай. Но НЕ придумывай цифры, параметры и детали, "
        "которых нет в цитируемом фрагменте — приводи только дословно проверяемое. "
        "Напиши ровно "
        f"\"{REFUSAL_TEXT}\" только если фрагменты про совершенно другую тему."
    )
    if rich:
        # Композер-режим: просим развёрнутый, вовлечённый ответ (крупная модель
        # это тянет). НЕ ослабляет заземление — только глубину раскрытия.
        parts.append(
            "Дай РАЗВЁРНУТЫЙ и связный ответ: раскрой ВСЕ относящиеся к вопросу "
            "детали из фрагментов (не только верхнюю выжимку), сгруппируй по "
            "смысловым блокам с подзаголовками, сопоставь фрагменты и сделай выводы "
            "— но строго по источникам, без воды и без выдумок."
        )
    user = "\n\n".join(parts)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# Второй проход при отказе (refusal-recovery). По данным eval (tools/eval/
# FINDINGS.md) крупнейший драйвер потерь — ЛОЖНЫЙ отказ на factoid-вопросах:
# модель пишет «нет ответа», хотя нужный факт лежит в её же фрагментах. Этот
# промпт запускается ТОЛЬКО когда обычный ответ оказался отказом при непустых
# источниках: он запрещает отказ и заставляет модель извлечь всё релевантное,
# сохраняя те же требования к заземлению ([N] + дословная проверяемость), чтобы
# не разменять отказ на выдумку.
RECOVERY_SYSTEM_PROMPT = (
    "Ты — ассистент, который ИЗВЛЕКАЕТ из источников блокнота всё, что относится "
    "к вопросу. Предыдущая попытка ответить закончилась отказом «нет ответа», хотя "
    "во фрагментах ниже почти наверняка есть релевантные факты — их упустили.\n"
    "Правила:\n"
    "1. НЕ отказывай. В этом режиме фраза про «нет ответа» запрещена, если во "
    "фрагментах есть хоть что-то относящееся к вопросу.\n"
    "2. Пройди по КАЖДОМУ пронумерованному фрагменту и выпиши всё, что упоминает "
    "любую сущность/термин/тему из вопроса — даже косвенно, со ссылкой [N].\n"
    "3. Собери из этих фактов максимально прямой ответ. Если чего-то не хватает — "
    "сначала дай, что есть (со ссылками [N]), затем ОДНОЙ фразой отметь, какой "
    "именно детали в источниках нет.\n"
    "4. Каждое число, название, параметр, порог и дата в ответе ДОЛЖНЫ дословно "
    "присутствовать в том фрагменте, на который ты ссылаешься. Ничего не выдумывай: "
    "лучше меньше фактов, но все со ссылкой [N] и 100% проверяемые.\n"
    "5. Отвечай на языке вопроса, по делу, без описания своих размышлений."
)


def build_recovery_messages(
    question: str,
    contexts: list[ContextItem],
    *,
    schema: str = "",
) -> list[dict[str, str]]:
    """Собрать messages для восстановительного (анти-отказного) прохода: тот же
    блок [Источники], но с извлекающим system-промптом, запрещающим отказ."""
    system = RECOVERY_SYSTEM_PROMPT
    s = schema.strip()[:2000]
    if s:
        system += "\n\n[Контекст домена этого блокнота]\n" + s
    if contexts:
        ctx_block = "\n\n".join(
            f"[{c.n}] ({c.display})\n{_strip_headers(c.text)}" for c in contexts
        )
    else:
        ctx_block = "(источники не найдены)"
    user = "\n\n".join([
        "[Источники]\n" + ctx_block,
        "[Вопрос]\n" + question.strip(),
        ("Извлеки из фрагментов выше ВСЁ, что относится к вопросу хотя бы косвенно, "
         "и сформулируй ответ со ссылками [N] на использованные фрагменты. НЕ "
         "отказывай и не пиши «нет ответа»: если прямого ответа нет — приведи "
         "ближайшие связанные факты со ссылками и отметь одной фразой, чего именно "
         "не хватает. Не придумывай цифры, названия и детали, которых нет в "
         "цитируемом фрагменте."),
    ])
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


LEAKED_REASONING_TEXT = (
    "Похоже, выбранная модель вернула размышления вместо готового ответа "
    "(так делают маленькие/reasoning-модели, напр. gemma-4-e2b). Переключитесь "
    "на более крупную модель (12b/26b) в меню подключения к нейросети."
)

# Маркеры «мыслей вслух»: маленькие модели иногда пишут мета-рассуждение о задаче
# прямо в контент (без тегов <think>, которые срезает processor). Grounded-ответ
# так НИКОГДА не начинается — на eval это 0 ложных срабатываний на 74 ответах
# 12b/26b и ловит протёкший CoT e2b (7/12).
_LEAKED_REASONING_MARKERS = (
    "пользователь просит", "пользователь спрашивает", "пользователь хочет",
    "я должен", "мне нужно проанализировать", "проанализирую источник",
    "просмотрю источник", "рассмотрю источник", "давайте проанализируем",
    "the user asks", "the user wants", "the user is asking",
    "let me analyze", "i need to analyze", "i should analyze", "i will analyze",
)


def looks_like_leaked_reasoning(answer: str) -> bool:
    """True, если ответ НАЧИНАЕТСЯ с мета-рассуждения о задаче (протёкший CoT),
    а не с самого ответа. Консервативно: только по началу строки."""
    if not answer:
        return False
    head = answer.strip().lower()
    return any(head.startswith(m) for m in _LEAKED_REASONING_MARKERS)


# --------------------------------------------------------------------------- #
#  B1: архитектурный анти-выдумки — grounding-verify (детерминированный).
#
#  Промпт уже просит «не придумывай цифры/параметры». Этот слой ПРОВЕРЯЕТ, что
#  модель не проигнорировала инструкцию: находит в ответе высокоспецифичные
#  «дословно-копируемые» токены (CVE, @handle, дата, составной идентификатор —
#  хост/путь/версия-с-буквой), которых НЕТ ни в одном извлечённом фрагменте, и
#  помечает их для пользователя. НИКОГДА не переписывает ответ и не добавляет
#  фактов — только добавляет предупреждение (прошлые попытки B2/S3.1 роняли гейт
#  именно тем, что ПЕРЕПИСЫВАЛИ и плодили новую выдумку). Голые числа/проценты
#  СОЗНАТЕЛЬНО исключены: агрегаты-подсчёты (deep «разобрано 28 единиц») — это
#  легитимный вывод модели, не выдумка → нулевой FP на них.
# --------------------------------------------------------------------------- #
_GROUND_CVE = re.compile(r"CVE-\d{4}-\d+", re.I)
_GROUND_HANDLE = re.compile(r"@[A-Za-z0-9_]{3,}")
_GROUND_DATE = re.compile(r"\b\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b")
# составной идентификатор: hostname/path/dotted-file/version — токены, склеенные
# из ≥2 частей через .-_/@ (db-node-07, meta_cache.pkl, gemma-4-12b).
_GROUND_IDENT = re.compile(r"\b[A-Za-z0-9]+(?:[._\-/@][A-Za-z0-9]+)+\b")


def _ground_norm(s: str) -> str:
    # слэши унифицируем (путь из ответа `a/b` vs Windows-путь источника `a\b`)
    return re.sub(r"\s+", " ", (s or "").replace("\\", "/")).lower()


def extract_verifiable_tokens(answer: str) -> list[str]:
    """Высокоспецифичные токены ответа, которые почти всегда КОПИРУЮТСЯ из
    источника, а не сочиняются: CVE / @handle / полная дата / составной
    идентификатор (обязательно с буквой → числовые диапазоны и годы-2021
    исключаются). Чистая функция. [N]-маркеры цитат срезаются, чтобы не ловить
    номер источника как «токен»."""
    a = re.sub(r"\[\d+\]", " ", answer or "")
    toks: list[str] = []
    toks += _GROUND_CVE.findall(a)
    toks += _GROUND_HANDLE.findall(a)
    toks += _GROUND_DATE.findall(a)
    for m in _GROUND_IDENT.findall(a):
        if any(ch.isalpha() for ch in m):  # только идентификаторы с буквой
            toks.append(m)
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        k = t.lower()
        if k in seen or len(t) < 4:
            continue
        seen.add(k)
        out.append(t)
    return out


def _date_triple(tok: str) -> tuple[int, int, int] | None:
    """Дату привести к (год, месяц, день) для формат-независимого сравнения:
    `2024-03-15`, `15.03.2024`, `5.3.2024` → одинаковый кортеж. Не-дата → None.
    RU-локаль: точечный/слэш-формат считаем day-first."""
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", tok)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.fullmatch(r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})", tok)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return (y + 2000 if y < 100 else y, mo, d)
    return None


def verify_grounding(answer: str, contexts: list[ContextItem]) -> list[str]:
    """Токены ответа (extract_verifiable_tokens), которых НЕТ дословно ни в одном
    фрагменте → кандидаты на выдумку. Пусто, если всё заземлено. Хэйстек — сырой
    текст фрагментов (с заголовками [FILE_PATH]/[SOURCE_URL]): хост/путь, названный
    в ответе из заголовка источника, честно считается заземлённым. Даты сверяются
    по (год,месяц,день), а не дословно — переформатирование (ISO↔DD.MM.YYYY,
    ведущие нули) НЕ считается выдумкой (иначе ложные срабатывания)."""
    toks = extract_verifiable_tokens(answer)
    if not toks:
        return []
    raw = " ".join((getattr(c, "text", "") or "") for c in contexts)
    haystack = _ground_norm(raw)
    ctx_dates = {tr for m in _GROUND_DATE.findall(raw) if (tr := _date_triple(m))}
    out: list[str] = []
    for t in toks:
        tri = _date_triple(t)
        if tri is not None:
            if tri not in ctx_dates and _ground_norm(t) not in haystack:
                out.append(t)
        elif _ground_norm(t) not in haystack:
            out.append(t)
    return out


_GROUNDING_CAVEAT_MAX = 6


def append_grounding_caveat(
    answer: str, contexts: list[ContextItem]
) -> tuple[str, list[str]]:
    """Если в ответе есть незаземлённые специфичные токены — дописать компактное
    нейтральное предупреждение и вернуть (answer_с_пометкой, список_токенов).
    Ничего не удаляет из ответа. Возвращает исходный ответ, если всё чисто."""
    ungrounded = verify_grounding(answer, contexts)
    if not ungrounded:
        return answer, []
    shown = ungrounded[:_GROUNDING_CAVEAT_MAX]
    tail = " …" if len(ungrounded) > len(shown) else ""
    note = ("\n\n_⚠ Не найдено дословно в источниках: "
            + ", ".join(f"«{t}»" for t in shown) + tail
            + " — проверьте эти детали._")
    return answer + note, ungrounded


class _RefusalGate:
    """Придерживает потоковый вывод, пока накопленный ответ короче порога отказа
    (``is_refusal`` требует краткости) — то есть пока он ещё МОЖЕТ оказаться
    отказом, который мы собираемся перегенерировать (refusal-recovery). Как только
    длина превысила порог, отказом ответ уже не будет → сбрасываем придержанный
    буфер в реальный sink и дальше пропускаем токены напрямую.

    Если стрим закончился, не превысив порог, буфер остаётся у вызывающего: он сам
    решит показать его (обычный короткий ответ / честный отказ) через ``flush()``
    или подавить (запуская восстановительный проход начисто).
    """

    __slots__ = ("_sink", "_threshold", "_buf", "_len", "_open")

    def __init__(self, sink: Callable[[str], None], threshold: int) -> None:
        self._sink = sink
        self._threshold = threshold
        self._buf: list[str] = []
        self._len = 0
        self._open = False

    def __call__(self, delta: str) -> None:
        if self._open:
            self._sink(delta)
            return
        self._buf.append(delta)
        self._len += len(delta)
        if self._len > self._threshold:
            self.flush()

    def flush(self) -> None:
        """Сбросить придержанный буфер в sink и перейти в режим прямой передачи."""
        if self._open:
            return
        self._open = True
        if self._buf:
            self._sink("".join(self._buf))
            self._buf.clear()

    @property
    def emitted(self) -> bool:
        """True, если хоть что-то уже ушло в реальный sink (буфер сброшен)."""
        return self._open


async def answer_question(
    notebook: Any,
    question: str,
    *,
    base_url: str,
    api_key: str,
    chat_model: str,
    composer_model: str = "",
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
    deep_depth: str = "full",
    recover_on_refusal: bool = True,
    recovery_min_score: float = 0.0,
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

    async def _llm_stream(msgs: list[dict[str, str]], max_tokens: int) -> str:
        # Стриминговый вызов для финального reduce — токены летят в пузырь.
        from processor import call_llm_stream
        return await _await_with_stop(
            call_llm_stream(msgs, chat_model, base_url, api_key,
                            max_tokens=max_tokens, api_mode=api_mode,
                            on_token=on_token, stop_flag=_stopped),
            _stopped,
        )

    # Композер: для ФИНАЛЬНОГО синтеза (простой ответ + deep-reduce) используем
    # более крупную композер-модель, если она задана — богаче/вовлечённее ответ;
    # дешёвые вспом-вызовы (expansion/rerank/deep-map) остаются на быстрой chat_model.
    synth_model = (composer_model or "").strip() or chat_model

    async def _synth(msgs: list[dict[str, str]], max_tokens: int) -> str:
        return await _await_with_stop(
            call_llm(msgs, synth_model, base_url, api_key, semaphore,
                     max_tokens=max_tokens, api_mode=api_mode, prefer_reasoning_off=True),
            _stopped,
        )

    async def _synth_stream(msgs: list[dict[str, str]], max_tokens: int) -> str:
        from processor import call_llm_stream
        return await _await_with_stop(
            call_llm_stream(msgs, synth_model, base_url, api_key,
                            max_tokens=max_tokens, api_mode=api_mode,
                            on_token=on_token, stop_flag=_stopped),
            _stopped,
        )

    # Композер активен → РАСШИРЯЕМ контекст и вовлечённость ответа: больше
    # извлечённых фрагментов в синтез + больше места под развёрнутый ответ
    # (крупная модель это тянет; на быстрой chat-модели значения не трогаем).
    using_composer = synth_model != chat_model
    if using_composer:
        top_k = max(top_k, 24)
        max_context_tokens = max(max_context_tokens, 16000)
        max_answer_tokens = max(max_answer_tokens, 3000)

    def _retrieve(q: str, k: int | None = None) -> list[Any]:
        return notebook.query(q, base_url=base_url, api_key=api_key,
                              embedding_model=embedding_model, top_k=k or top_k)

    # --- Глубокий анализ (map-reduce над всем следом сущности/темы) ----------
    import deep_analysis as _da
    want_deep = (deep_mode == "on") or (
        deep_mode == "auto" and _da.is_analytical_question(question))
    if want_deep:
        # Глубина: «полно» (400 юнитов, все пачки, ~11 мин) vs «быстро» (150
        # юнитов, плотнее пачки → без иерархии, ~4-5 мин).
        fast = (deep_depth == "fast")
        try:
            res = await _run_deep_analysis(
                notebook, question, _llm=_llm, _retrieve=_retrieve,
                _llm_stream=(_llm_stream if on_token else None),
                _reduce_llm=_synth,  # финальный reduce — на композер-модели
                _reduce_llm_stream=(_synth_stream if on_token else None),
                stopped=_stopped, log=_log, cancelled=_cancelled_result,
                chat_model=chat_model, on_token=on_token,
                max_context_tokens=max_context_tokens, max_answer_tokens=max_answer_tokens,
                cap_units=(150 if fast else 400),
                max_batch_tokens=(8000 if fast else 6000),
                depth_label=("быстрый" if fast else "полный"),
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
        # Шире per-query recall: узкий top_k=16 упускал канонические сообщения
        # (напр. явный список интеграций) — берём по 60 на запрос и 80 на сущность.
        hit_lists = [_retrieve(q, 60) for q in queries]
        for ent in entities:
            hit_lists.append(_retrieve(ent, 80))
        hits = _re.merge_hits(hit_lists, cap=60)
        # Держим шире окно после реранка — перечень/сводку не собрать из 16
        # фрагментов (нужны десятки).
        rerank_keep = max(top_k, 40) if entities else max(top_k, 28)
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
    # В «Точном поиске» держим больше фрагментов в контексте (перечни/факты
    # тонут при 16); базовый режим — компактнее.
    ctx_max = 20 if enhanced else max(12, top_k)
    contexts = select_contexts(hits, max_tokens=max_context_tokens, max_items=ctx_max)

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
    messages = build_chat_messages(question, contexts, history, schema=schema,
                                   rich=using_composer)

    raw: str | None = None
    # Гейт отказа: пока стрим короче порога отказа, придерживаем токены, чтобы
    # ложный отказ, который мы собираемся перегенерировать (refusal-recovery), не
    # мелькнул в пузыре и не был затем заменён — пользователь видит только финал.
    gate: _RefusalGate | None = None
    # C4: потоковый вывод. Стрим идёт по openai-совместимому пути (LM Studio его
    # отдаёт и в native-режиме), кроме «точного поиска» (там свой много-вызовный
    # цикл). Любой сбой стрима → тихий откат на обычный call_llm.
    if on_token and not enhanced:
        from processor import call_llm_stream
        gate = _RefusalGate(on_token, len(REFUSAL_TEXT) + 40) if recover_on_refusal else None
        sink = gate if gate is not None else on_token
        try:
            # Оборачиваем в _await_with_stop: при «Стоп» задача отменяется и рвёт
            # in-flight стрим даже если модель зависла между токенами (иначе Стоп
            # ждал бы read-timeout). stop_flag внутри тоже проверяется по-строчно.
            raw = await _await_with_stop(
                call_llm_stream(
                    messages, synth_model, base_url, api_key,
                    max_tokens=max_answer_tokens, api_mode=api_mode,
                    on_token=sink, stop_flag=_stopped,
                ),
                _stopped,
            )
        except ChatCancelled:
            return _cancelled_result(contexts)
        except Exception as exc:  # noqa: BLE001
            _log(f"стриминг недоступен, обычный режим: {exc}")
            raw = None
            gate = None  # откат на не-стрим путь: гейт больше не участвует
        if _stopped():
            return _cancelled_result(contexts)

    if raw is None:
        try:
            raw = await _await_with_stop(
                call_llm(
                    messages,
                    synth_model,
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
    if looks_like_leaked_reasoning(answer):
        if gate is not None:
            gate.flush()  # придержанное показать (финальный рендер всё равно заменит)
        return ChatResult(
            answer=LEAKED_REASONING_TEXT,
            citations=[],
            contexts=[c.to_citation() for c in contexts],
            refused=False,
            model=chat_model,
            extra={"leaked_reasoning": True, "raw_answer": answer[:2000]},
        )
    refused = is_refusal(answer)

    # Refusal-recovery (второй проход). Отказ при НЕПУСТЫХ фрагментах — крупнейший
    # драйвер потерь по данным eval (tools/eval/FINDINGS.md): модель пишет «нет
    # ответа», хотя факт лежит в её же фрагментах. Делаем ОДИН строгий извлекающий
    # проход и принимаем его ТОЛЬКО если он не отказ, не протёкший CoT и заземлён
    # (есть хоть одна цитата [N]); иначе честный отказ остаётся. Легитимный отказ
    # (источники совсем про другое) переживает: повтор тоже откажет/не заземлится.
    recovered = False
    if (refused and recover_on_refusal and contexts and not _stopped()
            and float(getattr(contexts[0], "score", 0.0) or 0.0) >= recovery_min_score):
        _log("refusal-recovery: отказ при непустых источниках → извлекающий проход")
        rec_msgs = build_recovery_messages(question, contexts, schema=schema)
        rec_answer = ""
        try:
            # Первый (отказной) вывод в пузырь не попал (гейт придержал буфер) →
            # стримим восстановительный ответ начисто. Если гейт уже сбросил буфер
            # (крайний случай) или это путь без стрима — берём ответ без стрима.
            if gate is not None and not gate.emitted:
                rec_answer = (await _synth_stream(rec_msgs, max_answer_tokens) or "").strip()
            else:
                rec_answer = (await _synth(rec_msgs, max_answer_tokens) or "").strip()
        except ChatCancelled:
            return _cancelled_result(contexts)
        except Exception as exc:  # noqa: BLE001
            _log(f"refusal-recovery пропущен: {exc}")
        if _stopped():
            # call_llm_stream при «Стоп» ВОЗВРАЩАЕТ частичный текст (не бросает),
            # поэтому _await_with_stop мог не поднять ChatCancelled — как на
            # основном стрим-пути, добираем проверку здесь (иначе остановленный
            # ход отрисовался бы и осел в истории).
            return _cancelled_result(contexts)
        if (rec_answer and not is_refusal(rec_answer)
                and not looks_like_leaked_reasoning(rec_answer)
                and parse_used_citations(rec_answer, contexts)):
            answer, refused, recovered = rec_answer, False, True
        # Повтор не принят → остаётся исходный честный отказ (уже в `answer`). Буфер
        # гейта НЕ сбрасываем: отклонённый повтор мог уже уйти в поток, а финальный
        # рендер покажет отказ из ChatResult — так двойного текста не будет.
    elif gate is not None and not gate.emitted:
        gate.flush()  # обычный короткий ответ уместился в буфер — показать его

    ungrounded: list[str] = []
    if not refused:  # B1: пометить незаземлённые специфичные токены (не переписывая)
        answer, ungrounded = append_grounding_caveat(answer, contexts)
    used = parse_used_citations(answer, contexts)
    extra: dict[str, Any] = {}
    if ungrounded:
        extra["ungrounded"] = ungrounded
    if recovered:
        extra["recovered"] = True
    return ChatResult(
        answer=answer,
        citations=used,
        contexts=[c.to_citation() for c in contexts],
        refused=refused,
        model=chat_model,
        extra=extra,
    )


async def _run_deep_analysis(
    notebook: Any,
    question: str,
    *,
    _llm: Callable[[list[dict[str, str]], int], Any],
    _retrieve: Callable[..., list[Any]],
    _llm_stream: Callable[[list[dict[str, str]], int], Any] | None = None,
    _reduce_llm: Callable[[list[dict[str, str]], int], Any] | None = None,
    _reduce_llm_stream: Callable[[list[dict[str, str]], int], Any] | None = None,
    stopped: Callable[[], bool],
    log: Callable[[str], None],
    cancelled: Callable[[list[ContextItem]], ChatResult],
    chat_model: str,
    on_token: Callable[[str], None] | None,
    max_context_tokens: int,
    max_answer_tokens: int,
    cap_units: int = 400,
    max_batch_tokens: int = 6000,
    max_batches: int = 60,
    depth_label: str = "полный",
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
            est = "≈4-5 мин" if depth_label == "быстрый" else "≈10-12 мин"
            on_token(f"🔬 Глубокий анализ ({depth_label}, {est}): собираю данные "
                     f"по корпусу, анализирую по частям…\n")
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

    # 2) Широкий сбор: семантика по вопросу + ПЕРЕФРАЗИРОВКИ (другой словарь —
    # ловят иную лексику причин/контекста, которую исходная формулировка упускает)
    # + лексика по каждой сущности.
    hit_lists = [_retrieve(question, 150)]
    for q in queries[1:4]:  # перефразировки без оригинала
        hit_lists.append(_retrieve(q, 120))
    for ent in entities[:4]:
        hit_lists.append(_retrieve(ent, 250))
    hits = _re.merge_hits(hit_lists, cap=max(cap_units - 100, 200))
    log(f"deep: сущности={entities or '—'}, перефраз={max(0, len(queries) - 1)}, кандидатов={len(hits)}")
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
    # max_batches теперь — предохранитель от runaway, а не рабочая обрезка: при
    # плотных пачках (6k токенов) все ~400 юнитов укладываются в лимит и
    # обрабатываются полностью; иерархический reduce сводит все выжимки.
    truncated = 0
    if len(batches) > max_batches:
        truncated = len(batches) - max_batches
        batches = batches[:max_batches]
    log(f"deep: единиц={len(units)}, пачек={len(batches)}"
        + (f" (+{truncated} за предохранителем)" if truncated else ""))

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

    # 6) REDUCE — иерархический: если выжимок много, сначала сводим их группами
    # (по FANIN), потом финальный синтез. Так в ответ попадают ВСЕ пачки, а не
    # первые сколько-то, и промпт финального reduce не переполняется.
    FANIN = 10
    tier = 0
    while len(summaries) > FANIN:
        tier += 1
        groups = _da.group_list(summaries, FANIN)
        merged: list[str] = []
        for gi, g in enumerate(groups, 1):
            if stopped():
                raise ChatCancelled()
            try:
                r = await _llm(_da.build_merge_messages(question, g), 800)
            except ChatCancelled:
                raise
            except Exception:  # слияние группы сорвалось — сохраняем сырьё группы
                r = "\n".join(g)
            merged.append((r or "").strip() or "\n".join(g))
            log(f"deep: merge t{tier} {gi}/{len(groups)}")
        summaries = merged

    log(f"deep: финальный reduce из {len(summaries)} выжимок")
    reduce_msgs = _da.build_reduce_messages(question, summaries, schema=schema)
    # Финальный синтез — на композер-модели (богаче), если она передана; map/merge
    # выше остаются на быстрой chat_model. Фолбэк на обычный _llm при отсутствии.
    reduce_llm = _reduce_llm or _llm
    reduce_llm_stream = _reduce_llm_stream or _llm_stream
    try:
        # Финал стримим в пузырь (если есть on_token-канал); сбой стрима → обычный.
        if reduce_llm_stream is not None:
            try:
                raw = await reduce_llm_stream(reduce_msgs, max_answer_tokens)
            except ChatCancelled:
                raise
            except Exception:  # noqa: BLE001
                raw = await reduce_llm(reduce_msgs, max_answer_tokens)
        else:
            raw = await reduce_llm(reduce_msgs, max_answer_tokens)
    except ChatCancelled:
        raise
    answer = (raw or "").strip()
    if not answer:
        return None
    if looks_like_leaked_reasoning(answer):
        return ChatResult(
            answer=LEAKED_REASONING_TEXT, citations=[], contexts=[],
            refused=False, model=chat_model,
            extra={"deep": True, "leaked_reasoning": True, "raw_answer": answer[:2000]},
        )

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
    refused = is_refusal(answer)
    ungrounded: list[str] = []
    final = answer
    if not refused:  # B1: заземление против всех разобранных единиц (не только цитат)
        final, ungrounded = append_grounding_caveat(answer, list(numbered_all.values()))
    if truncated:
        final += (f"\n\n_(разобрано {len(numbered_all)} единиц; часть корпуса за лимитом — "
                  f"уточните сущность/период для полного охвата)_")
    extra = {"deep": True, "units": len(numbered_all), "batches": len(batches)}
    if ungrounded:
        extra["ungrounded"] = ungrounded
    return ChatResult(
        answer=final,
        citations=[c.to_citation() for c in used_ctx],
        contexts=[c.to_citation() for c in used_ctx],
        refused=refused,
        model=chat_model,
        extra=extra,
    )
