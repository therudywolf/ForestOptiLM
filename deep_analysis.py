# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
# ForestOptiLM is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
Глубокий анализ: map-reduce над всем следом сущности/темы в корпусе.

Обычный RAG («найди top-N → ответь одним промптом») структурно проваливает
АГРЕГИРУЮЩИЕ вопросы («портрет человека X», «сводка по теме Y», «как устроен Z»):
ответ размазан по сотням фрагментов, а в промпт влезают десятки. Здесь — как
работал бы аналитик: собрать СОТНИ релевантных единиц (+ соседей по источнику
для контекста), выжать сигналы по пачкам (map), свести в цельный разбор (reduce).

Формат-агностично: «единица» — любой чанк (реплика чата, раздел PDF/DOCX, строка
таблицы, запись JSON, кусок кода); «соседи» — смежные чанки того же источника
(chunk_index ± radius). Для чата это соседние сообщения, для PDF — соседние
абзацы: один механизм на все форматы.

Здесь только чистые функции (классификатор, сбор соседей, разбиение на пачки,
сборка промптов, парсинг). LLM-вызовы и retrieval делает вызывающий
(notebook_chat), что упрощает тесты.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
#  Классификатор: аналитический (агрегирующий) вопрос или обычный factoid
# ---------------------------------------------------------------------------

# Маркеры вопросов, ответ на которые нужно СИНТЕЗИРОВАТЬ из многих фрагментов,
# а не найти в одном. Кириллица + латиница (корпус может быть любым).
_ANALYTICAL_MARKERS = (
    "портрет", "проанализир", "анализ", "охарактеризу", "характеристик",
    "сводк", "обзор", "резюмиру", "резюме", "суммир", "итог", "выжимк",
    "как устроен", "как работает", "что вообще", "что происходил", "чем занимает",
    "все упоминани", "всех упоминани", "все случаи", "весь контекст", "по всем",
    "полная картина", "общая картина", "динамик", "тенденц", "паттерн",
    "стиль обще", "стиль работ", "как взаимодейств", "как общаться", "чего нельзя",
    "profile", "analyze", "analysis", "summariz", "summary", "overview",
    "everything about", "all mentions", "characteriz", "how does", "patterns",
)
# Явно НЕ аналитические: короткие точечные «сколько/какой/когда/где/кто именно».
_FACTOID_HINT = re.compile(r"^\s*(сколько|когда|где|какой айпи|во сколько|"
                           r"how many|when did|what time)\b", re.IGNORECASE)


def is_analytical_question(q: str) -> bool:
    """Эвристика: похоже ли на агрегирующий вопрос (нужен глубокий разбор).

    Дёшево и без LLM: ловит явные маркеры синтеза. Ложные срабатывания дешевле
    пропусков — но короткие factoid-подсказки в начале строки исключаем."""
    s = (q or "").strip().lower()
    if not s:
        return False
    if _FACTOID_HINT.match(s):
        return False
    return any(m in s for m in _ANALYTICAL_MARKERS)


# ---------------------------------------------------------------------------
#  Единица анализа (формат-агностична)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Unit:
    chunk_id: str
    source_path: str
    text: str
    chunk_index: int | None = None
    score: float = 0.0
    is_neighbor: bool = False  # добран как сосед (для контекста), не как хит

    @staticmethod
    def _ci(meta: dict) -> int | None:
        try:
            v = meta.get("chunk_index")
            return int(v) if v is not None else None
        except Exception:
            return None

    @classmethod
    def from_hit(cls, h: Any) -> "Unit":
        meta = getattr(h, "metadata", None) or {}
        return cls(
            chunk_id=str(getattr(h, "chunk_id", "") or ""),
            source_path=str(getattr(h, "source_path", "") or ""),
            text=str(getattr(h, "text", "") or ""),
            chunk_index=cls._ci(meta),
            score=float(getattr(h, "score", 0.0) or 0.0),
        )

    @classmethod
    def from_meta(cls, m: dict, *, is_neighbor: bool = False) -> "Unit":
        return cls(
            chunk_id=str(m.get("chunk_id") or ""),
            source_path=str(m.get("source_path") or ""),
            text=str(m.get("text") or ""),
            chunk_index=cls._ci(m.get("metadata") or {}),
            score=0.0,
            is_neighbor=is_neighbor,
        )


def expand_with_neighbors(
    hits: list[Any],
    meta: list[dict],
    *,
    radius: int = 1,
    cap: int = 500,
) -> list[Unit]:
    """К найденным чанкам добавить соседей того же источника (chunk_index ± radius).

    «Соседи» дают диалоговый/документный контекст, которого нет в самом хите: для
    чата — на что человек отвечал и что ответили ему; для PDF — окружающие абзацы.
    Дедуп по chunk_id, хиты идут первыми (в порядке релевантности), затем соседи.
    Ограничение cap применяется к итогу. radius=0 отключает добор соседей.
    """
    units: list[Unit] = [Unit.from_hit(h) for h in hits]
    if radius <= 0:
        # только дедуп + cap
        seen: set[str] = set()
        out: list[Unit] = []
        for u in units:
            if u.chunk_id and u.chunk_id not in seen:
                seen.add(u.chunk_id)
                out.append(u)
            if len(out) >= cap:
                break
        return out

    # Индекс meta только по источникам, встретившимся в хитах (не по всему корпусу).
    src_of_interest = {u.source_path for u in units if u.source_path}
    by_src_idx: dict[tuple[str, int], dict] = {}
    for m in meta:
        sp = str(m.get("source_path") or "")
        if sp not in src_of_interest:
            continue
        ci = Unit._ci(m.get("metadata") or {})
        if ci is not None:
            by_src_idx[(sp, ci)] = m

    seen = set()
    out = []

    def _add(u: Unit) -> bool:
        if not u.chunk_id or u.chunk_id in seen:
            return True  # уже есть — не стоп
        seen.add(u.chunk_id)
        out.append(u)
        return len(out) < cap

    for u in units:
        if not _add(u):
            break
        if u.chunk_index is None or not u.source_path:
            continue
        for d in range(1, radius + 1):
            for ci in (u.chunk_index - d, u.chunk_index + d):
                nb = by_src_idx.get((u.source_path, ci))
                if nb is not None:
                    if not _add(Unit.from_meta(nb, is_neighbor=True)):
                        return out
    return out


# ---------------------------------------------------------------------------
#  Разбиение на пачки под map (по бюджету токенов)
# ---------------------------------------------------------------------------

def batch_units(
    units: list[Unit],
    *,
    max_batch_tokens: int = 2500,
    count_tokens: Callable[[str], int] | None = None,
) -> list[list[tuple[int, Unit]]]:
    """Сгруппировать единицы в пачки под бюджет токенов, сохраняя ГЛОБАЛЬНЫЙ
    номер [n] каждой единицы (сквозная нумерация для цитат сквозь map→reduce)."""
    if count_tokens is None:
        def count_tokens(t: str) -> int:  # грубая оценка без tiktoken (для тестов)
            return max(1, len(t) // 4)
    batches: list[list[tuple[int, Unit]]] = []
    cur: list[tuple[int, Unit]] = []
    used = 0
    for i, u in enumerate(units, 1):
        tok = count_tokens(u.text)
        if cur and used + tok > max_batch_tokens:
            batches.append(cur)
            cur, used = [], 0
        cur.append((i, u))
        used += tok
    if cur:
        batches.append(cur)
    return batches


def _fmt_units(numbered: list[tuple[int, Unit]], per_unit_chars: int = 1100) -> str:
    lines = []
    for n, u in numbered:
        t = u.text.strip()
        if len(t) > per_unit_chars:
            t = t[:per_unit_chars] + "…"
        tag = " (контекст)" if u.is_neighbor else ""
        lines.append(f"[{n}]{tag} {t}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
#  Промпты map / reduce
# ---------------------------------------------------------------------------

_MAP_SYSTEM = (
    "Ты — внимательный аналитик. Тебе дают ПАЧКУ фрагментов из корпуса данных и "
    "цель анализа. Выпиши из ЭТОЙ пачки только факты и наблюдения, относящиеся к "
    "цели, каждое со ссылкой на номер фрагмента [N]. Не выдумывай, опирайся строго "
    "на текст. Фрагменты с пометкой (контекст) — окружение, используй их для "
    "понимания, но факт приписывай тому, кто его сказал/о ком он. Если в пачке нет "
    "ничего по цели — ответь ровно: НЕТ."
)

_REDUCE_SYSTEM = (
    "Ты — аналитик. Тебе дают цель анализа и НАБЛЮДЕНИЯ, собранные по всему "
    "корпусу (из разных пачек, со ссылками [N]). Синтезируй цельный, структурный, "
    "развёрнутый ответ по цели: сгруппируй по темам, выдели паттерны, приведи "
    "показательные примеры со ссылками [N]. Опирайся ТОЛЬКО на наблюдения. Если "
    "чего-то в данных нет — честно отметь пробел, не домысливай."
)

_MAP_EMPTY = "НЕТ"


def build_map_messages(goal: str, numbered: list[tuple[int, Unit]]) -> list[dict[str, str]]:
    user = (
        f"[Цель анализа]\n{goal.strip()}\n\n"
        f"[Фрагменты]\n{_fmt_units(numbered)}\n\n"
        f"Выпиши относящиеся к цели факты/наблюдения со ссылками [N]. "
        f"Если ничего нет — ответь ровно: {_MAP_EMPTY}."
    )
    return [
        {"role": "system", "content": _MAP_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_map_result(raw: str) -> str:
    """Вернуть выжимку map, или '' если пачка пустая/ничего не дала."""
    s = (raw or "").strip()
    if not s:
        return ""
    # снять think-обёртки на всякий случай (call_llm обычно уже снял)
    compact = re.sub(r"\s+", " ", s).strip().upper()
    if compact == _MAP_EMPTY or compact.startswith(_MAP_EMPTY + " ") or compact in ("НЕТ.", "NONE", "N/A"):
        return ""
    return s


def build_reduce_messages(
    goal: str, summaries: list[str], *, schema: str = ""
) -> list[dict[str, str]]:
    system = _REDUCE_SYSTEM
    sc = (schema or "").strip()[:2000]
    if sc:
        system += "\n\n[Контекст домена]\n" + sc
    joined = "\n\n".join(f"— {s.strip()}" for s in summaries if s.strip())
    user = (
        f"[Цель анализа]\n{goal.strip()}\n\n"
        f"[Наблюдения по корпусу]\n{joined}\n\n"
        f"Синтезируй развёрнутый структурный разбор по цели. Сохрани ссылки [N] "
        f"на конкретные наблюдения."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@dataclass(slots=True)
class DeepPlan:
    """Разложение задачи глубокого анализа (для оркестратора)."""
    goal: str
    entities: list[str] = field(default_factory=list)
    units: list[Unit] = field(default_factory=list)
    batches: list[list[tuple[int, Unit]]] = field(default_factory=list)
