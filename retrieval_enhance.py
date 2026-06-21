# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
# ForestOptiLM is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
Усиление retrieval по мотивам LLM-Wiki Карпатого / qmd: query-expansion и
listwise LLM-реранкинг.

Идея (qmd): сгенерировать 1-2 перефразировки вопроса и искать по всем (выше
recall — добивает «нет ответа»), а затем переранжировать top-N кандидатов одним
LLM-вызовом по реальной релевантности (выше precision — модель видит меньше
шума). Здесь — ТОЛЬКО чистые функции (сборка промптов, парсинг ответа, слияние
кандидатов): сами LLM-вызовы делает notebook_chat, что упрощает тесты.
"""
from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
#  Query expansion
# ---------------------------------------------------------------------------

_EXPAND_SYSTEM = (
    "Ты помогаешь искать по базе знаний. По заданному вопросу придумай "
    "альтернативные формулировки для поиска: синонимы, другой ракурс, ключевые "
    "термины. Это НЕ ответ на вопрос, а только переформулировки запроса."
)


def build_expansion_messages(question: str, n: int = 2) -> list[dict[str, str]]:
    """messages для генерации n перефразировок (модель вернёт JSON-массив строк)."""
    user = (
        f"Вопрос: {question.strip()}\n\n"
        f"Верни РОВНО {n} альтернативные формулировки этого вопроса в виде "
        f"JSON-массива строк, например [\"...\", \"...\"]. Только JSON, без пояснений."
    )
    return [
        {"role": "system", "content": _EXPAND_SYSTEM},
        {"role": "user", "content": user},
    ]


def _extract_json_array(raw: str) -> list[Any]:
    """Достать первый JSON-массив из текста модели (терпимо к обёрткам/мусору)."""
    if not raw:
        return []
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        val = json.loads(m.group(0))
        return val if isinstance(val, list) else []
    except Exception:
        return []


def parse_expansions(raw: str, question: str, n: int = 2) -> list[str]:
    """Список запросов: [оригинал] + до n уникальных непустых перефразировок."""
    base = question.strip()
    out = [base]
    seen = {base.lower()}
    for item in _extract_json_array(raw):
        s = str(item).strip()
        if not s or s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
        if len(out) >= n + 1:
            break
    return out


def merge_hits(hit_lists: list[list[Any]], cap: int = 30) -> list[Any]:
    """Слить кандидатов из нескольких запросов: дедуп по chunk_id (берём лучший
    скор), сортировка по убыванию скора, ограничение до cap."""
    best: dict[str, Any] = {}
    for hits in hit_lists:
        for h in hits or []:
            cid = str(getattr(h, "chunk_id", "") or id(h))
            score = float(getattr(h, "score", 0.0) or 0.0)
            prev = best.get(cid)
            if prev is None or score > float(getattr(prev, "score", 0.0) or 0.0):
                best[cid] = h
    merged = sorted(best.values(), key=lambda h: float(getattr(h, "score", 0.0) or 0.0), reverse=True)
    return merged[:cap]


# ---------------------------------------------------------------------------
#  Listwise LLM re-ranking
# ---------------------------------------------------------------------------

_RERANK_SYSTEM = (
    "Ты — точный реранкер поиска. Тебе дают вопрос и пронумерованные фрагменты. "
    "Оцени, какие фрагменты реально помогают ответить на вопрос."
)


def build_rerank_messages(question: str, candidates: list[Any], snippet_chars: int = 320) -> list[dict[str, str]]:
    """messages для listwise-реранка: модель вернёт JSON-массив номеров (1-based)
    от самого релевантного к наименее, только релевантные."""
    lines = []
    for i, h in enumerate(candidates, 1):
        text = str(getattr(h, "text", "") or "").replace("\n", " ").strip()
        if len(text) > snippet_chars:
            text = text[:snippet_chars] + "…"
        lines.append(f"[{i}] {text}")
    user = (
        f"Вопрос: {question.strip()}\n\n"
        "Фрагменты:\n" + "\n".join(lines) + "\n\n"
        "Верни JSON-массив НОМЕРОВ фрагментов от самого релевантного к наименее. "
        "Включи только те, что реально относятся к вопросу. Только JSON, например [3,1,7]."
    )
    return [
        {"role": "system", "content": _RERANK_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_rerank_order(raw: str, n: int) -> list[int]:
    """0-based порядок из ответа модели: валидные уникальные индексы в пределах n."""
    order: list[int] = []
    seen: set[int] = set()
    for item in _extract_json_array(raw):
        try:
            idx = int(item) - 1  # модель отдаёт 1-based
        except Exception:
            continue
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            order.append(idx)
    return order


def apply_rerank(candidates: list[Any], order: list[int], top_k: int) -> list[Any]:
    """Переупорядочить кандидатов по order (релевантные сперва), добить хвостом в
    исходном порядке (recall не теряем), обрезать до top_k. Пустой order →
    исходный порядок (фолбэк на RRF)."""
    if not candidates:
        return []
    used: set[int] = set()
    out: list[Any] = []
    for idx in order:
        if 0 <= idx < len(candidates) and idx not in used:
            used.add(idx)
            out.append(candidates[idx])
    for i, h in enumerate(candidates):  # хвост: то, что модель не назвала
        if i not in used:
            out.append(h)
    return out[:top_k]
