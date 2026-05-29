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
Детерминированная агрегация результатов MAP (доменно-нейтральная).

MAP-модель извлекает элементы; счёт/группировка/дедуп считаются ЗДЕСЬ, в коде,
а не галлюцинируются моделью, и передаются в REDUCE как «истина». Никаких
доменных понятий (severity/CVE) — фасеты нейтральные:
  category, item, source, level (опц.), value (опц.), entities.

Поддерживает и текущую MAP-схему (findings[].{type,explanation,evidence_refs}),
и нейтральную (records/items[].{item,category,value,context,source,...}).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator

# Универсальные сущности (нейтральные): id-коды, числа, даты, email, url.
_ENTITY_RES: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "id": re.compile(r"\b(?=[\w-]*\d)[A-Za-z][\w]*(?:-[\w]+){1,5}\b"),
    "date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}\.\d{2}\.\d{4}\b"),
}

_FIRST = "item point claim name title".split()
_CATEGORY = "category type kind class".split()
_LEVEL = "level severity importance priority".split()
_VALUE = "value amount count metric".split()
_SOURCE = "source asset host file".split()


def _first_present(d: dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return str(v)
    return ""


def _first_evidence_source(rec: dict[str, Any]) -> str:
    for er in rec.get("evidence_refs") or []:
        if isinstance(er, dict):
            f = str(er.get("file") or er.get("source") or "").strip()
            if f:
                return f
    return ""


def _record_text(rec: dict[str, Any]) -> str:
    parts: list[str] = []
    for k, v in rec.items():
        if k == "evidence_refs":
            for er in v or []:
                if isinstance(er, dict):
                    parts.append(str(er.get("quote") or ""))
            continue
        if isinstance(v, str):
            parts.append(v)
    return " \n".join(parts)


def extract_facets(rec: dict[str, Any], outer_source: str = "") -> dict[str, Any]:
    """Нейтральные фасеты из записи MAP (любой схемы).

    Учитывает опциональный словарь `fields` (поля под задачу из query-adaptive
    MAP): его значения дополняют верхнеуровневые поля, не перекрывая их.
    """
    flat = dict(rec)
    extra = rec.get("fields")
    if isinstance(extra, dict):
        for k, v in extra.items():
            flat.setdefault(k, v)
    category = _first_present(flat, _CATEGORY)
    item = _first_present(flat, _FIRST)
    if not item:
        expl = _first_present(flat, ["explanation", "observation", "context", "rationale", "reason"])
        item = expl[:80]
    source = _first_present(flat, _SOURCE) or _first_evidence_source(rec) or outer_source
    level = _first_present(flat, _LEVEL).lower()
    value = _first_present(flat, _VALUE)
    blob = _record_text(flat)
    entities: list[str] = []
    for pat in _ENTITY_RES.values():
        for m in pat.findall(blob):
            tok = m if isinstance(m, str) else m[0]
            if tok and tok not in entities:
                entities.append(tok)
    return {
        "category": category.lower(),
        "item": item.lower().strip(),
        "source": source,
        "level": level,
        "value": value,
        "entities": entities[:8],
        "has_source": bool(source),
    }


def iter_records_from_map(map_results: list[str]) -> Iterator[dict[str, Any]]:
    """Распарсить MAP-JSON строки → записи с прикреплёнными нейтральными фасетами."""
    for raw in map_results:
        if not raw or not raw.strip():
            continue
        s = raw.strip()
        if "<results>" in s.lower():
            m = re.search(r"<results>\s*([\s\S]*?)\s*</results>", s, re.I)
            if m:
                s = m.group(1).strip()
        if s.startswith("```"):
            s = re.sub(r"^```\w*\n?", "", s)
            s = re.sub(r"\n?```\s*$", "", s)
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("no_relevant_data"):
            continue
        outer_source = str(obj.get("source") or obj.get("file") or "")
        records = obj.get("findings") or obj.get("items") or obj.get("records") or []
        for r in records:
            if not isinstance(r, dict):
                continue
            yield {**r, "_facets": extract_facets(r, outer_source=outer_source)}


def _dedup_key(facets: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, ...]:
    parts: list[str] = []
    for k in keys:
        v = facets.get(k)
        if isinstance(v, list):
            parts.append(",".join(map(str, v)))
        else:
            parts.append(str(v or ""))
    return tuple(parts)


@dataclass(slots=True)
class AggregateSummary:
    total: int = 0
    unique: int = 0
    by_category: list[tuple[str, int]] = field(default_factory=list)
    axis: str | None = None
    by_axis: list[tuple[str, int]] = field(default_factory=list)
    by_level: list[tuple[str, int]] = field(default_factory=list)
    top_entities: list[tuple[str, int]] = field(default_factory=list)
    with_source: int = 0


def categorize(
    records: list[dict[str, Any]],
    dedup_keys: tuple[str, ...] = ("category", "item"),
    axis: str | None = None,
    top_n: int = 15,
) -> AggregateSummary:
    cat_counter: Counter[str] = Counter()
    axis_counter: Counter[str] = Counter()
    level_counter: Counter[str] = Counter()
    ent_counter: Counter[str] = Counter()
    seen: set[tuple[str, ...]] = set()
    with_src = 0

    for r in records:
        facets = r.get("_facets") or extract_facets(r)
        key = _dedup_key(facets, dedup_keys)
        if key in seen:
            continue
        seen.add(key)
        if facets.get("category"):
            cat_counter[facets["category"]] += 1
        if facets.get("has_source"):
            with_src += 1
        if facets.get("level"):
            level_counter[facets["level"]] += 1
        for e in facets.get("entities") or []:
            ent_counter[e] += 1
        if axis:
            av = facets.get(axis)
            if isinstance(av, list):
                for x in av:
                    axis_counter[str(x)] += 1
            elif av:
                axis_counter[str(av)] += 1

    return AggregateSummary(
        total=len(records),
        unique=len(seen),
        by_category=cat_counter.most_common(top_n),
        axis=axis if axis and axis != "category" else None,
        by_axis=axis_counter.most_common(top_n) if axis and axis != "category" else [],
        by_level=level_counter.most_common(top_n),
        top_entities=ent_counter.most_common(top_n),
        with_source=with_src,
    )


def summary_markdown(summary: AggregateSummary, language: str = "ru") -> str:
    if summary.unique == 0:
        return ""
    ru = language == "ru"
    lines: list[str] = []
    head = ("### Детерминированная сводка (числа — истина, не пересчитывай)" if ru
            else "### Deterministic summary (treat these numbers as ground truth)")
    lines.append(head)
    if ru:
        lines.append(f"- Уникальных элементов: {summary.unique} (из {summary.total} до дедупа)")
        lines.append(f"- С указанием источника: {summary.with_source}")
    else:
        lines.append(f"- Unique items: {summary.unique} (of {summary.total} pre-dedup)")
        lines.append(f"- With source: {summary.with_source}")
    if summary.by_category:
        label = "По категориям" if ru else "By category"
        lines.append(f"- {label}: " + ", ".join(f"{k}: {v}" for k, v in summary.by_category))
    if summary.by_axis:
        label = f"По «{summary.axis}»" if ru else f"By {summary.axis}"
        lines.append(f"- {label}: " + ", ".join(f"{k}: {v}" for k, v in summary.by_axis))
    if summary.by_level:
        label = "По уровню" if ru else "By level"
        lines.append(f"- {label}: " + ", ".join(f"{k}: {v}" for k, v in summary.by_level))
    if summary.top_entities:
        label = "Топ сущностей" if ru else "Top entities"
        lines.append(f"- {label}: " + ", ".join(f"{k} ({n})" for k, n in summary.top_entities))
    return "\n".join(lines)


def build_category_block(
    map_results: list[str],
    dedup_keys: tuple[str, ...] = ("category", "item"),
    axis: str | None = None,
    language: str = "ru",
) -> str:
    records = list(iter_records_from_map(map_results))
    if not records:
        return ""
    return summary_markdown(categorize(records, dedup_keys=dedup_keys, axis=axis), language=language)
