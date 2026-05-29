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
QueryPlan — понимание задачи пользователя (доменно-нейтральное).

Концепт продукта: «дал большой файл / кучу разных файлов → вбил задачу →
получил итог, оптимизированно». Поэтому ядро ничего не знает про конкретный
домен (безопасность, юр-доки, логи, код — что угодно). Из запроса выводится:
- intent (что вообще нужно сделать),
- схема извлечения (ЧТО тащить из каждого фрагмента под эту задачу),
- ключевые термины/сущности (для scout и точного поиска),
- ось группировки и форма ответа.

Ядро детерминированное (правила), тестируется без LLM; схема извлечения может
дополнительно обогащаться composer-моделью.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Намерения (нейтральные). Порядок = приоритет при множественном совпадении ──
_INTENT_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("compare", (
        "сравн", "сравнен", "diff", "различ", "разниц", "было стало", "было/стало",
        "что изменил", "по сравнению", "versus", " vs ", "сопостав",
    )),
    ("classify", (
        "классифиц", "категориз", "по типам", "по категори", "сгруппируй по тип",
        "таксоном", "classify", "categor", "разбей на", "распредели по",
    )),
    ("count", (
        "сколько", "количеств", "посчитай", "подсчитай", "count", "статистик",
        "распределен", "how many", "breakdown", "по числу", "частот",
    )),
    ("prioritize", (
        "приорит", "что важн", "в первую очередь", "самые", "наиболее", "top",
        "топ", "ранжир", "rank", "prioriti", "важнейш", "ключев",
    )),
    ("explain", (
        "почему", "причин", "объясни", "как работает", "как устроен", "explain",
        "why", "обоснуй", "root cause", "первопричин",
    )),
    ("extract", (
        "извлеки", "собери", "вытащи", "выпиши", "найди все", "перечисл", "список",
        "extract", "collect", "list all", "выгрузи", "достань", "все упоминан",
    )),
    ("filter", (
        "только", "лишь", "исключ", "где ", "where", "фильтр", "отбери", "с услови",
        "содержащие", "у которых",
    )),
    ("summarize", (
        "кратко", "резюме", "summary", "обзор", "вкратце", "summarize", "саммари",
        "tl;dr", "общая картина", "overview", "о чём", "суть",
    )),
]

# Нейтральные оси группировки. group_by хранит «сырое» слово после «по/by»;
# facet_axis() приводит его к доступному фасету агрегации.
_GROUP_BY_RE = re.compile(
    r"(?:сгруппир\w*\s+по|группир\w*\s+по|по\s+кажд\w*|разбей\s+по|по|by|per|group\s+by)\s+"
    r"([\wа-яё/.\-]{3,40})",
    re.I,
)

_AXIS_SYNONYMS: dict[str, tuple[str, ...]] = {
    "source": ("файл", "файлам", "файлу", "источник", "источникам", "file", "files",
               "source", "хост", "хостам", "узел", "узлам", "host", "asset", "сервер"),
    "category": ("тип", "типам", "типу", "категори", "type", "category", "класс", "kind"),
    "level": ("уровн", "степен", "level", "severity", "priorit", "критичн", "важност"),
    "entity": ("сущност", "id", "идентификатор", "ключ", "entity", "code", "номер"),
    "date": ("дат", "дате", "датам", "date", "день", "месяц", "year", "год"),
}

_INTENT_OUTPUT_STYLE: dict[str, str] = {
    "compare": "comparison",
    "classify": "matrix",
    "count": "stats",
    "prioritize": "ranked_list",
    "extract": "table",
    "filter": "table",
    "summarize": "brief",
    "explain": "narrative",
    "analyze": "report",
}

# Схема извлечения под задачу (нейтральные поля записи MAP).
_INTENT_EXTRACTION_FIELDS: dict[str, list[str]] = {
    "compare":    ["item", "category", "state", "context", "source"],
    "classify":   ["item", "category", "rationale", "source"],
    "count":      ["item", "category", "source"],
    "prioritize": ["item", "category", "importance", "rationale", "source"],
    "extract":    ["item", "category", "value", "context", "source"],
    "filter":     ["item", "category", "value", "context", "source"],
    "summarize":  ["point", "category", "source"],
    "explain":    ["claim", "reason", "evidence", "source"],
    "analyze":    ["item", "category", "observation", "context", "source"],
}

# Универсальные детекторы сущностей (нейтральные, не доменные).
_ENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "url": re.compile(r"\bhttps?://[^\s)]+", re.I),
    "ip": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    # код/идентификатор: буквы+цифры с разделителями (CVE-…, INV-2024-7, A1B2-C3)
    "id": re.compile(r"\b(?=[\w-]*\d)[A-Za-z][\w]*(?:-[\w]+){1,5}\b"),
    "number": re.compile(r"\b\d[\d.,]{2,}\b"),
    "date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}\.\d{2}\.\d{4}\b"),
}

_QUOTED_RE = re.compile(r"[\"'«»“”]([^\"'«»“”]{2,60})[\"'«»“”]")

_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "all", "any", "with", "from", "this", "that", "find",
    "show", "list", "what", "which", "into", "each",
    "найди", "покажи", "выведи", "это", "для", "все", "всех", "что", "как",
    "сделай", "дай", "есть", "или", "так", "там", "над", "мне", "нужно",
    "собери", "извлеки", "посчитай", "сгруппируй",
})


@dataclass(slots=True)
class QueryPlan:
    """Структурный план задачи, выведенный из запроса пользователя."""

    query: str
    language: str = "en"                 # "ru" | "en"
    intent: str = "analyze"
    intents: list[str] = field(default_factory=list)
    key_terms: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)
    group_by: str | None = None          # «сырое» слово оси из запроса
    output_style: str = "report"
    extraction_fields: list[str] = field(default_factory=list)
    extraction_directive: str = ""

    # ── derived helpers ────────────────────────────────────────────
    def facet_axis(self) -> str | None:
        """Привести group_by к доступному фасету агрегации (source/category/level/...)."""
        if not self.group_by:
            return None
        g = self.group_by.lower()
        for facet, syns in _AXIS_SYNONYMS.items():
            if any(g.startswith(s) or s in g for s in syns):
                return facet
        return "category"

    def dedup_keys(self) -> tuple[str, ...]:
        """Нейтральные фасеты для дедупликации/агрегации под эту задачу."""
        axis = self.facet_axis()
        if axis:
            return (axis, "item")
        if self.intent in ("compare", "prioritize"):
            return ("item", "source")
        return ("category", "item")

    def language_hint(self) -> str:
        return "Ответь строго на русском языке." if self.language == "ru" else ""

    def summary(self) -> str:
        bits = [f"intent={self.intent}", f"out={self.output_style}"]
        if self.group_by:
            bits.append(f"group_by={self.group_by}->{self.facet_axis()}")
        if self.extraction_fields:
            bits.append("fields=" + ",".join(self.extraction_fields))
        if self.key_terms:
            bits.append("terms=" + ",".join(self.key_terms[:6]))
        n_ent = sum(len(v) for v in self.entities.values())
        if n_ent:
            bits.append(f"entities={n_ent}")
        return " ".join(bits)


def _detect_language(query: str) -> str:
    return "ru" if re.search(r"[А-Яа-яЁё]", query or "") else "en"


def _detect_intents(low: str) -> list[str]:
    found: list[str] = []
    for intent, pats in _INTENT_PATTERNS:
        if any(p in low for p in pats):
            found.append(intent)
    return found


def _detect_group_by(query: str) -> str | None:
    m = _GROUP_BY_RE.search(query or "")
    if not m:
        return None
    token = m.group(1).strip().strip(".,")
    if token.lower() in _STOPWORDS or len(token) < 3:
        return None
    return token


def _extract_entities(query: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, pat in _ENTITY_PATTERNS.items():
        vals = sorted({m if isinstance(m, str) else m[0] for m in pat.findall(query)})
        if vals:
            out[name] = vals[:20]
    return out


def _extract_key_terms(query: str, entities: dict[str, list[str]]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def _add(t: str) -> None:
        k = t.lower()
        if k and k not in seen and k not in _STOPWORDS and len(k) >= 3:
            seen.add(k)
            terms.append(t)

    for m in _QUOTED_RE.findall(query):
        _add(m.strip())
    for vals in entities.values():
        for v in vals:
            _add(v)
    # Технические токены: с разделителями или цифрами (lodash@1.2, payment_service)
    for tok in re.findall(r"[A-Za-zА-Яа-яЁё][\w.\-/@]{2,}", query):
        tok = tok.rstrip(".,/")
        if any(c in tok for c in "._-/@") or any(ch.isdigit() for ch in tok):
            _add(tok)
    # Содержательные слова длиной 4+ (без стоп-слов) — для scout/поиска
    for tok in re.findall(r"[A-Za-zА-Яа-яЁё]{4,}", query):
        if tok.lower() not in _STOPWORDS:
            _add(tok)
        if len(terms) >= 24:
            break
    return terms[:24]


def _derive_extraction(intent: str, group_by: str | None, language: str) -> tuple[list[str], str]:
    fields = list(_INTENT_EXTRACTION_FIELDS.get(intent, _INTENT_EXTRACTION_FIELDS["analyze"]))
    if language == "ru":
        directive = (
            "Извлеки из фрагмента элементы, относящиеся к задаче пользователя. "
            "Для каждого элемента заполни поля: " + ", ".join(fields) + ". "
            "Опирайся только на текст фрагмента, ничего не выдумывай."
        )
        if group_by:
            directive += f" По возможности укажи значение для группировки по «{group_by}»."
    else:
        directive = (
            "Extract items from the fragment relevant to the user's task. "
            "For each item fill the fields: " + ", ".join(fields) + ". "
            "Rely only on the fragment text; do not invent anything."
        )
        if group_by:
            directive += f" Where possible, include a value to group by '{group_by}'."
    return fields, directive


def build_query_plan(query: str) -> QueryPlan:
    """Детерминированный разбор запроса в нейтральный QueryPlan (без LLM)."""
    q = (query or "").strip()
    low = q.lower()
    language = _detect_language(q)
    entities = _extract_entities(q)
    intents = _detect_intents(low)
    intent = intents[0] if intents else "analyze"
    group_by = _detect_group_by(q)
    key_terms = _extract_key_terms(q, entities)
    output_style = _INTENT_OUTPUT_STYLE.get(intent, "report")
    fields, directive = _derive_extraction(intent, group_by, language)
    return QueryPlan(
        query=q,
        language=language,
        intent=intent,
        intents=intents or ["analyze"],
        key_terms=key_terms,
        entities=entities,
        group_by=group_by,
        output_style=output_style,
        extraction_fields=fields,
        extraction_directive=directive,
    )


# Контракт ответа под форму (нейтральный) — для REDUCE.
_OUTPUT_STYLE_DIRECTIVE: dict[str, str] = {
    "comparison": (
        "Сформируй ответ как сравнение: что появилось, что пропало, что совпадает; "
        "подсвети различия."
    ),
    "ranked_list": (
        "Выведи результат ранжированным списком по важности/приоритету, сверху — самое значимое."
    ),
    "matrix": "Сгруппируй результат в таблицу/матрицу по запрошенной оси; не теряй элементы.",
    "stats": "Дай количественную сводку: итоговые числа и распределения, затем краткие выводы.",
    "brief": "Дай сжатое резюме: ключевые пункты и общая картина, без воды.",
    "table": "Представь извлечённые элементы таблицей с заполненными полями.",
    "narrative": "Дай связное объяснение с опорой на источники.",
    "report": "",
}


def output_style_directive(plan: QueryPlan) -> str:
    base = _OUTPUT_STYLE_DIRECTIVE.get(plan.output_style, "")
    if plan.group_by:
        base = (base + f" Ось группировки: {plan.group_by}.").strip()
    return base
