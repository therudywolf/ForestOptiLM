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
QueryPlan — понимание запроса пользователя (Столп 1).

Строит из произвольного запроса (ru/en) структурный план прогона: намерение
(intent), ключевые сущности для scout/поиска, фильтр severity, ось группировки
и стиль ответа. Ядро детерминированное (правила), тестируется без LLM; опционально
обогащается composer-моделью (extraction_focus).

План — единый носитель «что хочет пользователь» для MAP/merge/REDUCE: им
управляются дедуп-ключ агрегации, контракт ответа и приоритеты scout.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Severity нормализация (ru/en) ───────────────────────────────────
_SEVERITY_TERMS: dict[str, tuple[str, ...]] = {
    "critical": ("critical", "критич", "критическ"),
    "high": ("high", "высок", "важн"),
    "medium": ("medium", "средн", "умерен"),
    "low": ("low", "низк", "незначительн"),
    "info": ("info", "informational", "информац", "информационн"),
}

# ── Намерения (intent). Порядок = приоритет при множественном совпадении ──
_INTENT_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("compare", (
        "сравн", "сравнен", "diff", "различ", "разниц", "было стало", "было/стало",
        "новые", "новых", "появил", "regress", "регресс", "что изменил", "since last",
        "по сравнению", "fixed", "исправлен", "устранен",
    )),
    ("prioritize", (
        "приорит", "что чинить", "что исправл", "в первую очередь", "самые",
        "наиболее", "top", "топ", "ранжир", "rank", "prioriti", "важнейш",
        "критичн", "по риску", "by risk", "exploitab", "эксплуатируем",
    )),
    ("rootcause", (
        "почему", "причин", "root cause", "первопричин", "из-за", "как эксплуат",
        "exploit", "вектор атак", "attack vector", "как использ",
    )),
    ("map_standard", (
        "owasp", "cwe", "mitre", "att&ck", "attack", "pci", "complian",
        "соответств", "стандарт", "gost", "гост", "категори",
    )),
    ("count", (
        "сколько", "количеств", "count", "статистик", "распределен",
        "how many", "breakdown", "по числу",
    )),
    ("filter", (
        "только", "лишь", "исключ", "where", "фильтр", "отбери", "выбери все",
        "по хост", "по сервис", "по пакет", "на хосте",
    )),
    ("enumerate", (
        "перечисл", "список", "все находк", "все уязвим", "list ", "enumerate",
        "выведи все", "покажи все", "каждую",
    )),
    ("summarize", (
        "кратко", "резюме", "summary", "обзор", "вкратце", "summarize",
        "общая картина", "overview", "tl;dr", "саммари",
    )),
]

# ── Ось группировки ─────────────────────────────────────────────────
_GROUP_BY_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("asset", ("по хост", "по узл", "по актив", "по серверам", "by host", "by asset", "per host")),
    ("cve", ("по cve", "by cve", "по уязвим", "per cve")),
    ("cwe", ("по cwe", "by cwe", "по типу слабост")),
    ("severity", ("по severity", "по уровн", "по критичн", "by severity", "по степен")),
    ("component", ("по сервис", "по компонент", "по пакет", "by service", "by component", "by package")),
    ("file", ("по файл", "by file", "per file")),
    ("tool", ("по сканер", "по инструмент", "by tool", "по источник")),
]

_INTENT_OUTPUT_STYLE: dict[str, str] = {
    "compare": "diff_table",
    "prioritize": "ranked_list",
    "enumerate": "matrix",
    "count": "stats",
    "summarize": "brief",
    "rootcause": "report",
    "map_standard": "matrix",
    "analyze": "report",
}

# Дедуп-ключ агрегации по оси группировки (для Столпа 3).
_GROUP_DEDUP_KEY: dict[str, tuple[str, ...]] = {
    "asset": ("asset", "cve"),
    "cve": ("cve",),
    "cwe": ("cwe",),
    "component": ("component", "cve"),
    "file": ("file", "type"),
    "severity": ("severity", "type", "explanation"),
    "tool": ("tool", "cve"),
}

_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
_CWE_RE = re.compile(r"\bCWE-\d{1,6}\b", re.I)
_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_HOST_RE = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.I)
_QUOTED_RE = re.compile(r"[\"'«»“”]([^\"'«»“”]{2,60})[\"'«»“”]")

_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "all", "any", "with", "from", "this", "that", "find",
    "show", "list", "what", "which", "report", "scan", "analyze", "analysis",
    "найди", "покажи", "выведи", "это", "для", "все", "всех", "что", "как",
    "отчет", "отчёт", "отчете", "анализ", "проанализируй", "сделай", "дай",
    "есть", "по", "на", "из", "при", "или", "так", "там", "над",
})


@dataclass(slots=True)
class QueryPlan:
    """Структурный план прогона, выведенный из запроса пользователя."""

    query: str
    language: str = "en"                 # "ru" | "en"
    intent: str = "analyze"             # основной intent
    intents: list[str] = field(default_factory=list)
    key_terms: list[str] = field(default_factory=list)
    cve_ids: list[str] = field(default_factory=list)
    cwe_ids: list[str] = field(default_factory=list)
    hosts: list[str] = field(default_factory=list)
    severity_filter: list[str] = field(default_factory=list)
    group_by: str | None = None
    output_style: str = "report"
    extraction_focus: str = ""          # обогащается composer-моделью (опц.)

    def dedup_keys(self) -> tuple[str, ...]:
        """Поля для дедупликации/агрегации находок под этот запрос."""
        if self.group_by and self.group_by in _GROUP_DEDUP_KEY:
            return _GROUP_DEDUP_KEY[self.group_by]
        if self.cve_ids or self.intent in ("compare", "prioritize"):
            return ("cve", "asset")
        return ("severity", "type", "explanation")

    def language_hint(self) -> str:
        return "Ответь строго на русском языке." if self.language == "ru" else ""

    def summary(self) -> str:
        bits = [f"intent={self.intent}", f"out={self.output_style}"]
        if self.group_by:
            bits.append(f"group_by={self.group_by}")
        if self.severity_filter:
            bits.append("sev=" + "/".join(self.severity_filter))
        if self.cve_ids:
            bits.append(f"cve={len(self.cve_ids)}")
        if self.key_terms:
            bits.append("terms=" + ",".join(self.key_terms[:6]))
        return " ".join(bits)


def _detect_language(query: str) -> str:
    return "ru" if re.search(r"[А-Яа-яЁё]", query or "") else "en"


def _detect_intents(low: str) -> list[str]:
    found: list[str] = []
    for intent, pats in _INTENT_PATTERNS:
        if not pats:
            continue
        if any(p in low for p in pats):
            found.append(intent)
    return found


def _detect_group_by(low: str) -> str | None:
    for axis, pats in _GROUP_BY_PATTERNS:
        if any(p in low for p in pats):
            return axis
    return None


def _detect_severity(low: str) -> list[str]:
    out: list[str] = []
    for sev, terms in _SEVERITY_TERMS.items():
        if any(t in low for t in terms):
            out.append(sev)
    return out


def _extract_key_terms(query: str, cve: list[str], cwe: list[str], hosts: list[str]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def _add(t: str) -> None:
        k = t.lower()
        if k and k not in seen and k not in _STOPWORDS and len(k) >= 3:
            seen.add(k)
            terms.append(t)

    for m in _QUOTED_RE.findall(query):
        _add(m.strip())
    for t in cve + cwe + hosts:
        _add(t)
    # Технические токены: с разделителями (lodash@1.2, log4j-core, payment_service)
    for tok in re.findall(r"[A-Za-z][\w.\-/@]{2,}", query):
        tok = tok.rstrip(".,/")
        if any(c in tok for c in "._-/@") or any(ch.isdigit() for ch in tok):
            _add(tok)
    return terms[:24]


def build_query_plan(query: str) -> QueryPlan:
    """Детерминированный разбор запроса в QueryPlan (без LLM)."""
    q = (query or "").strip()
    low = q.lower()
    language = _detect_language(q)

    cve_ids = sorted({m.upper() for m in _CVE_RE.findall(q)})
    cwe_ids = sorted({m.upper() for m in _CWE_RE.findall(q)})
    hosts = sorted({*(_IPV4_RE.findall(q)), *(_HOST_RE.findall(q))})

    intents = _detect_intents(low)
    intent = intents[0] if intents else "analyze"
    group_by = _detect_group_by(low)
    severity_filter = _detect_severity(low)
    key_terms = _extract_key_terms(q, cve_ids, cwe_ids, hosts)
    output_style = _INTENT_OUTPUT_STYLE.get(intent, "report")

    return QueryPlan(
        query=q,
        language=language,
        intent=intent,
        intents=intents or ["analyze"],
        key_terms=key_terms,
        cve_ids=cve_ids,
        cwe_ids=cwe_ids,
        hosts=hosts,
        severity_filter=severity_filter,
        group_by=group_by,
        output_style=output_style,
        extraction_focus="",
    )


# Человекочитаемые инструкции под стиль ответа — для REDUCE-контракта (Столп 3).
_OUTPUT_STYLE_DIRECTIVE: dict[str, str] = {
    "diff_table": (
        "Сформируй сравнение в виде таблицы: что появилось (NEW), что исправлено "
        "(FIXED), что осталось (PERSISTENT). Подсвети изменения по severity."
    ),
    "ranked_list": (
        "Выведи находки ранжированным списком по приоритету (severity + "
        "эксплуатируемость + наличие фикса), сверху — что чинить первым."
    ),
    "matrix": (
        "Сгруппируй находки в матрицу/таблицу по запрошенной оси; не теряй ни одной."
    ),
    "stats": (
        "Дай количественную сводку: распределение по severity и по запрошенной оси, "
        "итоговые числа, затем короткие выводы."
    ),
    "brief": (
        "Дай сжатое резюме: 5–10 ключевых пунктов и общий уровень риска, без воды."
    ),
    "report": "",
}


def output_style_directive(plan: QueryPlan) -> str:
    base = _OUTPUT_STYLE_DIRECTIVE.get(plan.output_style, "")
    if plan.group_by:
        base = (base + f" Ось группировки: {plan.group_by}.").strip()
    if plan.severity_filter:
        base = (base + " Сфокусируйся на severity: " + ", ".join(plan.severity_filter) + ".").strip()
    return base
