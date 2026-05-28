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
Детерминированная агрегация и категорирование находок (Столп 3).

MAP-модель извлекает находки; категорирование (counts по severity/оси,
дедуп, топы) считается ЗДЕСЬ, в коде, а не галлюцинируется моделью. Готовая
сводка передаётся в REDUCE как «истина», чтобы числа в отчёте были верными
на любом объёме и для разных файлов.

Facet'ы (cve/cwe/asset/component) извлекаются обобщённо регулярками из полей
находки и evidence — без привязки к схеме конкретного сканера.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator

_SEVERITY_RANK = {"critical": 5, "high": 4, "medium": 3, "low": 2, "info": 1, "unknown": 0}
_SEVERITY_ORDER = ["critical", "high", "medium", "low", "info", "unknown"]

_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
_CWE_RE = re.compile(r"\bCWE-\d{1,6}\b", re.I)
_PKG_RE = re.compile(r"\b([A-Za-z][\w.\-]{1,40})@([\w.\-]+)\b")


def normalize_severity(value: Any) -> str:
    """Привести severity к critical/high/medium/low/info/unknown (ru/en/числа/CVSS)."""
    if value is None:
        return "unknown"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        f = float(value)
        if f >= 9.0:
            return "critical"
        if f >= 7.0:
            return "high"
        if f >= 4.0:
            return "medium"
        if f > 0.0:
            return "low"
        return "info"
    s = str(value).strip().lower()
    if not s:
        return "unknown"
    table = {
        "critical": "critical", "крит": "critical", "критич": "critical",
        "high": "high", "выс": "high", "важн": "high",
        "medium": "medium", "moderate": "medium", "сред": "medium",
        "low": "low", "низк": "low", "minor": "low",
        "info": "info", "informational": "info", "information": "info", "информ": "info",
        "none": "info", "negligible": "low",
        "3": "high", "2": "medium", "1": "low", "0": "info",  # ZAP riskcode
    }
    for key, norm in table.items():
        if s.startswith(key):
            return norm
    # CVSS-число строкой
    try:
        return normalize_severity(float(s.replace(",", ".")))
    except ValueError:
        return "unknown"


def _texts_of(finding: dict[str, Any]) -> str:
    parts: list[str] = [
        str(finding.get("type") or ""),
        str(finding.get("explanation") or ""),
        str(finding.get("title") or ""),
    ]
    for er in finding.get("evidence_refs") or []:
        if isinstance(er, dict):
            parts.append(str(er.get("quote") or ""))
            parts.append(str(er.get("file") or ""))
    return " \n".join(parts)


def _first_evidence_file(finding: dict[str, Any]) -> str:
    for er in finding.get("evidence_refs") or []:
        if isinstance(er, dict):
            f = str(er.get("file") or "").strip()
            if f:
                return f
    return ""


def extract_facets(finding: dict[str, Any], outer_file: str = "") -> dict[str, Any]:
    """Обобщённо извлечь facet'ы из находки (без привязки к формату отчёта)."""
    blob = _texts_of(finding)
    cves = sorted({m.upper() for m in _CVE_RE.findall(blob)})
    cwes = sorted({m.upper() for m in _CWE_RE.findall(blob)})
    pkgs = sorted({f"{m[0]}@{m[1]}" for m in _PKG_RE.findall(blob)})
    asset = _first_evidence_file(finding) or outer_file or "unknown"
    component = pkgs[0].split("@", 1)[0] if pkgs else ""
    return {
        "severity": normalize_severity(finding.get("severity")),
        "type": str(finding.get("type") or "").strip().lower(),
        "explanation": str(finding.get("explanation") or "")[:80].strip().lower(),
        "cve": cves,
        "cwe": cwes,
        "asset": asset,
        "component": component,
        "has_evidence": bool(_first_evidence_file(finding)),
    }


def iter_findings_from_map(map_results: list[str]) -> Iterator[dict[str, Any]]:
    """Распарсить MAP-JSON строки → находки с прикреплёнными facet'ами."""
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
        outer_file = str(obj.get("file") or "")
        for f in obj.get("findings") or []:
            if not isinstance(f, dict):
                continue
            facets = extract_facets(f, outer_file=outer_file)
            yield {**f, "_facets": facets}


def _dedup_key(facets: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, ...]:
    parts: list[str] = []
    for k in keys:
        v = facets.get(k)
        if isinstance(v, list):
            parts.append(",".join(v))
        else:
            parts.append(str(v or ""))
    return tuple(parts)


@dataclass(slots=True)
class CategorySummary:
    total: int = 0
    unique: int = 0
    by_severity: dict[str, int] = field(default_factory=dict)
    axis: str = "severity"
    by_axis: list[tuple[str, int]] = field(default_factory=list)
    top_cves: list[tuple[str, int]] = field(default_factory=list)
    top_cwes: list[tuple[str, int]] = field(default_factory=list)
    with_evidence: int = 0


def categorize(
    findings: list[dict[str, Any]],
    dedup_keys: tuple[str, ...] = ("severity", "type", "explanation"),
    axis: str | None = None,
    top_n: int = 15,
) -> CategorySummary:
    """Посчитать распределения и топы детерминированно."""
    sev_counter: Counter[str] = Counter()
    cve_counter: Counter[str] = Counter()
    cwe_counter: Counter[str] = Counter()
    axis_counter: Counter[str] = Counter()
    seen: set[tuple[str, ...]] = set()
    with_ev = 0
    axis_name = axis or "severity"

    for f in findings:
        facets = f.get("_facets") or extract_facets(f)
        key = _dedup_key(facets, dedup_keys)
        if key in seen:
            continue
        seen.add(key)
        sev_counter[facets["severity"]] += 1
        if facets["has_evidence"]:
            with_ev += 1
        for c in facets["cve"]:
            cve_counter[c] += 1
        for c in facets["cwe"]:
            cwe_counter[c] += 1
        av = facets.get(axis_name)
        if isinstance(av, list):
            for x in av:
                axis_counter[x] += 1
        elif av:
            axis_counter[str(av)] += 1

    by_sev = {s: sev_counter.get(s, 0) for s in _SEVERITY_ORDER if sev_counter.get(s, 0)}
    return CategorySummary(
        total=len(findings),
        unique=len(seen),
        by_severity=by_sev,
        axis=axis_name,
        by_axis=axis_counter.most_common(top_n),
        top_cves=cve_counter.most_common(top_n),
        top_cwes=cwe_counter.most_common(top_n),
        with_evidence=with_ev,
    )


def summary_markdown(summary: CategorySummary, language: str = "ru") -> str:
    """Готовый детерминированный блок «фактов» для REDUCE."""
    if summary.unique == 0:
        return ""
    ru = language == "ru"
    sev_line = ", ".join(f"{k}: {v}" for k, v in summary.by_severity.items()) or "—"
    lines: list[str] = []
    if ru:
        lines.append("### Детерминированная сводка (числа — истина, не пересчитывай)")
        lines.append(f"- Уникальных находок: {summary.unique} (из {summary.total} до дедупа)")
        lines.append(f"- С доказательствами (file+quote): {summary.with_evidence}")
        lines.append(f"- По severity: {sev_line}")
        if summary.by_axis and summary.axis != "severity":
            top = ", ".join(f"{k}: {v}" for k, v in summary.by_axis)
            lines.append(f"- По «{summary.axis}»: {top}")
        if summary.top_cves:
            lines.append("- Топ CVE: " + ", ".join(f"{c} ({n})" for c, n in summary.top_cves))
        if summary.top_cwes:
            lines.append("- Топ CWE: " + ", ".join(f"{c} ({n})" for c, n in summary.top_cwes))
    else:
        lines.append("### Deterministic summary (treat these numbers as ground truth)")
        lines.append(f"- Unique findings: {summary.unique} (of {summary.total} pre-dedup)")
        lines.append(f"- With evidence (file+quote): {summary.with_evidence}")
        lines.append(f"- By severity: {sev_line}")
        if summary.by_axis and summary.axis != "severity":
            top = ", ".join(f"{k}: {v}" for k, v in summary.by_axis)
            lines.append(f"- By {summary.axis}: {top}")
        if summary.top_cves:
            lines.append("- Top CVE: " + ", ".join(f"{c} ({n})" for c, n in summary.top_cves))
        if summary.top_cwes:
            lines.append("- Top CWE: " + ", ".join(f"{c} ({n})" for c, n in summary.top_cwes))
    return "\n".join(lines)


def build_category_block(
    map_results: list[str],
    dedup_keys: tuple[str, ...] = ("severity", "type", "explanation"),
    axis: str | None = None,
    language: str = "ru",
) -> str:
    """map JSON → детерминированный markdown-блок категорирования (или '')."""
    findings = list(iter_findings_from_map(map_results))
    if not findings:
        return ""
    summary = categorize(findings, dedup_keys=dedup_keys, axis=axis)
    return summary_markdown(summary, language=language)
