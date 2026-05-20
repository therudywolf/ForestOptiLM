# SPDX-License-Identifier: AGPL-3.0-or-later
"""Иерархическое детерминированное слияние MAP JSON: file → directory → corpus."""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

_FILE_PATH_RE = re.compile(r"\[FILE_PATH:\s*([^\]]+)\]", re.I)


def _merge_findings_cap() -> int:
    """Верхний предел числа находок на уровне merge (env NOCTURNE_MERGE_FINDINGS_CAP)."""
    raw = os.getenv("NOCTURNE_MERGE_FINDINGS_CAP", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return 1000


def _extract_file_path(chunk_text: str, fallback: str = "") -> str:
    m = _FILE_PATH_RE.search(chunk_text)
    if m:
        return m.group(1).strip().strip('"')
    return fallback


def _parse_map_item(raw: str) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _merge_findings(
    items: list[dict[str, Any]], max_findings: int | None = None,
) -> list[dict[str, Any]]:
    cap = max_findings if max_findings is not None else _merge_findings_cap()
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for obj in items:
        for f in obj.get("findings") or []:
            if not isinstance(f, dict):
                continue
            key = (
                str(f.get("severity") or "").lower(),
                str(f.get("type") or "").lower(),
                str(f.get("explanation") or "")[:80].lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
            if len(out) >= cap:
                return out
    return out


def _merge_recommendations(items: list[dict[str, Any]]) -> list[str]:
    recs: list[str] = []
    for obj in items:
        for r in obj.get("recommendations") or []:
            if isinstance(r, str) and r.strip() and r.strip() not in recs:
                recs.append(r.strip())
    return recs


def _rollup_level(
    grouped: dict[str, list[dict[str, Any]]],
    level_name: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, objs in grouped.items():
        any_relevant = any(not bool(o.get("no_relevant_data")) for o in objs)
        findings = _merge_findings(objs)
        result[key] = {
            "level": level_name,
            "key": key,
            "file": key,
            "query_alignment": next(
                (str(o.get("query_alignment") or "") for o in objs if o.get("query_alignment")),
                "",
            ),
            "no_relevant_data": not any_relevant and not findings,
            "findings": findings,
            "recommendations": _merge_recommendations(objs),
            "child_count": len(objs),
        }
    return result


def hierarchical_merge_map_results(
    map_json_strings: list[str],
    chunk_texts: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Сгруппировать MAP JSON по файлу и директории; вернуть corpus JSON и дерево summaries.
    """
    parsed: list[tuple[str, dict[str, Any]]] = []
    for i, raw in enumerate(map_json_strings):
        obj = _parse_map_item(raw)
        if not obj:
            continue
        fp = str(obj.get("file") or "")
        if not fp and chunk_texts and i < len(chunk_texts):
            fp = _extract_file_path(chunk_texts[i])
        parsed.append((fp or f"chunk_{i}", obj))

    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fp, obj in parsed:
        by_file[fp].append(obj)

    file_rollups = _rollup_level(by_file, "file")

    by_dir: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fp, fobj in file_rollups.items():
        parent = str(Path(fp).parent).replace("\\", "/")
        if parent in (".", ""):
            parent = "/"
        by_dir[parent].append(fobj)

    dir_rollups = _rollup_level(by_dir, "directory")

    corpus_findings = _merge_findings(list(file_rollups.values()))
    corpus: dict[str, Any] = {
        "level": "corpus",
        "file": "CORPUS",
        "query_alignment": next(
            (str(o.get("query_alignment") or "") for o in file_rollups.values() if o.get("query_alignment")),
            "",
        ),
        "no_relevant_data": not corpus_findings,
        "findings": corpus_findings,
        "recommendations": _merge_recommendations(list(file_rollups.values())),
        "files_with_findings": sum(1 for f in file_rollups.values() if f.get("findings")),
        "directories": len(dir_rollups),
    }

    tree = {
        "corpus": corpus,
        "directories": dir_rollups,
        "files": file_rollups,
    }
    return json.dumps(corpus, ensure_ascii=False), tree


def top_evidence_from_tree(tree: dict[str, Any], limit: int = 40) -> list[dict[str, str]]:
    """Топ цитат для reduce/composer."""
    rows: list[dict[str, str]] = []
    corpus = tree.get("corpus") or {}
    for f in corpus.get("findings") or []:
        if not isinstance(f, dict):
            continue
        for er in f.get("evidence_refs") or []:
            if not isinstance(er, dict):
                continue
            rows.append({
                "file": str(er.get("file") or ""),
                "chunk": str(er.get("chunk") or ""),
                "quote": str(er.get("quote") or "")[:120],
            })
            if len(rows) >= limit:
                return rows
    return rows
