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
Память находок (Столп 5) — персистентный SQLite-store нормализованных находок
по прогонам + диффы между сканами одного источника.

Вдохновлено agentmemory: «фоновый» слой памяти, но под наш домен — не сессии
агента, а находки безопасности. Даёт:
- сравнение сканов во времени (новые / исправленные / сохраняющиеся);
- follow-up без повторного MAP (находки уже разобраны и лежат в БД).

Никогда не роняет основной прогон: все операции best-effort.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("nocturne")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    scan_id     TEXT PRIMARY KEY,
    job_id      TEXT,
    source_path TEXT,
    query       TEXT,
    created_at  INTEGER NOT NULL,
    totals_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_scans_source ON scans(source_path, created_at);
CREATE TABLE IF NOT EXISTS scan_findings (
    scan_id     TEXT NOT NULL,
    finding_key TEXT NOT NULL,
    severity    TEXT,
    type        TEXT,
    asset       TEXT,
    cve         TEXT,
    cwe         TEXT,
    component   TEXT,
    explanation TEXT,
    evidence    TEXT,
    PRIMARY KEY (scan_id, finding_key)
);
CREATE INDEX IF NOT EXISTS idx_sf_scan ON scan_findings(scan_id);
"""

_conn: sqlite3.Connection | None = None
_conn_path: str | None = None


def findings_memory_enabled() -> bool:
    return os.getenv("NOCTURNE_FINDINGS_MEMORY", "1").strip() != "0"


def _db_path() -> Path:
    override = os.getenv("NOCTURNE_CACHE_DIR", "").strip()
    base = Path(override).expanduser() if override else (Path(__file__).resolve().parent / ".nocturne_cache")
    return base / "findings.db"


def reset_connection() -> None:
    global _conn, _conn_path
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
    _conn = None
    _conn_path = None


def _ensure_db() -> sqlite3.Connection | None:
    global _conn, _conn_path
    path = str(_db_path())
    if _conn is not None and _conn_path == path:
        return _conn
    reset_connection()
    try:
        _db_path().parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(path, check_same_thread=False)
        _conn.executescript(_SCHEMA)
        _conn.commit()
        _conn_path = path
        return _conn
    except Exception as exc:
        logger.warning("findings_store init failed: %s", exc)
        return None


def canonical_finding_key(facets: dict[str, Any]) -> str:
    """
    Стабильный ключ «той же находки» для диффа между сканами (не зависит от запроса).
    Приоритет: CVE+asset; иначе type+asset+explanation[:60].
    """
    cve = facets.get("cve")
    cve_s = ",".join(cve) if isinstance(cve, list) else str(cve or "")
    asset = str(facets.get("asset") or "").lower()
    if cve_s:
        basis = f"cve:{cve_s.lower()}|asset:{asset}"
    else:
        typ = str(facets.get("type") or "").lower()
        expl = str(facets.get("explanation") or "")[:60].lower()
        basis = f"type:{typ}|asset:{asset}|expl:{expl}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20]


def _finding_row(facets: dict[str, Any]) -> dict[str, Any]:
    return {
        "finding_key": canonical_finding_key(facets),
        "severity": str(facets.get("severity") or "unknown"),
        "type": str(facets.get("type") or ""),
        "asset": str(facets.get("asset") or ""),
        "cve": ",".join(facets.get("cve") or []),
        "cwe": ",".join(facets.get("cwe") or []),
        "component": str(facets.get("component") or ""),
        "explanation": str(facets.get("explanation") or "")[:300],
        "evidence": "1" if facets.get("has_evidence") else "",
    }


def record_scan(
    *,
    job_id: str,
    source_path: str,
    query: str,
    findings: list[dict[str, Any]],
    totals: dict[str, Any] | None = None,
) -> str | None:
    """Сохранить прогон и его находки. Возвращает scan_id или None."""
    conn = _ensure_db()
    if conn is None:
        return None
    ts = int(time.time())
    # uuid гарантирует уникальность scan_id даже для двух прогонов в одну секунду.
    scan_id = hashlib.sha256(f"{job_id}|{source_path}|{ts}|{uuid.uuid4().hex}".encode()).hexdigest()[:20]
    try:
        conn.execute(
            "INSERT OR REPLACE INTO scans (scan_id, job_id, source_path, query, created_at, totals_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (scan_id, job_id, source_path, (query or "")[:500], ts,
             json.dumps(totals or {}, ensure_ascii=False)),
        )
        seen: set[str] = set()
        for f in findings:
            facets = f.get("_facets") or {}
            row = _finding_row(facets)
            if row["finding_key"] in seen:
                continue
            seen.add(row["finding_key"])
            conn.execute(
                "INSERT OR REPLACE INTO scan_findings "
                "(scan_id, finding_key, severity, type, asset, cve, cwe, component, explanation, evidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (scan_id, row["finding_key"], row["severity"], row["type"], row["asset"],
                 row["cve"], row["cwe"], row["component"], row["explanation"], row["evidence"]),
            )
        conn.commit()
        return scan_id
    except Exception as exc:
        logger.warning("record_scan failed: %s", exc)
        return None


def previous_scan_id(source_path: str, exclude_scan_id: str) -> str | None:
    conn = _ensure_db()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT scan_id FROM scans WHERE source_path = ? AND scan_id != ? "
            "ORDER BY created_at DESC LIMIT 1",
            (source_path, exclude_scan_id),
        ).fetchone()
        return str(row[0]) if row else None
    except Exception:
        return None


def _keys_with_meta(scan_id: str) -> dict[str, dict[str, str]]:
    conn = _ensure_db()
    if conn is None:
        return {}
    out: dict[str, dict[str, str]] = {}
    try:
        for r in conn.execute(
            "SELECT finding_key, severity, asset, cve, type, explanation "
            "FROM scan_findings WHERE scan_id = ?",
            (scan_id,),
        ):
            out[str(r[0])] = {
                "severity": str(r[1] or ""), "asset": str(r[2] or ""),
                "cve": str(r[3] or ""), "type": str(r[4] or ""),
                "explanation": str(r[5] or ""),
            }
    except Exception:
        return {}
    return out


def diff_scans(old_scan_id: str, new_scan_id: str) -> dict[str, list[dict[str, str]]]:
    """Вернуть {new, fixed, persistent} по canonical_finding_key."""
    old = _keys_with_meta(old_scan_id)
    new = _keys_with_meta(new_scan_id)
    old_keys, new_keys = set(old), set(new)
    return {
        "new": [new[k] for k in (new_keys - old_keys)],
        "fixed": [old[k] for k in (old_keys - new_keys)],
        "persistent": [new[k] for k in (new_keys & old_keys)],
    }


def list_scans(source_path: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    conn = _ensure_db()
    if conn is None:
        return []
    try:
        if source_path:
            rows = conn.execute(
                "SELECT scan_id, job_id, source_path, query, created_at, totals_json "
                "FROM scans WHERE source_path = ? ORDER BY created_at DESC LIMIT ?",
                (source_path, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT scan_id, job_id, source_path, query, created_at, totals_json "
                "FROM scans ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    except Exception:
        return []
    return [
        {"scan_id": r[0], "job_id": r[1], "source_path": r[2], "query": r[3],
         "created_at": int(r[4]), "totals": json.loads(r[5] or "{}")}
        for r in rows
    ]


def _sev_rank(s: str) -> int:
    return {"critical": 5, "high": 4, "medium": 3, "low": 2, "info": 1}.get(s, 0)


def diff_markdown(diff: dict[str, list[dict[str, str]]], language: str = "ru") -> str:
    new, fixed, persistent = diff["new"], diff["fixed"], diff["persistent"]
    if not (new or fixed):
        return ""
    ru = language == "ru"
    top_new = sorted(new, key=lambda d: _sev_rank(d.get("severity", "")), reverse=True)[:8]
    lines: list[str] = []
    if ru:
        lines.append("### Изменения с прошлого скана")
        lines.append(f"- Новых находок: {len(new)}")
        lines.append(f"- Исправлено (нет в текущем): {len(fixed)}")
        lines.append(f"- Сохраняется: {len(persistent)}")
        if top_new:
            lines.append("- Новые (топ по severity):")
            for d in top_new:
                tag = d.get("cve") or d.get("type") or "—"
                lines.append(f"  - [{d.get('severity','?')}] {tag} @ {d.get('asset','?')}")
    else:
        lines.append("### Changes since last scan")
        lines.append(f"- New findings: {len(new)}")
        lines.append(f"- Fixed (absent now): {len(fixed)}")
        lines.append(f"- Persistent: {len(persistent)}")
        if top_new:
            lines.append("- New (top by severity):")
            for d in top_new:
                tag = d.get("cve") or d.get("type") or "-"
                lines.append(f"  - [{d.get('severity','?')}] {tag} @ {d.get('asset','?')}")
    return "\n".join(lines)


def record_and_diff(
    *,
    job_id: str,
    source_path: str,
    query: str,
    findings: list[dict[str, Any]],
    totals: dict[str, Any] | None = None,
    language: str = "ru",
) -> str:
    """Сохранить текущий скан и вернуть markdown-дифф vs предыдущий (или '')."""
    if not findings_memory_enabled() or not source_path:
        return ""
    scan_id = record_scan(
        job_id=job_id, source_path=source_path, query=query,
        findings=findings, totals=totals,
    )
    if not scan_id:
        return ""
    prev = previous_scan_id(source_path, scan_id)
    if not prev:
        return ""
    try:
        return diff_markdown(diff_scans(prev, scan_id), language=language)
    except Exception as exc:
        logger.warning("scan diff failed: %s", exc)
        return ""
