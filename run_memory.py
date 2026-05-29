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
Память прогонов (доменно-нейтральная) — персистентный SQLite-store извлечённых
элементов по прогонам + сравнение двух прогонов над одним источником.

Даёт «дал тот же корпус ещё раз → увидел, что изменилось» для ЛЮБОЙ задачи:
added / removed / unchanged по нейтральному ключу элемента (без понятий
scan/vuln/fixed). Никогда не роняет основной прогон: всё best-effort.
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
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    job_id      TEXT,
    source_path TEXT,
    query       TEXT,
    created_at  INTEGER NOT NULL,
    totals_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_source ON runs(source_path, created_at);
CREATE TABLE IF NOT EXISTS run_items (
    run_id    TEXT NOT NULL,
    item_key  TEXT NOT NULL,
    category  TEXT,
    item      TEXT,
    source    TEXT,
    level     TEXT,
    entities  TEXT,
    PRIMARY KEY (run_id, item_key)
);
CREATE INDEX IF NOT EXISTS idx_ri_run ON run_items(run_id);
"""

_conn: sqlite3.Connection | None = None
_conn_path: str | None = None


def run_memory_enabled() -> bool:
    # NOCTURNE_RUN_MEMORY (нейтральное имя). Старое NOCTURNE_FINDINGS_MEMORY — алиас.
    val = os.getenv("NOCTURNE_RUN_MEMORY", os.getenv("NOCTURNE_FINDINGS_MEMORY", "1"))
    return str(val).strip() != "0"


def _db_path() -> Path:
    override = os.getenv("NOCTURNE_CACHE_DIR", "").strip()
    base = Path(override).expanduser() if override else (Path(__file__).resolve().parent / ".nocturne_cache")
    return base / "run_memory.db"


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
        logger.warning("run_memory init failed: %s", exc)
        return None


def canonical_item_key(facets: dict[str, Any]) -> str:
    """
    Стабильный ключ «того же элемента» для сравнения прогонов (не зависит от запроса).
    Приоритет: сущности(id/email/date)+источник; иначе category+item+source.
    """
    ents = facets.get("entities")
    ents_s = ",".join(sorted(ents)) if isinstance(ents, list) else str(ents or "")
    source = str(facets.get("source") or "").lower()
    if ents_s:
        basis = f"ent:{ents_s.lower()}|src:{source}"
    else:
        cat = str(facets.get("category") or "").lower()
        item = str(facets.get("item") or "")[:60].lower()
        basis = f"cat:{cat}|item:{item}|src:{source}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20]


def _item_row(facets: dict[str, Any]) -> dict[str, Any]:
    ents = facets.get("entities") or []
    return {
        "item_key": canonical_item_key(facets),
        "category": str(facets.get("category") or ""),
        "item": str(facets.get("item") or "")[:300],
        "source": str(facets.get("source") or ""),
        "level": str(facets.get("level") or ""),
        "entities": ",".join(ents) if isinstance(ents, list) else str(ents or ""),
    }


def record_run(
    *,
    job_id: str,
    source_path: str,
    query: str,
    records: list[dict[str, Any]],
    totals: dict[str, Any] | None = None,
) -> str | None:
    """Сохранить прогон и его извлечённые элементы. Возвращает run_id или None."""
    conn = _ensure_db()
    if conn is None:
        return None
    ts = int(time.time())
    run_id = hashlib.sha256(f"{job_id}|{source_path}|{ts}|{uuid.uuid4().hex}".encode()).hexdigest()[:20]
    try:
        conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, job_id, source_path, query, created_at, totals_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, job_id, source_path, (query or "")[:500], ts,
             json.dumps(totals or {}, ensure_ascii=False)),
        )
        seen: set[str] = set()
        for r in records:
            facets = r.get("_facets") or {}
            row = _item_row(facets)
            if row["item_key"] in seen:
                continue
            seen.add(row["item_key"])
            conn.execute(
                "INSERT OR REPLACE INTO run_items "
                "(run_id, item_key, category, item, source, level, entities) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, row["item_key"], row["category"], row["item"],
                 row["source"], row["level"], row["entities"]),
            )
        conn.commit()
        return run_id
    except Exception as exc:
        logger.warning("record_run failed: %s", exc)
        return None


def previous_run_id(source_path: str, exclude_run_id: str) -> str | None:
    conn = _ensure_db()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT run_id FROM runs WHERE source_path = ? AND run_id != ? "
            "ORDER BY created_at DESC LIMIT 1",
            (source_path, exclude_run_id),
        ).fetchone()
        return str(row[0]) if row else None
    except Exception:
        return None


def _items_with_meta(run_id: str) -> dict[str, dict[str, str]]:
    conn = _ensure_db()
    if conn is None:
        return {}
    out: dict[str, dict[str, str]] = {}
    try:
        for r in conn.execute(
            "SELECT item_key, category, item, source, level FROM run_items WHERE run_id = ?",
            (run_id,),
        ):
            out[str(r[0])] = {
                "category": str(r[1] or ""), "item": str(r[2] or ""),
                "source": str(r[3] or ""), "level": str(r[4] or ""),
            }
    except Exception:
        return {}
    return out


def diff_runs(old_run_id: str, new_run_id: str) -> dict[str, list[dict[str, str]]]:
    """Вернуть {added, removed, unchanged} по canonical_item_key."""
    old = _items_with_meta(old_run_id)
    new = _items_with_meta(new_run_id)
    ok, nk = set(old), set(new)
    return {
        "added": [new[k] for k in (nk - ok)],
        "removed": [old[k] for k in (ok - nk)],
        "unchanged": [new[k] for k in (nk & ok)],
    }


def list_runs(source_path: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    conn = _ensure_db()
    if conn is None:
        return []
    try:
        if source_path:
            rows = conn.execute(
                "SELECT run_id, job_id, source_path, query, created_at, totals_json "
                "FROM runs WHERE source_path = ? ORDER BY created_at DESC LIMIT ?",
                (source_path, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT run_id, job_id, source_path, query, created_at, totals_json "
                "FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    except Exception:
        return []
    return [
        {"run_id": r[0], "job_id": r[1], "source_path": r[2], "query": r[3],
         "created_at": int(r[4]), "totals": json.loads(r[5] or "{}")}
        for r in rows
    ]


def diff_markdown(diff: dict[str, list[dict[str, str]]], language: str = "ru") -> str:
    added, removed, unchanged = diff["added"], diff["removed"], diff["unchanged"]
    if not (added or removed):
        return ""
    ru = language == "ru"
    top_added = added[:8]
    lines: list[str] = []
    if ru:
        lines.append("### Изменения с прошлого прогона")
        lines.append(f"- Появилось: {len(added)}")
        lines.append(f"- Исчезло: {len(removed)}")
        lines.append(f"- Без изменений: {len(unchanged)}")
        if top_added:
            lines.append("- Новые (примеры):")
            for d in top_added:
                tag = d.get("item") or d.get("category") or "—"
                lines.append(f"  - {tag} @ {d.get('source','?')}")
    else:
        lines.append("### Changes since last run")
        lines.append(f"- Added: {len(added)}")
        lines.append(f"- Removed: {len(removed)}")
        lines.append(f"- Unchanged: {len(unchanged)}")
        if top_added:
            lines.append("- New (examples):")
            for d in top_added:
                tag = d.get("item") or d.get("category") or "-"
                lines.append(f"  - {tag} @ {d.get('source','?')}")
    return "\n".join(lines)


def record_and_diff(
    *,
    job_id: str,
    source_path: str,
    query: str,
    records: list[dict[str, Any]],
    totals: dict[str, Any] | None = None,
    language: str = "ru",
) -> str:
    """Сохранить текущий прогон и вернуть markdown-дифф vs предыдущий (или '')."""
    if not run_memory_enabled() or not source_path:
        return ""
    run_id = record_run(
        job_id=job_id, source_path=source_path, query=query,
        records=records, totals=totals,
    )
    if not run_id:
        return ""
    prev = previous_run_id(source_path, run_id)
    if not prev:
        return ""
    try:
        return diff_markdown(diff_runs(prev, run_id), language=language)
    except Exception as exc:
        logger.warning("run diff failed: %s", exc)
        return ""
