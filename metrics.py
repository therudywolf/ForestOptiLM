# SPDX-License-Identifier: AGPL-3.0-or-later
"""SQLite run metrics for observability."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / ".nocturne_cache" / "metrics.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    started_at INTEGER NOT NULL,
    finished_at INTEGER,
    duration_s REAL,
    query_preview TEXT,
    models_json TEXT,
    chunks_total INTEGER,
    chunks_ok INTEGER,
    chunks_failed INTEGER,
    scout_skipped INTEGER,
    retries INTEGER,
    warnings_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_run_metrics_job ON run_metrics(job_id);
"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    c.executescript(_SCHEMA)
    return c


def record_run_start(job_id: str, query: str, models: dict[str, str]) -> int:
    preview = query[:240]
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO run_metrics (job_id, started_at, query_preview, models_json) VALUES (?, ?, ?, ?)",
            (job_id, int(time.time()), preview, json.dumps(models, ensure_ascii=False)),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def record_run_finish(
    row_id: int,
    *,
    duration_s: float,
    chunks_total: int,
    chunks_ok: int,
    chunks_failed: int,
    scout_skipped: int = 0,
    retries: int = 0,
    warnings: list[str] | None = None,
) -> None:
    with _conn() as conn:
        conn.execute(
            """UPDATE run_metrics SET finished_at=?, duration_s=?, chunks_total=?, chunks_ok=?,
               chunks_failed=?, scout_skipped=?, retries=?, warnings_json=? WHERE id=?""",
            (
                int(time.time()),
                duration_s,
                chunks_total,
                chunks_ok,
                chunks_failed,
                scout_skipped,
                retries,
                json.dumps(warnings or [], ensure_ascii=False),
                row_id,
            ),
        )
        conn.commit()


def list_recent_runs(limit: int = 20) -> list[dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, job_id, started_at, duration_s, query_preview, chunks_ok, chunks_failed, scout_skipped "
            "FROM run_metrics ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "id": r[0],
            "job_id": r[1],
            "started_at": r[2],
            "duration_s": r[3],
            "query_preview": r[4],
            "chunks_ok": r[5],
            "chunks_failed": r[6],
            "scout_skipped": r[7],
        }
        for r in rows
    ]
