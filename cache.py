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
Nocturne Data Forge — кэш чекпоинтов MAP-фазы (SQLite).
"""
from __future__ import annotations

import logging
import os
import sqlite3
import hashlib
from pathlib import Path

logger = logging.getLogger("nocturne")

CACHE_DIR = Path(__file__).resolve().parent / ".nocturne_cache"
DB_PATH = CACHE_DIR / "cache.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS map_chunks (
    job_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    response_text TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (job_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_map_chunks_job ON map_chunks(job_id);
"""

_conn: sqlite3.Connection | None = None


def build_job_id(file_path: Path, user_query: str) -> str:
    """job_id = hash(file_path + mtime + user_query)."""
    try:
        stat = file_path.stat()
        payload = f"{file_path!s}{stat.st_mtime}{user_query}"
    except OSError:
        payload = f"{file_path!s}{user_query}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _ensure_db() -> sqlite3.Connection | None:
    global _conn
    if _conn is not None:
        return _conn
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.executescript(_SCHEMA)
        _conn.commit()
        return _conn
    except Exception as e:
        logger.warning("Cache DB init failed: %s", e)
        return None


def _cache_ttl_seconds() -> int:
    """TTL in seconds, configured via NOCTURNE_CACHE_TTL_DAYS (default 7 days, 0 = no expiry)."""
    try:
        days = int(os.getenv("NOCTURNE_CACHE_TTL_DAYS", "7"))
        return days * 86400 if days > 0 else 0
    except (ValueError, TypeError):
        return 7 * 86400


def get_cached_response(job_id: str, chunk_index: int) -> str | None:
    """Вернуть сохранённый ответ по job_id и chunk_index или None.
    Записи старше NOCTURNE_CACHE_TTL_DAYS дней считаются устаревшими.
    """
    try:
        conn = _ensure_db()
        if conn is None:
            return None
        ttl = _cache_ttl_seconds()
        if ttl > 0:
            import time as _time
            cutoff = int(_time.time()) - ttl
            row = conn.execute(
                "SELECT response_text FROM map_chunks "
                "WHERE job_id = ? AND chunk_index = ? AND created_at > ?",
                (job_id, chunk_index, cutoff),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT response_text FROM map_chunks WHERE job_id = ? AND chunk_index = ?",
                (job_id, chunk_index),
            ).fetchone()
        return row[0] if row else None
    except Exception as e:
        logger.warning("Cache get failed: %s", e)
        return None


def set_cached_response(job_id: str, chunk_index: int, response_text: str) -> None:
    """Сохранить ответ в кэш."""
    try:
        import time
        conn = _ensure_db()
        if conn is None:
            return
        conn.execute(
            """INSERT OR REPLACE INTO map_chunks (job_id, chunk_index, response_text, created_at)
               VALUES (?, ?, ?, ?)""",
            (job_id, chunk_index, response_text, int(time.time())),
        )
        conn.commit()
    except Exception as e:
        logger.warning("Cache set failed: %s", e)
