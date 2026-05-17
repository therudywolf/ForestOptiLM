# SPDX-License-Identifier: AGPL-3.0-or-later
"""On-disk store for normalized MAP JSON (bounded RAM before merge)."""
from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


def normalize_spill_threshold() -> int:
    raw = os.getenv("NOCTURNE_MAP_NORMALIZE_SPILL", "2500").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 2500


def _try_parse_map_json(norm: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(norm)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class MapResultStore:
    """Collect normalized MAP JSON; spill to SQLite when chunk count is large."""

    def __init__(self, job_id: str, *, threshold: int | None = None, chunk_count: int = 0) -> None:
        self.job_id = job_id
        thr = threshold if threshold is not None else normalize_spill_threshold()
        self._spilled = bool(job_id) and thr > 0 and chunk_count >= thr
        self._ram: list[str] = []
        self._file_groups_ram: dict[str, list[str]] = defaultdict(list)
        self._db_path = (
            Path(__file__).resolve().parent
            / ".nocturne_cache"
            / "map_results"
            / f"{job_id}.db"
        )
        self._conn: sqlite3.Connection | None = None
        self.nonempty_count = 0
        self.metrics: dict[str, int] = {
            "relevant_chunks": 0,
            "findings_count": 0,
            "evidence_refs_count": 0,
        }

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS map_norm (
                       idx INTEGER PRIMARY KEY,
                       file_key TEXT NOT NULL,
                       body TEXT NOT NULL
                   )"""
            )
            self._conn.commit()
        return self._conn

    def _update_metrics(self, parsed: dict[str, Any]) -> None:
        if parsed.get("no_relevant_data"):
            return
        self.metrics["relevant_chunks"] += 1
        for f in parsed.get("findings") or []:
            if isinstance(f, dict):
                self.metrics["findings_count"] += 1
                for er in f.get("evidence_refs") or []:
                    if isinstance(er, dict) and (er.get("quote") or er.get("file")):
                        self.metrics["evidence_refs_count"] += 1

    def add(self, index: int, norm: str, *, parsed: dict[str, Any] | None = None) -> None:
        if not norm or not norm.strip():
            return
        if parsed is None:
            parsed = _try_parse_map_json(norm)
        if parsed is not None:
            self._update_metrics(parsed)
        fkey = (parsed.get("file") or "unknown") if parsed else "unknown"
        self.nonempty_count += 1
        if self._spilled:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT OR REPLACE INTO map_norm (idx, file_key, body) VALUES (?, ?, ?)",
                (index, fkey, norm),
            )
            conn.commit()
        else:
            self._ram.append(norm)
            if parsed is None or not parsed.get("no_relevant_data"):
                if parsed is None or parsed.get("findings"):
                    self._file_groups_ram[fkey].append(norm)

    def iter_nonempty(self) -> Iterator[str]:
        if not self._spilled:
            yield from self._ram
            return
        conn = self._ensure_conn()
        for row in conn.execute("SELECT body FROM map_norm ORDER BY idx"):
            yield str(row[0])

    def iter_for_merge(self) -> Iterator[str]:
        for body in self.iter_nonempty():
            parsed = _try_parse_map_json(body)
            if parsed is None:
                yield body
            elif not parsed.get("no_relevant_data") and parsed.get("findings"):
                yield body

    def build_file_groups(self) -> dict[str, list[str]]:
        if not self._spilled:
            return {k: list(v) for k, v in self._file_groups_ram.items()}
        groups: dict[str, list[str]] = defaultdict(list)
        conn = self._ensure_conn()
        for row in conn.execute("SELECT file_key, body FROM map_norm ORDER BY idx"):
            body = str(row[1])
            parsed = _try_parse_map_json(body)
            if parsed is not None and parsed.get("no_relevant_data"):
                continue
            if parsed is not None and not parsed.get("findings"):
                continue
            groups[str(row[0])].append(body)
        return dict(groups)

    def cleanup(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._spilled and self._db_path.is_file():
            try:
                self._db_path.unlink()
            except OSError:
                pass
