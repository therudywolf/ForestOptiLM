# SPDX-License-Identifier: AGPL-3.0-or-later
"""On-disk chunk store for large corpora (bounded RAM during MAP)."""
from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path


def max_chunks_in_ram() -> int:
    raw = os.getenv("NOCTURNE_MAX_CHUNKS_IN_RAM", "12000").strip()
    try:
        return max(100, int(raw))
    except ValueError:
        return 12000


class ChunkStore:
    """list-like: append chunks; spill to SQLite when len > max_in_ram."""

    def __init__(self, job_id: str, *, max_in_ram: int | None = None) -> None:
        self.job_id = job_id
        self.max_in_ram = max_in_ram if max_in_ram is not None else max_chunks_in_ram()
        self._ram: list[str] = []
        self._db_path = Path(__file__).resolve().parent / ".nocturne_cache" / "chunk_jobs" / f"{job_id}.db"
        self._conn: sqlite3.Connection | None = None
        self._spilled = False
        self._disk_count = 0

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS chunks (idx INTEGER PRIMARY KEY, text TEXT NOT NULL)"
            )
            self._conn.commit()
        return self._conn

    def _spill_ram_to_disk(self) -> None:
        if self._spilled or not self._ram:
            return
        conn = self._ensure_conn()
        base = self._disk_count
        for i, text in enumerate(self._ram):
            conn.execute(
                "INSERT OR REPLACE INTO chunks (idx, text) VALUES (?, ?)",
                (base + i, text),
            )
        self._disk_count += len(self._ram)
        self._ram.clear()
        conn.commit()
        self._spilled = True

    def append(self, text: str) -> None:
        if not self._spilled and len(self._ram) >= self.max_in_ram:
            self._spill_ram_to_disk()
        if self._spilled:
            conn = self._ensure_conn()
            conn.execute(
                "INSERT OR REPLACE INTO chunks (idx, text) VALUES (?, ?)",
                (self._disk_count, text),
            )
            self._disk_count += 1
            conn.commit()
        else:
            self._ram.append(text)

    def extend(self, items: list[str]) -> None:
        for item in items:
            self.append(item)

    def __len__(self) -> int:
        return len(self._ram) + self._disk_count

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> str:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if not self._spilled:
            return self._ram[index]
        if index < self._disk_count:
            conn = self._ensure_conn()
            row = conn.execute("SELECT text FROM chunks WHERE idx=?", (index,)).fetchone()
            if row:
                return str(row[0])
            raise IndexError(index)
        return self._ram[index - self._disk_count]

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def cleanup(self) -> None:
        self.close()
        if self._db_path.is_file():
            try:
                self._db_path.unlink()
            except OSError:
                pass
        parent = self._db_path.parent
        if parent.is_dir() and not any(parent.iterdir()):
            try:
                parent.rmdir()
            except OSError:
                pass

    @classmethod
    def from_list(cls, job_id: str, chunks: list[str]) -> ChunkStore | list[str]:
        """Return list if small enough, else spill to disk."""
        if len(chunks) <= max_chunks_in_ram():
            return chunks
        store = cls(job_id)
        store.extend(chunks)
        return store
