# SPDX-License-Identifier: AGPL-3.0-or-later
"""First-run setup marker and wizard helpers."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _marker_path() -> Path:
    """Маркер «первый запуск пройден» — в ПЕРСИСТЕНТНОМ месте.

    В упакованном .exe пишем рядом с бинарником (в NocturneData), а НЕ в _internal:
    тот пересоздаётся при каждой пересборке → мастер всплывал бы заново.
    """
    cache = os.getenv("NOCTURNE_CACHE_DIR", "").strip()
    if cache:  # .../NocturneData/.nocturne_cache → .../NocturneData/first_run_done.json
        return Path(cache).expanduser().resolve().parent / "first_run_done.json"
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "NocturneData" / "first_run_done.json"
    return Path(__file__).resolve().parent / ".local" / "first_run_done.json"


def is_first_run() -> bool:
    return not _marker_path().is_file()


def mark_first_run_complete(config: dict[str, str]) -> None:
    p = _marker_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
