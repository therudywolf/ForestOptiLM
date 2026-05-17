# SPDX-License-Identifier: AGPL-3.0-or-later
"""First-run setup marker and wizard helpers."""
from __future__ import annotations

import json
from pathlib import Path

_MARKER = Path(__file__).resolve().parent / ".local" / "first_run_done.json"


def is_first_run() -> bool:
    return not _MARKER.is_file()


def mark_first_run_complete(config: dict[str, str]) -> None:
    _MARKER.parent.mkdir(parents=True, exist_ok=True)
    _MARKER.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
