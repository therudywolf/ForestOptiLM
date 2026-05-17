# SPDX-License-Identifier: AGPL-3.0-or-later
"""Session/project grouping for cache, index, and outputs."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_SESSIONS_ROOT = Path(__file__).resolve().parent / ".local" / "sessions"


def create_session(name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())[:64] or "session"
    path = _SESSIONS_ROOT / f"{safe}_{int(time.time())}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "cache").mkdir(exist_ok=True)
    (path / "index").mkdir(exist_ok=True)
    (path / "output").mkdir(exist_ok=True)
    meta = {"name": name, "created_at": int(time.time())}
    (path / "session.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_sessions() -> list[dict[str, Any]]:
    if not _SESSIONS_ROOT.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(_SESSIONS_ROOT.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_dir():
            continue
        meta_path = p / "session.json"
        meta: dict[str, Any] = {"path": str(p), "name": p.name}
        if meta_path.is_file():
            try:
                meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        out.append(meta)
    return out
