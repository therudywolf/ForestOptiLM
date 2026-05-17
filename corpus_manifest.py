# SPDX-License-Identifier: AGPL-3.0-or-later
"""Project corpus manifest without LLM calls."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline import _iter_files


def _lang_guess(path: Path) -> str:
    ext = path.suffix.lower()
    mapping = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".java": "java", ".go": "go", ".rs": "rust", ".md": "markdown",
        ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    }
    return mapping.get(ext, ext.lstrip(".") or "unknown")


def build_corpus_manifest(paths: list[Path]) -> dict[str, Any]:
    files = _iter_files(paths)
    by_lang: Counter[str] = Counter()
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for fp in files:
        try:
            st = fp.stat()
            size = st.st_size
            total_bytes += size
        except OSError:
            size = 0
        lang = _lang_guess(fp)
        by_lang[lang] += 1
        try:
            rel = str(fp)
        except Exception:
            rel = fp.name
        entries.append({"path": rel.replace("\\", "/"), "size_bytes": size, "lang": lang})
    return {
        "files_total": len(entries),
        "total_bytes": total_bytes,
        "languages": dict(by_lang.most_common()),
        "files": entries[:5000],
        "truncated": len(entries) > 5000,
    }


def manifest_to_json(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2)
