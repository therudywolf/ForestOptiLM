# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prefetch the tiktoken cl100k_base encoding into .build/tiktoken_cache.

Bundling this cache makes the packaged app tokenize fully offline. Safe to fail
(no network) — the app will fetch the encoding on first run instead.
"""
from __future__ import annotations

import os
from pathlib import Path


def main() -> int:
    cache = Path(__file__).resolve().parents[1] / ".build" / "tiktoken_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
    try:
        import tiktoken

        tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # noqa: BLE001 — offline build is allowed
        print(f"tiktoken prefetch skipped ({exc}); the app will fetch it on first run.")
        return 0
    print(f"tiktoken cached at {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
