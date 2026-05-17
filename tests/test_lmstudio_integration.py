# SPDX-License-Identifier: AGPL-3.0-or-later
"""Integration smoke against a live LM Studio instance (opt-in)."""
from __future__ import annotations

import os

import pytest

from lmstudio_config import get_connection_defaults
from processor import run_lmstudio_smoke_test


@pytest.mark.integration
def test_lmstudio_smoke_live() -> None:
    if os.getenv("NOCTURNE_RUN_INTEGRATION", "").strip() != "1":
        pytest.skip("NOCTURNE_RUN_INTEGRATION not set")
    base_url, api_key, _ = get_connection_defaults()
    ok, detail = run_lmstudio_smoke_test(
        base_url,
        api_key,
        chat_model=os.getenv("NOCTURNE_SMOKE_CHAT_MODEL", "").strip() or None,
        embedding_model=os.getenv("NOCTURNE_SMOKE_EMBED_MODEL", "").strip() or None,
    )
    low = detail.lower()
    if not ok and any(x in low for x in ("connection", "connect", "timed out", "refused")):
        pytest.skip(f"LM Studio not reachable: {detail}")
    assert ok, detail
