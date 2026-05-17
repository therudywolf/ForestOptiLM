# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pytest configuration: integration tests are opt-in only."""
from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.getenv("NOCTURNE_RUN_INTEGRATION", "").strip() == "1":
        return
    skip = pytest.mark.skip(reason="Set NOCTURNE_RUN_INTEGRATION=1 to run live LM Studio tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
