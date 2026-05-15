# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
"""Regression tests for LM Studio URL normalization."""
from __future__ import annotations

import unittest

from lmstudio_config import lmstudio_root_url, normalize_lmstudio_base_url


class TestLMStudioUrlNormalization(unittest.TestCase):
    def test_root_url_gets_openai_base_suffix(self) -> None:
        self.assertEqual(
            normalize_lmstudio_base_url("http://127.0.0.1:1234"),
            "http://127.0.0.1:1234/v1",
        )

    def test_native_base_url_maps_to_openai_base(self) -> None:
        self.assertEqual(
            normalize_lmstudio_base_url("http://127.0.0.1:1234/api/v1"),
            "http://127.0.0.1:1234/v1",
        )

    def test_copied_endpoint_url_maps_to_root(self) -> None:
        self.assertEqual(
            lmstudio_root_url("http://127.0.0.1:1234/v1/chat/completions"),
            "http://127.0.0.1:1234",
        )
        self.assertEqual(
            lmstudio_root_url("http://127.0.0.1:1234/api/v1/models/load"),
            "http://127.0.0.1:1234",
        )
        self.assertEqual(
            lmstudio_root_url("http://127.0.0.1:1234/api/v1/models"),
            "http://127.0.0.1:1234",
        )


if __name__ == "__main__":
    unittest.main()
