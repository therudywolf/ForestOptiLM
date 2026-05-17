# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from processor import categorize_models


class TestCategorizeModels(unittest.TestCase):
    @patch("processor.fetch_models_catalog")
    def test_splits_by_type_and_vision(self, catalog: MagicMock) -> None:
        catalog.return_value = [
            {"key": "llm-7b", "type": "llm"},
            {"key": "embed-small", "type": "embedding"},
            {
                "key": "vision-llm",
                "type": "llm",
                "capabilities": {"vision": True},
            },
            {
                "key": "qwen/qwen3.6-27b",
                "type": "llm",
                "capabilities": {
                    "reasoning": {"allowed_options": ["off", "on"], "default": "on"},
                },
            },
        ]
        out = categorize_models("http://127.0.0.1:1234/v1", "k")
        self.assertIn("llm-7b", out["chat"])
        self.assertIn("embed-small", out["embedding"])
        self.assertIn("vision-llm", out["vision"])
        self.assertIn("qwen/qwen3.6-27b", out["reasoning"])


if __name__ == "__main__":
    unittest.main()
