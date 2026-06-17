# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from processor import categorize_models, model_name_suggests_vision


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

    @patch("processor.fetch_models_catalog")
    def test_vision_detected_by_name_without_capabilities(self, catalog: MagicMock) -> None:
        # Ollama / OpenAI-compatible: plain id list, no capabilities → detect by name.
        catalog.return_value = [{"id": "llava:13b"}, {"id": "qwen2.5-vl-7b"}, {"id": "llama3.1:8b"}]
        out = categorize_models("http://127.0.0.1:11434/v1", "")
        self.assertIn("llava:13b", out["vision"])
        self.assertIn("qwen2.5-vl-7b", out["vision"])
        self.assertIn("llava:13b", out["chat"])  # vision models are chat too
        self.assertNotIn("llama3.1:8b", out["vision"])

    def test_vision_name_heuristic(self) -> None:
        for v in ("llava", "qwen2.5-vl-7b", "minicpm-v", "moondream", "pixtral-12b",
                  "google/gemma-4-e2b", "internvl2", "smolvlm"):
            self.assertTrue(model_name_suggests_vision(v), v)
        for n in ("llama-3.1-8b", "google/gemma-2-9b", "deepseek-r1", "supervision-x",
                  "text-embedding-nomic"):
            self.assertFalse(model_name_suggests_vision(n), n)


if __name__ == "__main__":
    unittest.main()
