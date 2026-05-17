# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import unittest

from lm_client import classify_http_400, extract_chat_response_content, extract_model_ids


class TestLmClient(unittest.TestCase):
    def test_extract_model_ids_openai_shape(self) -> None:
        data = {"data": [{"id": "m1"}, {"id": "m2"}]}
        self.assertEqual(extract_model_ids(data), ["m1", "m2"])

    def test_classify_context(self) -> None:
        self.assertEqual(classify_http_400("n_ctx exceeded"), "context_limit")

    def test_classify_reasoning(self) -> None:
        self.assertEqual(classify_http_400("invalid reasoning param"), "unsupported_option")

    def test_extract_text_output_item(self) -> None:
        data = {"output": [{"type": "text", "content": "answer"}]}
        self.assertEqual(extract_chat_response_content(data)[0], "answer")

    def test_extract_native_message(self) -> None:
        data = {"output": [{"type": "message", "content": "OK"}]}
        content, reasoning = extract_chat_response_content(data)
        self.assertEqual(content, "OK")
        self.assertEqual(reasoning, "")

    def test_extract_native_reasoning_only(self) -> None:
        data = {"output": [{"type": "reasoning", "content": "thinking…"}]}
        content, reasoning = extract_chat_response_content(data)
        self.assertEqual(content, "")
        self.assertIn("thinking", reasoning)


if __name__ == "__main__":
    unittest.main()
