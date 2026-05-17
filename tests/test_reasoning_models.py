# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import unittest

from lm_client import extract_chat_response_content, is_unsupported_reasoning_response
from reasoning_models import (
    clear_no_reasoning_param_cache,
    model_has_reasoning_capability,
    model_id_suggests_reasoning,
    native_reasoning_payload,
    parse_reasoning_capability,
    refresh_model_catalog_cache,
)


class TestReasoningCapability(unittest.TestCase):
    def test_parse_v1_catalog_shape(self) -> None:
        cap = parse_reasoning_capability(
            {"reasoning": {"allowed_options": ["off", "on"], "default": "on"}},
        )
        self.assertIsNotNone(cap)
        assert cap is not None
        self.assertTrue(cap.supports_off())
        self.assertEqual(native_reasoning_payload("qwen/qwen3.6-27b", capabilities={"reasoning": {"allowed_options": ["off", "on"], "default": "on"}}), {"reasoning": "off"})

    def test_heuristic_qwen3(self) -> None:
        self.assertTrue(model_id_suggests_reasoning("qwen/qwen3.6-27b"))
        self.assertFalse(model_id_suggests_reasoning("google/gemma-2-9b"))

    def test_blocked_skips_param(self) -> None:
        clear_no_reasoning_param_cache()
        refresh_model_catalog_cache(
            [{"key": "m1", "capabilities": {"reasoning": {"allowed_options": ["off", "on"], "default": "on"}}}],
        )
        self.assertEqual(native_reasoning_payload("m1"), {"reasoning": "off"})
        from reasoning_models import mark_no_reasoning_param

        mark_no_reasoning_param("m1")
        self.assertEqual(native_reasoning_payload("m1"), {})

    def test_has_reasoning_from_catalog(self) -> None:
        self.assertTrue(
            model_has_reasoning_capability(
                "x",
                {"reasoning": {"allowed_options": ["off", "on"], "default": "on"}},
            ),
        )


class TestExtractChatResponse(unittest.TestCase):
    def test_native_message(self) -> None:
        data = {"output": [{"type": "message", "content": "OK"}]}
        self.assertEqual(extract_chat_response_content(data), ("OK", ""))

    def test_native_reasoning_only(self) -> None:
        data = {"output": [{"type": "reasoning", "content": "thinking…"}]}
        self.assertEqual(extract_chat_response_content(data), ("", "thinking…"))

    def test_openai_shape(self) -> None:
        data = {"choices": [{"message": {"content": "hi", "reasoning_content": "think"}}]}
        self.assertEqual(extract_chat_response_content(data), ("hi", "think"))


class TestUnsupportedReasoningResponse(unittest.TestCase):
    def test_detects_param_error(self) -> None:
        class R:
            status_code = 400
            text = '{"error":{"param":"reasoning","code":"invalid_value"}}'

            def json(self):
                return {"error": {"param": "reasoning", "code": "invalid_value"}}

        self.assertTrue(is_unsupported_reasoning_response(R()))


if __name__ == "__main__":
    unittest.main()
