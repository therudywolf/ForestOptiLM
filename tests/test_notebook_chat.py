# SPDX-License-Identifier: AGPL-3.0-or-later
"""Grounded notebook chat: context selection, prompt build, citations, refusal."""
from __future__ import annotations

import asyncio
import unittest
from unittest import mock

import notebook_chat as nc


class _Hit:
    def __init__(self, text: str, source_path: str, score: float = 0.9,
                 chunk_id: str = "c", title: str = "") -> None:
        self.text = text
        self.source_path = source_path
        self.score = score
        self.chunk_id = chunk_id
        self.metadata = {"title": title} if title else {}


class _FakeNotebook:
    def __init__(self, hits: list[_Hit]) -> None:
        self._hits = hits
        self.last_query: str | None = None

    def query(self, question: str, **_kw) -> list[_Hit]:
        self.last_query = question
        return self._hits


class TestPureFunctions(unittest.TestCase):
    def test_strip_headers(self) -> None:
        text = "[FILE_PATH: a.txt]\n[CHUNK_INDEX: 0]\nреальное содержимое"
        self.assertEqual(nc._strip_headers(text), "реальное содержимое")

    def test_select_contexts_numbers_and_budget(self) -> None:
        hits = [_Hit("a" * 100, "C:/x/a.txt"), _Hit("b" * 100, "C:/x/b.txt")]
        ctx = nc.select_contexts(hits, max_tokens=100000, max_items=12)
        self.assertEqual([c.n for c in ctx], [1, 2])
        self.assertEqual(ctx[0].source_path, "C:/x/a.txt")

    def test_select_contexts_keeps_first_even_if_over_budget(self) -> None:
        hits = [_Hit("word " * 5000, "C:/x/a.txt"), _Hit("b", "C:/x/b.txt")]
        ctx = nc.select_contexts(hits, max_tokens=10)
        self.assertEqual(len(ctx), 1)

    def test_select_contexts_skips_empty(self) -> None:
        ctx = nc.select_contexts([_Hit("   ", "C:/x/a.txt"), _Hit("real", "C:/x/b.txt")],
                                 max_tokens=1000)
        self.assertEqual([c.text for c in ctx], ["real"])

    def test_display_uses_basename_and_title(self) -> None:
        ctx = nc.select_contexts([_Hit("text", "C:/dumps/log.txt", title="Отчёт")], max_tokens=1000)
        self.assertIn("log.txt", ctx[0].display)

    def test_build_messages_has_sources_and_question(self) -> None:
        ctx = nc.select_contexts([_Hit("содержимое", "C:/x/a.txt")], max_tokens=1000)
        msgs = nc.build_chat_messages("мой вопрос", ctx, history=[{"role": "user", "content": "ранее"}])
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("[Источники]", msgs[1]["content"])
        self.assertIn("[1]", msgs[1]["content"])
        self.assertIn("мой вопрос", msgs[1]["content"])
        self.assertIn("Предыдущий диалог", msgs[1]["content"])

    def test_build_messages_no_history_block_when_empty(self) -> None:
        ctx = nc.select_contexts([_Hit("x", "C:/x/a.txt")], max_tokens=1000)
        msgs = nc.build_chat_messages("q", ctx, history=[])
        self.assertNotIn("Предыдущий диалог", msgs[1]["content"])

    def test_parse_used_citations_dedup_and_order(self) -> None:
        ctx = nc.select_contexts([_Hit("a", "C:/x/a.txt"), _Hit("b", "C:/x/b.txt")], max_tokens=1000)
        used = nc.parse_used_citations("сначала [2], потом [1], снова [2], и [9] нет такого", ctx)
        self.assertEqual([c["n"] for c in used], [2, 1])

    def test_parse_used_citations_quote_present(self) -> None:
        ctx = nc.select_contexts([_Hit("[FILE_PATH: a.txt]\nважный факт", "C:/x/a.txt")], max_tokens=1000)
        used = nc.parse_used_citations("вывод [1]", ctx)
        self.assertEqual(used[0]["quote"], "важный факт")
        self.assertEqual(used[0]["source_path"], "C:/x/a.txt")

    def test_is_refusal(self) -> None:
        self.assertTrue(nc.is_refusal(nc.REFUSAL_TEXT))
        self.assertFalse(nc.is_refusal("Это полноценный ответ [1] с цитатой."))


class TestAnswerQuestion(unittest.TestCase):
    def test_refuses_when_no_contexts(self) -> None:
        nb = _FakeNotebook([])
        res = asyncio.run(nc.answer_question(
            nb, "вопрос", base_url="u", api_key="", chat_model="m"))
        self.assertTrue(res.refused)
        self.assertEqual(res.answer, nc.REFUSAL_TEXT)
        self.assertEqual(res.citations, [])

    def test_empty_model_output_is_graceful(self) -> None:
        # Small reasoning models with reasoning:off sometimes return empty; the
        # chat must not crash — it returns a friendly message instead.
        nb = _FakeNotebook([_Hit("some grounding text", "C:/x/a.txt")])

        async def empty_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            raise RuntimeError("Model returned empty content (possibly reasoning-only output)")

        with mock.patch("processor.call_llm", new=empty_call_llm):
            res = asyncio.run(nc.answer_question(
                nb, "вопрос", base_url="u", api_key="", chat_model="m"))
        self.assertFalse(res.refused)
        self.assertEqual(res.citations, [])
        self.assertEqual(len(res.contexts), 1)
        self.assertTrue(res.extra.get("empty_output"))
        self.assertIn("пуст", res.answer.lower())

    def test_other_runtime_errors_propagate(self) -> None:
        nb = _FakeNotebook([_Hit("text", "C:/x/a.txt")])

        async def boom(messages, model, base_url, api_key, semaphore, **kw):
            raise RuntimeError("connection refused")

        with mock.patch("processor.call_llm", new=boom):
            with self.assertRaises(RuntimeError):
                asyncio.run(nc.answer_question(
                    nb, "q", base_url="u", api_key="", chat_model="m"))

    def test_grounded_answer_with_citations(self) -> None:
        nb = _FakeNotebook([_Hit("xz backdoor CVE-2024-3094", "C:/x/a.txt")])

        async def fake_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            # Убедимся, что промпт действительно содержит источник.
            assert "[Источники]" in messages[1]["content"]
            return "Это бэкдор в xz [1]."

        with mock.patch("processor.call_llm", new=fake_call_llm):
            res = asyncio.run(nc.answer_question(
                nb, "что такое xz?", base_url="u", api_key="", chat_model="m"))
        self.assertFalse(res.refused)
        self.assertEqual([c["n"] for c in res.citations], [1])
        self.assertEqual(len(res.contexts), 1)
        self.assertEqual(nb.last_query, "что такое xz?")


if __name__ == "__main__":
    unittest.main()
