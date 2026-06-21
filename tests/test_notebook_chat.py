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
        self.last_top_k: int | None = None

    def query(self, question: str, **_kw) -> list[_Hit]:
        self.last_query = question
        self.last_top_k = _kw.get("top_k")
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

    def test_build_messages_includes_schema(self) -> None:
        ctx = nc.select_contexts([_Hit("содержимое", "C:/x/a.txt")], max_tokens=1000)
        msgs = nc.build_chat_messages("q", ctx, schema="Домен: архитектура; сущности — ВМ и подсистемы.")
        self.assertIn("[Контекст домена этого блокнота]", msgs[0]["content"])
        self.assertIn("сущности — ВМ", msgs[0]["content"])

    def test_build_messages_caps_long_schema(self) -> None:
        # Ревью beta.11 #3: длинная схема не должна раздувать system-промпт.
        ctx = nc.select_contexts([_Hit("c", "C:/x/a.txt")], max_tokens=1000)
        msgs = nc.build_chat_messages("q", ctx, schema="Д" * 5000)
        # system = базовый промпт + блок схемы (≤2000 символов схемы)
        self.assertLessEqual(msgs[0]["content"].count("Д"), 2000)

    def test_build_messages_no_schema_block_when_empty(self) -> None:
        ctx = nc.select_contexts([_Hit("x", "C:/x/a.txt")], max_tokens=1000)
        msgs = nc.build_chat_messages("q", ctx, schema="   ")
        self.assertNotIn("Контекст домена", msgs[0]["content"])

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

    def test_default_top_k_retrieves_more(self) -> None:
        # Регрессия: с маленькими чанками нужное «размазано» по большему числу
        # фрагментов, поэтому по умолчанию забираем top_k=16 (а не 8), иначе на
        # больших блокнотах релевантный фрагмент не попадает в контекст → отказ.
        nb = _FakeNotebook([_Hit("grounding", "C:/x/a.txt")])

        async def fake_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            return "ответ [1]."

        with mock.patch("processor.call_llm", new=fake_call_llm):
            asyncio.run(nc.answer_question(nb, "q", base_url="u", api_key="", chat_model="m"))
        self.assertEqual(nb.last_top_k, 16)

    def test_chat_uses_reasoning_off(self) -> None:
        # Для grounded-чата prefer_reasoning_off=True даёт чистый ответ (проверено
        # живьём: gemma-4 с reasoning:on парротит ограничения промпта в ответ).
        nb = _FakeNotebook([_Hit("grounding", "C:/x/a.txt")])
        captured: dict = {}

        async def fake_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            captured.update(kw)
            return "ответ [1]."

        with mock.patch("processor.call_llm", new=fake_call_llm):
            asyncio.run(nc.answer_question(
                nb, "q", base_url="u", api_key="", chat_model="google/gemma-4-12b-qat"))
        self.assertTrue(captured.get("prefer_reasoning_off"))

    def test_stop_flag_cancels_in_flight_request(self) -> None:
        # «Стоп» во время ответа модели → кооперативная отмена, не падение.
        nb = _FakeNotebook([_Hit("grounding", "C:/x/a.txt")])
        stop = {"v": False}

        async def slow_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            stop["v"] = True  # имитируем нажатие «Стоп» сразу после старта запроса
            await asyncio.sleep(5)  # должен быть отменён задолго до конца
            return "не должно дойти"

        with mock.patch("processor.call_llm", new=slow_call_llm):
            res = asyncio.run(nc.answer_question(
                nb, "q", base_url="u", api_key="", chat_model="m",
                stop_flag=lambda: stop["v"]))
        self.assertTrue(res.extra.get("cancelled"))
        self.assertEqual(res.answer, nc.CANCELLED_TEXT)
        self.assertEqual(len(res.contexts), 1)  # источники всё равно показываем

    def test_stop_flag_before_llm_skips_request(self) -> None:
        # Уже остановлено к моменту поиска → запрос к модели не уходит вовсе.
        nb = _FakeNotebook([_Hit("grounding", "C:/x/a.txt")])
        called = {"v": False}

        async def must_not_run(messages, model, base_url, api_key, semaphore, **kw):
            called["v"] = True
            return "x"

        with mock.patch("processor.call_llm", new=must_not_run):
            res = asyncio.run(nc.answer_question(
                nb, "q", base_url="u", api_key="", chat_model="m",
                stop_flag=lambda: True))
        self.assertTrue(res.extra.get("cancelled"))
        self.assertFalse(called["v"])

    def test_enhanced_runs_expansion_and_rerank(self) -> None:
        # «Точный поиск»: expansion (3 запроса) → rerank → grounded-ответ.
        nb = _FakeNotebook([_Hit("xz backdoor", "C:/x/a.txt", chunk_id="c1"),
                            _Hit("другой фрагмент", "C:/x/b.txt", chunk_id="c2")])
        calls = {"expansion": 0, "rerank": 0, "answer": 0}

        async def dispatch(messages, model, base_url, api_key, semaphore, **kw):
            sys = messages[0]["content"]
            if "альтернативные формулировки" in sys:
                calls["expansion"] += 1
                return '["xz уязвимость", "cve бэкдор"]'
            if "реранкер" in sys:
                calls["rerank"] += 1
                return "[2, 1]"
            calls["answer"] += 1
            return "Это бэкдор [1]."

        with mock.patch("processor.call_llm", new=dispatch):
            res = asyncio.run(nc.answer_question(
                nb, "что такое xz?", base_url="u", api_key="", chat_model="m", enhanced=True))
        self.assertEqual(calls["expansion"], 1)
        self.assertEqual(calls["rerank"], 1)
        self.assertEqual(calls["answer"], 1)
        self.assertFalse(res.refused)
        self.assertEqual([c["n"] for c in res.citations], [1])

    def test_enhanced_falls_back_on_bad_json(self) -> None:
        # Модель вернула не-JSON на expansion/rerank → мягкий фолбэк, ответ всё равно есть.
        nb = _FakeNotebook([_Hit("grounding", "C:/x/a.txt", chunk_id="c1")])

        async def dispatch(messages, model, base_url, api_key, semaphore, **kw):
            sys = messages[0]["content"]
            if "альтернативные формулировки" in sys or "реранкер" in sys:
                return "извините, не понял"
            return "ответ [1]."

        with mock.patch("processor.call_llm", new=dispatch):
            res = asyncio.run(nc.answer_question(
                nb, "q", base_url="u", api_key="", chat_model="m", enhanced=True))
        self.assertFalse(res.refused)
        self.assertEqual(len(res.contexts), 1)

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


class TestSSEParse(unittest.TestCase):
    def test_extracts_delta_content(self) -> None:
        import processor
        line = 'data: {"choices":[{"delta":{"content":"Привет"}}]}'
        self.assertEqual(processor.parse_sse_delta(line), "Привет")

    def test_done_and_blank_and_garbage_return_none(self) -> None:
        import processor
        self.assertIsNone(processor.parse_sse_delta("data: [DONE]"))
        self.assertIsNone(processor.parse_sse_delta(""))
        self.assertIsNone(processor.parse_sse_delta(": keep-alive"))
        self.assertIsNone(processor.parse_sse_delta("data: not-json"))
        self.assertIsNone(processor.parse_sse_delta('data: {"choices":[{"delta":{}}]}'))


class TestStreamingChat(unittest.TestCase):
    def test_streams_on_openai_and_calls_on_token(self) -> None:
        nb = _FakeNotebook([_Hit("grounding", "C:/x/a.txt")])
        tokens: list[str] = []

        async def fake_stream(messages, model, base_url, api_key, *, on_token, **kw):
            for t in ["Это ", "ответ ", "[1]."]:
                on_token(t)
            return "Это ответ [1]."

        with mock.patch("processor.call_llm_stream", new=fake_stream):
            res = asyncio.run(nc.answer_question(
                nb, "q", base_url="u", api_key="", chat_model="m", api_mode="openai",
                on_token=tokens.append))
        self.assertEqual("".join(tokens), "Это ответ [1].")
        self.assertEqual([c["n"] for c in res.citations], [1])

    def test_falls_back_to_call_llm_when_stream_errors(self) -> None:
        nb = _FakeNotebook([_Hit("grounding", "C:/x/a.txt")])

        async def boom_stream(messages, model, base_url, api_key, *, on_token, **kw):
            raise RuntimeError("stream blew up")

        async def fake_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            return "обычный ответ [1]."

        with mock.patch("processor.call_llm_stream", new=boom_stream), \
                mock.patch("processor.call_llm", new=fake_call_llm):
            res = asyncio.run(nc.answer_question(
                nb, "q", base_url="u", api_key="", chat_model="m", api_mode="openai",
                on_token=lambda _t: None))
        self.assertEqual(res.answer, "обычный ответ [1].")
        self.assertFalse(res.refused)

    def test_streams_even_in_native_mode(self) -> None:
        # LM Studio отдаёт openai-совместимый эндпоинт и в native-режиме → стримим.
        nb = _FakeNotebook([_Hit("grounding", "C:/x/a.txt")])
        used = {"stream": False}

        async def fake_stream(messages, model, base_url, api_key, *, on_token, **kw):
            used["stream"] = True
            on_token("ответ [1].")
            return "ответ [1]."

        with mock.patch("processor.call_llm_stream", new=fake_stream):
            asyncio.run(nc.answer_question(
                nb, "q", base_url="u", api_key="", chat_model="m", api_mode="native",
                on_token=lambda _t: None))
        self.assertTrue(used["stream"])

    def test_no_streaming_in_precise_mode(self) -> None:
        # «Точный поиск» (enhanced) использует свой много-вызовный путь, не стрим.
        nb = _FakeNotebook([_Hit("grounding", "C:/x/a.txt", chunk_id="c1")])
        used = {"stream": False}

        async def fake_stream(messages, model, base_url, api_key, *, on_token, **kw):
            used["stream"] = True
            return "x"

        async def fake_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            sys = messages[0]["content"]
            if "формулировк" in sys or "реранкер" in sys:
                return "[]"
            return "ответ [1]."

        with mock.patch("processor.call_llm_stream", new=fake_stream), \
                mock.patch("processor.call_llm", new=fake_call_llm):
            asyncio.run(nc.answer_question(
                nb, "q", base_url="u", api_key="", chat_model="m", api_mode="native",
                enhanced=True, on_token=lambda _t: None))
        self.assertFalse(used["stream"])


if __name__ == "__main__":
    unittest.main()
