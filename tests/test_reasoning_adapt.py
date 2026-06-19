# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reasoning-model adaptation in call_llm: escalation, <think> stripping, preference."""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx

import processor
import reasoning_models


class _Resp:
    def __init__(self, data: dict, status: int = 200) -> None:
        self._data = data
        self.status_code = status
        self.text = ""
        self.request = None

    def json(self) -> dict:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)  # type: ignore[arg-type]


def _msg(content: str) -> dict:
    return {"output": [{"type": "message", "content": content}]}


class _FakeClient:
    """Records outgoing payloads; returns queued responses in order."""

    def __init__(self, responses: list[_Resp]) -> None:
        self._responses = list(responses)
        self.payloads: list[dict] = []

    async def post(self, url, json=None, headers=None):  # noqa: A002
        self.payloads.append(json or {})
        return self._responses.pop(0) if self._responses else _Resp(_msg("x"))


REASONING_MODEL = "deepseek-r1"  # matches the reasoning id heuristic


def _call(model, client, **kw):
    return asyncio.run(processor.call_llm(
        [{"role": "user", "content": "вопрос"}],
        model, "http://x/v1", "", asyncio.Semaphore(1),
        api_mode="native", client=client, max_retries=4, **kw,
    ))


class TestReasoningAdapt(unittest.TestCase):
    def setUp(self) -> None:
        reasoning_models.clear_no_reasoning_param_cache()
        reasoning_models.refresh_model_catalog_cache([])

    def test_reasoning_model_gets_reasoning_off_first(self) -> None:
        client = _FakeClient([_Resp(_msg("ответ"))])
        out = _call(REASONING_MODEL, client)
        self.assertEqual(out, "ответ")
        self.assertEqual(client.payloads[0].get("reasoning"), "off")

    def test_classify_model_loading_400(self) -> None:
        # «Тяжёлая модель грузится» нужно отличать от прочих 400, чтобы дождаться её.
        self.assertEqual(
            processor._classify_http_400("Failed to load model X. Operation canceled."),
            "model_loading")
        self.assertEqual(processor._classify_http_400("the model is loading"), "model_loading")
        # не путаем с другими 400
        self.assertEqual(processor._classify_http_400("context length exceeded"), "context_limit")

    @patch("processor.asyncio.sleep", new_callable=AsyncMock)
    def test_retries_while_model_loading_then_answers(self, _sleep: AsyncMock) -> None:
        # 400 «модель грузится» → НЕ падаем, ждём и повторяем; со второй попытки ответ.
        r400 = _Resp({}, status=400)
        r400.text = '{"error":{"message":"Failed to load model google/gemma-4-26b. Operation canceled."}}'
        client = _FakeClient([r400, _Resp(_msg("Ответ по источникам [1]."))])
        out = _call("google/gemma-4-26b-a4b-qat", client)
        self.assertEqual(out, "Ответ по источникам [1].")
        self.assertEqual(len(client.payloads), 2)  # повторил после ошибки загрузки
        self.assertTrue(_sleep.await_count >= 1)    # подождал перед повтором

    def test_escalates_to_reasoning_on_when_empty(self) -> None:
        # Empty under reasoning:off -> retry WITHOUT reasoning:off (let it think).
        client = _FakeClient([_Resp(_msg("")), _Resp(_msg("Подумал и ответил"))])
        out = _call(REASONING_MODEL, client)
        self.assertEqual(out, "Подумал и ответил")
        self.assertGreaterEqual(len(client.payloads), 2)
        self.assertEqual(client.payloads[0].get("reasoning"), "off")
        self.assertNotIn("reasoning", client.payloads[1])  # escalated

    def test_prefer_reasoning_off_false_skips_off(self) -> None:
        client = _FakeClient([_Resp(_msg("сразу ответ"))])
        out = _call(REASONING_MODEL, client, prefer_reasoning_off=False)
        self.assertEqual(out, "сразу ответ")
        self.assertNotIn("reasoning", client.payloads[0])

    def test_natural_language_salvages_reasoning_channel(self) -> None:
        # Natural-language site (prefer_reasoning_off=False): a small reasoning
        # model may put the whole answer in the reasoning channel with empty
        # content — we must recover it, not return empty.
        resp = {"output": [{"type": "reasoning", "content": "Ответ из reasoning-канала"}]}
        client = _FakeClient([_Resp(resp)])
        out = _call(REASONING_MODEL, client, prefer_reasoning_off=False)
        self.assertEqual(out, "Ответ из reasoning-канала")

    def test_strips_inline_think_block(self) -> None:
        client = _FakeClient([_Resp(_msg("<think>рассуждаю долго</think>Чистый ответ"))])
        out = _call(REASONING_MODEL, client)
        self.assertEqual(out, "Чистый ответ")

    def test_gemma4_detected_and_escalates(self) -> None:
        # gemma-4* are reasoning models (must be detected even without a fresh
        # catalog) so the empty-under-reasoning:off escalation kicks in.
        self.assertTrue(reasoning_models.model_id_suggests_reasoning("google/gemma-4-e2b"))
        self.assertFalse(reasoning_models.model_id_suggests_reasoning("google/gemma-2-9b"))
        client = _FakeClient([_Resp(_msg("")), _Resp(_msg("ответ после раздумий"))])
        out = _call("google/gemma-4-e2b", client)
        self.assertEqual(out, "ответ после раздумий")
        self.assertEqual(client.payloads[0].get("reasoning"), "off")
        self.assertNotIn("reasoning", client.payloads[1])

    def test_non_reasoning_model_no_reasoning_param(self) -> None:
        client = _FakeClient([_Resp(_msg("ok"))])
        out = _call("llama-3.1-8b", client)
        self.assertEqual(out, "ok")
        self.assertNotIn("reasoning", client.payloads[0])


if __name__ == "__main__":
    unittest.main()
