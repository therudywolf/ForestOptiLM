# SPDX-License-Identifier: AGPL-3.0-or-later
"""resolve_runtime_model_context: interruptible wait + shorter default."""
from __future__ import annotations

import inspect
import unittest
from unittest.mock import MagicMock, patch

from processor import resolve_runtime_model_context


class TestContextStop(unittest.TestCase):
    def test_default_wait_is_short(self) -> None:
        sig = inspect.signature(resolve_runtime_model_context)
        # Дефолт снижен со 180с, чтобы мёртвый/медленный сервер не вешал UI.
        self.assertLessEqual(sig.parameters["max_wait_seconds"].default, 60.0)
        self.assertIn("stop_flag", sig.parameters)

    @patch("processor.httpx.Client")
    def test_stop_flag_short_circuits_before_any_http(self, client_cls: MagicMock) -> None:
        # stop_flag True на входе → выходим до единого сетевого запроса.
        ctx, source, state = resolve_runtime_model_context(
            "http://127.0.0.1:1234",
            "k",
            "model-x",
            wait_for_loaded=True,
            max_wait_seconds=999.0,
            stop_flag=lambda: True,
        )
        self.assertIsNone(ctx)
        self.assertEqual(source, "unavailable")
        self.assertEqual(state, "stopped")
        client_cls.assert_not_called()

    @patch("processor.httpx.Client")
    def test_stop_flag_breaks_after_polls(self, client_cls: MagicMock) -> None:
        # Сервер отдаёт модель «не загружена» → цикл крутился бы до дедлайна;
        # stop_flag, ставший True после первой итерации, прерывает ожидание.
        client = client_cls.return_value.__enter__.return_value
        not_loaded = MagicMock()
        not_loaded.raise_for_status = MagicMock()
        not_loaded.json.return_value = {
            "models": [{"key": "model-x", "loaded_instances": []}]
        }
        client.get.return_value = not_loaded
        client.post.return_value = MagicMock(status_code=200)

        calls = {"n": 0}

        def _stop() -> bool:
            calls["n"] += 1
            return calls["n"] > 1  # False первый раз, потом True

        ctx, source, state = resolve_runtime_model_context(
            "http://127.0.0.1:1234",
            "k",
            "model-x",
            wait_for_loaded=True,
            max_wait_seconds=999.0,
            poll_interval_seconds=0.01,
            stop_flag=_stop,
        )
        self.assertEqual(state, "stopped")
        # Должны были выйти задолго до дедлайна (мало проверок stop_flag).
        self.assertLessEqual(calls["n"], 4)


if __name__ == "__main__":
    unittest.main()
