# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mocked LM Studio smoke and connection tests (no live server)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from processor import check_lmstudio_connection, run_lmstudio_smoke_test


class TestSmokeMocked(unittest.TestCase):
    @patch("processor.fetch_models", return_value=["chat-7b", "embed-model"])
    @patch("processor.httpx.Client")
    def test_smoke_success_native(self, client_cls: MagicMock, _fm: MagicMock) -> None:
        client = client_cls.return_value.__enter__.return_value
        chat_resp = MagicMock()
        chat_resp.raise_for_status = MagicMock()
        chat_resp.json.return_value = {
            "output": [{"type": "message", "content": "OK"}],
        }
        emb_resp = MagicMock()
        emb_resp.raise_for_status = MagicMock()
        emb_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
        client.post.side_effect = [chat_resp, emb_resp]

        ok, detail = run_lmstudio_smoke_test(
            "http://127.0.0.1:1234/v1",
            "key",
            chat_model="chat-7b",
            embedding_model="embed-model",
        )
        self.assertTrue(ok, detail)
        self.assertIn("models=2", detail)
        self.assertIn("chat=chat-7b", detail)
        self.assertIn("embed=embed-model", detail)

    @patch("processor.fetch_models", return_value=["m1"])
    def test_connection_quick_ok(self, _fm: MagicMock) -> None:
        ok, msg = check_lmstudio_connection("http://127.0.0.1:1234/v1", "k", full_smoke=False)
        self.assertTrue(ok)
        self.assertIn("моделей", msg)

    @patch("processor.run_lmstudio_smoke_test", return_value=(True, "Smoke OK"))
    def test_connection_full_smoke_delegates(self, smoke: MagicMock) -> None:
        ok, msg = check_lmstudio_connection(
            "http://127.0.0.1:1234/v1", "k", full_smoke=True, chat_model="m1",
        )
        self.assertTrue(ok)
        smoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
