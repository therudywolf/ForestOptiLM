# SPDX-License-Identifier: AGPL-3.0-or-later
"""EmbeddingClient must not spawn a new LM Studio instance on every request."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import embeddings
from embeddings import EmbeddingClient


def _resp(status: int, payload: dict | None = None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload or {}
    r.text = ""
    return r


class TestEmbeddingLoadOnce(unittest.TestCase):
    def setUp(self) -> None:
        embeddings._EMB_LOAD_REQUESTED.clear()

    def tearDown(self) -> None:
        embeddings._EMB_LOAD_REQUESTED.clear()

    def _client(self):
        c = EmbeddingClient("http://h:1234", "k", "emb-1")
        c._client = MagicMock()
        return c

    def test_loads_once_when_not_loaded(self) -> None:
        c = self._client()
        c._client.get.return_value = _resp(200, {"models": [{"key": "emb-1", "loaded_instances": []}]})
        c._client.post.return_value = _resp(200)
        c._try_load_model()
        self.assertEqual(c._client.post.call_count, 1)  # одна загрузка
        url = c._client.post.call_args.args[0]
        self.assertTrue(url.endswith("/api/v1/models/load"))

    def test_skips_load_when_already_loaded(self) -> None:
        c = self._client()
        c._client.get.return_value = _resp(200, {
            "models": [{"key": "emb-1", "loaded_instances": [{"id": "emb-1"}, {"id": "emb-1:2"}]}]})
        c._try_load_model()
        c._client.post.assert_not_called()  # инстанс уже есть → не грузим

    def test_second_client_does_not_reload(self) -> None:
        c1 = self._client()
        c1._client.get.return_value = _resp(200, {"models": [{"key": "emb-1", "loaded_instances": []}]})
        c1._client.post.return_value = _resp(200)
        c1._try_load_model()
        # второй клиент той же модели — процесс-гард не даёт грузить снова
        c2 = self._client()
        c2._try_load_model()
        c2._client.get.assert_not_called()
        c2._client.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
