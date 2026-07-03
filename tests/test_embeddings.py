# SPDX-License-Identifier: AGPL-3.0-or-later
"""EmbeddingClient must not spawn a new LM Studio instance on every request."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import embeddings
from embeddings import EmbeddingClient, _task_prefix, embedding_prefix_scheme


def _resp(status: int, payload: dict | None = None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload or {}
    r.text = ""
    r.headers = {}
    return r


def _emb_resp(status: int, n: int = 1):
    return _resp(status, {"data": [{"embedding": [0.1, 0.2]} for _ in range(n)]})


class TestNomicPrefixes(unittest.TestCase):
    def test_prefix_scheme(self) -> None:
        self.assertEqual(embedding_prefix_scheme("text-embedding-nomic-embed-text-v1.5"), "nomic-v2")
        self.assertEqual(embedding_prefix_scheme("bge-m3"), "none")

    def test_task_prefix_only_for_nomic(self) -> None:
        self.assertEqual(_task_prefix("nomic-x", "document"), "search_document: ")
        self.assertEqual(_task_prefix("nomic-x", "query"), "search_query: ")
        self.assertEqual(_task_prefix("bge-m3", "document"), "")   # не-nomic не трогаем
        self.assertEqual(_task_prefix("nomic-x", None), "")

    def test_embed_texts_applies_document_prefix(self) -> None:
        c = EmbeddingClient("http://h:1234", "k", "text-embedding-nomic-embed-text-v1.5")
        c._client = MagicMock()
        c._load_attempted = True  # пропустить авто-load
        c._client.post.return_value = _emb_resp(200, n=1)
        c.embed_texts(["привет"], task="document")
        sent = c._client.post.call_args.kwargs["json"]["input"]
        self.assertEqual(sent, ["search_document: привет"])


class TestEmbedConcurrency(unittest.TestCase):
    """Конвейерный embed_texts обязан сохранять порядок векторов = порядку текстов."""

    def _client_with_echo(self) -> EmbeddingClient:
        c = EmbeddingClient("http://h:1234", "k", "bge-m3")  # без префиксов
        c._load_attempted = True

        def fake_batch(batch: list[str]) -> list[list[float]]:
            # вектор кодирует исходный текст: [номер] — порядок проверяем по нему
            import time as _t
            import random as _r
            _t.sleep(_r.random() * 0.01)  # перемешать завершение батчей
            return [[float(t.split("#")[1])] for t in batch]

        c._embed_batch = fake_batch  # type: ignore[method-assign]
        return c

    def test_order_preserved_under_concurrency(self) -> None:
        c = self._client_with_echo()
        texts = [f"txt#{i}" for i in range(101)]  # не кратно batch_size
        vecs = c.embed_texts(texts, batch_size=7, concurrency=4)
        self.assertEqual([v[0] for v in vecs], [float(i) for i in range(101)])

    def test_progress_reaches_total(self) -> None:
        c = self._client_with_echo()
        seen: list[tuple[int, int]] = []
        c.embed_texts([f"txt#{i}" for i in range(23)], batch_size=5, concurrency=3,
                      on_batch=lambda d, t: seen.append((d, t)))
        self.assertEqual(seen[-1], (23, 23))
        self.assertEqual(len(seen), 5)  # по одному колбэку на батч

    def test_serial_path_identical(self) -> None:
        c = self._client_with_echo()
        texts = [f"txt#{i}" for i in range(10)]
        self.assertEqual(c.embed_texts(texts, batch_size=3, concurrency=1),
                         c.embed_texts(texts, batch_size=3, concurrency=4))

    def test_empty_input(self) -> None:
        c = self._client_with_echo()
        self.assertEqual(c.embed_texts([], batch_size=4, concurrency=3), [])


class TestEmbeddingRetry(unittest.TestCase):
    @patch("embeddings.time.sleep", return_value=None)
    def test_retries_transient_500_then_succeeds(self, _sleep: MagicMock) -> None:
        c = EmbeddingClient("http://h:1234", "k", "nomic-embed")
        c._client = MagicMock()
        c._load_attempted = True
        c._client.post.side_effect = [_emb_resp(500), _emb_resp(200, n=1)]
        out = c._embed_batch(["x"])
        self.assertEqual(len(out), 1)
        self.assertEqual(c._client.post.call_count, 2)  # один 500 не валит сборку

    @patch("embeddings.time.sleep", return_value=None)
    def test_permanent_4xx_fails_fast(self, _sleep: MagicMock) -> None:
        c = EmbeddingClient("http://h:1234", "k", "nomic-embed")
        c._client = MagicMock()
        c._load_attempted = True
        c._client.post.return_value = _emb_resp(404)  # модель не найдена → без повторов
        with self.assertRaises(RuntimeError):
            c._embed_batch(["x"])
        self.assertEqual(c._client.post.call_count, 1)


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
