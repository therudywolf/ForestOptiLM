# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hybrid (vector + BM25 / RRF) retrieval over a real FAISS index with fake vectors."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import faiss  # noqa: F401
    _HAVE_FAISS = True
except Exception:
    _HAVE_FAISS = False

from models import DocumentChunk


@unittest.skipUnless(_HAVE_FAISS, "faiss not installed")
class TestHybridSearch(unittest.TestCase):
    def _store(self, index_dir: Path):
        from retrieval import LocalFaissStore

        chunks = [
            DocumentChunk("c0", "a.txt", "memory leak in cache module", 6, {}),
            DocumentChunk("c1", "b.txt", "reflected xss in login form", 6, {}),
            DocumentChunk("c2", "c.txt", "remote code execution CVE-2024-3094 xz backdoor", 7, {}),
        ]
        # Fake unit vectors: query will be closest to c0 by vector, but the exact
        # CVE term lives only in c2 — BM25 must pull c2 up via RRF.
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        store = LocalFaissStore(index_dir=index_dir)
        store.build(chunks, vectors, embedding_model="fake")
        return store

    def test_exact_cve_found_via_bm25_even_with_offtarget_vector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = self._store(Path(td))
            # Vector points at c0; lexical query is the CVE that's only in c2.
            hits = store.hybrid_search(
                query_text="CVE-2024-3094", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=3,
            )
            ids = [h.chunk_id for h in hits]
            self.assertIn("c2", ids)

    def test_bm25_only_when_no_vector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = self._store(Path(td))
            hits = store.hybrid_search(query_text="xss login", query_vector=None, top_k=2)
            self.assertTrue(hits)
            self.assertEqual(hits[0].chunk_id, "c1")

    def test_min_score_ratio_trims_tail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = self._store(Path(td))
            full = store.hybrid_search(
                query_text="memory leak", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=3,
            )
            trimmed = store.hybrid_search(
                query_text="memory leak", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=3,
                min_score_ratio=0.95,  # агрессивно: оставить лишь почти-лучшие
            )
            self.assertLessEqual(len(trimmed), len(full))
            self.assertTrue(trimmed)  # лучший фрагмент никогда не выкидываем
            # Все оставшиеся — не ниже порога относительно лучшего.
            top = trimmed[0].score
            self.assertTrue(all(h.score >= top * 0.95 - 1e-9 for h in trimmed))

    def test_store_for_reuses_instance_per_dir(self) -> None:
        import pipeline
        with tempfile.TemporaryDirectory() as td:
            pipeline._STORE_CACHE.clear()
            a = pipeline._store_for(Path(td))
            b = pipeline._store_for(Path(td))
            self.assertIs(a, b)  # тот же инстанс → кэш FAISS/BM25 переживает запросы

    def test_missing_index_mid_query_returns_empty(self) -> None:
        # Ревью beta.8: блокнот удалили в соседнем потоке между .exists() и stat()
        # → раньше летел FileNotFoundError. Теперь — пустой результат, без падения.
        with tempfile.TemporaryDirectory() as td:
            store = self._store(Path(td))
            store.index_file.unlink()
            store.meta_file.unlink()
            self.assertEqual(store.hybrid_search("x", [1.0, 0.0, 0.0, 0.0], top_k=3), [])
            self.assertEqual(store.search([1.0, 0.0, 0.0, 0.0], top_k=3), [])

    def test_evict_store_drops_cache_entry(self) -> None:
        import pipeline
        with tempfile.TemporaryDirectory() as td:
            pipeline._STORE_CACHE.clear()
            a = pipeline._store_for(Path(td))
            pipeline._evict_store(Path(td))
            b = pipeline._store_for(Path(td))
            self.assertIsNot(a, b)  # после пересборки кэш сброшен → новый инстанс

    def test_cache_invalidates_on_same_mtime_content_change(self) -> None:
        # Регрессия (ревью beta.8): на FS с грубым mtime (FAT32/exFAT/USB/сеть)
        # пересборка может попасть в тот же mtime. Инвалидация по mtime+РАЗМЕРУ
        # файлов ловит смену содержимого даже без сдвига mtime → не отдаём устаревший
        # корпус (та самая боль «нет ответа» после переиндексации).
        import os

        from retrieval import LocalFaissStore
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            store = self._store(d)  # c0,c1,c2
            store.hybrid_search("memory leak", [1.0, 0.0, 0.0, 0.0], top_k=3)  # прогрев кэша
            old_idx = store.index_file.stat().st_mtime_ns
            old_meta = store.meta_file.stat().st_mtime_ns

            # Пересобираем НА ДИСКЕ другим содержимым (1 чанк вместо 3 → другой размер).
            LocalFaissStore(index_dir=d).build(
                [DocumentChunk("z0", "z.txt", "completely different zebra content", 5, {})],
                [[0.0, 0.0, 0.0, 1.0]], embedding_model="fake")
            # Имитируем грубый mtime: возвращаем файлам СТАРОЕ время.
            os.utime(store.index_file, ns=(old_idx, old_idx))
            os.utime(store.meta_file, ns=(old_meta, old_meta))

            # Тот же закэшированный стор обязан увидеть НОВОЕ содержимое (размер сменился).
            hits = store.hybrid_search("zebra", [0.0, 0.0, 0.0, 1.0], top_k=3)
            ids = [h.chunk_id for h in hits]
            self.assertIn("z0", ids)
            self.assertNotIn("c0", ids)


if __name__ == "__main__":
    unittest.main()
