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


if __name__ == "__main__":
    unittest.main()
