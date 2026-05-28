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


if __name__ == "__main__":
    unittest.main()
