# SPDX-License-Identifier: AGPL-3.0-or-later
"""BM25 lexical index + RRF fusion (no external deps)."""
from __future__ import annotations

import unittest

from bm25 import BM25Index, reciprocal_rank_fusion, tokenize


class TestTokenize(unittest.TestCase):
    def test_keeps_security_tokens_whole(self) -> None:
        toks = tokenize("Vuln CVE-2024-3094 in lodash@4.17.20 on app.example.com")
        self.assertIn("cve-2024-3094", toks)
        self.assertIn("lodash@4.17.20", toks)
        self.assertIn("app.example.com", toks)

    def test_cyrillic(self) -> None:
        self.assertIn("уязвимость", tokenize("Критическая уязвимость"))


class TestBM25(unittest.TestCase):
    def _index(self) -> BM25Index:
        docs = [
            "memory leak in the cache module",
            "reflected xss vulnerability in login form",
            "remote code execution CVE-2024-3094 in xz backdoor",
        ]
        return BM25Index().fit([f"c{i}" for i in range(len(docs))], docs)

    def test_exact_cve_match_ranks_first(self) -> None:
        bm = self._index()
        hits = bm.search("CVE-2024-3094", top_k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], "c2")

    def test_semantic_term(self) -> None:
        bm = self._index()
        hits = bm.search("xss login", top_k=3)
        self.assertEqual(hits[0][0], "c1")

    def test_no_match_returns_empty(self) -> None:
        bm = self._index()
        self.assertEqual(bm.search("zzzznotpresent"), [])

    def test_empty_index(self) -> None:
        self.assertEqual(BM25Index().search("anything"), [])


class TestRRF(unittest.TestCase):
    def test_fusion_orders_by_combined_rank(self) -> None:
        vec = ["a", "b", "c"]
        lex = ["c", "a", "d"]
        fused = reciprocal_rank_fusion([vec, lex])
        ids = [i for i, _ in fused]
        # 'a' (ranks 1 & 2) and 'c' (ranks 3 & 1) top the list above b/d
        self.assertEqual(set(ids[:2]), {"a", "c"})
        self.assertIn("d", ids)

    def test_top_k(self) -> None:
        fused = reciprocal_rank_fusion([["a", "b", "c", "d"]], top_k=2)
        self.assertEqual(len(fused), 2)


if __name__ == "__main__":
    unittest.main()
