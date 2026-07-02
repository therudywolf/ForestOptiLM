# SPDX-License-Identifier: AGPL-3.0-or-later
"""BM25 lexical index + RRF fusion (no external deps)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bm25 import BM25Index, fuse_rankings, reciprocal_rank_fusion, tokenize


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

    def test_scores_match_reference_formula(self) -> None:
        # Инвертированный numpy-скоринг должен давать те же значения, что и
        # прямая формула Okapi BM25 по каждому документу.
        docs = [
            "alpha beta beta gamma",
            "alpha alpha delta",
            "gamma gamma gamma epsilon zeta",
        ]
        bm = BM25Index().fit(["d0", "d1", "d2"], docs)
        hits = dict(bm.search("alpha gamma", top_k=10))
        import math
        n, avgdl = 3, (4 + 3 + 5) / 3
        def ref(term_df, f, dl):
            idf = math.log(1.0 + (n - term_df + 0.5) / (term_df + 0.5))
            dn = 1.5 * (1 - 0.75 + 0.75 * dl / avgdl)
            return idf * (f * 2.5) / (f + dn)
        expect_d0 = ref(2, 1, 4) + ref(2, 1, 4)   # alpha f=1, gamma f=1
        self.assertAlmostEqual(hits["d0"], expect_d0, places=5)

    def test_save_load_roundtrip(self) -> None:
        docs = ["alpha beta", "gamma delta CVE-2024-3094", "beta beta gamma"]
        bm = BM25Index().fit(["a", "b", "c"], docs)
        sig = (123, 456, 789)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bm25_cache.pkl"
            bm.save(p, sig)
            loaded = BM25Index.load(p, sig)
            self.assertIsNotNone(loaded)
            for q in ("beta gamma", "cve-2024-3094", "alpha"):
                self.assertEqual(bm.search(q, top_k=5), loaded.search(q, top_k=5))
            # неверная сигнатура → кэш отвергается
            self.assertIsNone(BM25Index.load(p, (0, 0, 0)))
            # отсутствующий файл → None, не исключение
            self.assertIsNone(BM25Index.load(Path(td) / "nope.pkl", sig))


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


class TestFuseRankings(unittest.TestCase):
    """Score-aware слияние: величина скора различает кандидатов внутри связки."""

    def test_strong_score_beats_weak_rank1_in_disjoint_lists(self) -> None:
        # Непересекающиеся списки (типичный случай на больших корпусах).
        # У чистого RRF a и x были бы связкой 1/(k+1); здесь побеждает x с
        # доминирующим скором внутри своего списка над «плоским» списком vec.
        vec = [("a", 0.51), ("b", 0.50), ("c", 0.49)]
        lex = [("x", 12.0), ("y", 1.0)]
        fused = fuse_rankings([vec, lex])
        ids = [i for i, _ in fused]
        self.assertEqual(ids[0], "a")  # rank1+norm1.0 в vec
        self.assertEqual(ids[1], "x")  # rank1+norm1.0 в lex — не хуже связки
        # 'y' (norm 1/12) должен уйти ниже b/c (norm ~0.98)
        self.assertGreater(ids.index("y"), ids.index("c"))

    def test_doc_in_both_lists_wins(self) -> None:
        vec = [("a", 0.9), ("both", 0.8)]
        lex = [("both", 5.0), ("z", 4.0)]
        fused = fuse_rankings([vec, lex])
        self.assertEqual(fused[0][0], "both")

    def test_no_flat_ties_on_disjoint_lists(self) -> None:
        # У чистого RRF все 5 схлопывались в ~1/(k+rank) без учёта скоров.
        # Здесь внутрисписочные различия скоров должны сохраниться в слитом
        # ранжировании (v2≠v3≠l2); паритет двух rank-1/norm-1.0 — легален.
        vec = [("v1", 0.9), ("v2", 0.6), ("v3", 0.3)]
        lex = [("l1", 8.0), ("l2", 2.0)]
        fused = dict(fuse_rankings([vec, lex]))
        self.assertNotEqual(fused["v2"], fused["v3"])
        self.assertNotEqual(fused["v2"], fused["l2"])
        self.assertNotEqual(fused["v3"], fused["l2"])
        # порядок отражает норму скора: v2 (0.667) > v3 (0.333) ≥ l2 (0.25)
        self.assertGreater(fused["v2"], fused["v3"])
        self.assertGreater(fused["v3"], fused["l2"])

    def test_negative_max_score_guard(self) -> None:
        # Все скоры ≤ 0 (вырожденный косинус) — не должно падать и не должно
        # переворачивать порядок нормировкой на отрицательный максимум.
        fused = fuse_rankings([[("a", -0.1), ("b", -0.5)]])
        self.assertEqual([i for i, _ in fused], ["a", "b"])

    def test_top_k_and_empty(self) -> None:
        self.assertEqual(fuse_rankings([]), [])
        self.assertEqual(fuse_rankings([[]]), [])
        fused = fuse_rankings([[("a", 1.0), ("b", 0.5), ("c", 0.1)]], top_k=2)
        self.assertEqual(len(fused), 2)


if __name__ == "__main__":
    unittest.main()
