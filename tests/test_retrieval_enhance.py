# SPDX-License-Identifier: AGPL-3.0-or-later
"""Усиление retrieval (по мотивам qmd/LLM-Wiki): expansion, merge, listwise rerank."""
from __future__ import annotations

import unittest

import retrieval_enhance as re_enh


class _Hit:
    def __init__(self, chunk_id: str, score: float, text: str = "") -> None:
        self.chunk_id = chunk_id
        self.score = score
        self.text = text


class TestExpansion(unittest.TestCase):
    def test_parse_keeps_original_first_and_dedups(self) -> None:
        out = re_enh.parse_expansions('["на каких ВМ", "серверы подсистемы"]', "где развёрнуто?")
        self.assertEqual(out[0], "где развёрнуто?")
        self.assertEqual(len(out), 3)

    def test_parse_caps_at_n(self) -> None:
        out = re_enh.parse_expansions('["a","b","c","d"]', "q", n=2)
        self.assertEqual(len(out), 3)  # original + 2

    def test_parse_drops_duplicate_of_original(self) -> None:
        out = re_enh.parse_expansions('["Q", "другой"]', "q")  # "Q" == "q" без регистра
        self.assertEqual(out, ["q", "другой"])

    def test_parse_garbage_returns_only_original(self) -> None:
        self.assertEqual(re_enh.parse_expansions("я не понял задачу", "q"), ["q"])
        self.assertEqual(re_enh.parse_expansions("", "q"), ["q"])

    def test_parse_tolerates_prose_wrapper(self) -> None:
        out = re_enh.parse_expansions('Вот варианты: ["x", "y"]. Готово.', "q")
        self.assertEqual(out, ["q", "x", "y"])

    def test_build_messages_shape(self) -> None:
        msgs = re_enh.build_expansion_messages("вопрос?")
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("вопрос?", msgs[1]["content"])


class TestMergeHits(unittest.TestCase):
    def test_dedup_by_chunk_id_keeps_max_score(self) -> None:
        a = [_Hit("c1", 0.3), _Hit("c2", 0.9)]
        b = [_Hit("c1", 0.7), _Hit("c3", 0.5)]
        merged = re_enh.merge_hits([a, b])
        by = {h.chunk_id: h.score for h in merged}
        self.assertEqual(by["c1"], 0.7)  # лучший скор сохранён
        self.assertEqual(set(by), {"c1", "c2", "c3"})

    def test_sorted_desc_and_capped(self) -> None:
        hits = [_Hit(f"c{i}", i / 10) for i in range(10)]
        merged = re_enh.merge_hits([hits], cap=3)
        self.assertEqual([h.chunk_id for h in merged], ["c9", "c8", "c7"])


class TestRerank(unittest.TestCase):
    def test_parse_order_one_based_to_zero(self) -> None:
        self.assertEqual(re_enh.parse_rerank_order("[3,1,2]", 3), [2, 0, 1])

    def test_parse_order_filters_out_of_range_and_dups(self) -> None:
        self.assertEqual(re_enh.parse_rerank_order("[3, 9, 3, 1]", 3), [2, 0])

    def test_parse_order_garbage_empty(self) -> None:
        self.assertEqual(re_enh.parse_rerank_order("не знаю", 5), [])

    def test_apply_reorders_and_fills_tail(self) -> None:
        cands = [_Hit("a", 0), _Hit("b", 0), _Hit("c", 0)]
        out = re_enh.apply_rerank(cands, [2, 0], top_k=3)  # c, a, затем хвостом b
        self.assertEqual([h.chunk_id for h in out], ["c", "a", "b"])

    def test_apply_empty_order_is_rrf_fallback(self) -> None:
        cands = [_Hit("a", 0), _Hit("b", 0)]
        out = re_enh.apply_rerank(cands, [], top_k=2)
        self.assertEqual([h.chunk_id for h in out], ["a", "b"])  # исходный порядок

    def test_apply_truncates_to_top_k(self) -> None:
        cands = [_Hit(str(i), 0) for i in range(5)]
        self.assertEqual(len(re_enh.apply_rerank(cands, [4, 3], top_k=2)), 2)

    def test_build_rerank_messages_numbers_candidates(self) -> None:
        msgs = re_enh.build_rerank_messages("q", [_Hit("a", 0, "первый"), _Hit("b", 0, "второй")])
        self.assertIn("[1] первый", msgs[1]["content"])
        self.assertIn("[2] второй", msgs[1]["content"])


if __name__ == "__main__":
    unittest.main()
