# SPDX-License-Identifier: AGPL-3.0-or-later
"""Глубокий анализ (map-reduce над следом сущности): чистые функции."""
from __future__ import annotations

import unittest

import deep_analysis as da


class _Hit:
    def __init__(self, cid, sp, text, ci=None, score=1.0):
        self.chunk_id = cid
        self.source_path = sp
        self.text = text
        self.metadata = {"chunk_index": ci} if ci is not None else {}
        self.score = score


def _meta(cid, sp, ci, text=""):
    return {"chunk_id": cid, "source_path": sp, "text": text or f"t{cid}",
            "metadata": {"chunk_index": ci}}


class TestClassifier(unittest.TestCase):
    def test_analytical_true(self):
        for q in ["составь психологический портрет Иванова",
                  "проанализируй переписку по проекту",
                  "дай сводку по всем инцидентам",
                  "как устроен модуль оплаты",
                  "give me an overview of the incidents",
                  "все упоминания системы X"]:
            self.assertTrue(da.is_analytical_question(q), q)

    def test_enumeration_is_analytical(self):
        # перечни лучше собирает deep (широкий сбор) — авто-детект их ловит
        for q in ["перечисли все интеграции модуля",
                  "какой список внешних систем связан с аналитикой",
                  "какие все версии ПО нужны"]:
            self.assertTrue(da.is_analytical_question(q), q)

    def test_factoid_false(self):
        for q in ["на каком сервере крутится API",
                  "сколько стоит лицензия",
                  "когда был последний релиз",
                  "какой айпи у прод-сервера"]:
            self.assertFalse(da.is_analytical_question(q), q)

    def test_empty(self):
        self.assertFalse(da.is_analytical_question(""))
        self.assertFalse(da.is_analytical_question("   "))


class TestNeighbors(unittest.TestCase):
    def _corpus(self):
        # один источник a.txt с чанками 0..5, второй b.txt с 0..2
        meta = [_meta(f"a{i}", "a.txt", i) for i in range(6)]
        meta += [_meta(f"b{i}", "b.txt", i) for i in range(3)]
        return meta

    def test_adds_adjacent_same_source(self):
        meta = self._corpus()
        hits = [_Hit("a2", "a.txt", "hit", ci=2)]
        units = da.expand_with_neighbors(hits, meta, radius=1, cap=50)
        ids = [u.chunk_id for u in units]
        self.assertEqual(ids[0], "a2")           # хит первым
        self.assertIn("a1", ids)                 # сосед слева
        self.assertIn("a3", ids)                 # сосед справа
        self.assertNotIn("b0", ids)              # чужой источник не тянем

    def test_radius_zero_is_dedup_only(self):
        meta = self._corpus()
        hits = [_Hit("a2", "a.txt", "x", ci=2), _Hit("a2", "a.txt", "x", ci=2)]
        units = da.expand_with_neighbors(hits, meta, radius=0, cap=50)
        self.assertEqual([u.chunk_id for u in units], ["a2"])

    def test_cap_respected(self):
        meta = self._corpus()
        hits = [_Hit(f"a{i}", "a.txt", "x", ci=i) for i in range(6)]
        units = da.expand_with_neighbors(hits, meta, radius=2, cap=3)
        self.assertEqual(len(units), 3)

    def test_boundary_no_negative_index(self):
        meta = self._corpus()
        hits = [_Hit("a0", "a.txt", "x", ci=0)]
        units = da.expand_with_neighbors(hits, meta, radius=1, cap=50)
        ids = [u.chunk_id for u in units]
        self.assertEqual(ids[0], "a0")
        self.assertIn("a1", ids)                 # только правый сосед существует
        self.assertNotIn("a-1", ids)


class TestBatching(unittest.TestCase):
    def test_global_numbering_and_budget(self):
        units = [da.Unit(f"c{i}", "s", "слово " * 10, chunk_index=i) for i in range(5)]
        # ~ каждый ~ (60 симв /4)=15 ток; бюджет 30 → ~2 юнита/пачку
        batches = da.batch_units(units, max_batch_tokens=30)
        # сквозная нумерация 1..5, без повторов и пропусков
        nums = [n for b in batches for n, _ in b]
        self.assertEqual(nums, [1, 2, 3, 4, 5])
        self.assertGreater(len(batches), 1)

    def test_single_giant_unit_gets_own_batch(self):
        units = [da.Unit("c1", "s", "x" * 5000, chunk_index=0),
                 da.Unit("c2", "s", "y", chunk_index=1)]
        batches = da.batch_units(units, max_batch_tokens=50)
        self.assertEqual(len(batches), 2)


class TestPrompts(unittest.TestCase):
    def test_map_messages_number_and_goal(self):
        numbered = [(1, da.Unit("c1", "s", "текст один", chunk_index=0)),
                    (2, da.Unit("c2", "s", "текст два", chunk_index=1, is_neighbor=True))]
        msgs = da.build_map_messages("портрет X", numbered)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("портрет X", msgs[1]["content"])
        self.assertIn("[1]", msgs[1]["content"])
        self.assertIn("(контекст)", msgs[1]["content"])  # сосед помечен

    def test_parse_map_empty_variants(self):
        self.assertEqual(da.parse_map_result("НЕТ"), "")
        self.assertEqual(da.parse_map_result("  нет  "), "")
        self.assertEqual(da.parse_map_result("NONE"), "")
        self.assertEqual(da.parse_map_result(""), "")
        self.assertEqual(da.parse_map_result("Факт: [1] важное"), "Факт: [1] важное")

    def test_reduce_messages_include_schema_and_summaries(self):
        msgs = da.build_reduce_messages("цель", ["набл 1", "набл 2"], schema="домен чатов")
        self.assertIn("домен чатов", msgs[0]["content"])
        self.assertIn("набл 1", msgs[1]["content"])
        self.assertIn("цель", msgs[1]["content"])

    def test_reduce_prompt_warns_against_name_conflation(self):
        # анти-склейка имён (ник ≠ ФИО)
        sysmsg = da.build_reduce_messages("цель", ["x"])[0]["content"].lower()
        self.assertIn("ник", sysmsg)

    def test_merge_messages_preserve_citations_hint(self):
        msgs = da.build_merge_messages("цель", ["факт [1]", "факт [2]"])
        self.assertIn("[N]", msgs[0]["content"])       # инструкция сохранять ссылки
        self.assertIn("факт [1]", msgs[1]["content"])
        self.assertIn("цель", msgs[1]["content"])


class TestGroupList(unittest.TestCase):
    def test_groups_by_size(self):
        self.assertEqual(da.group_list([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])

    def test_exact_multiple(self):
        self.assertEqual(da.group_list([1, 2, 3, 4], 2), [[1, 2], [3, 4]])

    def test_empty_and_size_guard(self):
        self.assertEqual(da.group_list([], 3), [])
        self.assertEqual(da.group_list([1, 2], 0), [[1], [2]])  # size≥1


if __name__ == "__main__":
    unittest.main()
