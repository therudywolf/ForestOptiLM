# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic, domain-neutral query understanding (QueryPlan) — no LLM."""
from __future__ import annotations

import unittest

from query_plan import build_query_plan, output_style_directive


class TestLanguage(unittest.TestCase):
    def test_ru(self) -> None:
        self.assertEqual(build_query_plan("собери все договоры").language, "ru")
        self.assertIn("русск", build_query_plan("собери данные").language_hint())

    def test_en(self) -> None:
        p = build_query_plan("extract all contracts")
        self.assertEqual(p.language, "en")
        self.assertEqual(p.language_hint(), "")


class TestIntent(unittest.TestCase):
    def test_extract(self) -> None:
        p = build_query_plan("извлеки все упоминания контрагентов и суммы")
        self.assertEqual(p.intent, "extract")
        self.assertEqual(p.output_style, "table")

    def test_compare(self) -> None:
        p = build_query_plan("сравни две версии договора, что изменилось")
        self.assertEqual(p.intent, "compare")
        self.assertEqual(p.output_style, "comparison")

    def test_count(self) -> None:
        p = build_query_plan("сколько обращений по каждому типу")
        self.assertEqual(p.intent, "count")
        self.assertEqual(p.output_style, "stats")

    def test_classify(self) -> None:
        p = build_query_plan("классифицируй документы по типам")
        self.assertEqual(p.intent, "classify")

    def test_default_analyze(self) -> None:
        p = build_query_plan("опиши общее состояние дел в данных")
        self.assertEqual(p.intent, "analyze")
        self.assertEqual(p.output_style, "report")


class TestExtractionSchema(unittest.TestCase):
    def test_fields_and_directive(self) -> None:
        p = build_query_plan("извлеки контрагентов и суммы")
        self.assertIn("item", p.extraction_fields)
        self.assertIn("source", p.extraction_fields)
        self.assertIn("item", p.extraction_directive)

    def test_explain_schema(self) -> None:
        p = build_query_plan("объясни почему упала выручка")
        self.assertEqual(p.intent, "explain")
        self.assertIn("claim", p.extraction_fields)


class TestEntitiesAndTerms(unittest.TestCase):
    def test_generic_entities(self) -> None:
        p = build_query_plan(
            "выгрузи документ INV-2024-7 на email a@b.com за дату 2024-05-01"
        )
        self.assertIn("a@b.com", p.entities.get("email", []))
        self.assertIn("INV-2024-7", p.entities.get("id", []))
        self.assertIn("2024-05-01", p.entities.get("date", []))

    def test_key_terms_quoted_and_token(self) -> None:
        p = build_query_plan('найди упоминания "Газпром" и файла report_2024.csv')
        terms_l = [t.lower() for t in p.key_terms]
        self.assertIn("газпром", terms_l)
        self.assertTrue(any("report_2024" in t for t in terms_l))


class TestGroupByNeutral(unittest.TestCase):
    def test_group_by_source(self) -> None:
        p = build_query_plan("сгруппируй по файлам")
        self.assertIsNotNone(p.group_by)
        self.assertEqual(p.facet_axis(), "source")

    def test_group_by_category_en(self) -> None:
        p = build_query_plan("group by type")
        self.assertEqual(p.facet_axis(), "category")

    def test_dedup_keys_axis(self) -> None:
        p = build_query_plan("сгруппируй по файлам")
        self.assertEqual(p.dedup_keys(), ("source", "item"))

    def test_dedup_keys_default(self) -> None:
        p = build_query_plan("опиши элементы")
        self.assertEqual(p.dedup_keys(), ("category", "item"))


class TestOutput(unittest.TestCase):
    def test_directive_comparison(self) -> None:
        p = build_query_plan("сравни два набора")
        self.assertTrue(output_style_directive(p))

    def test_directive_report_empty(self) -> None:
        p = build_query_plan("опиши данные")
        self.assertEqual(output_style_directive(p), "")


if __name__ == "__main__":
    unittest.main()
