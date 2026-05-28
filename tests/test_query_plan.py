# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic query understanding (QueryPlan) — no LLM."""
from __future__ import annotations

import unittest

from query_plan import build_query_plan, output_style_directive


class TestLanguage(unittest.TestCase):
    def test_ru(self) -> None:
        self.assertEqual(build_query_plan("найди критичные уязвимости").language, "ru")
        self.assertIn("русск", build_query_plan("найди уязвимости").language_hint())

    def test_en(self) -> None:
        p = build_query_plan("find critical vulnerabilities")
        self.assertEqual(p.language, "en")
        self.assertEqual(p.language_hint(), "")


class TestIntent(unittest.TestCase):
    def test_compare(self) -> None:
        p = build_query_plan("сравни два скана, что нового появилось")
        self.assertEqual(p.intent, "compare")
        self.assertEqual(p.output_style, "diff_table")

    def test_prioritize(self) -> None:
        p = build_query_plan("что чинить в первую очередь, самые критичные")
        self.assertEqual(p.intent, "prioritize")
        self.assertEqual(p.output_style, "ranked_list")

    def test_count(self) -> None:
        p = build_query_plan("сколько уязвимостей по severity")
        self.assertEqual(p.intent, "count")
        self.assertEqual(p.output_style, "stats")

    def test_default_analyze(self) -> None:
        p = build_query_plan("опиши состояние безопасности")
        self.assertEqual(p.intent, "analyze")
        self.assertEqual(p.output_style, "report")


class TestExtraction(unittest.TestCase):
    def test_severity_filter(self) -> None:
        p = build_query_plan("покажи только критичные и high находки")
        self.assertIn("critical", p.severity_filter)
        self.assertIn("high", p.severity_filter)

    def test_cve_cwe(self) -> None:
        p = build_query_plan("есть ли CVE-2024-3094 и проблемы CWE-89?")
        self.assertIn("CVE-2024-3094", p.cve_ids)
        self.assertIn("CWE-89", p.cwe_ids)

    def test_group_by_host(self) -> None:
        self.assertEqual(build_query_plan("сгруппируй по хостам").group_by, "asset")
        self.assertEqual(build_query_plan("group by cwe please").group_by, "cwe")

    def test_key_terms_quoted_and_pkg(self) -> None:
        p = build_query_plan('найди уязвимости в "log4j" и пакете lodash@4.17.20')
        terms_l = [t.lower() for t in p.key_terms]
        self.assertIn("log4j", terms_l)
        self.assertTrue(any("lodash" in t for t in terms_l))

    def test_hosts(self) -> None:
        p = build_query_plan("уязвимости на app.example.com и 10.0.0.5")
        self.assertIn("app.example.com", p.hosts)
        self.assertIn("10.0.0.5", p.hosts)


class TestDedupAndOutput(unittest.TestCase):
    def test_dedup_keys_cve(self) -> None:
        p = build_query_plan("приоритизируй уязвимость CVE-2024-3094")
        self.assertEqual(p.dedup_keys(), ("cve", "asset"))

    def test_dedup_keys_group_cwe(self) -> None:
        p = build_query_plan("сгруппируй по cwe")
        self.assertEqual(p.dedup_keys(), ("cwe",))

    def test_dedup_keys_default(self) -> None:
        p = build_query_plan("опиши находки")
        self.assertEqual(p.dedup_keys(), ("severity", "type", "explanation"))

    def test_output_directive(self) -> None:
        p = build_query_plan("сравни сканы по хостам")
        d = output_style_directive(p)
        self.assertTrue(d)
        self.assertIn("asset", d)


if __name__ == "__main__":
    unittest.main()
