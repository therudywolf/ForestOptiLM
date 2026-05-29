# SPDX-License-Identifier: AGPL-3.0-or-later
"""Domain-neutral deterministic aggregation (no LLM, no security semantics)."""
from __future__ import annotations

import json
import unittest

from aggregate import (
    build_category_block,
    categorize,
    extract_facets,
    iter_records_from_map,
)


def _map(source: str, items: list[dict]) -> str:
    # Uses the neutral "items" key; legacy "findings" is also supported.
    return json.dumps(
        {"source": source, "no_relevant_data": False, "items": items},
        ensure_ascii=False,
    )


class TestFacets(unittest.TestCase):
    def test_neutral_record(self) -> None:
        f = extract_facets({
            "item": "Договор №INV-2024-7",
            "category": "contract",
            "value": "1 000 000",
            "source": "docs/a.pdf",
        })
        self.assertEqual(f["category"], "contract")
        self.assertEqual(f["source"], "docs/a.pdf")
        self.assertIn("INV-2024-7", f["entities"])
        self.assertTrue(f["has_source"])

    def test_query_adaptive_fields(self) -> None:
        # query-adaptive MAP puts task-specific data under "fields"
        f = extract_facets({
            "type": "row",
            "explanation": "счёт",
            "fields": {"category": "invoice", "value": "100", "item": "Acme INV-7"},
        })
        self.assertEqual(f["category"], "invoice")
        self.assertEqual(f["value"], "100")
        self.assertEqual(f["item"], "acme inv-7")
        self.assertIn("INV-7", f["entities"])

    def test_legacy_finding_record(self) -> None:
        # current MAP schema still works
        f = extract_facets({
            "type": "issue",
            "explanation": "Something about 2024-05-01 deadline",
            "evidence_refs": [{"file": "x.log", "quote": "2024-05-01"}],
        })
        self.assertEqual(f["category"], "issue")
        self.assertEqual(f["source"], "x.log")
        self.assertIn("2024-05-01", f["entities"])


class TestCategorize(unittest.TestCase):
    def _records(self) -> list[dict]:
        m1 = _map("a.pdf", [
            {"item": "alpha", "category": "contract"},
            {"item": "beta", "category": "invoice"},
        ])
        m2 = _map("b.pdf", [
            {"item": "alpha", "category": "contract"},  # dup by (category,item)
            {"item": "gamma", "category": "invoice"},
        ])
        return list(iter_records_from_map([m1, m2]))

    def test_dedup_by_category_item(self) -> None:
        recs = self._records()
        summ = categorize(recs, dedup_keys=("category", "item"))
        self.assertEqual(summ.unique, 3)  # alpha/contract dedup'd
        cats = dict(summ.by_category)
        self.assertEqual(cats.get("contract"), 1)
        self.assertEqual(cats.get("invoice"), 2)

    def test_group_by_source_keeps_both(self) -> None:
        recs = self._records()
        summ = categorize(recs, dedup_keys=("source", "item"), axis="source")
        self.assertEqual(summ.unique, 4)  # same item on 2 sources kept
        srcs = dict(summ.by_axis)
        self.assertEqual(srcs.get("a.pdf"), 2)
        self.assertEqual(srcs.get("b.pdf"), 2)

    def test_block_has_numbers(self) -> None:
        block = build_category_block(
            [_map("h", [{"item": "x", "category": "note"}])],
            dedup_keys=("category", "item"), language="ru",
        )
        self.assertIn("категориям", block.lower())
        self.assertIn("note: 1", block)


class TestEmpty(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(build_category_block([]), "")
        nrd = json.dumps({"no_relevant_data": True, "items": []})
        self.assertEqual(build_category_block([nrd]), "")


if __name__ == "__main__":
    unittest.main()
