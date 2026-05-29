# SPDX-License-Identifier: AGPL-3.0-or-later
"""Domain-neutral run memory & cross-run diff (no security semantics)."""
from __future__ import annotations

import os
import tempfile
import unittest

import run_memory as rm


def _r(item: str, category: str = "note", source: str = "a.txt", entities: list[str] | None = None) -> dict:
    return {
        "_facets": {
            "category": category,
            "item": item,
            "source": source,
            "level": "",
            "entities": entities or [],
            "has_source": True,
        }
    }


class TestRunMemory(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        os.environ["NOCTURNE_CACHE_DIR"] = self._tmp.name
        rm.reset_connection()

    def tearDown(self) -> None:
        rm.reset_connection()
        os.environ.pop("NOCTURNE_CACHE_DIR", None)

    def test_key_prefers_entities(self) -> None:
        k1 = rm.canonical_item_key({"entities": ["INV-1"], "source": "s", "item": "x"})
        k2 = rm.canonical_item_key({"entities": ["INV-1"], "source": "s", "item": "DIFFERENT"})
        k3 = rm.canonical_item_key({"entities": ["INV-2"], "source": "s", "item": "x"})
        self.assertEqual(k1, k2)        # same entity+source → same item
        self.assertNotEqual(k1, k3)

    def test_first_run_no_diff(self) -> None:
        block = rm.record_and_diff(
            job_id="j", source_path="/c", query="q", records=[_r("alpha"), _r("beta")],
        )
        self.assertEqual(block, "")
        self.assertEqual(len(rm.list_runs("/c")), 1)

    def test_diff_added_removed(self) -> None:
        r1 = rm.record_run(job_id="j1", source_path="/c", query="q",
                           records=[_r("alpha"), _r("beta")])
        r2 = rm.record_run(job_id="j2", source_path="/c", query="q",
                           records=[_r("beta"), _r("gamma")])
        assert r1 and r2
        d = rm.diff_runs(r1, r2)
        self.assertEqual({x["item"] for x in d["added"]}, {"gamma"})
        self.assertEqual({x["item"] for x in d["removed"]}, {"alpha"})
        self.assertEqual({x["item"] for x in d["unchanged"]}, {"beta"})

    def test_record_and_diff_second_run(self) -> None:
        rm.record_and_diff(job_id="j1", source_path="/c", query="q", records=[_r("alpha")])
        block = rm.record_and_diff(job_id="j2", source_path="/c", query="q",
                                   records=[_r("omega")], language="ru")
        self.assertIn("Изменения с прошлого прогона", block)
        self.assertIn("omega", block)

    def test_disabled(self) -> None:
        os.environ["NOCTURNE_RUN_MEMORY"] = "0"
        try:
            self.assertEqual(
                rm.record_and_diff(job_id="j", source_path="/c", query="q", records=[_r("a")]),
                "",
            )
        finally:
            os.environ.pop("NOCTURNE_RUN_MEMORY", None)


if __name__ == "__main__":
    unittest.main()
