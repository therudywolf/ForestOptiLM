# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import json
import os
import unittest

from map_result_store import MapResultStore, normalize_spill_threshold


class TestMapResultStore(unittest.TestCase):
    def test_spill_builds_file_groups(self) -> None:
        os.environ["NOCTURNE_MAP_NORMALIZE_SPILL"] = "2"
        store = MapResultStore("spilljob", chunk_count=5)
        try:
            a = json.dumps({
                "file": "a.py",
                "findings": [{"evidence_refs": [{"file": "a.py", "quote": "x"}]}],
            })
            b = json.dumps({"file": "b.py", "no_relevant_data": True, "findings": []})
            store.add(0, a, parsed=json.loads(a))
            store.add(1, b, parsed=json.loads(b))
            store.add(2, a, parsed=json.loads(a))
            self.assertTrue(store._spilled)
            groups = store.build_file_groups()
            self.assertIn("a.py", groups)
            self.assertEqual(len(groups["a.py"]), 2)
            self.assertEqual(store.metrics["relevant_chunks"], 2)
        finally:
            store.cleanup()
            os.environ.pop("NOCTURNE_MAP_NORMALIZE_SPILL", None)

    def test_ram_mode_under_threshold(self) -> None:
        thr = normalize_spill_threshold()
        store = MapResultStore("ramjob", chunk_count=max(0, thr - 1))
        try:
            payload = json.dumps({
                "file": "x.txt",
                "findings": [{"evidence_refs": [{"quote": "q"}]}],
            })
            parsed = json.loads(payload)
            store.add(0, payload, parsed=parsed)
            self.assertFalse(store._spilled)
            self.assertEqual(list(store.iter_nonempty()), [payload])
        finally:
            store.cleanup()


if __name__ == "__main__":
    unittest.main()
