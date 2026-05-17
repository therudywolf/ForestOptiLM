# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import json
import unittest

from merge_hierarchy import hierarchical_merge_map_results, top_evidence_from_tree


class TestMergeHierarchy(unittest.TestCase):
    def test_rollup_two_files(self) -> None:
        a = json.dumps({
            "file": "src/a.py",
            "no_relevant_data": False,
            "findings": [{
                "severity": "high",
                "type": "bug",
                "explanation": "x",
                "evidence_refs": [{"file": "src/a.py", "chunk": "1", "quote": "bad"}],
            }],
            "recommendations": ["fix a"],
        })
        b = json.dumps({
            "file": "src/b.py",
            "no_relevant_data": True,
            "findings": [],
            "recommendations": [],
        })
        corpus_json, tree = hierarchical_merge_map_results([a, b])
        corpus = json.loads(corpus_json)
        self.assertFalse(corpus.get("no_relevant_data"))
        self.assertEqual(len(corpus.get("findings") or []), 1)
        self.assertIn("src/a.py", tree.get("files", {}))
        ev = top_evidence_from_tree(tree, limit=5)
        self.assertGreaterEqual(len(ev), 1)


if __name__ == "__main__":
    unittest.main()
