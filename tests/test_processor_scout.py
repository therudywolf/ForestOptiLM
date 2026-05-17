# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import unittest

from processor import (
    _parse_scout_json_payload,
    filter_indices_by_scout_scores,
    scout_relevance_score,
)


class TestScoutParsing(unittest.TestCase):
    def test_parse_wrapped_results(self) -> None:
        raw = "<results>{\"relevance_score\": 0.8, \"relevant\": true, \"no_relevant_data\": false}</results>"
        obj = _parse_scout_json_payload(raw)
        self.assertIsNotNone(obj)
        assert obj is not None
        self.assertGreaterEqual(scout_relevance_score(obj), 0.8)

    def test_filter_respects_threshold(self) -> None:
        scores = {0: 0.9, 1: 0.1, 2: 0.35}
        deep, skipped = filter_indices_by_scout_scores([0, 1, 2], scores, 0.35)
        self.assertEqual(deep, [0, 2])
        self.assertEqual(skipped, [1])


if __name__ == "__main__":
    unittest.main()
