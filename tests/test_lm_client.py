# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import unittest

from lm_client import classify_http_400, extract_model_ids


class TestLmClient(unittest.TestCase):
    def test_extract_model_ids_openai_shape(self) -> None:
        data = {"data": [{"id": "m1"}, {"id": "m2"}]}
        self.assertEqual(extract_model_ids(data), ["m1", "m2"])

    def test_classify_context(self) -> None:
        self.assertEqual(classify_http_400("n_ctx exceeded"), "context_limit")

    def test_classify_reasoning(self) -> None:
        self.assertEqual(classify_http_400("invalid reasoning param"), "unsupported_option")


if __name__ == "__main__":
    unittest.main()
