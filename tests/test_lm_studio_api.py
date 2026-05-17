# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import unittest

from lm_studio_api import V1_MODELS, is_lm_studio_api_token, v1_url


class TestLmStudioApi(unittest.TestCase):
    def test_v1_url(self) -> None:
        self.assertEqual(v1_url("http://127.0.0.1:1234", V1_MODELS), "http://127.0.0.1:1234/api/v1/models")

    def test_token_detect(self) -> None:
        self.assertTrue(is_lm_studio_api_token("sk-lm-abc:def"))
        self.assertFalse(is_lm_studio_api_token("forest"))


if __name__ == "__main__":
    unittest.main()
