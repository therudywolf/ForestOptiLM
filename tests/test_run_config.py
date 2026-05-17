# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import unittest

from run_config import RunConfig


class TestRunConfig(unittest.TestCase):
    def test_from_gui_defaults_scout_to_chat(self) -> None:
        cfg = RunConfig.from_gui(
            base_url="http://127.0.0.1:1234/v1",
            api_key="k",
            chat_model="chat-7b",
            vision_model=None,
            composer_model=None,
            scout_model=None,
            embedding_model="embed",
            api_mode="native",
            low_vram=True,
            workers=3,
            context_budget=8192,
            max_chunk_tokens=6000,
            max_reduce_input_tokens=24000,
            scout_mode=True,
            scout_threshold=0.4,
        )
        self.assertEqual(cfg.scout_model, "chat-7b")
        self.assertTrue(cfg.scout_mode)


if __name__ == "__main__":
    unittest.main()
