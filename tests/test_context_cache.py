# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lmstudio_config import load_ui_runtime_state, save_ui_runtime_state


class TestContextCache(unittest.TestCase):
    def test_roundtrip_model_context_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ui_path = Path(tmp) / "ui_runtime.json"
            with patch("lmstudio_config._runtime_ui_path", return_value=ui_path):
                save_ui_runtime_state({
                    "selected_model": "m1",
                    "model_context_cache": {"m1": 16384, "m2": 8192},
                })
                loaded = load_ui_runtime_state()
                mcc = loaded.get("model_context_cache")
                self.assertIsInstance(mcc, dict)
                assert isinstance(mcc, dict)
                self.assertEqual(mcc.get("m1"), 16384)


if __name__ == "__main__":
    unittest.main()
