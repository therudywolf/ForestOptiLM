# SPDX-License-Identifier: AGPL-3.0-or-later
"""App-loaded model registry + best-effort unload-on-close (no live server)."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import processor


class TestModelRegistry(unittest.TestCase):
    def setUp(self) -> None:
        processor._APP_LOADED_MODELS.clear()

    def tearDown(self) -> None:
        processor._APP_LOADED_MODELS.clear()

    def test_note_records_real_models_only(self) -> None:
        processor.note_app_loaded_model("gemma-3-12b")
        processor.note_app_loaded_model("  nomic-embed  ")
        processor.note_app_loaded_model("")  # ignored
        processor.note_app_loaded_model("(нажмите Обновить модели)")  # placeholder ignored
        self.assertEqual(processor.app_loaded_models(), {"gemma-3-12b", "nomic-embed"})

    def test_app_loaded_models_returns_copy(self) -> None:
        processor.note_app_loaded_model("m1")
        snap = processor.app_loaded_models()
        snap.add("m2")
        self.assertEqual(processor.app_loaded_models(), {"m1"})

    def test_unload_app_models_unloads_each_and_clears(self) -> None:
        processor.note_app_loaded_model("m1")
        processor.note_app_loaded_model("m2")
        with patch("processor._try_unload_model", return_value=True) as unload:
            n = processor.unload_app_models("http://127.0.0.1:1234", "k")
        self.assertEqual(n, 2)
        self.assertEqual(unload.call_count, 2)
        self.assertEqual(processor.app_loaded_models(), set())

    def test_unload_app_models_survives_errors(self) -> None:
        processor.note_app_loaded_model("m1")
        processor.note_app_loaded_model("m2")
        with patch("processor._try_unload_model", side_effect=[RuntimeError("boom"), True]):
            n = processor.unload_app_models("http://127.0.0.1:1234", "k")
        self.assertEqual(n, 1)
        self.assertEqual(processor.app_loaded_models(), set())

    def test_unload_empty_registry_is_noop(self) -> None:
        with patch("processor._try_unload_model", return_value=True) as unload:
            n = processor.unload_app_models("http://127.0.0.1:1234", "k")
        self.assertEqual(n, 0)
        unload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
