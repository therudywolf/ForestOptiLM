# SPDX-License-Identifier: AGPL-3.0-or-later
"""Connection presets: provider quick-setup and detection."""
from __future__ import annotations

import unittest

import connection_presets as cp


class TestPresets(unittest.TestCase):
    def test_known_keys_present(self) -> None:
        keys = {p.key for p in cp.all_presets()}
        self.assertTrue({"lmstudio", "lmstudio_openai", "ollama", "openai_compatible", "custom"} <= keys)

    def test_labels_unique_and_lookup(self) -> None:
        labels = cp.preset_labels()
        self.assertEqual(len(labels), len(set(labels)))
        for label in labels:
            p = cp.preset_by_label(label)
            self.assertIsNotNone(p)
            self.assertEqual(cp.get_preset(p.key).label, label)  # type: ignore[union-attr]

    def test_ollama_preset_shape(self) -> None:
        o = cp.get_preset("ollama")
        self.assertIsNotNone(o)
        self.assertEqual(o.api_mode, "openai")  # type: ignore[union-attr]
        self.assertIn("11434", o.base_url)  # type: ignore[union-attr]
        self.assertFalse(o.needs_api_key)  # type: ignore[union-attr]
        self.assertTrue(o.autofills_url)  # type: ignore[union-attr]

    def test_lmstudio_native_default(self) -> None:
        p = cp.get_preset("lmstudio")
        self.assertEqual(p.api_mode, "native")  # type: ignore[union-attr]
        self.assertIn("1234", p.base_url)  # type: ignore[union-attr]

    def test_manual_and_openai_do_not_force_url(self) -> None:
        self.assertEqual(cp.get_preset("custom").base_url, "")  # type: ignore[union-attr]
        self.assertEqual(cp.get_preset("openai_compatible").base_url, "")  # type: ignore[union-attr]
        self.assertFalse(cp.get_preset("custom").autofills_url)  # type: ignore[union-attr]


class TestDetect(unittest.TestCase):
    def test_detect_ollama_by_port(self) -> None:
        self.assertEqual(cp.detect_preset("http://127.0.0.1:11434", "openai"), "ollama")
        self.assertEqual(cp.detect_preset("http://localhost:11434/v1", "native"), "ollama")

    def test_detect_lmstudio_modes(self) -> None:
        self.assertEqual(cp.detect_preset("http://127.0.0.1:1234", "native"), "lmstudio")
        self.assertEqual(cp.detect_preset("http://127.0.0.1:1234/v1", "openai"), "lmstudio_openai")

    def test_detect_openai_compatible(self) -> None:
        self.assertEqual(cp.detect_preset("http://10.0.0.5:8000/v1", "openai"), "openai_compatible")

    def test_remote_1234_is_not_lmstudio(self) -> None:
        # vLLM/llama.cpp на :1234 НЕ на localhost → не записываем в LM Studio.
        self.assertEqual(cp.detect_preset("http://10.0.0.5:1234/v1", "openai"), "openai_compatible")

    def test_remote_11434_is_not_ollama(self) -> None:
        self.assertEqual(cp.detect_preset("http://10.0.0.5:11434", "openai"), "openai_compatible")

    def test_detect_empty_defaults_to_lmstudio(self) -> None:
        self.assertEqual(cp.detect_preset("", "native"), "lmstudio")

    def test_detect_custom_fallback(self) -> None:
        self.assertEqual(cp.detect_preset("http://10.0.0.5:9999", "native"), "custom")


class TestVisionSwapWarning(unittest.TestCase):
    def test_different_models_warns(self) -> None:
        self.assertIsNotNone(cp.vision_swap_warning("gemma-12b", "gemma-27b"))

    def test_same_model_no_warn(self) -> None:
        self.assertIsNone(cp.vision_swap_warning("gemma-12b", "gemma-12b"))
        self.assertIsNone(cp.vision_swap_warning("Gemma-12B", "gemma-12b"))  # регистр

    def test_empty_or_placeholder_vision_no_warn(self) -> None:
        self.assertIsNone(cp.vision_swap_warning("gemma-12b", ""))
        self.assertIsNone(cp.vision_swap_warning("gemma-12b", "   "))
        self.assertIsNone(cp.vision_swap_warning("gemma", "(нажмите Обновить модели)"))

    def test_workers_hint_mentions_tradeoff(self) -> None:
        self.assertIn("2×", cp.WORKERS_HINT)
        self.assertIn("vision", cp.WORKERS_HINT.lower())


if __name__ == "__main__":
    unittest.main()
