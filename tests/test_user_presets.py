# SPDX-License-Identifier: AGPL-3.0-or-later
"""Пользовательские пресеты подключения: сохранение/загрузка/upsert/delete."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import user_presets as up


class TestUserPresets(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.path = Path(self._td.name) / "connection_presets.json"

    def tearDown(self):
        self._td.cleanup()

    def _p(self, name, **kw):
        return up.ConnectionPreset(name=name, **kw)

    def test_missing_file_is_empty(self):
        self.assertEqual(up.load_presets(self.path), [])
        self.assertIsNone(up.get_preset("x", self.path))

    def test_upsert_and_load_roundtrip(self):
        up.upsert_preset(self._p("Домашний LM", base_url="http://10.0.0.2:1234",
                                 api_key="secret", llm_model="gemma"), self.path)
        got = up.get_preset("домашний lm", self.path)  # регистронезависимо
        self.assertIsNotNone(got)
        self.assertEqual(got.base_url, "http://10.0.0.2:1234")
        self.assertEqual(got.api_key, "secret")
        self.assertEqual(got.llm_model, "gemma")

    def test_upsert_overwrites_same_name(self):
        up.upsert_preset(self._p("Сервер", base_url="a"), self.path)
        up.upsert_preset(self._p("сервер", base_url="b"), self.path)  # тот же по регистру
        presets = up.load_presets(self.path)
        self.assertEqual(len(presets), 1)
        self.assertEqual(presets[0].base_url, "b")

    def test_sorted_by_name(self):
        up.upsert_preset(self._p("Яндекс"), self.path)
        up.upsert_preset(self._p("Альфа"), self.path)
        names = [p.name for p in up.load_presets(self.path)]
        self.assertEqual(names, sorted(names, key=str.lower))

    def test_delete(self):
        up.upsert_preset(self._p("A"), self.path)
        up.upsert_preset(self._p("B"), self.path)
        up.delete_preset("A", self.path)
        self.assertEqual([p.name for p in up.load_presets(self.path)], ["B"])

    def test_empty_name_rejected(self):
        with self.assertRaises(ValueError):
            up.upsert_preset(self._p("  "), self.path)

    def test_corrupt_file_ignored(self):
        self.path.write_text("{ не json", encoding="utf-8")
        self.assertEqual(up.load_presets(self.path), [])

    def test_atomic_write_no_tmp_left(self):
        up.upsert_preset(self._p("X", base_url="u"), self.path)
        self.assertFalse(self.path.with_suffix(".tmp").exists())
        self.assertTrue(self.path.exists())

    def test_from_dict_defaults(self):
        pr = up.ConnectionPreset.from_dict({"name": "N"})
        self.assertEqual(pr.api_mode, "native")
        self.assertEqual(pr.base_url, "")


if __name__ == "__main__":
    unittest.main()
