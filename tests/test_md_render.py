# SPDX-License-Identifier: AGPL-3.0-or-later
"""md_render.to_plain — markdown → чистый читаемый plain-text для чат-пузыря."""
from __future__ import annotations

import unittest

from md_render import to_plain


class TestToPlain(unittest.TestCase):
    def test_strips_bold_and_code(self) -> None:
        self.assertEqual(to_plain("Вот **важный** факт и `код`."), "Вот важный факт и код.")
        self.assertEqual(to_plain("__тоже жирный__"), "тоже жирный")

    def test_headers_become_clean_lines(self) -> None:
        out = to_plain("### Заголовок\nтекст")
        self.assertNotIn("#", out)
        self.assertIn("Заголовок", out)
        self.assertTrue(out.startswith("Заголовок"))

    def test_bullets_normalized(self) -> None:
        out = to_plain("* один\n- два\n+ три")
        self.assertEqual(out.count("•"), 3)
        self.assertNotIn("* один", out)
        for w in ("один", "два", "три"):
            self.assertIn(w, out)

    def test_numbered_list_kept_as_bullets(self) -> None:
        out = to_plain("1. первый\n2. второй")
        self.assertIn("первый", out)
        self.assertIn("второй", out)

    def test_citations_preserved(self) -> None:
        self.assertIn("[1][3]", to_plain("Факт **X** [1][3]."))

    def test_hr_removed(self) -> None:
        self.assertEqual(to_plain("текст\n---\nещё"), "текст\nещё")

    def test_header_gets_blank_line_before(self) -> None:
        out = to_plain("Абзац текста.\n## Раздел\nсодержимое")
        self.assertIn("Абзац текста.\n\nРаздел", out)

    def test_empty_and_none_safe(self) -> None:
        self.assertEqual(to_plain(""), "")
        self.assertEqual(to_plain(None), "")  # type: ignore[arg-type]

    def test_collapses_excess_blank_lines(self) -> None:
        self.assertNotIn("\n\n\n", to_plain("a\n\n\n\n\nb"))

    def test_plain_text_unchanged(self) -> None:
        s = "Просто предложение без разметки."
        self.assertEqual(to_plain(s), s)

    def test_multiplication_star_not_eaten(self) -> None:
        # одиночная * между числами (2*2) не должна ломаться как курсив
        self.assertIn("2*2", to_plain("Пример 2*2 в тексте"))


if __name__ == "__main__":
    unittest.main()
