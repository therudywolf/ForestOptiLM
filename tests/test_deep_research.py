# SPDX-License-Identifier: AGPL-3.0-or-later
"""Дипресёрч (deep_research.py): сборка grounded-промпта из веб-страниц.

Чистая функция build_research_messages — без сети/LLM."""
from __future__ import annotations

import unittest
from dataclasses import dataclass

import deep_research as dr


@dataclass
class _Page:
    title: str
    final_url: str
    text: str


class TestBuildResearchMessages(unittest.TestCase):
    def _pages(self):
        return [
            _Page("OWASP", "https://owasp.org/x", "Инъекции — критичный риск."),
            _Page("Habr", "https://habr.com/y", "Broken access control на первом месте."),
        ]

    def test_structure_and_citations(self) -> None:
        msgs = dr.build_research_messages("Что входит в OWASP Top 10?", self._pages())
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("[N]", msgs[0]["content"])            # правило цитирования
        self.assertIn("не выдумывай", msgs[0]["content"].lower())
        user = msgs[1]["content"]
        self.assertIn("[Веб-источники]", user)
        self.assertIn("[1] OWASP (https://owasp.org/x)", user)   # нумерованный источник
        self.assertIn("[2] Habr (https://habr.com/y)", user)
        self.assertIn("Инъекции — критичный риск", user)         # текст страницы попал
        self.assertIn("Что входит в OWASP Top 10?", user)        # сам вопрос
        self.assertIn("[N]", user)                                # инструкция цитировать

    def test_per_source_char_cap(self) -> None:
        pages = [_Page("T", "https://u", "x" * 10000)]
        user = dr.build_research_messages("q", pages, per_source_chars=100)[1]["content"]
        self.assertLess(user.count("x"), 200)                    # текст усечён


class TestMapReduceBuilders(unittest.TestCase):
    """W4-улучшение: map-reduce по многим источникам. Чистые билдеры."""

    def _page(self):
        return _Page("OWASP", "https://owasp.org/x", "Инъекции — критичный риск.")

    def test_map_message_has_question_and_one_source(self) -> None:
        msgs = dr.build_map_messages("Что критично?", self._page(), 3)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("одной", msgs[0]["content"].lower())          # про одну страницу
        u = msgs[1]["content"]
        self.assertIn("[Источник 3]", u)                            # номер источника сохранён
        self.assertIn("Что критично?", u)
        self.assertIn("Инъекции — критичный риск", u)

    def test_map_char_cap(self) -> None:
        big = _Page("T", "https://u", "z" * 20000)
        u = dr.build_map_messages("q", big, 1, per_source_chars=100)[1]["content"]
        self.assertLess(u.count("z"), 200)

    def test_reduce_message_numbers_notes_and_asks_citations(self) -> None:
        notes = [{"n": 1, "title": "A", "url": "https://a", "text": "факт-1"},
                 {"n": 5, "title": "B", "url": "https://b", "text": "факт-2"}]
        msgs = dr.build_reduce_messages("Вопрос?", notes)
        u = msgs[1]["content"]
        self.assertIn("[1] A (https://a)", u)
        self.assertIn("[5] B (https://b)", u)                       # оригинальные номера
        self.assertIn("факт-1", u)
        self.assertIn("[N]", u)                                      # инструкция цитировать
        self.assertIn("[N]", msgs[0]["content"])

    def test_looks_empty_note(self) -> None:
        for empty in ("", "  ", "НЕТ", "нет.", "No", "n/a", "-"):
            self.assertTrue(dr._looks_empty_note(empty), empty)
        for real in ("Инъекции критичны", "нет единого мнения, но..."):
            self.assertFalse(dr._looks_empty_note(real), real)


class TestSourcesToCitations(unittest.TestCase):
    """W5-адаптер: source-дикты дипресёрча → чиповые цитаты чата блокнота."""

    def test_maps_fields(self) -> None:
        cits = dr.sources_to_citations([
            {"n": 1, "url": "https://owasp.org/Top10/", "title": "OWASP Top 10"},
            {"n": 2, "url": "https://habr.com/ru/articles/1/", "title": ""},
        ])
        self.assertEqual(len(cits), 2)
        self.assertEqual(cits[0]["n"], 1)
        self.assertEqual(cits[0]["display"], "OWASP Top 10")
        self.assertEqual(cits[0]["source_path"], "https://owasp.org/Top10/")  # клик → браузер
        self.assertEqual(cits[0]["locator"], "🌐 веб")
        # без title показываем хост, а не пустую строку
        self.assertEqual(cits[1]["display"], "habr.com")

    def test_empty(self) -> None:
        self.assertEqual(dr.sources_to_citations([]), [])


if __name__ == "__main__":
    unittest.main()
