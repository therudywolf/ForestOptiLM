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


if __name__ == "__main__":
    unittest.main()
