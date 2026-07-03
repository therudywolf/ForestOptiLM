# SPDX-License-Identifier: AGPL-3.0-or-later
"""Веб-fetch (web_fetch.py): извлечение основного текста и валидация URL.

Чистые функции — без сети (сетевой fetch проверяется e2e)."""
from __future__ import annotations

import unittest

import web_fetch as wf

_PAGE = """
<html><head><title>Заголовок статьи</title></head>
<body>
  <header>Шапка сайта</header>
  <nav>меню навигации</nav>
  <script>var x = 1;</script>
  <style>.a{color:red}</style>
  <main>
    <article>
      <h1>Основной заголовок</h1>
      <p>Первый содержательный абзац про инъекции.</p>
      <p>Второй абзац с деталями.</p>
      <aside>реклама сбоку</aside>
    </article>
  </main>
  <footer>Подвал с копирайтом</footer>
</body></html>
"""


class TestExtractMainText(unittest.TestCase):
    def test_extracts_article_strips_noise(self) -> None:
        title, text = wf.extract_main_text(_PAGE)
        self.assertEqual(title, "Заголовок статьи")
        self.assertIn("содержательный абзац про инъекции", text)
        self.assertIn("Второй абзац", text)
        # шум удалён
        for noise in ("меню навигации", "Шапка сайта", "Подвал", "var x", "color:red", "реклама"):
            self.assertNotIn(noise, text)

    def test_empty_html(self) -> None:
        title, text = wf.extract_main_text("")
        self.assertEqual(title, "")
        self.assertEqual(text, "")

    def test_fallback_to_body_without_article(self) -> None:
        title, text = wf.extract_main_text(
            "<html><title>T</title><body><p>Только body-контент.</p></body></html>")
        self.assertIn("body-контент", text)


class TestFetchGuards(unittest.TestCase):
    def test_fetch_rejects_non_http(self) -> None:
        with self.assertRaises(wf.FetchError):
            wf.fetch("ftp://example.com/x")
        with self.assertRaises(wf.FetchError):
            wf.fetch("not a url")

    def test_fetch_safe_returns_none_on_bad_url(self) -> None:
        self.assertIsNone(wf.fetch_safe("not a url"))


if __name__ == "__main__":
    unittest.main()
