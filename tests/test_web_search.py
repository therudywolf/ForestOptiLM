# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keyless веб-поиск (web_search.py): парсер DDG-выдачи и декодирование URL.

Только чистые функции — без сети (сетевой search_ddg тестируется вручную/e2e)."""
from __future__ import annotations

import unittest

import web_search as ws

# Урезанная, но структурно верная выдача DuckDuckGo HTML (2 результата + дубль).
_FIXTURE = """
<div class="result results_links web-result">
  <div class="result__body">
    <h2 class="result__title">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fowasp.org%2Ftop10%2F&rut=x">
        OWASP Top 10</a>
    </h2>
    <a class="result__snippet" href="#">Критичные риски веб-приложений.</a>
  </div>
</div>
<div class="result web-result">
  <div class="result__body">
    <h2 class="result__title">
      <a class="result__a" href="https://example.com/direct">Прямая ссылка</a>
    </h2>
    <a class="result__snippet">Пример сниппета.</a>
  </div>
</div>
<div class="result web-result">
  <div class="result__body">
    <h2 class="result__title">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fowasp.org%2Ftop10%2F">Дубль</a>
    </h2>
  </div>
</div>
"""


class TestDdgParse(unittest.TestCase):
    def test_decode_ddg_redirect(self) -> None:
        self.assertEqual(
            ws._decode_ddg_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fowasp.org%2Fx&rut=y"),
            "https://owasp.org/x")
        # прямые ссылки — без изменений
        self.assertEqual(ws._decode_ddg_url("https://example.com/p"), "https://example.com/p")
        self.assertEqual(ws._decode_ddg_url(""), "")

    def test_parse_results_and_dedup(self) -> None:
        res = ws.parse_ddg_html(_FIXTURE)
        self.assertEqual(len(res), 2)                       # дубль по URL схлопнут
        self.assertEqual(res[0].url, "https://owasp.org/top10/")   # редирект раскодирован
        self.assertEqual(res[0].title, "OWASP Top 10")
        self.assertIn("Критичные риски", res[0].snippet)
        self.assertEqual(res[1].url, "https://example.com/direct")  # прямая ссылка
        self.assertEqual(res[1].snippet, "Пример сниппета.")

    def test_max_results_cap(self) -> None:
        self.assertEqual(len(ws.parse_ddg_html(_FIXTURE, max_results=1)), 1)

    def test_empty_query_returns_empty(self) -> None:
        self.assertEqual(ws.search("   "), [])              # не ходим в сеть на пустой запрос


if __name__ == "__main__":
    unittest.main()
