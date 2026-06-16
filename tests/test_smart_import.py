# SPDX-License-Identifier: AGPL-3.0-or-later
"""Smart import: Telegram HTML export → clean per-message dialogue."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import smart_import as si

# Минимальная, но структурно точная выгрузка Telegram Desktop:
# service-разделитель, обычное сообщение, сгруппированное (joined, без from_name),
# и медиа-сообщение (стикер) без текста.
TELEGRAM_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>
<div class="page_wrap"><div class="page_body"><div class="history">
  <div class="message service"><div class="body details">3 October 2025</div></div>
  <div class="message default clearfix" id="message1">
    <div class="pull_left userpic_wrap"><div class="userpic userpic7"><div class="initials">SM</div></div></div>
    <div class="body">
      <div class="pull_right date details" title="03.10.2025 12:34:18 UTC+03:00">12:34</div>
      <div class="from_name">Sergey Medvedev</div>
      <div class="text">Привет, коллеги!<br>Есть вопрос по статусам</div>
    </div>
  </div>
  <div class="message default clearfix joined" id="message2">
    <div class="body">
      <div class="pull_right date details" title="03.10.2025 12:35:00 UTC+03:00">12:35</div>
      <div class="text">Второе сообщение подряд</div>
    </div>
  </div>
  <div class="message default clearfix" id="message3">
    <div class="body">
      <div class="pull_right date details" title="03.10.2025 12:36:00 UTC+03:00">12:36</div>
      <div class="from_name">Anna</div>
      <div class="media_wrap clearfix"><div class="media clearfix pull_left media_photo">
        <div class="body"><div class="title bold">Sticker</div><div class="status details">512x512</div></div>
      </div></div>
    </div>
  </div>
</div></div></div>
</body></html>"""

PLAIN_HTML = "<html><head><title>T</title></head><body><h1>Заголовок</h1><p>Обычная страница</p></body></html>"


def _write(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.write_text(content, encoding="utf-8")
    return p


class TestTelegramImporter(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_detect_telegram(self) -> None:
        p = _write(self.tmp, "messages.html", TELEGRAM_HTML)
        self.assertEqual(si.detect_format(p), "telegram_html")

    def test_detect_rejects_plain_html(self) -> None:
        p = _write(self.tmp, "page.html", PLAIN_HTML)
        self.assertIsNone(si.detect_format(p))

    def test_detect_rejects_non_html_suffix(self) -> None:
        p = _write(self.tmp, "notes.txt", TELEGRAM_HTML)
        self.assertIsNone(si.detect_format(p))

    def test_extract_carries_sender_forward(self) -> None:
        p = _write(self.tmp, "messages.html", TELEGRAM_HTML)
        text = si.smart_extract_text(p)
        self.assertIsNotNone(text)
        blocks = text.split("\n\n")
        self.assertEqual(len(blocks), 3)  # service пропущено
        # Сгруппированное сообщение наследует автора предыдущего.
        self.assertIn("Sergey Medvedev", blocks[1])
        self.assertNotIn("Unknown", text)

    def test_extract_skips_service(self) -> None:
        p = _write(self.tmp, "messages.html", TELEGRAM_HTML)
        text = si.smart_extract_text(p) or ""
        self.assertNotIn("3 October 2025", text)

    def test_extract_media_marker(self) -> None:
        p = _write(self.tmp, "messages.html", TELEGRAM_HTML)
        text = si.smart_extract_text(p) or ""
        self.assertIn("[медиа: Sticker]", text)
        self.assertIn("Anna", text)

    def test_extract_includes_dates_and_text(self) -> None:
        p = _write(self.tmp, "messages.html", TELEGRAM_HTML)
        text = si.smart_extract_text(p) or ""
        self.assertIn("[03.10.2025 12:34:18 UTC+03:00] Sergey Medvedev:", text)
        self.assertIn("Есть вопрос по статусам", text)

    def test_plain_html_returns_none(self) -> None:
        p = _write(self.tmp, "page.html", PLAIN_HTML)
        self.assertIsNone(si.smart_extract_text(p))


class TestIntegrationWithExtractContent(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_extract_content_uses_smart_import(self) -> None:
        from file_extractors import extract_content

        p = _write(self.tmp, "messages.html", TELEGRAM_HTML)
        kind, content = extract_content(p)
        self.assertEqual(kind, "text")
        self.assertIn("[медиа: Sticker]", content)
        self.assertIn("Sergey Medvedev", content)
        # Сырого HTML/служебных классов в результате быть не должно.
        self.assertNotIn("page_wrap", content)
        self.assertNotIn("media_wrap", content)

    def test_extract_content_falls_back_for_plain_html(self) -> None:
        from file_extractors import extract_content

        p = _write(self.tmp, "page.html", PLAIN_HTML)
        kind, content = extract_content(p)
        self.assertEqual(kind, "text")
        self.assertIn("Обычная страница", content)


if __name__ == "__main__":
    unittest.main()
