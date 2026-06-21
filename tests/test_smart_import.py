# SPDX-License-Identifier: AGPL-3.0-or-later
"""Smart import: Telegram HTML export → clean per-message dialogue."""
from __future__ import annotations

import json
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

# Ответ на сообщение + пересланное (forwarded) — проверка сохранения структуры треда.
TELEGRAM_REPLY_FWD = """<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>
<div class="history">
  <div class="message default clearfix" id="m4"><div class="body">
    <div class="pull_right date details" title="03.10.2025 13:00:00 UTC+03:00">13:00</div>
    <div class="from_name">Boris</div>
    <div class="reply_to details">In reply to <a href="#m1">Sergey</a></div>
    <div class="text">Согласен</div>
  </div></div>
  <div class="message default clearfix" id="m5"><div class="body">
    <div class="pull_right date details" title="03.10.2025 13:01:00 UTC+03:00">13:01</div>
    <div class="from_name">Anna</div>
    <div class="forwarded body">
      <div class="from_name">External Source</div>
      <div class="text">Пересланный текст</div>
    </div>
  </div></div>
</div></body></html>"""


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

    def test_reply_and_forwarded_markers(self) -> None:
        p = _write(self.tmp, "messages.html", TELEGRAM_REPLY_FWD)
        text = si.smart_extract_text(p) or ""
        # reply сохранён
        self.assertIn("Boris:", text)
        self.assertIn("[In reply to Sergey]", text)
        self.assertIn("Согласен", text)
        # forwarded: автор = пересылающий (Anna), плюс атрибуция источника
        self.assertIn("Anna:", text)
        self.assertIn("[переслано от: External Source]", text)
        self.assertIn("Пересланный текст", text)
        # имя источника пересылки НЕ стало автором сообщения
        self.assertNotIn("External Source:", text)

    def test_plain_html_returns_none(self) -> None:
        p = _write(self.tmp, "page.html", PLAIN_HTML)
        self.assertIsNone(si.smart_extract_text(p))


WHATSAPP_TXT = (
    "[12.10.2023, 14:30:15] Сергей Медведев: Привет, выкатываем релиз v0.7?\n"
    "[12.10.2023, 14:31:02] Анна: Да, и я проверю кириллицу\n"
    "это вторая строка моего сообщения\n"
    "[12.10.2023, 14:32:00] Сергей Медведев: Отлично 👍\n"
    "12/10/2023, 14:33 - Анна: Собираю под Fedora\n"
    "12/10/2023, 14:34 - Messages and calls are end-to-end encrypted.\n"
)

REGULAR_LOG = "\n".join(
    f"2023-10-12 14:3{i}:00 ERROR module: something failed code={i}" for i in range(9)
)

SLACK_JSON = json.dumps([
    {"type": "message", "user": "U1", "user_profile": {"real_name": "Сергей"},
     "text": "Готовим релиз v0.7", "ts": "1697112600.000100"},
    {"type": "message", "username": "anna", "text": "Проверю кириллицу", "ts": "1697112660.000200"},
    {"type": "message", "subtype": "channel_join", "user": "U3", "text": "has joined", "ts": "1697112700.0"},
], ensure_ascii=False)

DISCORD_JSON = json.dumps({
    "guild": {"name": "G"}, "channel": {"name": "general"},
    "messages": [
        {"author": {"name": "sergey", "nickname": "Сергей"},
         "timestamp": "2023-10-12T14:30:00.000+00:00", "content": "Релиз готов"},
        {"author": {"name": "anna"}, "timestamp": "2023-10-12T14:31:00.000+00:00", "content": "Проверю"},
    ],
}, ensure_ascii=False)


class TestWhatsAppSlackDiscord(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_whatsapp_detect_and_extract(self) -> None:
        p = _write(self.tmp, "WhatsApp Chat.txt", WHATSAPP_TXT)
        self.assertEqual(si.detect_format(p), "whatsapp_txt")
        text = si.smart_extract_text(p) or ""
        self.assertIn("[12.10.2023, 14:30:15] Сергей Медведев:", text)
        self.assertIn("это вторая строка моего сообщения", text)  # multiline joined
        self.assertNotIn("end-to-end encrypted", text)  # system line skipped
        self.assertIn("Собираю под Fedora", text)  # dash format too

    def test_regular_log_not_whatsapp(self) -> None:
        p = _write(self.tmp, "app.log", REGULAR_LOG)
        self.assertIsNone(si.detect_format(p))  # must NOT hijack ordinary logs

    def test_slack_detect_and_extract(self) -> None:
        p = _write(self.tmp, "general.json", SLACK_JSON)
        self.assertEqual(si.detect_format(p), "slack_json")
        text = si.smart_extract_text(p) or ""
        self.assertIn("Сергей:", text)
        self.assertIn("Готовим релиз", text)
        self.assertNotIn("has joined", text)  # channel_join skipped
        self.assertEqual(text.count("\n\n"), 1)  # two messages

    def test_discord_detect_and_extract(self) -> None:
        p = _write(self.tmp, "export.json", DISCORD_JSON)
        self.assertEqual(si.detect_format(p), "discord_json")
        text = si.smart_extract_text(p) or ""
        self.assertIn("Сергей:", text)  # nickname preferred
        self.assertIn("Релиз готов", text)

    def test_plain_json_not_matched(self) -> None:
        p = _write(self.tmp, "data.json", '{"foo": 1, "bar": [2, 3]}')
        self.assertIsNone(si.detect_format(p))


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


_CHATLOG = (
    "Аня: привет, ты видела новый отчёт?\n"
    "Борис: да, посмотрел утром\n"
    "Аня: что думаешь про раздел по ВМ?\n"
    "Борис: норм, но не хватает деталей\n"
    "это можно дописать позже\n"
    "Аня: согласна, добавим\n"
    "Борис: ок, тогда я возьму\n"
)
_CONFIG = (
    "host: localhost\nport: 8080\ntimeout: 30\nretries: 5\n"
    "name: service\nlevel: debug\npath: /var/log\n"
)
_PROSE = (
    "Это обычный текст. Вот мысль: иногда бывает двоеточие.\n"
    "Ещё предложение без структуры реплик.\n"
)


class TestGenericChatLog(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_detects_chatlog(self) -> None:
        p = _write(self.tmp, "chat.txt", _CHATLOG)
        self.assertEqual(si.detect_format(p), "generic_chatlog")

    def test_rejects_config_unique_keys(self) -> None:
        # Конфиг: уникальные ключи, нет повторяющихся «собеседников» → не чат.
        p = _write(self.tmp, "app.conf.txt", _CONFIG)
        self.assertIsNone(si.detect_format(p))

    def test_rejects_prose(self) -> None:
        p = _write(self.tmp, "notes.txt", _PROSE)
        self.assertIsNone(si.detect_format(p))

    def test_extract_merges_continuation_lines(self) -> None:
        p = _write(self.tmp, "chat.txt", _CHATLOG)
        text = si.smart_extract_text(p)
        self.assertIn("Аня:", text)
        self.assertIn("Борис:", text)
        # строка-продолжение приклеена к реплике Бориса, а не отдельным блоком
        self.assertIn("не хватает деталей\nэто можно дописать позже", text)

    def test_whatsapp_still_wins_over_generic(self) -> None:
        # Файл с таймстампами WhatsApp должен опознаться как whatsapp_txt, не generic.
        wa = (
            "[12.10.2023, 14:30:15] Аня: привет\n"
            "[12.10.2023, 14:31:00] Борис: здарова\n"
            "[12.10.2023, 14:32:00] Аня: как дела\n"
            "[12.10.2023, 14:33:00] Борис: норм\n"
            "[12.10.2023, 14:34:00] Аня: ок\n"
            "[12.10.2023, 14:35:00] Борис: пока\n"
        )
        p = _write(self.tmp, "_chat.txt", wa)
        self.assertEqual(si.detect_format(p), "whatsapp_txt")


if __name__ == "__main__":
    unittest.main()
