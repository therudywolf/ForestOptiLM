# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
# ForestOptiLM is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ForestOptiLM is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public
# License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ForestOptiLM. If not, see <https://www.gnu.org/licenses/>.
"""
Умный импорт — распознаёт формат выгрузки и приводит его к чистому,
оптимальному для LLM тексту ДО чанкинга/индексации.

Зачем: сырой HTML-экспорт (например, из Telegram Desktop) при наивном
``get_text()`` превращается в кашу из навигации, дат, реакций и разметки.
Умный импортёр вместо этого извлекает диалог «по сообщениям»
(``[дата] автор:\\nтекст``) — ровно тот вид, что хорошо заходит в NotebookLM-
подобные сценарии.

Архитектура — реестр импортёров (``Importer``). Каждый умеет ``detect(path)``
и ``extract(path) -> str``. ``smart_extract_text`` возвращает чистый текст
первого подходящего импортёра либо ``None`` (тогда вызывающий код использует
обычный путь). Встроено в :func:`file_extractors.extract_content`, поэтому
работает сразу в Map-Reduce, RAG и Блокнотах.

Добавить новый формат (WhatsApp ``_chat.txt``, Slack export JSON, …) — это
один новый класс в ``IMPORTERS`` ниже, без правок в остальном коде.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("nocturne")

_HTML_SUFFIXES = {".html", ".htm"}
_DETECT_BYTES = 262144  # читаем первые 256 KiB для определения формата
# Заглушки Telegram для не выгруженных вложений — это шум, не контент.
_TELEGRAM_NOISE = {
    "not included, change data exporting settings to download.",
    "not included, change data exporting settings to download",
}


@runtime_checkable
class Importer(Protocol):
    name: str

    def detect(self, path: Path) -> bool: ...

    def extract(self, path: Path) -> str: ...


def _read_head(path: Path, n: int = _DETECT_BYTES) -> str:
    try:
        with path.open("rb") as f:
            raw = f.read(n)
    except Exception:
        return ""
    try:
        from file_extractors import _decode

        return _decode(raw)
    except Exception:
        return raw.decode("utf-8", errors="replace")


class TelegramHtmlImporter:
    """Экспорт Telegram Desktop (messages*.html) → чистый диалог.

    Улучшения относительно наивного парсинга:
    - **перенос автора** для сгруппированных (``joined``) сообщений, у которых
      Telegram опускает ``from_name``;
    - пропуск служебных сообщений (``message service``: даты-разделители, пины);
    - маркер ``[медиа: …]`` для фото/файлов/стикеров (иначе теряются);
    - дата берётся из ``title`` (полная, с таймзоной).
    """

    name = "telegram_html"

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() not in _HTML_SUFFIXES:
            return False
        low = _read_head(path).lower()
        if not low:
            return False
        # Сигнатуры экспорта Telegram Desktop.
        if "message default clearfix" in low:
            return True
        return 'class="message' in low and "pull_right date details" in low

    # --- helpers -------------------------------------------------------- #
    @staticmethod
    def _msg_date(msg) -> str:
        d = msg.find("div", class_="date")
        if d is not None and d.has_attr("title"):
            return str(d["title"]).strip()
        return ""

    @staticmethod
    def _media_marker(msg) -> str:
        mw = msg.find("div", class_="media_wrap")
        if mw is None:
            return ""
        parts: list[str] = []
        title = mw.find("div", class_="title")
        if title is not None:
            t = title.get_text(strip=True)
            if t:
                parts.append(t)
        desc = mw.find("div", class_="description")
        if desc is not None:
            d = desc.get_text(strip=True)
            if d and d.lower() not in _TELEGRAM_NOISE:
                parts.append(d)
        label = " — ".join(parts) if parts else "вложение"
        return f"[медиа: {label}]"

    @staticmethod
    def _forwarded_marker(msg) -> str:
        fwd = msg.find("div", class_="forwarded")
        if fwd is None:
            return ""
        fn = fwd.find("div", class_="from_name")
        name = fn.get_text(strip=True) if fn is not None else ""
        return f"[переслано от: {name}]" if name else "[переслано]"

    @staticmethod
    def _reply_marker(msg) -> str:
        rt = msg.find("div", class_="reply_to")
        if rt is None:
            return ""
        t = rt.get_text(" ", strip=True)
        return f"[{t}]" if t else "[в ответ на сообщение]"

    def extract(self, path: Path) -> str:
        from bs4 import BeautifulSoup

        from file_extractors import _decode

        raw = path.read_bytes()
        try:
            soup = BeautifulSoup(_decode(raw), "lxml")   # C-парсер — кратно быстрее на больших экспортах
        except Exception:
            soup = BeautifulSoup(_decode(raw), "html.parser")
        messages = soup.find_all("div", class_="message")

        out: list[str] = []
        last_sender = "Unknown"
        kept = 0
        for msg in messages:
            classes = msg.get("class") or []
            if "service" in classes:
                continue  # даты-разделители, пины и т.п. — пропускаем

            # Автор — первый from_name ВНЕ forwarded-блока (иначе подхватили бы
            # имя источника пересылки вместо реального отправителя).
            fn = next((c for c in msg.find_all("div", class_="from_name")
                       if c.find_parent("div", class_="forwarded") is None), None)
            if fn is not None:
                sender = fn.get_text(strip=True) or last_sender
                last_sender = sender
            else:
                sender = last_sender  # сгруппированное сообщение — автор тот же

            date = self._msg_date(msg)

            body_parts: list[str] = []
            # Сохраняем структуру треда: на что отвечают и откуда переслано.
            reply = self._reply_marker(msg)
            if reply:
                body_parts.append(reply)
            fwd = self._forwarded_marker(msg)
            if fwd:
                body_parts.append(fwd)
            text_div = msg.find("div", class_="text")
            if text_div is not None:
                text = text_div.get_text(separator="\n").strip()
                if text:
                    body_parts.append(text)
            media = self._media_marker(msg)
            if media:
                body_parts.append(media)

            if not body_parts:
                continue  # нечего извлекать (например, чисто служебный блок)

            header = f"[{date}] {sender}:" if date else f"{sender}:"
            out.append(f"{header}\n" + "\n".join(body_parts))
            kept += 1

        logger.info("smart_import telegram_html: %s → %d messages", path.name, kept)
        return "\n\n".join(out)


def _fmt_unix(ts: Any) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%d.%m.%Y %H:%M")
    except Exception:
        return ""


class WhatsAppTxtImporter:
    """Экспорт чата WhatsApp (_chat.txt) → диалог по сообщениям.

    Форматы строк (зависят от платформы/локали)::

        [12.10.2023, 14:30:15] Имя: текст
        12/10/2023, 14:30 - Имя: текст

    Многострочные сообщения склеиваются (строки-продолжения без таймстампа).
    Системные строки (шифрование, «изменил тему» и т.п. — без «Имя:») опускаем.
    """

    name = "whatsapp_txt"
    _LINE = re.compile(
        r"^\[?(?P<ts>\d{1,2}[./]\d{1,2}[./]\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?"
        r"(?:\s?[APap][Mm])?)\]?\s+(?:-\s)?(?P<rest>.+)$"
    )

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() not in {".txt"}:
            return False
        head = _read_head(path, 65536)
        if not head:
            return False
        hits = 0
        for line in head.splitlines()[:80]:
            m = self._LINE.match(line)
            if m and ": " in m.group("rest")[:42]:
                hits += 1
        return hits >= 4  # WhatsApp-структура (дата с / или . + запятая); логи не задевает

    def extract(self, path: Path) -> str:
        from file_extractors import _decode

        text = _decode(path.read_bytes())
        out: list[str] = []
        cur: list[str] = []
        cur_header = ""

        def flush() -> None:
            if cur_header and cur:
                out.append(cur_header + "\n" + "\n".join(cur).strip())

        for line in text.splitlines():
            m = self._LINE.match(line)
            if m:
                rest = m.group("rest")
                if ": " in rest[:42]:
                    flush()
                    sender, msg = rest.split(":", 1)
                    cur_header = f"[{m.group('ts').strip()}] {sender.strip()}:"
                    cur = [msg.strip()]
                else:
                    # системное сообщение — закрываем предыдущее и пропускаем
                    flush()
                    cur, cur_header = [], ""
            elif cur_header:
                cur.append(line.rstrip())
        flush()
        logger.info("smart_import whatsapp_txt: %s → %d messages", path.name, len(out))
        return "\n\n".join(out)


class SlackJsonImporter:
    """Экспорт Slack (JSON массив сообщений канала) → диалог по сообщениям."""

    name = "slack_json"

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() != ".json":
            return False
        head = _read_head(path, 65536)
        return bool(head) and '"ts"' in head and '"text"' in head and (
            '"user"' in head or '"username"' in head or '"user_profile"' in head
        )

    def extract(self, path: Path) -> str:
        from file_extractors import _decode

        data = json.loads(_decode(path.read_bytes()))
        msgs = data if isinstance(data, list) else (data.get("messages") if isinstance(data, dict) else None)
        if not isinstance(msgs, list):
            return ""
        skip_subtypes = {"channel_join", "channel_leave", "group_join", "group_leave",
                         "channel_topic", "channel_purpose", "channel_name"}
        out: list[str] = []
        for m in msgs:
            if not isinstance(m, dict) or m.get("type", "message") != "message":
                continue
            if str(m.get("subtype") or "") in skip_subtypes:
                continue
            text = str(m.get("text") or "").strip()
            if not text:
                continue
            prof = m.get("user_profile") if isinstance(m.get("user_profile"), dict) else {}
            sender = str(prof.get("real_name") or m.get("username") or m.get("user") or "unknown")
            ts = _fmt_unix(m.get("ts"))
            header = f"[{ts}] {sender}:" if ts else f"{sender}:"
            out.append(f"{header}\n{text}")
        logger.info("smart_import slack_json: %s → %d messages", path.name, len(out))
        return "\n\n".join(out)


class DiscordJsonImporter:
    """Экспорт Discord (DiscordChatExporter JSON) → диалог по сообщениям."""

    name = "discord_json"

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() != ".json":
            return False
        head = _read_head(path, 65536)
        return bool(head) and '"messages"' in head and '"author"' in head and '"timestamp"' in head

    def extract(self, path: Path) -> str:
        from file_extractors import _decode

        data = json.loads(_decode(path.read_bytes()))
        msgs = data.get("messages") if isinstance(data, dict) else None
        if not isinstance(msgs, list):
            return ""
        out: list[str] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            author = m.get("author") if isinstance(m.get("author"), dict) else {}
            name = str(author.get("nickname") or author.get("name") or "unknown")
            content = str(m.get("content") or "").strip()
            if not content:
                continue
            ts = str(m.get("timestamp") or "")[:16].replace("T", " ")
            header = f"[{ts}] {name}:" if ts else f"{name}:"
            out.append(f"{header}\n{content}")
        logger.info("smart_import discord_json: %s → %d messages", path.name, len(out))
        return "\n\n".join(out)


class GenericChatLogImporter:
    """Неструктурированный чат-лог вида «Имя: сообщение» (без таймстампов).

    Срабатывает ПОСЛЕДНИМ и консервативно: нужен высокий процент строк-реплик,
    короткие имена-отправители И повторяющиеся собеседники — чтобы не перехватить
    конфиги (`ключ: значение`), прозу с двоеточиями или код. Многострочные реплики
    (строки без префикса «Имя:») приклеиваются к предыдущему сообщению.
    """

    name = "generic_chatlog"
    # Имя: 1–30 символов, начинается с буквы/цифры, без предложений/пунктуации-в-конце.
    _LINE = re.compile(r"^(?P<name>[\wА-Яа-яЁё][\w .\-]{0,29}):[ \t]+(?P<msg>\S.*)$")

    def _scan(self, head: str) -> tuple[int, int, dict[str, int]]:
        non_empty = 0
        matched = 0
        senders: dict[str, int] = {}
        for line in head.splitlines()[:80]:
            if not line.strip():
                continue
            non_empty += 1
            m = self._LINE.match(line)
            if m:
                matched += 1
                senders[m.group("name").strip()] = senders.get(m.group("name").strip(), 0) + 1
        return non_empty, matched, senders

    def detect(self, path: Path) -> bool:
        if path.suffix.lower() not in {".txt", ".log"}:
            return False
        head = _read_head(path, 65536)
        if not head:
            return False
        non_empty, matched, senders = self._scan(head)
        if non_empty < 5 or matched < 6:
            return False
        if matched / non_empty < 0.6:  # большинство строк — реплики
            return False
        # Минимум 2 разных собеседника, и кто-то повторяется (признак диалога, не конфига).
        recurring = [c for c in senders.values() if c >= 2]
        return len(senders) >= 2 and bool(recurring)

    def extract(self, path: Path) -> str:
        from file_extractors import _decode

        text = _decode(path.read_bytes())
        out: list[str] = []
        cur: list[str] = []
        cur_header = ""

        def flush() -> None:
            if cur_header and cur:
                out.append(cur_header + "\n" + "\n".join(cur).strip())

        for line in text.splitlines():
            m = self._LINE.match(line)
            if m:
                flush()
                cur_header = f"{m.group('name').strip()}:"
                cur = [m.group("msg").strip()]
            elif cur_header and line.strip():
                cur.append(line.rstrip())  # продолжение реплики
        flush()
        logger.info("smart_import generic_chatlog: %s → %d messages", path.name, len(out))
        return "\n\n".join(out)


# Порядок важен: первый совпавший импортёр выигрывает. Generic — последним
# (самый широкий и потому наименее приоритетный).
IMPORTERS: list[Importer] = [
    TelegramHtmlImporter(),
    WhatsAppTxtImporter(),
    SlackJsonImporter(),
    DiscordJsonImporter(),
    GenericChatLogImporter(),
]


def detect_format(path: Path) -> str | None:
    """Вернуть имя формата, если какой-то импортёр распознал файл."""
    p = Path(path)
    for imp in IMPORTERS:
        try:
            if imp.detect(p):
                return imp.name
        except Exception as exc:  # noqa: BLE001 — детектор не должен ронять извлечение
            logger.debug("smart_import detect %s failed: %s", getattr(imp, "name", "?"), exc)
    return None


def smart_extract_text(path: Path) -> str | None:
    """Чистый текст первого подходящего импортёра или ``None``.

    ``None`` означает «обычный путь извлечения» (никакой умный импортёр не
    подошёл). Пустая строка (импортёр сработал, но контента нет) тоже трактуется
    как ``None``, чтобы не индексировать пустышку.
    """
    p = Path(path)
    for imp in IMPORTERS:
        try:
            if not imp.detect(p):
                continue
        except Exception as exc:  # noqa: BLE001
            logger.debug("smart_import detect %s failed: %s", getattr(imp, "name", "?"), exc)
            continue
        try:
            text = imp.extract(p)
        except Exception as exc:  # noqa: BLE001 — падение умного парсера → откат к обычному
            logger.warning("smart_import %s extract failed for %s: %s",
                           getattr(imp, "name", "?"), p.name, exc)
            return None
        # Устойчивость к импортёрам, нарушившим контракт (вернувшим не-строку):
        # любой пустой/неожиданный результат → откат к обычному пути.
        if not text or not isinstance(text, str):
            return None
        return text.strip() or None
    return None
