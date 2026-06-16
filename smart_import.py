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

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

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

    def extract(self, path: Path) -> str:
        from bs4 import BeautifulSoup

        from file_extractors import _decode

        raw = path.read_bytes()
        soup = BeautifulSoup(_decode(raw), "html.parser")
        messages = soup.find_all("div", class_="message")

        out: list[str] = []
        last_sender = "Unknown"
        kept = 0
        for msg in messages:
            classes = msg.get("class") or []
            if "service" in classes:
                continue  # даты-разделители, пины и т.п. — пропускаем

            fn = msg.find("div", class_="from_name")
            if fn is not None:
                sender = fn.get_text(strip=True) or last_sender
                last_sender = sender
            else:
                sender = last_sender  # сгруппированное сообщение — автор тот же

            date = self._msg_date(msg)

            body_parts: list[str] = []
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


# Порядок важен: первый совпавший импортёр выигрывает.
IMPORTERS: list[Importer] = [TelegramHtmlImporter()]


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
