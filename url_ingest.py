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
Загрузка веб-страницы как источника блокнота.

Полностью локальная операция: только HTTP(S)-запрос к указанному URL и
извлечение основного текста (через ту же BeautifulSoup, что и остальной проект).
Никаких облачных сервисов; ничего не отправляется наружу, кроме самого GET.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("nocturne")

# Гард на размер ответа (по умолчанию 25 MiB), чтобы случайная ссылка на дамп
# не утянула гигабайты в память.
_MAX_URL_BYTES = int(os.getenv("NOCTURNE_URL_MAX_BYTES", str(25 * 1024 * 1024)))
_USER_AGENT = "NocturneDataForge/1.0 (+local; AGPL-3.0)"


class UrlIngestError(RuntimeError):
    """Понятная ошибка загрузки URL для показа в UI."""


@dataclass(slots=True)
class FetchedDoc:
    url: str
    title: str
    text: str
    content_type: str


def _validate_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise UrlIngestError("Пустой URL")
    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise UrlIngestError(f"Поддерживаются только http/https (получено: {parsed.scheme!r})")
    if not parsed.netloc:
        raise UrlIngestError("URL без хоста")
    return url


def _decode_bytes(raw: bytes, encoding_hint: str | None = None) -> str:
    if encoding_hint:
        try:
            return raw.decode(encoding_hint, errors="replace")
        except Exception:
            pass
    try:
        from file_extractors import _decode  # многокодировочный декодер проекта

        return _decode(raw)
    except Exception:
        return raw.decode("utf-8", errors="replace")


def _html_to_text(html: str) -> tuple[str, str]:
    """Вернуть (title, text). Заголовок — из <title>/<h1>; текст без скриптов."""
    title = ""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        for tag in soup(["script", "style", "noscript", "template", "svg"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
    except Exception:
        # Грубый fallback без bs4.
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            title = re.sub(r"\s+", " ", m.group(1)).strip()
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"[ \t]+", " ", text)
    # Схлопываем длинные простыни пустых строк.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text


def fetch_url(url: str, *, timeout: float = 30.0, max_bytes: int | None = None) -> FetchedDoc:
    """Скачать страницу и извлечь основной текст.

    Бросает :class:`UrlIngestError` с человекочитаемым сообщением.
    """
    url = _validate_url(url)
    cap = max_bytes if max_bytes is not None else _MAX_URL_BYTES
    headers = {"User-Agent": _USER_AGENT, "Accept": "*/*"}
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
            with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise UrlIngestError(f"HTTP {resp.status_code} при загрузке {url}")
                content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > cap:
                        raise UrlIngestError(
                            f"Ответ превысил лимит {cap // (1024 * 1024)} MiB "
                            "(NOCTURNE_URL_MAX_BYTES)"
                        )
                    chunks.append(chunk)
                raw = b"".join(chunks)
                encoding_hint = resp.encoding
    except UrlIngestError:
        raise
    except httpx.HTTPError as exc:
        raise UrlIngestError(f"Сетевая ошибка: {exc}") from exc

    text_body = _decode_bytes(raw, encoding_hint)
    is_html = "html" in content_type or (not content_type and "<html" in text_body[:2000].lower())
    if is_html:
        title, text = _html_to_text(text_body)
    else:
        title, text = "", text_body.strip()

    if not text.strip():
        raise UrlIngestError("Из страницы не удалось извлечь текст")

    if not title:
        # Заголовок из первой непустой строки.
        for line in text.splitlines():
            if line.strip():
                title = line.strip()[:120]
                break

    return FetchedDoc(url=url, title=title, text=text, content_type=content_type or "text/plain")
