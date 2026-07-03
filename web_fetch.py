# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
# ForestOptiLM is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""W2 веб-стека: скачать страницу и извлечь ОСНОВНОЙ текст (browser-имитация).

Keyless: обычный User-Agent (при наличии curl_cffi можно перейти на impersonate
Chrome для обхода бот-детекта — задел). Извлечение основного контента — эвристика
на bs4 (убираем nav/header/footer/aside/script/style, берём article/main/body):
без внешних readability-зависимостей. Веб-PDF — через существующий `_read_pdf`.

Гигиена: follow redirects, тайм-аут, лимит размера, никакого исполнения
скачанного. `extract_main_text` — чистая функция, тестируется без сети.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from web_search import _UA

logger = logging.getLogger("nocturne")

_NOISE_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside",
               "form", "iframe", "svg", "button", "template")


class FetchError(Exception):
    """Не удалось скачать/распарсить страницу."""


@dataclass(slots=True)
class FetchedPage:
    url: str
    final_url: str
    title: str
    text: str
    content_type: str = ""

    def to_dict(self) -> dict:
        return {"url": self.url, "final_url": self.final_url, "title": self.title,
                "text": self.text, "content_type": self.content_type}


def extract_main_text(html: str) -> tuple[str, str]:
    """Из HTML вернуть (title, основной_текст). Чистая функция: убираем шумовые
    теги и берём самый содержательный контейнер (article/main/[role=main]/body),
    схлопываем пустые строки."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html or "", "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup(list(_NOISE_TAGS)):
        tag.decompose()
    main = (soup.select_one("article") or soup.select_one("main")
            or soup.select_one("[role=main]") or soup.body or soup)
    text = main.get_text("\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # схлопнуть длинные повторы пустых строк уже сделано; лёгкая чистка пробелов
    return title, re.sub(r"[ \t]{2,}", " ", "\n".join(lines))


def _pdf_bytes_to_text(data: bytes) -> str:
    """Веб-PDF: сохранить во временный файл и переиспользовать штатный _read_pdf."""
    import tempfile
    from pathlib import Path
    from file_extractors import _read_pdf
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(data)
        tmp = Path(f.name)
    try:
        return _read_pdf(tmp)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def fetch(url: str, *, timeout: float = 20.0, max_bytes: int = 5_000_000) -> FetchedPage:
    """Скачать URL и извлечь основной текст. Бросает FetchError при сбое."""
    import httpx
    if not (url or "").startswith(("http://", "https://")):
        raise FetchError(f"неподдерживаемый URL: {url!r}")
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout,
                          headers={"User-Agent": _UA}) as client:
            r = client.get(url)
            r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"не удалось скачать {url}: {exc}") from exc
    ct = (r.headers.get("content-type") or "").lower()
    final = str(r.url)
    data = r.content[:max_bytes]
    if "application/pdf" in ct or url.lower().split("?")[0].endswith(".pdf"):
        text = _pdf_bytes_to_text(data)
        title = final.rstrip("/").rsplit("/", 1)[-1]
    elif "html" in ct or "xml" in ct or not ct:
        enc = r.encoding or "utf-8"
        title, text = extract_main_text(data.decode(enc, "replace"))
    else:
        raise FetchError(f"неподдерживаемый content-type: {ct or '?'}")
    if not (text or "").strip():
        raise FetchError(f"пустой контент со страницы {final}")
    return FetchedPage(url=url, final_url=final, title=title, text=text, content_type=ct)


def fetch_safe(url: str, **kw) -> FetchedPage | None:
    """Как fetch, но при сбое → None (для батч-обхода выдачи, чтобы одна битая
    ссылка не рушила дипресёрч)."""
    try:
        return fetch(url, **kw)
    except Exception as exc:  # noqa: BLE001
        logger.info("web_fetch: пропуск %s — %s", url, exc)
        return None
