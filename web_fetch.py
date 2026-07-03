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

import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

from web_search import _UA

logger = logging.getLogger("nocturne")

_MAX_REDIRECTS = 5

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
    """Веб-PDF: сохранить во временный файл и переиспользовать штатный _read_pdf.
    tmp биндится СРАЗУ после создания, чтобы сбой записи не оставил осиротевший
    файл (delete=False)."""
    import tempfile
    from pathlib import Path
    from file_extractors import _read_pdf
    f = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp = Path(f.name)
    try:
        f.write(data)
        f.close()
        return _read_pdf(tmp)
    finally:
        f.close()
        try:
            tmp.unlink()
        except OSError:
            pass


def _assert_public_url(url: str) -> None:
    """SSRF-защита: пропускать только http(s) на ПУБЛИЧНО-маршрутизируемый адрес.
    Резолвим хост и отбрасываем loopback/private/link-local/reserved (в т.ч.
    облачные метаданные 169.254.169.254 и локальные LLM-порты). Проверяется на
    КАЖДОМ хопе редиректа отдельно. Бросает FetchError, если адрес непубличный."""
    if not (url or "").startswith(("http://", "https://")):
        raise FetchError(f"неподдерживаемый URL: {url!r}")
    host = urlparse(url).hostname
    if not host:
        raise FetchError(f"нет хоста в URL: {url!r}")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise FetchError(f"не удалось разрешить {host}: {exc}") from exc
    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip.split("%")[0])  # срезаем zone-id IPv6
        except ValueError:
            raise FetchError(f"нераспознанный адрес {ip} для {host}")
        if not addr.is_global or addr.is_reserved or addr.is_multicast:
            raise FetchError(f"доступ к непубличному адресу запрещён: {host} ({ip})")


def fetch(url: str, *, timeout: float = 20.0, max_bytes: int = 5_000_000) -> FetchedPage:
    """Скачать URL и извлечь основной текст. Бросает FetchError при сбое.

    Гигиена: (1) SSRF-проверка хоста на каждом хопе (редиректы follow-им ВРУЧНУЮ,
    чтобы 30x не увёл на внутренний адрес); (2) тело читается ПОТОКОМ с обрывом по
    max_bytes — гигантский/бесконечный ответ не буферизуется в RAM целиком."""
    import httpx
    current = url
    try:
        with httpx.Client(follow_redirects=False, timeout=timeout,
                          headers={"User-Agent": _UA}) as client:
            for _hop in range(_MAX_REDIRECTS + 1):
                _assert_public_url(current)  # проверка ПЕРЕД каждым соединением
                with client.stream("GET", current) as r:
                    if r.is_redirect:
                        loc = r.headers.get("location")
                        if not loc:
                            raise FetchError(f"редирект без Location: {current}")
                        current = urljoin(str(r.url), loc)
                        continue
                    r.raise_for_status()
                    ct = (r.headers.get("content-type") or "").lower()
                    final = str(r.url)
                    buf = bytearray()
                    for chunk in r.iter_bytes():
                        buf += chunk
                        if len(buf) >= max_bytes:
                            break
                    data = bytes(buf[:max_bytes])
                    enc = r.encoding or "utf-8"
                    break
            else:
                raise FetchError(f"слишком много редиректов: {url}")
    except FetchError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"не удалось скачать {url}: {exc}") from exc
    if "application/pdf" in ct or current.lower().split("?")[0].endswith(".pdf"):
        text = _pdf_bytes_to_text(data)
        title = final.rstrip("/").rsplit("/", 1)[-1]
    elif "html" in ct or "xml" in ct or not ct:
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
