# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
# ForestOptiLM is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Keyless веб-поиск: провайдеры БЕЗ API-ключей.

W1 веб-стека. Пользователю не нужно ничего настраивать — поиск идёт через
HTML-эндпоинты, имитируя обычный браузер (User-Agent). Сейчас: DuckDuckGo HTML
(`html.duckduckgo.com/html`) — стабилен и не требует ключа. Google/Yandex SERP
можно добавить провайдерами (лучше через curl_cffi с impersonate — обход
бот-детекта), фолбэк-цепочка уже заложена.

Парсер (`parse_ddg_html`) — чистая функция, тестируется на фикстуре без сети.
Сетевой вызов изолирован. Сбой любого провайдера → пустой список (не падаем).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

logger = logging.getLogger("nocturne")

# UA обычного Chrome — иначе DDG/поисковики отдают капчу/пусто.
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
_DDG_HTML = "https://html.duckduckgo.com/html/"


@dataclass(slots=True)
class WebResult:
    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


def _decode_ddg_url(href: str) -> str:
    """DDG отдаёт ссылки как редирект `//duckduckgo.com/l/?uddg=<real>` —
    достаём реальный URL из параметра uddg. Прямые ссылки возвращаем как есть."""
    if not href:
        return ""
    full = href if href.startswith("http") else ("https:" + href if href.startswith("//") else href)
    if "uddg=" in full:
        q = parse_qs(urlparse(full).query)
        if q.get("uddg"):
            return unquote(q["uddg"][0])
    return full


def parse_ddg_html(html_text: str, max_results: int = 10) -> list[WebResult]:
    """Разобрать HTML-выдачу DuckDuckGo в список результатов. Чистая функция."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_text or "", "html.parser")
    out: list[WebResult] = []
    seen: set[str] = set()
    for res in soup.select("div.result, div.web-result, div.results_links"):
        a = res.select_one("a.result__a")
        if not a:
            continue
        url = _decode_ddg_url(a.get("href", ""))
        title = a.get_text(" ", strip=True)
        if not url or not title or url in seen:
            continue
        sn = res.select_one(".result__snippet")
        snippet = sn.get_text(" ", strip=True) if sn else ""
        seen.add(url)
        out.append(WebResult(title=title, url=url, snippet=snippet))
        if len(out) >= max_results:
            break
    return out


def search_ddg(query: str, max_results: int = 10, timeout: float = 15.0) -> list[WebResult]:
    """DuckDuckGo HTML-поиск (keyless). Бросает исключение при сетевом сбое —
    ловит вызывающий `search`."""
    import httpx
    r = httpx.post(_DDG_HTML, data={"q": query, "kl": "wt-wt"},
                   headers={"User-Agent": _UA}, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return parse_ddg_html(r.text, max_results)


# Провайдеры в порядке предпочтения (фолбэк-цепочка). Google/Yandex добавятся сюда.
_PROVIDERS = (("duckduckgo", search_ddg),)


def search(query: str, max_results: int = 10) -> list[WebResult]:
    """Keyless-поиск по цепочке провайдеров. Первый успешный непустой ответ —
    результат; при сбое всех → пустой список (не роняем вызывающий код)."""
    q = (query or "").strip()
    if not q:
        return []
    for name, provider in _PROVIDERS:
        try:
            hits = provider(q, max_results)
            if hits:
                return hits
            logger.info("web_search: провайдер %s вернул пусто", name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("web_search: провайдер %s не отработал — %s", name, exc)
    return []
