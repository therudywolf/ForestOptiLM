# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
# ForestOptiLM is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""W3 веб-стека: импорт внешних источников в блокнот (Wikipedia, GitLab).

Keyless для ПУБЛИЧНОГО контента: Wikipedia через её plain-text extracts API,
GitLab через публичный REST API (без токена для открытых репозиториев). Каждый
импортер возвращает материал, который штатный индексатор блокнота (pipeline.
build_index) съедает как любой другой источник — формат-агностично.

Confluence требует сервер + авторизацию (не keyless) → отдельно, здесь не делаем.
Парсеры (`_wiki_title_from_url`, `_gitlab_project_id`) — чистые, тестируются.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import quote, unquote, urlparse

logger = logging.getLogger("nocturne")

# Wikimedia/GitLab API-политика требует ОПИСАТЕЛЬНЫЙ User-Agent (фейковый
# браузерный UA Wikipedia отдаёт 403). Идентифицируем приложение честно.
_API_UA = "NocturneDataForge/0.7 (+https://github.com/therudywolf/ForestOptiLM)"

# код/текст, который есть смысл индексировать из репозитория (остальное — скип)
_GITLAB_TEXT_EXT = {
    ".md", ".rst", ".txt", ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".java", ".kt", ".rb", ".php", ".cs", ".swift",
    ".sh", ".bash", ".sql", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".json",
    ".xml", ".html", ".css", ".scala", ".lua", ".r", ".dart", ".vue",
}


@dataclass(slots=True)
class ImportedDoc:
    name: str          # человекочитаемое имя источника
    text: str
    origin: str = ""   # URL/путь происхождения


# --------------------------------------------------------------------------- #
#  Wikipedia
# --------------------------------------------------------------------------- #

def _wiki_title_from_url(s: str) -> tuple[str, str]:
    """Из ссылки/заголовка вернуть (title, lang). Понимает
    `https://ru.wikipedia.org/wiki/Заголовок` и голый заголовок (lang по умолч.)."""
    s = (s or "").strip()
    if s.startswith("http"):
        u = urlparse(s)
        first = u.netloc.split(".")[0] if u.netloc else ""
        # язык — только если это реальный языковой поддомен (ru./en./de.…), а не
        # www/m/канонический wikipedia.org; иначе "" → caller подставит дефолт
        # (иначе `https://wikipedia.org/wiki/Foo` дал бы host wikipedia.wikipedia.org).
        lang = first if ("wikipedia.org" in u.netloc and first not in ("wikipedia", "www", "m", "")) else ""
        path = u.path
        title = unquote(path.split("/wiki/", 1)[1]) if "/wiki/" in path else unquote(path.strip("/"))
        return title.replace("_", " "), lang
    return s, ""


def import_wikipedia(title_or_url: str, lang: str = "ru", timeout: float = 20.0) -> ImportedDoc:
    """Импортировать статью Wikipedia как чистый текст (keyless extracts API)."""
    import httpx
    title, url_lang = _wiki_title_from_url(title_or_url)
    lang = url_lang or lang
    api = f"https://{lang}.wikipedia.org/w/api.php"
    params = {"action": "query", "format": "json", "prop": "extracts",
              "explaintext": "1", "redirects": "1", "titles": title}
    r = httpx.get(api, params=params, headers={"User-Agent": _API_UA},
                  timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    pages = (r.json().get("query") or {}).get("pages") or {}
    for _pid, page in pages.items():
        text = (page.get("extract") or "").strip()
        if text:
            real_title = page.get("title") or title
            return ImportedDoc(name=f"Wikipedia — {real_title}", text=text,
                               origin=f"https://{lang}.wikipedia.org/wiki/{quote(real_title)}")
    raise ValueError(f"Wikipedia: статья не найдена — {title!r} ({lang})")


# --------------------------------------------------------------------------- #
#  GitLab (public repos, keyless)
# --------------------------------------------------------------------------- #

def _gitlab_project_id(repo_url: str) -> tuple[str, str]:
    """Из URL репозитория вернуть (url-encoded project path, host). Понимает
    `https://gitlab.com/group/sub/project` и `group/project`."""
    s = (repo_url or "").strip().rstrip("/")
    host = "gitlab.com"
    if s.startswith("http"):
        u = urlparse(s)
        host = u.netloc or host
        path = u.path.strip("/")
    else:
        path = s
    path = path.removesuffix(".git")
    return quote(path, safe=""), host


def import_gitlab_repo(repo_url: str, ref: str = "", *, max_files: int = 100,
                       max_bytes: int = 200_000, timeout: float = 25.0) -> list[ImportedDoc]:
    """Импортировать текстовые/кодовые файлы публичного GitLab-репо (keyless API)."""
    import base64
    import httpx
    from pathlib import PurePosixPath
    pid, host = _gitlab_project_id(repo_url)
    api = f"https://{host}/api/v4/projects/{pid}"
    hdr = {"User-Agent": _API_UA}
    with httpx.Client(timeout=timeout, headers=hdr, follow_redirects=True) as c:
        if not ref:
            info = c.get(api); info.raise_for_status()
            ref = info.json().get("default_branch") or "main"
        tree, page = [], 1
        while len(tree) < 2000:
            tr = c.get(f"{api}/repository/tree",
                       params={"recursive": "true", "per_page": 100, "page": page, "ref": ref})
            tr.raise_for_status()
            batch = tr.json()
            if not batch:
                break
            tree.extend(batch)
            page += 1
        out: list[ImportedDoc] = []
        for node in tree:
            if node.get("type") != "blob":
                continue
            path = node.get("path", "")
            if PurePosixPath(path).suffix.lower() not in _GITLAB_TEXT_EXT:
                continue
            fr = c.get(f"{api}/repository/files/{quote(path, safe='')}",
                       params={"ref": ref})
            if fr.status_code != 200:
                continue
            j = fr.json()
            if j.get("encoding") == "base64":
                raw = base64.b64decode(j.get("content") or "")[:max_bytes]
                text = raw.decode("utf-8", "replace")
            else:
                text = (j.get("content") or "")[:max_bytes]
            if text.strip():
                out.append(ImportedDoc(name=path, text=text,
                                       origin=f"{repo_url}#{path}"))
            if len(out) >= max_files:
                break
    if not out:
        raise ValueError(f"GitLab: не найдено текстовых файлов — {repo_url}")
    return out
