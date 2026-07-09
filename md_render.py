# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
"""Лёгкий рендер Markdown → читаемый plain-text для чат-пузыря.

CTkLabel не умеет rich-text, поэтому «сырые» `### заголовок`, `**жирный**`,
`* пункт` показывались буквально — некрасиво и нечитаемо. Здесь снимаем разметку
и приводим к чистому виду: заголовки — отдельной строкой, списки — маркером «•»,
инлайн-маркеры (**/__/`/*) убираем, ссылки [N] и текст сохраняем. Чистая функция —
тестируется без GUI и применяется и к финальному ответу, и к стримингу.
"""
from __future__ import annotations

import re

_H_RE = re.compile(r"^(\s{0,3})#{1,6}\s+(.*?)\s*#*\s*$")
_BULLET_RE = re.compile(r"^(\s*)(?:[*+\-]|\d+[.)])\s+(.*)$")
_HR_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")   # --- *** ___
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_CODE_RE = re.compile(r"`([^`]+)`")
# одиночный *курсив* — только когда явно парный и не список/умножение
_ITALIC_RE = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")


def _strip_inline(s: str) -> str:
    s = _BOLD_RE.sub(lambda m: m.group(1) if m.group(1) is not None else m.group(2), s)
    s = _CODE_RE.sub(r"\1", s)
    s = _ITALIC_RE.sub(r"\1", s)
    return s


def to_plain(text: str) -> str:
    """Markdown → чистый plain-text. Заголовок → своя строка (с пустой строкой до,
    чтобы визуально отделялся); списки → «•  …»; **/__/`/* сняты."""
    if not text:
        return ""
    out: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if _HR_RE.match(raw):
            continue  # горизонтальные линии --- просто убираем
        h = _H_RE.match(raw)
        if h:
            if out and out[-1].strip():
                out.append("")  # пустая строка перед заголовком — отделяем блок
            out.append(_strip_inline(h.group(2)).strip())
            continue
        b = _BULLET_RE.match(raw)
        if b:
            out.append(f"{b.group(1)}•  {_strip_inline(b.group(2)).rstrip()}")
            continue
        out.append(_strip_inline(raw).rstrip())
    # схлопнуть 3+ подряд пустых строк в одну
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return result
