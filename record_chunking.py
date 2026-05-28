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
Format-agnostic, structure-preserving record extraction & chunking.

Цель: превратить большие структурированные отчёты (JSON / JSONL / XML) в
MAP-чанки так, чтобы **одна запись никогда не рвалась между чанками**, и при
этом НЕ хардкодить схему конкретного сканера/отчёта. Мы распознаём *форму*
данных (массивы объектов / повторяющиеся XML-элементы), а не бренд.

Это решает две беды generic-пути:
- JSON вида {"Results":[{"Vulnerabilities":[...]}]} раньше превращался в одну
  строку pandas.json_normalize и был бесполезен;
- сырой текст/таблица рвали структуру находок по токенам.

Записи извлекаются из самого глубокого «листового» массива объектов, а скаляры
родительских уровней переносятся как контекст (@-поля). Богатые контейнеры
(много собственных скалярных полей, напр. ZAP alert) эмитятся как одна запись
с кратким резюме вложенных массивов — иначе один alert с 1000 instances дал бы
1000 записей.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

logger = logging.getLogger("nocturne")

RECORD_SUFFIXES = {".json", ".jsonl", ".ndjson", ".xml"}

_SCALAR_TYPES = (str, int, float, bool)
# Контейнер с таким числом собственных скалярных полей сам считается записью
# (его вложенные массивы сворачиваются в "[N items]"), а не разворачивается.
MIN_PARENT_SCALARS_TO_SUMMARIZE = 6
_MAX_DEPTH = 10
_MAX_VALUE_CHARS = 300
_MAX_CTX_VALUE_CHARS = 160


def record_aware_enabled() -> bool:
    return os.getenv("NOCTURNE_RECORD_AWARE", "1").strip() != "0"


def _record_aware_max_bytes() -> int:
    raw = os.getenv("NOCTURNE_RECORD_AWARE_MAX_BYTES", "").strip()
    if raw.isdigit():
        return int(raw)
    return 200 * 1024 * 1024  # 200 MiB: выше — отдаём потоковому текстовому пути


def _min_records() -> int:
    raw = os.getenv("NOCTURNE_RECORD_AWARE_MIN_RECORDS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return 2


# ------------------------------------------------------------------ #
#  JSON record extraction
# ------------------------------------------------------------------ #

def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, _SCALAR_TYPES)


def _scalars(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if _is_scalar(v)}


def _has_record_array(d: dict[str, Any]) -> bool:
    for v in d.values():
        if isinstance(v, list) and any(isinstance(x, dict) for x in v):
            return True
    return False


def _summarize_arrays(d: dict[str, Any]) -> dict[str, Any]:
    """Свернуть вложенные массивы объектов в "[N items]" (для записи-контейнера)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, list) and any(isinstance(x, dict) for x in v):
            out[k] = f"[{len(v)} items]"
        else:
            out[k] = v
    return out


def _merge_ctx(ctx: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    """Контекст родителей (@-поля) идёт первым; поля записи перекрывают при коллизии."""
    merged: dict[str, Any] = {}
    for k, v in ctx.items():
        merged[f"@{k}"] = v
    merged.update(rec)
    return merged


def _collect_json(
    node: Any,
    ctx: dict[str, Any],
    path: str,
    out: list[tuple[str, dict[str, Any]]],
    depth: int = 0,
) -> None:
    if depth > _MAX_DEPTH:
        return
    if isinstance(node, dict):
        ctx2 = {**ctx, **_scalars(node)}
        arrays = [
            (k, v) for k, v in node.items()
            if isinstance(v, list) and any(isinstance(x, dict) for x in v)
        ]
        nested_dicts = [(k, v) for k, v in node.items() if isinstance(v, dict)]
        if not arrays:
            # Одиночный объект без массивов записей: на верхнем уровне это сам по
            # себе один отчёт-запись (напр. файл с единственной находкой).
            if depth == 0 and _scalars(node):
                out.append((path or "$", dict(node)))
            for k, v in nested_dicts:
                _collect_json(v, ctx2, f"{path}.{k}" if path else k, out, depth + 1)
            return
        for k, arr in arrays:
            _collect_list(arr, ctx2, f"{path}.{k}" if path else k, out, depth + 1)
        for k, v in nested_dicts:
            if _has_record_array(v):
                _collect_json(v, ctx2, f"{path}.{k}" if path else k, out, depth + 1)
    elif isinstance(node, list):
        _collect_list(node, ctx, path, out, depth + 1)


def _collect_list(
    arr: list[Any],
    ctx: dict[str, Any],
    path: str,
    out: list[tuple[str, dict[str, Any]]],
    depth: int,
) -> None:
    for d in arr:
        if not isinstance(d, dict):
            continue
        if not _has_record_array(d):
            out.append((path, _merge_ctx(ctx, d)))
        elif len(_scalars(d)) >= MIN_PARENT_SCALARS_TO_SUMMARIZE:
            # Богатый контейнер (напр. ZAP alert): сам запись, вложенное — резюме.
            out.append((path, _merge_ctx(ctx, _summarize_arrays(d))))
        else:
            # Бедный контейнер (напр. Trivy Result: Target/Class/Type) — спускаемся
            # к настоящим записям, перенося его скаляры как контекст.
            _collect_json(d, {**ctx, **_scalars(d)}, path, out, depth + 1)


def extract_json_records(data: Any) -> list[tuple[str, dict[str, Any]]]:
    """Вернуть [(json_path, record_dict), ...] из произвольного JSON-дерева."""
    out: list[tuple[str, dict[str, Any]]] = []
    _collect_json(data, {}, "", out, 0)
    return out


# ------------------------------------------------------------------ #
#  XML record extraction (best-effort, generic)
# ------------------------------------------------------------------ #

def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _xml_leaf_fields(el: ET.Element) -> dict[str, Any]:
    """Атрибуты + текст прямых/неглубоких дочерних листьев элемента."""
    fields: dict[str, Any] = {}
    for ak, av in el.attrib.items():
        fields[_strip_ns(ak)] = av
    for child in el:
        tag = _strip_ns(child.tag)
        text = (child.text or "").strip()
        grandkids = list(child)
        if not grandkids:
            if text:
                fields[tag] = text
        else:
            # Вложенный повторяющийся блок (напр. instances) — сводим к счётчику.
            fields[tag] = f"[{len(grandkids)} items]"
    if el.text and el.text.strip() and not fields:
        fields["_text"] = el.text.strip()
    return fields


def extract_xml_records(root: ET.Element) -> list[tuple[str, dict[str, Any]]]:
    """
    Найти самый «мелкий» (ближе к корню) повторяющийся тип элемента с детьми
    и эмитить по записи на вхождение. Без привязки к конкретной схеме.
    """
    # tag -> (count, min_depth, elements)
    stats: dict[str, list[Any]] = {}

    def _walk(el: ET.Element, depth: int) -> None:
        for child in el:
            tag = _strip_ns(child.tag)
            if list(child):  # есть дети → потенциальная «запись»
                rec = stats.setdefault(tag, [0, depth, []])
                rec[0] += 1
                rec[1] = min(rec[1], depth)
                rec[2].append(child)
            _walk(child, depth + 1)

    _walk(root, 0)
    candidates = [
        (tag, cnt, mind, els) for tag, (cnt, mind, els) in stats.items() if cnt >= 2
    ]
    if not candidates:
        return []
    # Предпочитаем самый мелкий уровень; при равенстве — больший count.
    candidates.sort(key=lambda c: (c[2], -c[1]))
    tag, _cnt, _mind, els = candidates[0]
    out: list[tuple[str, dict[str, Any]]] = []
    for el in els:
        out.append((tag, _xml_leaf_fields(el)))
    return out


# ------------------------------------------------------------------ #
#  Rendering & chunking
# ------------------------------------------------------------------ #

def _render_value(v: Any, max_chars: int) -> str:
    if isinstance(v, dict):
        s = json.dumps(v, ensure_ascii=False)
    elif isinstance(v, list):
        if all(_is_scalar(x) for x in v):
            s = ", ".join(str(x) for x in v)
        else:
            s = f"[{len(v)} items]"
    else:
        s = str(v)
    s = " ".join(s.split())
    if len(s) > max_chars:
        s = s[:max_chars] + "…"
    return s


def render_record(path: str, rec: dict[str, Any]) -> str:
    lines = [f"- record @ {path}"]
    for k, v in rec.items():
        if v is None or v == "":
            continue
        is_ctx = k.startswith("@")
        s = _render_value(v, _MAX_CTX_VALUE_CHARS if is_ctx else _MAX_VALUE_CHARS)
        if not s:
            continue
        lines.append(f"  {k}: {s}")
    return "\n".join(lines)


def _meta_header_line(file_meta: dict[str, str] | None) -> str:
    if not file_meta:
        return ""
    parts: list[str] = []
    if file_meta.get("title"):
        parts.append(f"[FILE_TITLE: {file_meta['title']}]")
    if file_meta.get("labels"):
        parts.append(f"[FILE_LABELS: {file_meta['labels']}]")
    if file_meta.get("format"):
        parts.append(f"[FILE_FORMAT: {file_meta['format']}]")
    return "".join(parts) + "\n" if parts else ""


def records_to_chunks(
    records: list[tuple[str, dict[str, Any]]],
    display_path: str,
    chunk_size_tokens: int,
    file_meta: dict[str, str] | None = None,
) -> list[str]:
    """Упаковать целые записи в чанки под бюджет токенов (запись неделима)."""
    from parser import count_tokens

    budget = max(200, chunk_size_tokens)
    rendered = [render_record(p, r) for p, r in records]
    groups: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for txt in rendered:
        t = count_tokens(txt)
        if t > budget:
            # Одна запись больше бюджета — выносим отдельно (модель усечёт сама).
            if cur:
                groups.append("\n".join(cur))
                cur, cur_tokens = [], 0
            groups.append(txt)
            continue
        if cur and cur_tokens + t > budget:
            groups.append("\n".join(cur))
            cur, cur_tokens = [], 0
        cur.append(txt)
        cur_tokens += t
    if cur:
        groups.append("\n".join(cur))

    meta_line = _meta_header_line(file_meta)
    n = len(groups)
    out: list[str] = []
    for i, body in enumerate(groups):
        out.append(
            f"[FILE_PATH: {display_path}][FILE_PART: records {i + 1}/{n}][CHUNK_INDEX: {i + 1}]\n"
            f"{meta_line}"
            f"[Файл: {display_path}]\n{body}"
        )
    return out


# ------------------------------------------------------------------ #
#  File-level dispatch
# ------------------------------------------------------------------ #

def _read_text(path: Path) -> str:
    try:
        from file_extractors import _decode

        return _decode(path.read_bytes())
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace")


def extract_records_from_file(path: Path) -> list[tuple[str, dict[str, Any]]] | None:
    """
    Извлечь записи из структурированного файла без хардкода схемы.
    Возвращает None, если файл не похож на набор записей (тогда — обычный путь).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in RECORD_SUFFIXES:
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > _record_aware_max_bytes() and suffix not in (".jsonl", ".ndjson"):
        return None

    try:
        if suffix in (".jsonl", ".ndjson"):
            records: list[tuple[str, dict[str, Any]]] = []
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for ln, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        records.append((f"line[{ln}]", obj))
            return records if len(records) >= _min_records() else None

        if suffix == ".json":
            data = json.loads(_read_text(path))
            records = extract_json_records(data)
            return records if len(records) >= _min_records() else None

        if suffix == ".xml":
            root = ET.fromstring(_read_text(path))
            records = extract_xml_records(root)
            return records if len(records) >= _min_records() else None
    except (json.JSONDecodeError, ET.ParseError) as exc:
        logger.debug("record extraction failed for %s: %s", path, exc)
        return None
    except Exception as exc:
        logger.debug("record extraction error for %s: %s", path, exc)
        return None
    return None


def build_record_chunks(
    path: Path,
    chunk_size_tokens: int,
    root_dir: Path | None = None,
    file_meta: dict[str, str] | None = None,
) -> list[str] | None:
    """parse_file-совместимый вход: записи → MAP-чанки, или None для fallback."""
    path = Path(path)
    records = extract_records_from_file(path)
    if not records:
        return None
    if root_dir is not None:
        try:
            display_path = str(path.relative_to(root_dir)).replace("\\", "/")
        except ValueError:
            display_path = path.name
    else:
        display_path = path.name
    chunks = records_to_chunks(records, display_path, chunk_size_tokens, file_meta)
    logger.info(
        "Record-aware chunking: %s records=%s chunks=%s",
        display_path, len(records), len(chunks),
    )
    return chunks or None
