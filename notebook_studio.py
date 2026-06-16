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
Studio — генерация учебных материалов по корпусу блокнота.

Аналог Studio-панели NotebookLM: по содержимому источников блокнота строятся
учебный гайд, FAQ, таймлайн, краткий конспект (брифинг) и флеш-карточки. Всё
заземлено на реальные фрагменты из индекса; для больших корпусов берётся
равномерная выборка фрагментов под бюджет контекста (об этом честно пишем).

Чистые функции (``gather_corpus_digest`` / ``build_material_messages`` /
``parse_flashcards``) тестируются без сети; ``generate_material`` — async
оркестратор поверх ``processor.call_llm``.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("nocturne")


@dataclass(frozen=True, slots=True)
class MaterialSpec:
    kind: str
    title: str
    filename: str
    instructions: str
    is_json: bool = False


MATERIALS: dict[str, MaterialSpec] = {
    "study_guide": MaterialSpec(
        kind="study_guide",
        title="Учебный гайд",
        filename="study_guide.md",
        instructions=(
            "Составь подробное учебное пособие (study guide) по материалам. "
            "Структура (Markdown):\n"
            "## Краткий обзор — 3–5 предложений о чём корпус.\n"
            "## Ключевые понятия — список терминов с определениями.\n"
            "## Основные разделы — по каждой крупной теме абзац с сутью.\n"
            "## Главные выводы — маркированный список.\n"
            "## Вопросы для самопроверки — 5–8 вопросов."
        ),
    ),
    "faq": MaterialSpec(
        kind="faq",
        title="FAQ",
        filename="faq.md",
        instructions=(
            "Составь FAQ из 8–15 пар «Вопрос/Ответ», покрывающих самое важное в "
            "материалах. Формат Markdown: каждый пункт как '**В:** …' и '**О:** …'. "
            "Ответы — только по содержимому фрагментов."
        ),
    ),
    "timeline": MaterialSpec(
        kind="timeline",
        title="Таймлайн",
        filename="timeline.md",
        instructions=(
            "Построй хронологию (таймлайн) событий, версий, дат или этапов, "
            "упомянутых в материалах. Формат Markdown — маркированный список "
            "'- **<дата/этап>** — <что произошло>', по возрастанию. Если явных дат "
            "нет, упорядочь по логической последовательности этапов."
        ),
    ),
    "briefing": MaterialSpec(
        kind="briefing",
        title="Конспект",
        filename="briefing.md",
        instructions=(
            "Напиши краткий брифинг-конспект (синопсис) по материалам: 4–8 абзацев, "
            "которые передают суть корпуса так, чтобы человек без чтения источников "
            "понял главное. В конце — раздел '## Главное в одном абзаце'."
        ),
    ),
    "flashcards": MaterialSpec(
        kind="flashcards",
        title="Флеш-карточки",
        filename="flashcards.json",
        is_json=True,
        instructions=(
            "Сделай 10–20 флеш-карточек для запоминания по материалам. Верни ТОЛЬКО "
            "JSON-массив объектов вида {\"front\": \"вопрос/термин\", "
            "\"back\": \"ответ/определение\"}. Без пояснений и текста вне JSON."
        ),
    ),
}

MATERIAL_ORDER = ["study_guide", "faq", "timeline", "briefing", "flashcards"]

_STUDIO_SYSTEM_PROMPT = (
    "Ты — методист, который готовит учебные материалы СТРОГО на основе "
    "предоставленных фрагментов корпуса. Не добавляй фактов извне и не выдумывай. "
    "Если данных мало — отрази это честно. Пиши на языке материалов."
)


@dataclass(slots=True)
class CorpusDigest:
    text: str
    chunks_used: int
    chunks_total: int
    sampled: bool


def read_index_chunks(index_dir: Path) -> list[str]:
    """Прочитать тексты фрагментов из chunks_meta.jsonl индекса блокнота."""
    meta = Path(index_dir) / "chunks_meta.jsonl"
    if not meta.is_file():
        return []
    out: list[str] = []
    with meta.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = str(rec.get("text") or "").strip()
            if text:
                out.append(text)
    return out


def _count_tokens(text: str) -> int:
    try:
        from parser import count_tokens

        return count_tokens(text)
    except Exception:  # pragma: no cover
        return max(1, len(text) // 4)


def gather_corpus_digest(chunks: list[str], *, max_tokens: int = 12000) -> CorpusDigest:
    """Сжать корпус до бюджета токенов.

    Если всё помещается — конкатенируем. Иначе берём равномерную выборку
    фрагментов (стабильную, без рандома), помечая, что это выборка.
    """
    total = len(chunks)
    if total == 0:
        return CorpusDigest(text="", chunks_used=0, chunks_total=0, sampled=False)

    full = "\n\n---\n\n".join(chunks)
    if _count_tokens(full) <= max_tokens:
        return CorpusDigest(text=full, chunks_used=total, chunks_total=total, sampled=False)

    # Равномерная выборка: идём по корпусу с шагом, копим до бюджета.
    selected: list[str] = []
    used = 0
    # шаг подбираем так, чтобы охватить весь диапазон, но не упереться сразу
    step = max(1, total // 64) if total > 64 else 1
    idx = 0
    visited: set[int] = set()
    while idx < total and used < max_tokens:
        if idx in visited:
            idx += 1
            continue
        visited.add(idx)
        c = chunks[idx]
        tok = _count_tokens(c)
        if selected and used + tok > max_tokens:
            break
        selected.append(c)
        used += tok
        idx += step
    digest = "\n\n---\n\n".join(selected)
    return CorpusDigest(
        text=digest, chunks_used=len(selected), chunks_total=total, sampled=True
    )


def build_material_messages(
    spec: MaterialSpec, digest: CorpusDigest, *, notebook_name: str = ""
) -> list[dict[str, str]]:
    """Собрать messages для генерации одного материала (чистая функция)."""
    note = ""
    if digest.sampled:
        note = (
            f"\n\nПРИМЕЧАНИЕ: корпус большой, ниже — репрезентативная выборка "
            f"{digest.chunks_used} из {digest.chunks_total} фрагментов."
        )
    header = f"Блокнот: {notebook_name}\n\n" if notebook_name else ""
    user = (
        f"{header}{spec.instructions}{note}\n\n"
        "=== МАТЕРИАЛЫ КОРПУСА ===\n"
        f"{digest.text}\n"
        "=== КОНЕЦ МАТЕРИАЛОВ ==="
    )
    return [
        {"role": "system", "content": _STUDIO_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_flashcards(raw: str) -> list[dict[str, str]]:
    """Извлечь массив карточек из ответа модели (с/без ```json-ограждения)."""
    text = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    # Найти первый JSON-массив.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    cards: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        front = str(item.get("front") or item.get("q") or item.get("question") or "").strip()
        back = str(item.get("back") or item.get("a") or item.get("answer") or "").strip()
        if front and back:
            cards.append({"front": front, "back": back})
    return cards


def _flashcards_to_markdown(cards: list[dict[str, str]]) -> str:
    lines = ["# Флеш-карточки", ""]
    for i, c in enumerate(cards, 1):
        lines.append(f"**{i}. {c['front']}**")
        lines.append("")
        lines.append(c["back"])
        lines.append("")
    return "\n".join(lines)


async def generate_material(
    notebook: Any,
    kind: str,
    *,
    base_url: str,
    api_key: str,
    chat_model: str,
    api_mode: str = "native",
    max_tokens: int = 3000,
    max_context_tokens: int = 12000,
    on_log: Callable[[str], None] | None = None,
) -> tuple[Path, str]:
    """Сгенерировать материал ``kind`` и сохранить его в notes/ блокнота.

    Возвращает (путь_к_файлу, содержимое). Бросает RuntimeError, если индекс
    блокнота пуст.
    """
    import asyncio

    from processor import call_llm

    spec = MATERIALS.get(kind)
    if spec is None:
        raise ValueError(f"Неизвестный тип материала: {kind}")

    def _log(msg: str) -> None:
        if on_log:
            try:
                on_log(msg)
            except Exception:
                pass

    chunks = read_index_chunks(notebook.index_dir)
    if not chunks:
        raise RuntimeError("Индекс блокнота пуст — сначала постройте индекс")
    digest = gather_corpus_digest(chunks, max_tokens=max_context_tokens)
    _log(
        f"{spec.title}: дайджест {digest.chunks_used}/{digest.chunks_total} фрагментов"
        + (" (выборка)" if digest.sampled else "")
    )

    messages = build_material_messages(spec, digest, notebook_name=notebook.name)
    semaphore = asyncio.Semaphore(1)
    raw = await call_llm(
        messages,
        chat_model,
        base_url,
        api_key,
        semaphore,
        max_tokens=max_tokens,
        api_mode=api_mode,
    )
    raw = (raw or "").strip()

    if spec.is_json:
        cards = parse_flashcards(raw)
        if cards:
            json_path = notebook.save_note(
                spec.filename, json.dumps(cards, ensure_ascii=False, indent=2)
            )
            # Дублируем человекочитаемой версией для просмотра в панели.
            md = _flashcards_to_markdown(cards)
            notebook.save_note("flashcards.md", md)
            return json_path, md
        # Модель не отдала валидный JSON — сохраняем как есть в .md.
        path = notebook.save_note("flashcards.md", raw or "(пустой ответ)")
        return path, raw

    content = raw or "(пустой ответ модели)"
    path = notebook.save_note(spec.filename, content)
    return path, content
