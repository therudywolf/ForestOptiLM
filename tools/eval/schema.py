# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
"""Схема набора вопросов + gold-разметки для eval-харнеса.

Один вопрос = одна JSONL-строка. Разметка встроена в запись (проще, чем
отдельный файл). ВАЖНО: файлы вопросов содержат реальный контент корпуса
(имена/пути/хендлы) → лежат в gitignored eval_data/, в гит НЕ коммитятся.
Коммитится только КОД харнеса.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

# Типы задач — по ним группируем метрики и подбираем режим (deep/enhanced/…).
TASK_TYPES = ("enumeration", "causal", "portrait", "factoid")


@dataclass(slots=True)
class Question:
    id: str
    task_type: str
    question: str
    # gold для retrieval-метрик: id релевантных чанков и/или исходные файлы.
    gold_chunk_ids: list[str] = field(default_factory=list)
    gold_sources: list[str] = field(default_factory=list)
    # gold для оценки ОТВЕТА: ключевые факты, которые полный ответ обязан покрыть.
    gold_answer_points: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(
            id=str(d["id"]),
            task_type=str(d.get("task_type") or "factoid"),
            question=str(d["question"]),
            gold_chunk_ids=[str(x) for x in (d.get("gold_chunk_ids") or [])],
            gold_sources=[str(x) for x in (d.get("gold_sources") or [])],
            gold_answer_points=[str(x) for x in (d.get("gold_answer_points") or [])],
            notes=str(d.get("notes") or ""),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_type": self.task_type,
            "question": self.question,
            "gold_chunk_ids": self.gold_chunk_ids,
            "gold_sources": self.gold_sources,
            "gold_answer_points": self.gold_answer_points,
            "notes": self.notes,
        }


def load_questions(path: Path) -> list[Question]:
    out: list[Question] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Question.from_dict(json.loads(line)))
    return out


def save_questions(questions: Iterable[Question], path: Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
