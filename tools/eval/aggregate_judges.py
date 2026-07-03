# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
"""Мульти-судейство: K независимых оценок ОДНОГО ответа → стабильный вердикт.

Эмпирика (2026-07-03): генерация детерминирована — deep-режим при greedy-декоде
даёт БАЙТ-в-байт тот же ответ на тот же вход (проверено: 4 прогона e3 идентичны).
Значит источник шума в eval — не генерация, а САМ судья: Claude по-разному
оценивает пограничные случаи (напр. флаг «галлюцинация» на факте, часть которого
в обрезанном контексте). На главном вердикте (кто победил) судья устойчив
(e3: 4/4 «no»), но балл/флаг гуляют. Лечится усреднением K судей: мажоритарный
вердикт + средний балл + флаг разногласия (agree=False там, где судьи разошлись).

Чистая функция, без сети/LLM — юнит-тестируется. Сами оценки собирает Workflow
(K параллельных судей-агентов на вопрос), сюда приходит список их вердиктов.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence


def _mean3(d: dict) -> float:
    try:
        return (d["completeness"] + d["correctness"] + d["grounding"]) / 3.0
    except Exception:  # noqa: BLE001
        return 0.0


def aggregate(records: Sequence[dict]) -> dict[str, dict]:
    """Свести K оценок на вопрос в один устойчивый результат.

    Вход: список вердиктов вида {qid, project_beats_baseline, project:{c/c/g},
    baseline:{c/c/g}, project_hallucination}. Выход: {qid: {...}} с мажоритарным
    вердиктом, средними баллами и признаком согласия судей."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        qid = r.get("qid")
        if qid:
            by[qid].append(r)
    out: dict[str, dict] = {}
    for qid, rs in by.items():
        votes = Counter(r.get("project_beats_baseline") for r in rs)
        maj, maj_n = votes.most_common(1)[0]
        halls = sum(1 for r in rs if r.get("project_hallucination"))
        out[qid] = {
            "n_judges": len(rs),
            "beats": maj,                                   # мажоритарный вердикт
            "beats_votes": dict(votes),
            "agree": maj_n == len(rs),                      # судьи единогласны?
            "project_mean": round(sum(_mean3(r.get("project", {})) for r in rs) / len(rs), 2),
            "baseline_mean": round(sum(_mean3(r.get("baseline", {})) for r in rs) / len(rs), 2),
            "hallucination": halls * 2 > len(rs),           # по большинству
            "hallucination_votes": halls,
        }
    return out


def tally(agg: dict[str, dict]) -> dict:
    """Свод по всем вопросам: сколько побед/поражений/ничьих проекта, средние баллы,
    сколько вердиктов с разногласием судей (индикатор надёжности)."""
    beats = Counter(v["beats"] for v in agg.values())
    n = len(agg) or 1
    return {
        "n_questions": len(agg),
        "wins": beats.get("yes", 0),
        "losses": beats.get("no", 0),
        "ties": beats.get("tie", 0),
        "project_mean": round(sum(v["project_mean"] for v in agg.values()) / n, 2),
        "baseline_mean": round(sum(v["baseline_mean"] for v in agg.values()) / n, 2),
        "hallucinations": sum(1 for v in agg.values() if v["hallucination"]),
        "split_verdicts": sum(1 for v in agg.values() if not v["agree"]),
    }
