# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
"""Сборка карточек для мульти-судейства: baseline-ответ + ответ проекта + его
СОБСТВЕННЫЕ извлечённые контексты + вопрос → один JSON на вопрос.

Судья грунтует ответ проекта по его own_contexts (а не по чужой узкой выжимке —
иначе широкий deep-сбор штрафуется как «выдумки»). Код committable, данные
(gen/baseline/cards) — в gitignored eval_runs/. Чистые пути внутрь, без сети.
"""
from __future__ import annotations

import json
from pathlib import Path


def _q_from_ctx(ctx_dir: Path, qid: str) -> str:
    """Вопрос берём из первой строки baseline_ctx/{qid}.txt («# Q[..]: текст»)."""
    p = ctx_dir / f"{qid}.txt"
    if not p.is_file():
        return ""
    first = p.read_text(encoding="utf-8").splitlines()
    return first[0] if first else ""


def build_cards(gen_dir: Path, baseline_dir: Path, ctx_dir: Path, out_dir: Path) -> int:
    """Собрать карточки по всем {qid}.json в gen_dir. Возвращает число карточек.

    gen_dir/{qid}.json     — ответ проекта: {answer, refused, contexts:[{source,text}], ...}
    baseline_dir/{qid}.json — мой ручной ответ: {answer, ...}
    ctx_dir/{qid}.txt       — вопрос в первой строке + BM25-референс
    """
    gen_dir, baseline_dir, ctx_dir, out_dir = map(Path, (gen_dir, baseline_dir, ctx_dir, out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(gen_dir.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        qid = r.get("qid") or f.stem
        bpath = baseline_dir / f"{qid}.json"
        baseline_answer = ""
        if bpath.is_file():
            baseline_answer = json.loads(bpath.read_text(encoding="utf-8")).get("answer", "")
        card = {
            "qid": qid,
            "task_type": r.get("task_type", ""),
            "question": _q_from_ctx(ctx_dir, qid),
            "baseline_answer": baseline_answer,
            "project_answer": r.get("answer", ""),
            "project_refused": r.get("refused"),
            "project_own_contexts": r.get("contexts", []),
            "baseline_ctx_path": str((ctx_dir / f"{qid}.txt").resolve()),
        }
        (out_dir / f"{qid}.json").write_text(
            json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
        n += 1
    return n


if __name__ == "__main__":  # pragma: no cover
    import sys
    if len(sys.argv) != 5:
        print("usage: build_cards.py <gen_dir> <baseline_dir> <ctx_dir> <out_dir>")
        raise SystemExit(2)
    count = build_cards(*(Path(a) for a in sys.argv[1:5]))
    print(f"wrote {count} cards to {sys.argv[4]}")
