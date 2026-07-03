# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
"""Чистые IR-метрики для оценки ранжирования retrieval.

Без внешних зависимостей и без обращения к корпусу — берут список
*ранжированных* id (что вернул поиск, сверху вниз) и множество *релевантных*
id (gold) и считают качество. Так тюнинг ранжирования становится измеримым,
а не «на глаз».
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Доля gold-элементов, попавших в top-k. 1.0 = все релевантные найдены."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    hit = sum(1 for cid in ranked[:k] if cid in gold_set)
    return hit / len(gold_set)


def precision_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Доля top-k, которая релевантна. Штрафует за шум наверху выдачи."""
    if k <= 0:
        return 0.0
    gold_set = set(gold)
    hit = sum(1 for cid in ranked[:k] if cid in gold_set)
    return hit / k


def reciprocal_rank(ranked: Sequence[str], gold: Iterable[str]) -> float:
    """1/позиция первого релевантного (1-based). 0.0 — ни одного не нашли.

    MRR по набору вопросов = среднее этих значений; ловит «релевантное есть,
    но закопано вниз»."""
    gold_set = set(gold)
    for i, cid in enumerate(ranked):
        if cid in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    return sum(
        (1.0 / math.log2(i + 2)) for i, cid in enumerate(ranked[:k]) if cid in gold_set
    )


def ndcg_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Normalized DCG@k с бинарной релевантностью. Награждает релевантное ВЫШЕ.

    Отличается от recall тем, что учитывает ПОРЯДОК: тот же набор попаданий,
    но выше в списке → больше NDCG."""
    gold_set = set(gold)
    ideal_hits = min(len(gold_set), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg_at_k(ranked, gold_set, k) / idcg if idcg else 0.0


def score_ranking(ranked: Sequence[str], gold: Iterable[str], ks: Sequence[int] = (5, 10, 20)) -> dict:
    """Полный набор метрик для одного вопроса. `ks` — точки отсечки."""
    gold_set = set(gold)
    out: dict[str, float] = {"mrr": reciprocal_rank(ranked, gold_set), "n_gold": float(len(gold_set))}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, gold_set, k)
        out[f"precision@{k}"] = precision_at_k(ranked, gold_set, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, gold_set, k)
    return out


def aggregate(per_question: Sequence[dict]) -> dict:
    """Среднее каждой метрики по набору вопросов (макро-усреднение).

    Пропускает вопросы без gold (n_gold==0) для recall/ndcg/mrr, чтобы пустая
    разметка не занижала средние."""
    if not per_question:
        return {}
    scored = [q for q in per_question if q.get("n_gold", 0)]
    if not scored:
        return {"n_questions": float(len(per_question)), "n_scored": 0.0}
    keys = [k for k in scored[0] if k != "n_gold"]
    agg = {k: sum(q.get(k, 0.0) for q in scored) / len(scored) for k in keys}
    agg["n_questions"] = float(len(per_question))
    agg["n_scored"] = float(len(scored))
    return agg
