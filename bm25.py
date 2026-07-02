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
Lexical BM25 index + Reciprocal Rank Fusion (Столп 4, гибридный поиск).

Чистый Python, без внешних зависимостей. Нужен потому, что эмбеддинги плохо
находят ТОЧНЫЕ токены — `CVE-2024-3094`, `log4j`, имя хоста, `lodash@4.17.20` —
а аналитик безопасности ищет именно их. BM25 ловит точные совпадения, вектор —
семантику; RRF объединяет оба ранжирования.

Токенизатор сохраняет CVE/пакеты/хосты как единые токены (не режет по дефисам),
поддерживает кириллицу и латиницу.
"""
from __future__ import annotations

import logging
import math
import os
import pickle
import re
import uuid
from array import array
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger("nocturne")

# Версия формата дискового кэша BM25 — поднять при несовместимом изменении полей.
_CACHE_VERSION = 1

# Токен: начинается с буквы/цифры (lat/cyr), допускает . - _ / @ внутри —
# чтобы CVE-2024-3094, lodash@4.17.20, app.example.com оставались целыми.
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё][0-9A-Za-zА-Яа-яЁё._\-/@]*")

_STOP = frozenset({
    "the", "and", "for", "with", "from", "this", "that",
    "по", "на", "из", "при", "или", "как", "это", "для",
})


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for m in _TOKEN_RE.findall((text or "").lower()):
        if len(m) < 2 or m in _STOP:
            continue
        out.append(m)
    return out


class BM25Index:
    """Okapi BM25 (k1=1.5, b=0.75) над набором документов с внешними id.

    Внутри — инвертированный индекс (term → постинг-лист (doc_idx, tf)) и
    numpy-скоринг: при поиске оцениваются только документы, содержащие термины
    запроса, а не весь корпус. На 455k чанков это ~0.38s → ~0.01s на запрос.
    Поддерживает дисковый кэш (save/load), чтобы не пересчитывать fit
    (десятки секунд токенизации) при каждом холодном старте.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self._idf: dict[str, float] = {}
        # постинги: term → (массив doc_idx, массив tf); array('i') компактен
        self._post_docs: dict[str, array] = {}
        self._post_tf: dict[str, array] = {}
        self._denom_norm: np.ndarray | None = None  # k1*(1-b+b*dl/avgdl), float32 [n]
        self._avgdl: float = 0.0

    def fit(self, ids: list[str], texts: list[str]) -> "BM25Index":
        self.ids = list(ids)
        n = len(texts)
        doc_len = np.zeros(n, dtype=np.float32)
        post_docs: dict[str, array] = {}
        post_tf: dict[str, array] = {}
        for i, t in enumerate(texts):
            toks = tokenize(t)
            doc_len[i] = len(toks)
            for term, f in Counter(toks).items():
                d = post_docs.get(term)
                if d is None:
                    d = post_docs[term] = array("i")
                    post_tf[term] = array("i")
                d.append(i)
                post_tf[term].append(f)
        self._post_docs = post_docs
        self._post_tf = post_tf
        self._avgdl = float(doc_len.sum() / n) if n else 0.0
        if self._avgdl > 0.0:
            self._denom_norm = self.k1 * (1.0 - self.b + self.b * doc_len / self._avgdl)
        else:
            self._denom_norm = None
        self._idf = {}
        for term, docs in post_docs.items():
            df = len(docs)
            # BM25+ idf, всегда положительный
            self._idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        return self

    def search(self, query: str, top_k: int = 8) -> list[tuple[str, float]]:
        if not self.ids or self._avgdl == 0.0 or self._denom_norm is None:
            return []
        q_terms = [t for t in tokenize(query) if t in self._idf]
        if not q_terms:
            return []
        scores = np.zeros(len(self.ids), dtype=np.float32)
        k1p1 = self.k1 + 1.0
        for term in q_terms:
            docs = np.frombuffer(self._post_docs[term], dtype=np.intc)
            f = np.frombuffer(self._post_tf[term], dtype=np.intc).astype(np.float32)
            scores[docs] += self._idf[term] * (f * k1p1) / (f + self._denom_norm[docs])
        k = min(top_k, len(scores))
        # argpartition даёт top-k за O(n); сортируем только k элементов
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self.ids[i], float(scores[i])) for i in top_idx if scores[i] > 0.0]

    # --- дисковый кэш --------------------------------------------------- #
    def save(self, path: Path, signature: tuple = ()) -> None:
        """Сохранить построенный индекс на диск (pickle), с сигнатурой корпуса.

        Ошибка записи (диск полон, права) не критична — просто следующий
        холодный старт заплатит за fit заново.
        """
        try:
            payload = {
                "version": _CACHE_VERSION,
                "signature": tuple(signature),
                "k1": self.k1, "b": self.b,
                "ids": self.ids, "idf": self._idf,
                "post_docs": self._post_docs, "post_tf": self._post_tf,
                "denom_norm": self._denom_norm, "avgdl": self._avgdl,
            }
            p = Path(path)
            # Уникальный tmp (pid+uuid): конкурентные писатели (build/append на
            # отдельном инстансе, второй процесс рядом с exe) не топчут общий tmp.
            tmp = p.with_name(f"{p.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            try:
                with tmp.open("wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, p)
            finally:
                try:
                    tmp.unlink()  # если replace не забрал tmp — не оставляем мусор
                except OSError:
                    pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("bm25: не удалось сохранить кэш %s — %s", path, exc)

    @classmethod
    def load(cls, path: Path, signature: tuple = ()) -> "BM25Index | None":
        """Загрузить кэш, если он существует и сигнатура корпуса совпадает."""
        try:
            with Path(path).open("rb") as f:
                payload = pickle.load(f)
            if payload.get("version") != _CACHE_VERSION:
                return None
            if tuple(payload.get("signature") or ()) != tuple(signature):
                return None
            bm = cls(k1=payload["k1"], b=payload["b"])
            bm.ids = payload["ids"]
            bm._idf = payload["idf"]
            bm._post_docs = payload["post_docs"]
            bm._post_tf = payload["post_tf"]
            bm._denom_norm = payload["denom_norm"]
            bm._avgdl = payload["avgdl"]
            return bm
        except FileNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("bm25: кэш %s не читается (%s) — пересборка", path, exc)
            return None


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """
    Слить несколько ранжирований (списков id по убыванию релевантности) в одно.
    RRF: score(d) = Σ 1 / (k + rank_i(d)). Устойчив к разным шкалам скоров.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    out = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return out[:top_k] if top_k else out


def fuse_rankings(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
    top_k: int | None = None,
    score_weight: float = 1.0,
) -> list[tuple[str, float]]:
    """
    Score-aware слияние ранжирований: RRF-ранг + нормированный скор.

    Чистый RRF на больших корпусах вырождается: списки вектора и BM25 почти не
    пересекаются, все кандидаты получают ≈1/(k+1) и порядок внутри «связки»
    случаен — косинус 0.9 неотличим от 0.55. Здесь каждый документ получает
    score(d) = Σ_i [ 1/(k + rank_i) + score_weight × s_i/max_i ],
    где s_i/max_i — скор, нормированный на максимум своего списка. Ранговая
    часть сохраняет устойчивость RRF к шкалам, скоровая — различает сильные и
    слабые совпадения внутри одного списка и между списками.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        if not ranking:
            continue
        max_s = max(s for _, s in ranking)
        for rank, (doc_id, s) in enumerate(ranking):
            rrf = 1.0 / (k + rank + 1)
            norm = (s / max_s) if max_s > 0 else 0.0
            fused[doc_id] = fused.get(doc_id, 0.0) + rrf + score_weight * norm
    out = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return out[:top_k] if top_k else out
