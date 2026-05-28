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

import math
import re
from collections import Counter

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
    """Okapi BM25 (k1=1.5, b=0.75) над набором документов с внешними id."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._idf: dict[str, float] = {}
        self._doc_len: list[int] = []
        self._avgdl: float = 0.0

    def fit(self, ids: list[str], texts: list[str]) -> "BM25Index":
        self.ids = list(ids)
        self._tokens = [tokenize(t) for t in texts]
        self._tf = [Counter(toks) for toks in self._tokens]
        self._doc_len = [len(toks) for toks in self._tokens]
        n = len(self._tokens)
        self._avgdl = (sum(self._doc_len) / n) if n else 0.0
        self._df = Counter()
        for toks in self._tokens:
            for term in set(toks):
                self._df[term] += 1
        self._idf = {}
        for term, df in self._df.items():
            # BM25+ idf, всегда положительный
            self._idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        return self

    def search(self, query: str, top_k: int = 8) -> list[tuple[str, float]]:
        if not self.ids or self._avgdl == 0.0:
            return []
        q_terms = [t for t in tokenize(query) if t in self._idf]
        if not q_terms:
            return []
        scores: list[tuple[str, float]] = []
        for i, tf in enumerate(self._tf):
            dl = self._doc_len[i]
            denom_norm = self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                s += self._idf[term] * (f * (self.k1 + 1.0)) / (f + denom_norm)
            if s > 0.0:
                scores.append((self.ids[i], s))
        scores.sort(key=lambda kv: kv[1], reverse=True)
        return scores[:top_k]


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
