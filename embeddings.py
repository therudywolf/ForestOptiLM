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
from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import Any

import httpx

from lmstudio_config import lmstudio_root_url, normalize_lmstudio_base_url, sanitize_for_log

logger = logging.getLogger("nocturne")

_EMB_MAX_RETRIES = 4


def embedding_prefix_scheme(model: str) -> str:
    """Схема префиксов эмбеддера (пишется в индекс; при смене → пересборка)."""
    return "nomic-v1" if "nomic" in (model or "").lower() else "none"


def _task_prefix(model: str, task: str | None) -> str:
    """nomic-embed-text-v1.5 обучен с задачными префиксами; без них recall падает.

    Для документов и запросов — РАЗНЫЕ префиксы; критично использовать их и при
    индексации, и при поиске одинаково, иначе пространства не совпадут.
    """
    if task and "nomic" in (model or "").lower():
        return "search_document: " if task == "document" else "search_query: "
    return ""


def _emb_is_model_loading(text: str) -> bool:
    low = (text or "").lower()
    return any(s in low for s in (
        "failed to load", "operation cancel", "is loading", "currently loading",
        "model is not loaded", "no model loaded",
    ))


def _emb_retry_delay(resp: httpx.Response | None, attempt: int) -> float:
    if resp is not None:
        try:
            ra = resp.headers.get("Retry-After", "").strip()
            if ra:
                return max(0.5, float(ra))
        except Exception:
            pass
    return min(15.0, (2 ** attempt) * (0.6 + random.uniform(0.0, 0.6)))

# (root_url|model), для которых уже инициировали загрузку в этом процессе. Без него
# LM Studio плодит по новому инстансу embedding-модели (text-embedding-...:N) на
# КАЖДЫЙ EmbeddingClient (а он создаётся на каждый чат-запрос и пересборку индекса).
_EMB_LOAD_REQUESTED: set[str] = set()


def _embedding_already_loaded(client: httpx.Client, root_url: str,
                              headers: dict[str, str], model: str) -> bool:
    """Есть ли уже загруженный инстанс модели (чтобы не грузить второй)."""
    try:
        r = client.get(f"{root_url}/api/v1/models", headers=headers, timeout=10.0)
        if r.status_code >= 400:
            return False
        for m in (r.json().get("models") or []):
            mid = str(m.get("key") or m.get("id") or "")
            if mid == model and m.get("loaded_instances"):
                return True
    except Exception:
        return False
    return False


class EmbeddingClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = normalize_lmstudio_base_url(base_url).rstrip("/")
        self.root_url = lmstudio_root_url(base_url)
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)
        self._load_attempted = False

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _try_load_model(self) -> None:
        """Best-effort load for LM Studio REST v1; ignored for other compatible servers.

        Грузим embedding-модель МАКСИМУМ один раз за процесс и только если она ещё
        не загружена — иначе LM Studio плодит по инстансу на каждый запрос (их потом
        приходится выгружать вручную). После первой загрузки /v1/embeddings сам
        переиспользует инстанс (JIT).
        """
        if self._load_attempted or not self.root_url:
            return
        self._load_attempted = True
        key = f"{self.root_url}|{self.model}"
        if key in _EMB_LOAD_REQUESTED:
            return
        try:
            if _embedding_already_loaded(self._client, self.root_url, self._headers(), self.model):
                _EMB_LOAD_REQUESTED.add(key)
                return  # инстанс уже есть — переиспользуем, новый не создаём
            url = f"{self.root_url}/api/v1/models/load"
            r = self._client.post(url, headers=self._headers(), json={"model": self.model})
            _EMB_LOAD_REQUESTED.add(key)
            if r.status_code >= 400:
                logger.info(
                    "Embedding model auto-load skipped/failed: HTTP %s %s",
                    r.status_code,
                    sanitize_for_log(r.text[:220]),
                )
        except Exception as exc:
            logger.info("Embedding model auto-load unavailable: %s", sanitize_for_log(str(exc)))

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        task: str | None = None,
        on_batch: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """task='document' при индексации, 'query' при поиске — для nomic-префиксов.

        on_batch(done, total) — прогресс по обработанным текстам (эмбеддинг 200+
        чанков иначе читается как зависание).
        """
        try:  # пометить embedding-модель для выгрузки при закрытии (ленивый импорт)
            from processor import note_app_loaded_model

            note_app_loaded_model(self.model)
        except Exception:
            pass
        prefix = _task_prefix(self.model, task)
        total = len(texts)
        vectors: list[list[float]] = []
        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            if prefix:
                batch = [prefix + t for t in batch]
            vectors.extend(self._embed_batch(batch))
            if on_batch:
                try:
                    on_batch(min(start + batch_size, total), total)
                except Exception:
                    pass
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        self._try_load_model()
        last_err: str = ""
        for attempt in range(_EMB_MAX_RETRIES):
            try:
                r = self._client.post(url, headers=self._headers(), json=payload)
                # Временные ошибки сервера / «модель ещё грузится» → ждём и повторяем,
                # а не выбрасываем всю сборку индекса из-за одного 500.
                if r.status_code in (500, 502, 503) or (
                    r.status_code == 400 and _emb_is_model_loading(r.text)
                ):
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    delay = _emb_retry_delay(r, attempt)
                    logger.info(
                        "Embeddings transient %s — retry %d/%d in %.1fs",
                        r.status_code, attempt + 1, _EMB_MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                if r.status_code >= 400:
                    # Постоянная 4xx (нет модели/не та модель/плохой payload) — без повторов.
                    raise RuntimeError(
                        f"Embeddings request failed ({r.status_code}). "
                        f"Likely no embedding model loaded in LM Studio developer tab "
                        f"or selected model is chat-only. Detail: {r.text[:500]}"
                    )
                data = r.json()
                out: list[list[float]] = []
                for item in data.get("data", []):
                    emb = item.get("embedding")
                    if isinstance(emb, list):
                        out.append([float(v) for v in emb])
                if len(out) != len(texts):
                    raise RuntimeError(
                        f"Embedding response size mismatch: expected {len(texts)} vectors, "
                        f"got {len(out)}. Check that the embedding model is loaded correctly."
                    )
                return out
            except (httpx.TimeoutException, httpx.HTTPError) as e:
                last_err = f"{type(e).__name__}: {e}"
                delay = _emb_retry_delay(None, attempt)
                logger.info("Embeddings network error — retry %d/%d in %.1fs: %s",
                            attempt + 1, _EMB_MAX_RETRIES, delay, sanitize_for_log(str(e)))
                time.sleep(delay)
        raise RuntimeError(
            f"Embeddings failed after {_EMB_MAX_RETRIES} attempts (последняя ошибка: "
            f"{sanitize_for_log(last_err)}). Сервер эмбеддингов недоступен/перегружен."
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
