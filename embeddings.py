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
from typing import Any

import httpx

from lmstudio_config import lmstudio_root_url, normalize_lmstudio_base_url, sanitize_for_log

logger = logging.getLogger("nocturne")

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

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        try:  # пометить embedding-модель для выгрузки при закрытии (ленивый импорт)
            from processor import note_app_loaded_model

            note_app_loaded_model(self.model)
        except Exception:
            pass
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        self._try_load_model()
        r = self._client.post(url, headers=self._headers(), json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = r.text[:500]
            raise RuntimeError(
                f"Embeddings request failed ({r.status_code}). "
                f"Likely no embedding model loaded in LM Studio developer tab "
                f"or selected model is chat-only. Detail: {detail}"
            ) from e
        data = r.json()
        items = data.get("data", [])
        out: list[list[float]] = []
        for item in items:
            emb = item.get("embedding")
            if isinstance(emb, list):
                out.append([float(v) for v in emb])
        if len(out) != len(texts):
            raise RuntimeError(
                f"Embedding response size mismatch: expected {len(texts)} vectors, "
                f"got {len(out)}. Check that the embedding model is loaded correctly."
            )
        return out

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
