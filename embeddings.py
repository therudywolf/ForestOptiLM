from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger("nocturne")


class EmbeddingClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
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
