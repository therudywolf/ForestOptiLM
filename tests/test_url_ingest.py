# SPDX-License-Identifier: AGPL-3.0-or-later
"""URL ingestion: validation, HTML→text, mocked fetch (no network)."""
from __future__ import annotations

import unittest
from unittest import mock

import url_ingest as ui


class _FakeStream:
    def __init__(self, status: int, headers: dict[str, str], body: bytes,
                 encoding: str | None = "utf-8") -> None:
        self.status_code = status
        self.headers = headers
        self._body = body
        self.encoding = encoding

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *a) -> bool:
        return False

    def iter_bytes(self):
        # Бьём тело на пару кусков, чтобы проверить аккумуляцию.
        mid = max(1, len(self._body) // 2)
        yield self._body[:mid]
        yield self._body[mid:]


class _FakeClient:
    def __init__(self, stream: _FakeStream) -> None:
        self._stream = stream

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *a) -> bool:
        return False

    def stream(self, method: str, url: str) -> _FakeStream:
        return self._stream


def _patch_client(stream: _FakeStream):
    return mock.patch.object(ui.httpx, "Client", lambda **kw: _FakeClient(stream))


class TestValidation(unittest.TestCase):
    def test_adds_scheme(self) -> None:
        self.assertEqual(ui._validate_url("example.com/x"), "https://example.com/x")

    def test_rejects_non_http(self) -> None:
        with self.assertRaises(ui.UrlIngestError):
            ui._validate_url("file:///etc/passwd")
        with self.assertRaises(ui.UrlIngestError):
            ui._validate_url("")


class TestHtmlToText(unittest.TestCase):
    def test_extracts_title_strips_scripts(self) -> None:
        html = ("<html><head><title>Заголовок</title></head>"
                "<body><script>evil()</script><h1>Шапка</h1><p>Текст абзаца</p></body></html>")
        title, text = ui._html_to_text(html)
        self.assertEqual(title, "Заголовок")
        self.assertIn("Текст абзаца", text)
        self.assertNotIn("evil()", text)

    def test_title_fallback_h1(self) -> None:
        title, _ = ui._html_to_text("<html><body><h1>Только H1</h1><p>x</p></body></html>")
        self.assertEqual(title, "Только H1")


class TestFetchUrl(unittest.TestCase):
    def test_fetch_html(self) -> None:
        body = ("<html><head><title>T</title></head><body><p>Привет мир</p></body></html>"
                ).encode("utf-8")
        stream = _FakeStream(200, {"content-type": "text/html; charset=utf-8"}, body)
        with _patch_client(stream):
            doc = ui.fetch_url("https://example.com")
        self.assertEqual(doc.title, "T")
        self.assertIn("Привет мир", doc.text)
        self.assertEqual(doc.url, "https://example.com")

    def test_fetch_plain_text(self) -> None:
        stream = _FakeStream(200, {"content-type": "text/plain"}, "линия один\nлиния два".encode("utf-8"))
        with _patch_client(stream):
            doc = ui.fetch_url("https://example.com/raw.txt")
        self.assertIn("линия один", doc.text)
        self.assertTrue(doc.title)  # заголовок из первой строки

    def test_http_error(self) -> None:
        stream = _FakeStream(404, {}, b"nope")
        with _patch_client(stream):
            with self.assertRaises(ui.UrlIngestError):
                ui.fetch_url("https://example.com/missing")

    def test_size_guard(self) -> None:
        stream = _FakeStream(200, {"content-type": "text/plain"}, b"x" * 5000)
        with _patch_client(stream):
            with self.assertRaises(ui.UrlIngestError):
                ui.fetch_url("https://example.com/big", max_bytes=1000)

    def test_empty_body_errors(self) -> None:
        stream = _FakeStream(200, {"content-type": "text/html"}, b"<html><body></body></html>")
        with _patch_client(stream):
            with self.assertRaises(ui.UrlIngestError):
                ui.fetch_url("https://example.com/empty")


if __name__ == "__main__":
    unittest.main()
