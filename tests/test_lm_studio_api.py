# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from lm_studio_api import (
    V1_MODELS,
    V1_MODELS_DOWNLOAD,
    V1_MODELS_DOWNLOAD_STATUS,
    get_model_download_status,
    is_lm_studio_api_token,
    start_model_download,
    v1_url,
)


class TestLmStudioApi(unittest.TestCase):
    def test_v1_url(self) -> None:
        self.assertEqual(v1_url("http://127.0.0.1:1234", V1_MODELS), "http://127.0.0.1:1234/api/v1/models")

    def test_token_detect(self) -> None:
        self.assertTrue(is_lm_studio_api_token("sk-lm-abc:def"))
        self.assertFalse(is_lm_studio_api_token("forest"))

    def test_download_constants(self) -> None:
        self.assertEqual(V1_MODELS_DOWNLOAD, "/api/v1/models/download")
        self.assertEqual(V1_MODELS_DOWNLOAD_STATUS, "/api/v1/models/download/status")

    @patch("lm_studio_api.httpx.Client")
    def test_start_download_ok(self, client_cls: MagicMock) -> None:
        client = client_cls.return_value.__enter__.return_value
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b"{}"
        resp.json.return_value = {"job_id": "job-123"}
        client.post.return_value = resp

        out = start_model_download("http://127.0.0.1:1234", "k", "some/model")
        self.assertEqual(out["job_id"], "job-123")
        # endpoint and payload
        args, kwargs = client.post.call_args
        self.assertEqual(args[0], "http://127.0.0.1:1234/api/v1/models/download")
        self.assertEqual(kwargs["json"], {"model": "some/model"})

    @patch("lm_studio_api.httpx.Client")
    def test_start_download_http_error(self, client_cls: MagicMock) -> None:
        client = client_cls.return_value.__enter__.return_value
        resp = MagicMock()
        resp.status_code = 404
        resp.text = "not found"
        client.post.return_value = resp

        out = start_model_download("http://127.0.0.1:1234", "k", "x")
        self.assertIn("error", out)
        self.assertIn("404", out["error"])

    @patch("lm_studio_api.httpx.Client")
    def test_status_url_includes_job_id(self, client_cls: MagicMock) -> None:
        client = client_cls.return_value.__enter__.return_value
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b"{}"
        resp.json.return_value = {"status": "downloading", "progress": 0.5}
        client.get.return_value = resp

        out = get_model_download_status("http://127.0.0.1:1234", "k", "job-9")
        self.assertEqual(out["status"], "downloading")
        args, _ = client.get.call_args
        self.assertEqual(args[0], "http://127.0.0.1:1234/api/v1/models/download/status/job-9")

    @patch("lm_studio_api.httpx.Client", side_effect=RuntimeError("conn refused"))
    def test_status_exception_is_caught(self, _cls: MagicMock) -> None:
        out = get_model_download_status("http://127.0.0.1:1234", "k", "j")
        self.assertIn("error", out)


if __name__ == "__main__":
    unittest.main()
