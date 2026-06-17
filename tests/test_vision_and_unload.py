# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vision message build + capability fallback, and native model-unload HTTP path."""
from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from processor import (
    _build_vision_map_messages,
    _extract_loaded_instance_ids,
    _try_unload_model,
    check_vision_capability,
)

# 1x1 transparent PNG
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class TestVisionMessages(unittest.TestCase):
    def test_build_vision_map_messages_shape(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "img.png"
            p.write_bytes(_PNG)
            msgs = _build_vision_map_messages(
                "Опиши кадр", 7, "frame_007.png", p, language_hint="Отвечай по-русски.",
            )
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        content = msgs[1]["content"]
        text = next(c["text"] for c in content if c["type"] == "text")
        img = next(c["image_url"]["url"] for c in content if c["type"] == "image_url")
        self.assertIn("7", text)                      # chunk_index
        self.assertIn("frame_007.png", text)          # file label
        self.assertTrue(img.startswith("data:image/png;base64,"))

    def test_build_vision_custom_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.png"
            p.write_bytes(_PNG)
            msgs = _build_vision_map_messages("q", 0, "x.png", p, system_prompt="SYS")
        self.assertEqual(msgs[0]["content"], "SYS")


def _resp(status: int) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.text = f"body{status}"
    r.raise_for_status = MagicMock()
    return r


class TestVisionCapability(unittest.TestCase):
    @patch("processor.httpx.Client")
    def test_ok_first_try(self, client_cls: MagicMock) -> None:
        client = client_cls.return_value.__enter__.return_value
        client.post.return_value = _resp(200)
        ok, msg = check_vision_capability("http://127.0.0.1:1234", "k", "vl-model")
        self.assertTrue(ok, msg)
        self.assertEqual(client.post.call_count, 1)

    @patch("processor.httpx.Client")
    def test_fallback_on_400_then_200(self, client_cls: MagicMock) -> None:
        client = client_cls.return_value.__enter__.return_value
        client.post.side_effect = [_resp(400), _resp(200)]
        ok, msg = check_vision_capability("http://127.0.0.1:1234", "k", "vl-model")
        self.assertTrue(ok, msg)
        self.assertEqual(client.post.call_count, 2)  # alternate transport tried

    @patch("processor.httpx.Client")
    def test_both_transports_fail(self, client_cls: MagicMock) -> None:
        client = client_cls.return_value.__enter__.return_value
        client.post.side_effect = [_resp(400), _resp(400)]
        ok, msg = check_vision_capability("http://127.0.0.1:1234", "k", "vl-model")
        self.assertFalse(ok)
        self.assertIn("400", msg)


class TestNativeUnload(unittest.TestCase):
    _CATALOG = [{"key": "m1", "loaded_instances": [{"id": "a"}, {"id": "b"}]}]

    @patch("processor.fetch_models_catalog")
    def test_extract_loaded_instance_ids(self, fmc: MagicMock) -> None:
        fmc.return_value = self._CATALOG
        self.assertEqual(_extract_loaded_instance_ids("u", "k", "m1"), ["a", "b"])
        self.assertEqual(_extract_loaded_instance_ids("u", "k", "absent"), [])

    @patch("processor.fetch_models_catalog")
    @patch("processor.httpx.Client")
    def test_try_unload_posts_each_instance(self, client_cls: MagicMock, fmc: MagicMock) -> None:
        fmc.return_value = self._CATALOG
        client = client_cls.return_value.__enter__.return_value
        client.post.return_value = _resp(200)
        ok = _try_unload_model("http://127.0.0.1:1234", "k", "m1")
        self.assertTrue(ok)
        self.assertEqual(client.post.call_count, 2)
        sent = [c.kwargs["json"] for c in client.post.call_args_list]
        self.assertIn({"instance_id": "a"}, sent)
        self.assertIn({"instance_id": "b"}, sent)

    @patch("processor.fetch_models_catalog")
    @patch("processor.httpx.Client")
    def test_try_unload_no_instances_is_ok(self, client_cls: MagicMock, fmc: MagicMock) -> None:
        fmc.return_value = [{"key": "m1", "loaded_instances": []}]
        ok = _try_unload_model("http://127.0.0.1:1234", "k", "m1")
        self.assertTrue(ok)
        client_cls.return_value.__enter__.return_value.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
