# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestProjectFoss(unittest.TestCase):
    def test_license_file_present(self) -> None:
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("GNU AFFERO GENERAL PUBLIC LICENSE", text)

    def test_notice_and_third_party(self) -> None:
        self.assertTrue((ROOT / "NOTICE").is_file())
        tp = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        self.assertIn("httpx", tp)
        self.assertIn("faiss", tp.lower())

    def test_readme_agpl_and_concept(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("AGPL", readme)
        self.assertIn("Map-Reduce", readme)
        self.assertIn("forestoptilm", readme.lower())

    def test_pyproject_agpl_classifier(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("AGPL", pyproject)

    def test_spdx_on_python_sources(self) -> None:
        py_files = [
            p for p in ROOT.rglob("*.py")
            if ".venv" not in p.parts and "__pycache__" not in p.parts
        ]
        self.assertGreater(len(py_files), 20)
        missing = [
            p.relative_to(ROOT)
            for p in py_files
            if "SPDX-License-Identifier" not in p.read_text(encoding="utf-8", errors="replace")[:500]
        ]
        self.assertEqual(missing, [], f"Missing SPDX header: {missing[:5]}")


class TestLmstudioUrlValidation(unittest.TestCase):
    def test_validate_url(self) -> None:
        from lmstudio_config import validate_lmstudio_url

        self.assertFalse(validate_lmstudio_url("")[0])
        self.assertFalse(validate_lmstudio_url("ftp://x")[0])
        self.assertTrue(validate_lmstudio_url("http://127.0.0.1:1234")[0])


if __name__ == "__main__":
    unittest.main()
