# SPDX-License-Identifier: AGPL-3.0-or-later
"""Security tests for archive extraction: path traversal, decompression bombs."""
from __future__ import annotations

import gzip
import io
import os
import tarfile
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from file_extractors import (
    ParseError,
    _extract_archive_to_dir,
    _is_within,
    extract_content,
)


class TestIsWithin(unittest.TestCase):
    def test_nested_path_is_within(self) -> None:
        with TemporaryDirectory() as td:
            base = Path(td)
            self.assertTrue(_is_within(base, base / "sub" / "file.txt"))
            self.assertTrue(_is_within(base, base))

    def test_sibling_prefix_is_not_within(self) -> None:
        # /tmp/foobar must NOT count as inside /tmp/foo (prefix-match bug).
        with TemporaryDirectory() as td:
            root = Path(td)
            foo = root / "foo"
            foobar = root / "foobar"
            foo.mkdir()
            foobar.mkdir()
            self.assertFalse(_is_within(foo, foobar / "evil.txt"))

    def test_parent_escape_is_not_within(self) -> None:
        with TemporaryDirectory() as td:
            base = Path(td) / "dest"
            base.mkdir()
            self.assertFalse(_is_within(base, base / ".." / "evil.txt"))


class TestZipPathTraversal(unittest.TestCase):
    def test_zip_member_outside_destination_rejected(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "evil.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("good.txt", "safe content\n")
                zf.writestr("../../evil.txt", "pwned\n")

            dest = root / "dest"
            dest.mkdir()
            _extract_archive_to_dir(archive, dest)

            # Safe member extracted, traversal member skipped.
            self.assertTrue((dest / "good.txt").is_file())
            self.assertFalse((root / "evil.txt").exists())
            self.assertFalse((root.parent / "evil.txt").exists())


class TestTarLinkEscape(unittest.TestCase):
    def test_tar_symlink_member_skipped(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "evil.tar"
            with tarfile.open(archive, "w") as tf:
                # Regular safe file.
                data = b"safe content\n"
                info = tarfile.TarInfo(name="good.txt")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
                # Symlink pointing outside the destination.
                link = tarfile.TarInfo(name="escape")
                link.type = tarfile.SYMTYPE
                link.linkname = "../../../etc/passwd"
                tf.addfile(link)

            dest = root / "dest"
            dest.mkdir()
            _extract_archive_to_dir(archive, dest)

            self.assertTrue((dest / "good.txt").is_file())
            # Symlink member must not have been materialized.
            self.assertFalse((dest / "escape").exists())

    def test_tar_member_with_absolute_name_skipped(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "abs.tar"
            with tarfile.open(archive, "w") as tf:
                data = b"x\n"
                info = tarfile.TarInfo(name="/abs_evil.txt")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

            dest = root / "dest"
            dest.mkdir()
            _extract_archive_to_dir(archive, dest)
            self.assertFalse((dest / "abs_evil.txt").exists())


class TestDecompressionBomb(unittest.TestCase):
    def test_zip_over_uncompressed_limit_raises(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "bomb.zip"
            # Highly compressible payload — small on disk, large uncompressed.
            payload = b"0" * (256 * 1024)
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("big.txt", payload)

            dest = root / "dest"
            dest.mkdir()
            old = os.environ.get("NOCTURNE_MAX_UNCOMPRESSED_BYTES")
            os.environ["NOCTURNE_MAX_UNCOMPRESSED_BYTES"] = "1024"
            try:
                with self.assertRaises(ParseError):
                    _extract_archive_to_dir(archive, dest)
            finally:
                if old is None:
                    os.environ.pop("NOCTURNE_MAX_UNCOMPRESSED_BYTES", None)
                else:
                    os.environ["NOCTURNE_MAX_UNCOMPRESSED_BYTES"] = old

    def test_tar_over_uncompressed_limit_raises(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "bomb.tar"
            payload = b"0" * (256 * 1024)
            with tarfile.open(archive, "w") as tf:
                info = tarfile.TarInfo(name="big.txt")
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))

            dest = root / "dest"
            dest.mkdir()
            old = os.environ.get("NOCTURNE_MAX_UNCOMPRESSED_BYTES")
            os.environ["NOCTURNE_MAX_UNCOMPRESSED_BYTES"] = "1024"
            try:
                with self.assertRaises(ParseError):
                    _extract_archive_to_dir(archive, dest)
            finally:
                if old is None:
                    os.environ.pop("NOCTURNE_MAX_UNCOMPRESSED_BYTES", None)
                else:
                    os.environ["NOCTURNE_MAX_UNCOMPRESSED_BYTES"] = old

    def test_gz_over_uncompressed_limit_raises(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "bomb.gz"
            with gzip.open(archive, "wb") as gz:
                gz.write(b"0" * (256 * 1024))

            dest = root / "dest"
            dest.mkdir()
            old = os.environ.get("NOCTURNE_MAX_UNCOMPRESSED_BYTES")
            os.environ["NOCTURNE_MAX_UNCOMPRESSED_BYTES"] = "1024"
            try:
                with self.assertRaises(ParseError):
                    _extract_archive_to_dir(archive, dest)
            finally:
                if old is None:
                    os.environ.pop("NOCTURNE_MAX_UNCOMPRESSED_BYTES", None)
                else:
                    os.environ["NOCTURNE_MAX_UNCOMPRESSED_BYTES"] = old

    def test_within_limit_extracts_ok(self) -> None:
        with TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "ok.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("small.txt", "hi\n")

            dest = root / "dest"
            dest.mkdir()
            old = os.environ.get("NOCTURNE_MAX_UNCOMPRESSED_BYTES")
            os.environ["NOCTURNE_MAX_UNCOMPRESSED_BYTES"] = "1048576"
            try:
                _extract_archive_to_dir(archive, dest)
                self.assertTrue((dest / "small.txt").is_file())
            finally:
                if old is None:
                    os.environ.pop("NOCTURNE_MAX_UNCOMPRESSED_BYTES", None)
                else:
                    os.environ["NOCTURNE_MAX_UNCOMPRESSED_BYTES"] = old


class TestXmlConsistency(unittest.TestCase):
    def test_small_xml_extracted_as_text(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "config.xml"
            xml = "<root><item>hello world</item></root>\n"
            p.write_text(xml, encoding="utf-8")

            kind, content = extract_content(p)
            self.assertEqual(kind, "text")
            self.assertIn("hello world", str(content))


if __name__ == "__main__":
    unittest.main()
