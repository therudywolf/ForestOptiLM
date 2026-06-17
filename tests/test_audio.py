# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audio source ingestion: registration + graceful degradation without faster-whisper."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import audio_transcribe
from chunking import build_document_chunks
from file_extractors import TEXT_EXTRACTORS


class TestAudio(unittest.TestCase):
    def test_audio_extensions_registered(self) -> None:
        for ext in (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".opus"):
            self.assertIn(ext, TEXT_EXTRACTORS)
        self.assertIn(".mp3", audio_transcribe.AUDIO_EXTENSIONS)

    def test_audio_in_indexable_suffixes(self) -> None:
        from pipeline import _ALLOWED_SUFFIXES
        self.assertIn(".mp3", _ALLOWED_SUFFIXES)
        self.assertIn(".wav", _ALLOWED_SUFFIXES)

    def test_graceful_without_whisper(self) -> None:
        if audio_transcribe.is_available():
            self.skipTest("faster-whisper is installed — graceful-skip path not exercised")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "voice.mp3"
            p.write_bytes(b"\x00\x01\x02fake-audio-bytes")
            # build_document_chunks swallows ParseError → no chunks, no crash
            self.assertEqual(build_document_chunks(p, 2000), [])

    def test_transcribe_raises_clear_error_without_whisper(self) -> None:
        if audio_transcribe.is_available():
            self.skipTest("faster-whisper is installed")
        with self.assertRaises(RuntimeError) as ctx:
            audio_transcribe.transcribe("x.mp3")
        self.assertIn("faster-whisper", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
