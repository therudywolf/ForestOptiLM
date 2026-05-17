# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import unittest

from chunk_store import ChunkStore


class TestChunkStore(unittest.TestCase):
    def test_spill_to_disk(self) -> None:
        store = ChunkStore("testjob", max_in_ram=3)
        try:
            for i in range(8):
                store.append(f"chunk-{i}-" + ("x" * 40))
            self.assertEqual(len(store), 8)
            self.assertTrue(store._spilled)
            self.assertEqual(store[0], "chunk-0-" + ("x" * 40))
            self.assertEqual(store[7][:8], "chunk-7-")
        finally:
            store.cleanup()

    def test_iterate(self) -> None:
        store = ChunkStore("iterjob", max_in_ram=10)
        try:
            store.extend(["a", "b", "c"])
            self.assertEqual(list(store), ["a", "b", "c"])
        finally:
            store.cleanup()


if __name__ == "__main__":
    unittest.main()
