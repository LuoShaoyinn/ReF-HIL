from __future__ import annotations

import pickle
from pathlib import Path
import tempfile
import unittest

from actor.storage import RawTransitionWriter, RawTransitionWriterConfig


class RawTransitionWriterTest(unittest.TestCase):
    def test_append_uses_numeric_max_beyond_four_digits(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in ('9999.pkl', '10000.pkl'):
                (directory/name).write_bytes(b'preserve')
            writer = RawTransitionWriter(RawTransitionWriterConfig(directory=directory,chunk_size=1))
            writer.add({'new':True})
            self.assertEqual((directory/'10000.pkl').read_bytes(), b'preserve')
            self.assertTrue((directory/'10001.pkl').exists())

    def test_collision_never_overwrites_data(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            writer = RawTransitionWriter(RawTransitionWriterConfig(directory=directory,chunk_size=1))
            (directory/'0000.pkl').write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):
                writer.add({'new':True})
            self.assertEqual((directory/'0000.pkl').read_bytes(), b'preserve')

    def test_writes_exact_payload_and_continues_chunk_index(self) -> None:
        with tempfile.TemporaryDirectory(prefix="srt-actor-writer-") as temp:
            directory = Path(temp)
            existing = [{"existing": True}]
            with (directory / "0000.pkl").open("wb") as handle:
                pickle.dump(existing, handle)
            payload = {"continuous_gripper": 0.25}
            writer = RawTransitionWriter(
                RawTransitionWriterConfig(directory=directory, chunk_size=1)
            )
            writer.add(payload)
            with (directory / "0001.pkl").open("rb") as handle:
                self.assertEqual(pickle.load(handle), [payload])


if __name__ == "__main__":
    unittest.main()
