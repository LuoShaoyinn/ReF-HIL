from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

from shared.zmq import DictMessage


@dataclass(kw_only=True)
class RawTransitionWriterConfig:
    directory: Path
    chunk_size: int = 128


class RawTransitionWriter:
    """Append protocol transitions to canonical pickle chunks."""

    def __init__(self, config: RawTransitionWriterConfig) -> None:
        self.config = config
        self.config.directory.mkdir(parents=True, exist_ok=True)
        paths = sorted(self.config.directory.glob("*.pkl"))
        self.file_index = max((int(path.stem) for path in paths), default=-1) + 1
        self.pending: list[DictMessage] = []

    def add(self, transition: DictMessage) -> None:
        self.pending.append(transition)
        if len(self.pending) >= int(self.config.chunk_size):
            self.save()

    def save(self) -> None:
        if not self.pending:
            return
        path = self.config.directory / f"{self.file_index:04d}.pkl"
        print(f"Autosaving {len(self.pending)} transitions to {path}...")
        # A second writer or a late file must never overwrite recorded data.
        with path.open("xb") as handle:
            pickle.dump(self.pending, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.pending = []
        self.file_index += 1
