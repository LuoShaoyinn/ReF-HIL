from __future__ import annotations

import copy
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
import os
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class _NumpyTensor:
    """Tensor-free IPC envelope; never stored in the published checkpoint."""

    data: np.ndarray
    dtype: torch.dtype
    shape: tuple[int, ...]


def _pack_tensors(value: Any) -> Any:
    """Avoid PyTorch's per-storage file descriptors in multiprocessing queues.

    Raw bytes preserve all dense optimizer dtypes, including bfloat16, which
    NumPy cannot represent directly. The writer reconstructs ordinary CPU
    tensors before pickling, so existing readers and optimizer loading do not
    need a new checkpoint format.
    """
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError(
                "checkpoint transport requires dense, non-quantized tensors"
            )
        tensor = value.detach().cpu().resolve_conj().resolve_neg().contiguous()
        data = tensor.reshape(-1).view(torch.uint8).numpy().copy()
        return _NumpyTensor(data=data, dtype=tensor.dtype, shape=tuple(tensor.shape))
    if isinstance(value, dict):
        return {_pack_tensors(key): _pack_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_pack_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pack_tensors(item) for item in value)
    # Network NumPy arrays were already isolated by immutable_cpu_copy.
    return value


def _unpack_tensors(value: Any) -> Any:
    if isinstance(value, _NumpyTensor):
        if value.data.size == 0:
            return torch.empty(value.shape, dtype=value.dtype)
        return torch.from_numpy(value.data).view(value.dtype).reshape(value.shape)
    if isinstance(value, dict):
        return {
            _unpack_tensors(key): _unpack_tensors(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_unpack_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_unpack_tensors(item) for item in value)
    return value


def _write_transferred_checkpoint(path_string: str, state: dict[str, Any]) -> None:
    write_checkpoint_atomic(path_string, _unpack_tensors(state))


def immutable_cpu_copy(value: Any) -> Any:
    """Detach checkpoint state from live learner tensors before async I/O."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {
            immutable_cpu_copy(key): immutable_cpu_copy(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [immutable_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(immutable_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def write_checkpoint_atomic(path_string: str, state: dict[str, Any]) -> None:
    """Write one pickle and publish it only after the payload is complete."""

    path = Path(path_string)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CheckpointWriter:
    """Single-worker writer; periodic saves never queue stale snapshots."""

    def __init__(self, *, asynchronous: bool) -> None:
        self.asynchronous = bool(asynchronous)
        self._executor: ProcessPoolExecutor | None = None
        self._pending: Future[None] | None = None

    def _ensure_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
            )
        return self._executor

    def _finish_completed(self) -> None:
        if self._pending is not None and self._pending.done():
            self._pending.result()
            self._pending = None

    def available(self) -> bool:
        self._finish_completed()
        return self._pending is None

    def submit(
        self,
        path: Path,
        state: dict[str, Any],
        *,
        wait: bool,
    ) -> bool:
        self._finish_completed()
        if not self.asynchronous:
            write_checkpoint_atomic(str(path), state)
            return True
        if self._pending is not None:
            if not wait:
                return False
            self._pending.result()
            self._pending = None
        transferred_state = _pack_tensors(state)
        self._pending = self._ensure_executor().submit(
            _write_transferred_checkpoint,
            str(path),
            transferred_state,
        )
        if wait:
            self._pending.result()
            self._pending = None
        return True

    def close(self) -> None:
        try:
            if self._pending is not None:
                self._pending.result()
        finally:
            self._pending = None
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
