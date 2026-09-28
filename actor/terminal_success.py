"""Non-blocking terminal control for marking a physical rollout successful."""

from __future__ import annotations

import os
import select
import sys
import termios
import tty
from typing import Self


class TerminalSuccessKey:
    """Report Enter presses without blocking the actor control loop."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._fd: int | None = None
        self._previous_settings: list | None = None
        self._failure_pending = False
        self._success_pending = False

    def __enter__(self) -> Self:
        if not self.enabled:
            return self
        if not sys.stdin.isatty():
            print(
                "[ACTOR] Enter success override disabled: stdin is not a TTY",
                flush=True,
            )
            return self
        self._fd = sys.stdin.fileno()
        self._previous_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        termios.tcflush(self._fd, termios.TCIFLUSH)
        print(
            "[ACTOR] Enter: success; f: fail and terminate the current episode",
            flush=True,
        )
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self._fd is not None and self._previous_settings is not None:
            termios.tcsetattr(
                self._fd,
                termios.TCSADRAIN,
                self._previous_settings,
            )
        self._fd = None
        self._previous_settings = None

    def requested(self) -> bool:
        if self._fd is None:
            return False
        requested = False
        while select.select([self._fd], [], [], 0.0)[0]:
            chunk = os.read(self._fd, 4096)
            if not chunk:
                break
            requested = (b"\r" in chunk or b"\n" in chunk) or requested
            self._failure_pending |= b"f" in chunk.lower()
        return requested

    def failure_requested(self) -> bool:
        # Poll through requested(), retaining Enter for the next success poll.
        new_success = self.requested()
        self._success_pending = self._success_pending or new_success
        result = self._failure_pending
        self._failure_pending = False
        return result

    def success_requested(self) -> bool:
        result = self.requested() or self._success_pending
        self._success_pending = False
        return result

    def drain(self) -> None:
        """Discard keys queued before the next episode begins."""
        self._failure_pending = False
        self._success_pending = False
        if self._fd is not None:
            termios.tcflush(self._fd, termios.TCIFLUSH)
