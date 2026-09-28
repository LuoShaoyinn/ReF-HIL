"""Hang Double Strings 2 modeling binding."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from shared.task.manipulation_modeling import (
    Modeling as BaseModeling,
    ModelingConfig as BaseModelingConfig,
)
from .config import TaskConfig


@dataclass(kw_only=True)
class ModelingConfig(BaseModelingConfig):
    task: TaskConfig = field(default_factory=TaskConfig)


class Modeling(BaseModeling):
    def reset(self, send_action: object, read_observation: object) -> None:
        """Reset the robot, then reserve task-local time to arrange the string."""

        completed_resets = int(getattr(self, "_completed_resets", 0))
        print("\n[hang_double_strings_2] resetting robot...", flush=True)
        reset_started = time.monotonic()
        super().reset(send_action, read_observation)
        self._completed_resets = completed_resets + 1
        print(
            f"[hang_double_strings_2] robot reset complete in "
            f"{time.monotonic() - reset_started:.1f}s",
            flush=True,
        )
        if completed_resets == 0:
            return

        seconds = max(0, int(self.config.task.human_string_reset_seconds))
        if seconds == 0:
            return
        print(
            "[hang_double_strings_2] arrange the string before the next episode:",
            flush=True,
        )
        for remaining in range(seconds, 0, -1):
            print(
                f"[hang_double_strings_2] human reset: {remaining}s remaining",
                flush=True,
            )
            time.sleep(1.0)
        print("[hang_double_strings_2] human reset complete; starting episode", flush=True)


__all__ = ["Modeling", "ModelingConfig"]
