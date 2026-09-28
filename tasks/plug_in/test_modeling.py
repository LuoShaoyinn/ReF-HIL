from __future__ import annotations

import unittest
from unittest.mock import call, patch

import numpy as np

from shared.task.base import BaseModeling
from .modeling import Modeling, ModelingConfig


class PlugInResetTest(unittest.TestCase):
    def test_reset_waits_three_seconds_after_motion(self) -> None:
        modeling = object.__new__(Modeling)
        BaseModeling.__init__(modeling, config=ModelingConfig(device="cpu"))
        task = modeling.config.task

        with (
            patch(
                "shared.task.manipulation_modeling.np.random.uniform",
                return_value=np.zeros(3, dtype=np.float32),
            ),
            patch("shared.task.manipulation_modeling.time.sleep") as sleep,
        ):
            modeling.reset(
                lambda _action: None,
                lambda: {
                    "tcp_pose": np.asarray(
                        [task.reset_pos, task.zero_point_rot], dtype=np.float32
                    )
                },
            )

        self.assertEqual(
            sleep.call_args_list,
            [call(task.reset_lift_duration_s), call(3.0)],
        )


if __name__ == "__main__":
    unittest.main()
