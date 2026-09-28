from __future__ import annotations

import sys
from types import ModuleType
import unittest

import numpy as np

sys.modules.setdefault("rtde_control", ModuleType("rtde_control"))
sys.modules.setdefault("rtde_receive", ModuleType("rtde_receive"))
sys.modules.setdefault("pyrealsense2", ModuleType("pyrealsense2"))

from actor.robot.ur5e.ur_impedance_control import (  # noqa: E402
    URImpedanceControl,
    URImpedanceControlConfig,
)
class _Receive:
    def __init__(self, rotation: np.ndarray) -> None:
        self.pose = np.asarray([np.zeros(3), rotation], dtype=np.float32)

    def getActualTCPPose(self) -> np.ndarray:
        return self.pose.reshape(-1)

    def getActualTCPSpeed(self) -> np.ndarray:
        return np.zeros(6, dtype=np.float32)

    def getActualTCPForce(self) -> np.ndarray:
        return np.zeros(6, dtype=np.float32)


class _Control:
    def __init__(self) -> None:
        self.args: tuple | None = None

    def forceMode(self, *args: object) -> None:
        self.args = args


class ForceModeTest(unittest.TestCase):
    def _controller(self, rotation: np.ndarray) -> URImpedanceControl:
        controller = URImpedanceControl(URImpedanceControlConfig())
        controller.rtde_r = _Receive(rotation)
        controller.rtde_c = _Control()
        controller.target_pose = controller.get_actual_tcp_pose().copy()
        return controller

    def test_force_mode_is_compliant_on_all_six_axes(self) -> None:
        rotation = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
        controller = self._controller(rotation)
        controller.pid_step()
        assert controller.rtde_c.args is not None
        np.testing.assert_array_equal(
            controller.rtde_c.args[1], np.ones(6, dtype=np.int32)
        )

    def test_target_rotation_is_not_replaced_by_startup_rotation(self) -> None:
        rotation = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
        controller = self._controller(rotation)
        target = np.asarray([[0.1, 0.2, 0.3], [1.0, 1.0, 1.0]])
        controller.set_target_pose(target)
        np.testing.assert_allclose(controller.target_pose[1], target[1])


if __name__ == "__main__":
    unittest.main()
