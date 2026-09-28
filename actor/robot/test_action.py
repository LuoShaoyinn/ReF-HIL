from __future__ import annotations

import unittest

from actor.robot.action import (
    GRIPPER_COMMAND_FORCE,
    GRIPPER_COMMAND_SPEED,
    normalized_gripper_position,
    send_continuous_gripper_action,
)
from actor.robot.ur5e.robotiq_gripper import (
    LOW_FORCE_LIMIT,
    LOW_SPEED_LIMIT,
    RobotiqGripper,
    RobotiqGripperConfig,
)


class _Gripper:
    def __init__(self) -> None:
        self.command: tuple[int, int, int] | None = None

    def get_open_position(self) -> int:
        return 10

    def get_closed_position(self) -> int:
        return 210

    def move(self, position: int, speed: int, force: int) -> None:
        self.command = (position, speed, force)


class ContinuousGripperMappingTest(unittest.TestCase):
    def test_full_continuous_interval_maps_to_calibrated_positions(self) -> None:
        self.assertEqual(normalized_gripper_position(-1.0, 10, 210), 10)
        self.assertEqual(normalized_gripper_position(0.0, 10, 210), 110)
        self.assertEqual(normalized_gripper_position(0.25, 10, 210), 135)
        self.assertEqual(normalized_gripper_position(1.0, 10, 210), 210)

    def test_out_of_range_commands_are_clamped(self) -> None:
        self.assertEqual(normalized_gripper_position(-2.0, 10, 210), 10)
        self.assertEqual(normalized_gripper_position(2.0, 10, 210), 210)

    def test_fractional_action_is_sent_directly_at_low_speed(self) -> None:
        gripper = _Gripper()
        position = send_continuous_gripper_action(gripper, 0.25)
        self.assertEqual(position, 135)
        self.assertEqual(GRIPPER_COMMAND_SPEED, 64)
        self.assertEqual(GRIPPER_COMMAND_FORCE, 0)
        self.assertEqual(gripper.command, (135, 64, 0))

    def test_sdk_caps_all_move_spe_values_at_low_speed(self) -> None:
        gripper = RobotiqGripper(
            RobotiqGripperConfig(hold_detected_object=False)
        )
        sent: dict[str, int] = {}

        def capture(values: dict[str, int]) -> bool:
            sent.update(values)
            return True

        gripper._set_vars = capture  # type: ignore[method-assign]
        accepted, position = gripper.move(120, 255, 200)

        self.assertTrue(accepted)
        self.assertEqual(position, 120)
        self.assertEqual(LOW_SPEED_LIMIT, 64)
        self.assertEqual(LOW_FORCE_LIMIT, 0)
        self.assertEqual(sent[gripper.SPE], 64)
        self.assertEqual(sent[gripper.FOR], 0)

    def test_task_can_explicitly_enable_maximum_speed_and_force(self) -> None:
        gripper = RobotiqGripper(
            RobotiqGripperConfig(
                runtime_speed_limit=255,
                runtime_force_limit=255,
                hold_detected_object=False,
            )
        )
        sent: dict[str, int] = {}

        def capture(values: dict[str, int]) -> bool:
            sent.update(values)
            return True

        gripper._set_vars = capture  # type: ignore[method-assign]
        gripper.move(120, 255, 255)

        self.assertEqual(sent[gripper.SPE], 255)
        self.assertEqual(sent[gripper.FOR], 255)

    def test_detected_object_prevents_lower_position_until_full_open(self) -> None:
        gripper = RobotiqGripper(RobotiqGripperConfig())
        sent: dict[str, int] = {}
        status = gripper.ObjectStatus.STOPPED_OUTER_OBJECT.value

        def get_var(variable: str) -> int:
            if variable == gripper.OBJ:
                return status
            if variable == gripper.POS:
                return 140
            raise AssertionError(variable)

        def capture(values: dict[str, int]) -> bool:
            sent.update(values)
            return True

        gripper._get_var = get_var  # type: ignore[method-assign]
        gripper._set_vars = capture  # type: ignore[method-assign]

        accepted, position = gripper.move(100, 64, 0)
        self.assertTrue(accepted)
        self.assertEqual(position, 140)
        self.assertEqual(sent[gripper.POS], 140)

        # Keep the latch after the transient OBJ=1 report disappears.
        status = gripper.ObjectStatus.AT_DEST.value
        _, position = gripper.move(120, 64, 0)
        self.assertEqual(position, 140)

        # Exact fully-open is an intentional reset/release command.
        _, position = gripper.move(gripper.get_open_position(), 64, 0)
        self.assertEqual(position, gripper.get_open_position())
        self.assertIsNone(gripper._object_hold_position)


if __name__ == "__main__":
    unittest.main()
