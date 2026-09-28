from __future__ import annotations

import unittest

from .toggle import (
    resolve_gripper_command,
    spacemouse_gripper_buttons,
    spacemouse_has_intent,
)


class OperatorButtonMappingTest(unittest.TestCase):
    def test_physical_left_opens_and_right_closes(self) -> None:
        self.assertEqual(spacemouse_gripper_buttons([1, 0]), (False, True))
        self.assertEqual(spacemouse_gripper_buttons([0, 1]), (True, False))

    def test_buttons_are_absolute_commands(self) -> None:
        self.assertEqual(resolve_gripper_command([1, 0], 1.0), -1.0)
        self.assertEqual(resolve_gripper_command([0, 1], 0.0), 1.0)
        # Opening is the fail-safe behavior if both are pressed.
        self.assertEqual(resolve_gripper_command([1, 1], 0.5), -1.0)

    def test_no_button_preserves_measured_robot_position(self) -> None:
        for measured, expected in (
            (0.0, -1.0),
            (0.25, -0.5),
            (0.5, 0.0),
            (0.75, 0.5),
            (1.0, 1.0),
        ):
            with self.subTest(measured=measured):
                self.assertEqual(
                    resolve_gripper_command([0, 0], measured), expected
                )

    def test_missing_measurement_uses_last_command_fallback(self) -> None:
        self.assertEqual(
            resolve_gripper_command([0, 0], None, fallback=0.25), 0.25
        )

    def test_close_readback_holds_previous_command_without_buttons(self) -> None:
        self.assertEqual(resolve_gripper_command([0, 0], 0.99, last_command=1.0), 1.0)
        self.assertEqual(resolve_gripper_command([0, 0], 0.01, last_command=-1.0), -1.0)
        self.assertEqual(resolve_gripper_command([0, 0], 0.61, last_command=0.2), 0.2)

    def test_far_readback_follows_robot_and_buttons_always_override(self) -> None:
        self.assertEqual(resolve_gripper_command([0, 0], 0.7, last_command=1.0), 2 * 0.7 - 1)
        self.assertEqual(resolve_gripper_command([1, 0], 0.99, last_command=1.0), -1.0)
        self.assertEqual(resolve_gripper_command([0, 1], 0.01, last_command=-1.0), 1.0)

    def test_all_six_axes_or_either_button_trigger_intervention(self) -> None:
        for axis in range(6):
            translation = [0.0, 0.0, 0.0]
            rotation = [0.0, 0.0, 0.0]
            (translation if axis < 3 else rotation)[axis % 3] = 0.01
            with self.subTest(axis=axis):
                self.assertTrue(
                    spacemouse_has_intent(
                        translation, rotation, [0, 0]
                    )
                )
        self.assertTrue(
            spacemouse_has_intent([0, 0, 0], [0, 0, 0], [1, 0])
        )
        self.assertTrue(
            spacemouse_has_intent([0, 0, 0], [0, 0, 0], [0, 1])
        )
        self.assertFalse(
            spacemouse_has_intent([0, 0, 0], [0, 0, 0], [0, 0])
        )


if __name__ == "__main__":
    unittest.main()
