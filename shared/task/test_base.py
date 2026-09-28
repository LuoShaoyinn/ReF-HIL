from __future__ import annotations

import unittest

from .base import BaseModeling


class AbsoluteGripperButtonTest(unittest.TestCase):
    def test_operator_resolved_command_is_used_directly(self) -> None:
        action = BaseModeling.prepare_spacemouse_action(
            {"gripper": 0.4, "gripper_position": 0.05},
            last_gripper_command=-1.0,
        )
        self.assertEqual(action["gripper"], 0.4)

    def test_no_button_preserves_measured_position(self) -> None:
        for measured, expected in (
            (0.0, -1.0),
            (0.3, -0.4),
            (0.6, 0.2),
            (1.0, 1.0),
        ):
            with self.subTest(measured=measured, expected=expected):
                action = BaseModeling.prepare_spacemouse_action(
                    {"gripper_position": measured},
                    last_gripper_command=1.0,
                )
                self.assertAlmostEqual(action["gripper"], expected)

    def test_either_gripper_button_has_operator_intent(self) -> None:
        for key in ("gripper_open_pressed", "gripper_close_pressed"):
            with self.subTest(key=key):
                self.assertTrue(
                    BaseModeling.operator_action_has_intent(
                        object(),
                        {key: True},
                    )
                )

    def test_missing_feedback_uses_last_command(self) -> None:
        action = BaseModeling.prepare_spacemouse_action(
            {}, last_gripper_command=0.3
        )
        self.assertEqual(action["gripper"], 0.3)

    def test_all_rotation_axes_have_operator_intent(self) -> None:
        self.assertTrue(
            BaseModeling.operator_action_has_intent(
                object(),
                {
                    "delta_rot": [0.0, 0.01, 0.0],
                },
            )
        )


if __name__ == "__main__":
    unittest.main()
