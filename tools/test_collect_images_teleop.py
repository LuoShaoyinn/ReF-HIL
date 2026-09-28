from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock
import sys

import cv2
import numpy as np

from tools.collect_images_teleop import (
    _build_robot_action, _to_opencv_image, _next_episode_index, _save_images,
)


class ImageCollectorContractTest(unittest.TestCase):
    def test_collector_uses_live_calibration_for_button_release(self):
        from tools.collect_images_teleop import main
        from actor.operator.toggle import resolve_gripper_command
        robot, operator = Mock(), Mock()
        robot.read_observation.return_value = {
            'gripper': 248, 'gripper_action_range': (3,248),
            'images': {'classifier': np.zeros((2,2,3), dtype=np.uint8)},
        }
        def read_action(request):
            self.assertEqual(request['gripper_position'], 1.)
            return {'gripper': resolve_gripper_command([0,0], request['gripper_position'], last_command=1.)}
        operator.read_action.side_effect = read_action
        with TemporaryDirectory() as tmp, \
             patch.object(sys, 'argv', ['collect', '--task', 'rotate_knob', '--output', tmp, '--episodes', '1', '--steps', '3']), \
             patch('tools.collect_images_teleop.ZmqRobotClient', return_value=robot), \
             patch('tools.collect_images_teleop.ZmqOperatorClient', return_value=operator), \
             patch('tools.collect_images_teleop.cv2.imshow'), \
             patch('tools.collect_images_teleop.cv2.waitKey', return_value=-1), \
             patch('tools.collect_images_teleop.cv2.destroyAllWindows'), \
             patch('tools.collect_images_teleop.time.sleep'):
            main()
        self.assertEqual(robot.send_action.call_count, 3)
        for call in robot.send_action.call_args_list:
            self.assertEqual(call.args[0]['gripper'], 1.)

    def test_append_uses_numeric_max_and_preserves_existing_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(_next_episode_index(root, append=False), 0)
            for index in (2, 9999, 10000):
                (root / f"episode_{index:04d}").mkdir()
            labels = root / "labels.jsonl"
            labels.write_text("existing labels\n")
            with self.assertRaisesRegex(FileExistsError, "--append"):
                _next_episode_index(root, append=False)
            self.assertEqual(_next_episode_index(root, append=True), 10001)
            self.assertEqual(labels.read_text(), "existing labels\n")

    def test_images_roundtrip_and_cannot_be_overwritten(self) -> None:
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rgb = np.asarray([[[10, 20, 30]]], dtype=np.uint8)
            depth = np.asarray([[1234]], dtype=np.uint16)
            _save_images(directory, 0, {"camera": rgb, "camera_depth": depth})
            path = directory / "step_00000_camera_rgb.png"
            original = path.read_bytes()
            np.testing.assert_array_equal(cv2.imread(str(path)), rgb[..., ::-1])
            np.testing.assert_array_equal(
                cv2.imread(str(directory / "step_00000_camera_depth_depth.png"), cv2.IMREAD_UNCHANGED),
                depth,
            )
            with self.assertRaises(FileExistsError):
                _save_images(directory, 0, {"camera": np.zeros_like(rgb)})
            self.assertEqual(path.read_bytes(), original)

    def test_rgb_is_converted_to_bgr_for_opencv(self) -> None:
        rgb = np.asarray([[[10, 20, 30]]], dtype=np.uint8)
        np.testing.assert_array_equal(_to_opencv_image(rgb), [[[30, 20, 10]]])

    def test_translation_only_task_preserves_normalized_action(self) -> None:
        base = {
            "delta_pos": np.asarray([1.0, -1.0, 0.5], dtype=np.float32),
            "delta_rot": np.ones(3, dtype=np.float32),
        }
        opened = _build_robot_action({**base, "gripper": -1.0}, allow_rotation=False)
        closed = _build_robot_action({**base, "gripper": 1.0}, allow_rotation=False)
        self.assertEqual(opened["gripper"], -1.0)
        self.assertEqual(closed["gripper"], 1.0)
        np.testing.assert_array_equal(opened["delta_rot"], np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(opened["delta_pos"], [1.0, -1.0, 0.5])


if __name__ == "__main__":
    unittest.main()
