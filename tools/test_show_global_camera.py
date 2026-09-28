from __future__ import annotations

import unittest

import numpy as np

from tools.show_global_camera import (
    draw_configured_crop,
    normalize_crop,
    xywh_to_corners,
)


class GlobalCameraCropTest(unittest.TestCase):
    def test_crop_is_ordered_and_clamped(self) -> None:
        self.assertEqual(
            normalize_crop((500, 400, 100, -20), width=640, height=480),
            (100, 0, 500, 400),
        )

    def test_crop_has_at_least_one_pixel(self) -> None:
        self.assertEqual(
            normalize_crop((640, 480, 640, 480), width=640, height=480),
            (639, 479, 640, 480),
        )

    def test_task_crop_converts_width_and_height_without_approximation(self) -> None:
        self.assertEqual(
            xywh_to_corners((33, 142, 112, 112)),
            (33, 142, 145, 254),
        )

    def test_configured_crop_overlay_preserves_image_shape(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_configured_crop(
            image,
            (170, 10, 448, 448),
            label="global policy",
            color=(255, 180, 0),
        )
        self.assertEqual(image.shape, (480, 640, 3))
        self.assertGreater(int(image.sum()), 0)


if __name__ == "__main__":
    unittest.main()
