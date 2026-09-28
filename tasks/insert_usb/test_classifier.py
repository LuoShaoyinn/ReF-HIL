from __future__ import annotations

import unittest

import numpy as np

from .classifier import Classifier, ClassifierConfig, clip_rgbd
from .config import TaskConfig


class InsertUSBClassifierTest(unittest.TestCase):
    def test_configured_crop_is_used_for_rgb_and_depth(self) -> None:
        config = TaskConfig()
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.zeros((480, 640), dtype=np.uint8)
        clipped_rgb, clipped_depth = clip_rgbd(
            rgb,
            depth,
            config.success_conditions[0].clip,
        )
        self.assertEqual(clipped_rgb.shape, (56, 56, 3))
        self.assertEqual(clipped_depth.shape, (56, 56))

    def test_runtime_resizes_roi_to_classifier_input(self) -> None:
        classifier = object.__new__(Classifier)
        classifier.config = ClassifierConfig(device="cpu")
        observation = {
            "images": {
                "classifier": np.zeros((56, 56, 3), dtype=np.uint8),
                "classifier_depth": np.zeros((56, 56), dtype=np.uint8),
            }
        }
        self.assertEqual(classifier.modeling(observation).shape, (4, 224, 224))


if __name__ == "__main__":
    unittest.main()
