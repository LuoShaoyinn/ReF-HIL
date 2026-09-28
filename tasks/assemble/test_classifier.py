from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .classifier import Classifier, ClassifierConfig


class AssembleClassifierContractTest(unittest.TestCase):
    def test_runtime_uses_standard_resnet_input_size(self) -> None:
        classifier = object.__new__(Classifier)
        classifier.config = ClassifierConfig(device="cpu")
        observation = {
            "images": {
                "classifier": np.zeros((112, 112, 3), dtype=np.uint8),
                "classifier_depth": np.zeros((112, 112), dtype=np.uint8),
            }
        }
        self.assertEqual(classifier.config.condition.image_size, 224)
        self.assertEqual(classifier.modeling(observation).shape, (4, 224, 224))

    def test_new_checkpoint_records_and_validates_preprocessing(self) -> None:
        classifier = Classifier(config=ClassifierConfig(device="cpu", pretrained=False))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "classifier.pt"
            classifier.save(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu")
            self.assertEqual(payload["preprocessing"], classifier.preprocessing_contract())
            classifier.load(checkpoint, strict=True)


if __name__ == "__main__":
    unittest.main()
