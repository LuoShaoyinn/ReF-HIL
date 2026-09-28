from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from tasks.assemble.classifier import normalize_rgbd
from tasks.assemble.config import TaskConfig
from tools import train_classifier
from tools.train_classifier import RGBDPNGDataset, Sample, grouped_split, read_samples


class ClassifierPNGTrainingTest(unittest.TestCase):
    def test_training_only_path_saves_final_epoch_without_validation_metrics(self) -> None:
        class TinyClassifier:
            def __init__(self, config):
                self.model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 1))

            def save(self, path):
                torch.save(self.model.state_dict(), path)
                return path

        component = SimpleNamespace(Classifier=TinyClassifier, normalize_rgbd=normalize_rgbd,
                                    ClassifierConfig=lambda **kwargs: SimpleNamespace(resnet_name='test', **kwargs))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [self._write_pair(root / 'episode', i, label=label) for i, label in enumerate(['failure', 'success'])]
            labels = root / 'labels.jsonl'
            labels.write_text('\n'.join(map(json.dumps, rows)))
            with patch('sys.argv', ['train_classifier', '--labels', str(labels), '--output', str(root / 'classifier.pt'), '--device', 'cpu', '--epochs', '1', '--allow-incomplete']), patch.object(train_classifier, 'load_component', side_effect=lambda task, name: component if name == 'classifier' else SimpleNamespace(TaskConfig=TaskConfig)):
                with self.assertWarns(UserWarning):
                    train_classifier.main()
            metrics = json.loads((root / 'classifier_metrics.json').read_text())
            self.assertEqual(metrics['checkpoint_selection'], 'final-epoch')
            self.assertEqual(metrics['validation_samples'], 0)
            self.assertNotIn('val_loss', metrics['history'][0])
            self.assertTrue((root / 'classifier.pt').is_file())

    def test_incomplete_pairs_are_skipped_only_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = self._write_pair(root / "episode", 0, label="success")
            incomplete = self._write_pair(root / "episode", 1, label="failure")
            (root / "episode/step_00001_classifier_depth_depth.png").unlink()
            labels = root / "labels.jsonl"
            labels.write_text("\n".join(map(json.dumps, [good, incomplete, {"image": str(root / 'missing_rgb.png'), "label": "failure"}])))
            with self.assertRaises(FileNotFoundError):
                read_samples(labels)
            with self.assertWarns(UserWarning):
                samples = read_samples(labels, allow_incomplete=True)
            self.assertEqual(len(samples), 1)

    def test_incomplete_split_has_no_fake_validation_and_requires_two_classes(self) -> None:
        samples = [Sample(Path('episode/a_rgb.png'), Path('depth.png'), 0),
                   Sample(Path('episode/b_rgb.png'), Path('depth.png'), 1)]
        with self.assertRaises(ValueError):
            grouped_split(samples, val_fraction=0.2, seed=0)
        with self.assertWarns(UserWarning):
            train, validation = grouped_split(samples, val_fraction=0.2, seed=0, allow_incomplete=True)
        self.assertEqual(train, samples)
        self.assertEqual(validation, [])
        with self.assertRaisesRegex(ValueError, 'both success and failure'):
            grouped_split(samples[:1], val_fraction=0.2, seed=0, allow_incomplete=True)

    def _write_pair(self, episode: Path, step: int, *, label: str) -> dict[str, str]:
        episode.mkdir(parents=True, exist_ok=True)
        rgb_path = episode / f"step_{step:05d}_classifier_rgb.png"
        depth_path = episode / f"step_{step:05d}_classifier_depth_depth.png"
        rgb = np.asarray([[[10, 20, 30]]], dtype=np.uint8)
        depth = np.asarray([[40]], dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
        self.assertTrue(cv2.imwrite(str(depth_path), depth))
        return {"image": str(rgb_path), "label": label}

    def test_reader_pairs_rgb_with_depth_and_keeps_latest_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._write_pair(root / "episode_0000", 0, label="failure")
            labels_path = root / "labels.jsonl"
            labels_path.write_text(
                json.dumps(row) + "\n" + json.dumps({**row, "label": "success"}) + "\n"
            )
            samples = read_samples(labels_path)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].label, 1)
            self.assertEqual(samples[0].depth_path.name, "step_00000_classifier_depth_depth.png")

    def test_reader_pairs_named_condition_with_its_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode_0000"
            episode.mkdir()
            rgb_path = episode / "step_00000_classifier_released_rgb.png"
            depth_path = episode / "step_00000_classifier_released_depth_depth.png"
            self.assertTrue(cv2.imwrite(str(rgb_path), np.zeros((2, 2, 3), dtype=np.uint8)))
            self.assertTrue(cv2.imwrite(str(depth_path), np.zeros((2, 2), dtype=np.uint8)))
            labels_path = root / "labels.jsonl"
            labels_path.write_text(json.dumps({"image": str(rgb_path), "label": "success"}) + "\n")
            sample = read_samples(labels_path)[0]
            self.assertEqual(sample.depth_path, depth_path)

    def test_png_dataset_matches_shared_runtime_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self._write_pair(root / "episode_0000", 0, label="failure")
            labels_path = root / "labels.jsonl"
            labels_path.write_text(json.dumps(row) + "\n")
            sample = read_samples(labels_path)[0]
            tensor, label = RGBDPNGDataset(
                [sample], image_size=1, normalize_rgbd=normalize_rgbd
            )[0]
            expected = normalize_rgbd(
                np.asarray([[[10, 20, 30]]], dtype=np.uint8),
                np.asarray([[40]], dtype=np.uint8),
                image_size=1,
            )
            np.testing.assert_allclose(tensor.numpy(), expected)
            self.assertEqual(float(label), 0.0)

    def test_validation_split_holds_out_whole_episode_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows: list[dict[str, str]] = []
            for episode in range(4):
                rows.append(self._write_pair(root / f"episode_{episode:04d}", 0, label="failure"))
                rows.append(self._write_pair(root / f"episode_{episode:04d}", 1, label="success"))
            labels_path = root / "labels.jsonl"
            labels_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            train, validation = grouped_split(read_samples(labels_path), val_fraction=0.25, seed=0)
            self.assertTrue({sample.group for sample in train}.isdisjoint({sample.group for sample in validation}))
            self.assertEqual({sample.label for sample in train}, {0, 1})
            self.assertEqual({sample.label for sample in validation}, {0, 1})


if __name__ == "__main__":
    unittest.main()
