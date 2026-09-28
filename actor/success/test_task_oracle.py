from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from shared.task.manipulation_config import SuccessCondition, TCPPositionCondition
from .task_oracle import TaskSuccessOracle


class _Classifier:
    def __init__(self, condition: SuccessCondition, score: float) -> None:
        self.config = SimpleNamespace(condition=condition)
        self._score = score

    def score(self, *, raw_robot_observation: dict) -> float:
        del raw_robot_observation
        return self._score


class TaskSuccessOracleTest(unittest.TestCase):
    def test_gear_requires_both_images_and_tcp_distance(self):
        from tasks.insert_gear.config import TaskConfig
        task = TaskConfig()
        for a, b, offset, expected in ((0.8, 0.8, 0., True), (0.8, 0.8, .009, True),
                                      (0.8, 0.8, .011, False), (0.1, 0.8, 0., False),
                                      (0.8, 0.1, 0., False)):
            with self.subTest(a=a, b=b, offset=offset):
                oracle = TaskSuccessOracle(classifiers=tuple(_Classifier(c, s) for c, s in zip(task.success_conditions, (a, b))), tcp_condition=task.tcp_success_condition)
                pos = np.asarray(task.tcp_success_condition.target) + [offset, 0, 0]
                result = oracle.evaluate({"tcp_pose": [pos, [0.4, 0.5, 0.6]]})
                self.assertEqual(len(result.conditions), 3)
                self.assertEqual(result.success, expected)
                self.assertAlmostEqual(result.conditions[-1].score, offset)

    def test_tcp_distance_is_strict_and_euclidean(self):
        oracle = TaskSuccessOracle(classifiers=(_Classifier(SuccessCondition(name="a", clip=(0, 0, 8, 8)), 1.),),
                                   tcp_condition=TCPPositionCondition(target=(0., 0., 0.), radius=.010))
        self.assertFalse(oracle.evaluate({"tcp_pose": [[.010, 0, 0], [0, 0, 0]]}).success)
        self.assertFalse(oracle.evaluate({"tcp_pose": [[.008, .008, 0], [0, 0, 0]]}).success)
        self.assertTrue(oracle.evaluate({"tcp_pose": [.009, 0, 0, 1, 2, 3]}).success)

    def test_invalid_tcp_fails_closed(self):
        oracle = TaskSuccessOracle(classifiers=(_Classifier(SuccessCondition(name="a", clip=(0, 0, 8, 8)), 1.),),
                                   tcp_condition=TCPPositionCondition(target=(0., 0., 0.), radius=.010))
        for obs in ({}, {"tcp_pose": [0, 0, 0]}, {"tcp_pose": [[float("nan"), 0, 0], [0, 0, 0]]},
                    {"tcp_pose": [[float("inf"), 0, 0], [0, 0, 0]]}):
            self.assertFalse(oracle.evaluate(obs).success)

    def test_load_installs_tcp_gate_without_a_third_classifier(self):
        from tasks.insert_gear.config import TaskConfig
        task = TaskConfig()
        class LoadedClassifier(_Classifier):
            def __init__(self, config):
                super().__init__(config.condition, 1.)
            def load(self, strict=True):
                pass
        component = SimpleNamespace(Classifier=LoadedClassifier, ClassifierConfig=SimpleNamespace)
        with patch("actor.success.task_oracle.load_component", return_value=component):
            oracle = TaskSuccessOracle.load(task_name="insert_gear", task=task, experiment_name="test", device="cpu")
        self.assertEqual(len(oracle.classifiers), 2)
        self.assertEqual(oracle.tcp_condition, task.tcp_success_condition)
        self.assertFalse(oracle.inference(raw_robot_observation={}))

    def test_all_conditions_must_pass(self) -> None:
        first = SuccessCondition(name="inserted", clip=(0, 0, 8, 8), threshold=0.5)
        second = SuccessCondition(
            name="released",
            clip=(8, 0, 8, 8),
            threshold=0.7,
            image_key="classifier_released",
            depth_key="classifier_released_depth",
            checkpoint="classifiers/released.pt",
        )
        oracle = TaskSuccessOracle(
            classifiers=(_Classifier(first, 0.8), _Classifier(second, 0.6))
        )
        evaluation = oracle.evaluate({})
        self.assertFalse(evaluation.success)
        self.assertEqual([result.passed for result in evaluation.conditions], [True, False])

    def test_all_pass_is_success(self) -> None:
        first = SuccessCondition(name="a", clip=(0, 0, 8, 8), threshold=0.5)
        second = SuccessCondition(
            name="b",
            clip=(8, 0, 8, 8),
            image_key="classifier_b",
            depth_key="classifier_b_depth",
            checkpoint="classifiers/b.pt",
        )
        oracle = TaskSuccessOracle(
            classifiers=(_Classifier(first, 0.5), _Classifier(second, 0.9))
        )
        self.assertTrue(oracle.inference(raw_robot_observation={}))


if __name__ == "__main__":
    unittest.main()
