"""Task-configured independent success classifiers combined by logical AND."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from shared.task.manipulation_config import SuccessCondition, TCPPositionCondition, TaskConfig
from shared.zmq import DictMessage
from tasks import load_component


@dataclass(frozen=True)
class ConditionResult:
    name: str
    score: float
    threshold: float
    passed: bool
    image_key: str | None
    depth_key: str | None


@dataclass(frozen=True)
class SuccessEvaluation:
    conditions: tuple[ConditionResult, ...]

    @property
    def success(self) -> bool:
        return bool(self.conditions) and all(result.passed for result in self.conditions)


class TaskSuccessOracle:
    """Own one model per condition; task success requires every model to pass."""

    def __init__(self, *, classifiers: tuple[object, ...], tcp_condition: TCPPositionCondition | None = None) -> None:
        if not classifiers:
            raise ValueError("success oracle needs at least one classifier")
        self.classifiers = classifiers
        self.tcp_condition = tcp_condition

    @classmethod
    def load(
        cls,
        *,
        task_name: str,
        task: TaskConfig,
        experiment_name: str,
        device: str,
        strict: bool = True,
    ) -> "TaskSuccessOracle":
        component = load_component(task_name, "classifier")
        classifiers: list[object] = []
        for condition in task.success_conditions:
            classifier = component.Classifier(
                config=component.ClassifierConfig(
                    experiment_name=experiment_name,
                    device=device,
                    task=task,
                    condition=condition,
                )
            )
            classifier.load(strict=strict)
            classifiers.append(classifier)
        return cls(classifiers=tuple(classifiers), tcp_condition=task.tcp_success_condition)

    def evaluate(self, raw_observation: DictMessage) -> SuccessEvaluation:
        results: list[ConditionResult] = []
        for classifier in self.classifiers:
            condition: SuccessCondition = classifier.config.condition
            score = float(classifier.score(raw_robot_observation=raw_observation))
            threshold = float(condition.threshold)
            results.append(
                ConditionResult(
                    name=condition.name,
                    score=score,
                    threshold=threshold,
                    passed=score >= threshold,
                    image_key=condition.image_key,
                    depth_key=condition.depth_key,
                )
            )
        if self.tcp_condition is not None:
            condition = self.tcp_condition
            distance = float("inf")
            try:
                pose = np.asarray(raw_observation["tcp_pose"], dtype=np.float64)
                if pose.shape in ((2, 3), (6,)) and np.isfinite(pose).all():
                    distance = float(np.linalg.norm(pose.reshape(2, 3)[0] - condition.target))
            except (KeyError, TypeError, ValueError):
                pass  # Missing/invalid TCP must never be classified as success.
            results.append(ConditionResult(
                name="tcp_position", score=distance, threshold=condition.radius,
                passed=distance < condition.radius, image_key=None, depth_key=None,
            ))
        return SuccessEvaluation(conditions=tuple(results))

    def inference(self, *, raw_robot_observation: DictMessage) -> bool:
        return self.evaluate(raw_robot_observation).success
