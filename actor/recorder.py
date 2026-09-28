from __future__ import annotations

import time
import pickle
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from actor.storage import RawTransitionWriter, RawTransitionWriterConfig
from actor.success import SuccessEvaluation
from actor.transport import OperatorClient, RobotClient
from shared.protocol import validate_transition
from shared.task.base import BaseModeling
from shared.task.gripper_smoothness import tracker_for_task
from shared.zmq import DictMessage


@dataclass(kw_only=True)
class RecorderConfig:
    dataset_dir: Path = Path("outputs/default/buffer")
    dataset_chunk_size: int = 128
    max_episodes: int = 20
    append: bool = False
    max_steps_per_episode: int
    early_done_on_success: bool = True
    not_success_reward: float = -0.01
    loop_sleep_s: float = 0.1
    operator_start_poll_s: float = 0.005
    show_classifier_preview: bool = False
    classifier_preview_scale: int = 3


def validate_demo_destination(directory: Path, *, append: bool) -> None:
    paths = list(directory.glob('*.pkl'))
    if not paths:
        return
    if not append:
        raise FileExistsError(f'{directory} already contains data; use --append to add new demos')
    last = max(paths, key=lambda path: int(path.stem))
    with last.open('rb') as stream:
        rows = pickle.load(stream)
    if not rows or not bool(rows[-1].get('done', False)):
        raise ValueError(f'{last} does not end at an episode boundary; inspect it before appending')


def operator_has_intent(action: DictMessage) -> bool:
    """Return true once the operator supplies the first meaningful command."""

    translation = np.asarray(
        action.get("delta_pos", np.zeros(3, dtype=np.float32)), dtype=np.float32
    ).reshape(3)
    rotation = np.asarray(
        action.get("delta_rot", np.zeros(3, dtype=np.float32)), dtype=np.float32
    ).reshape(3)
    return bool(
        np.linalg.norm(translation) > 1e-3
        or np.linalg.norm(rotation) > 1e-3
        or action.get("gripper_pressed", False)
        or action.get("gripper_close_pressed", False)
        or action.get("gripper_open_pressed", False)
        or action.get("is_intervene", False)
    )


def build_classifier_preview(
    raw_observation: DictMessage,
    *,
    evaluation: SuccessEvaluation,
    scale: int = 3,
) -> np.ndarray:
    """Build one annotated RGB-D row per independent success condition."""

    images = raw_observation["images"]
    scale = max(1, int(scale))
    rows: list[np.ndarray] = []
    for result in evaluation.conditions:
        if result.image_key is None:
            row = np.zeros((48, 800, 3), dtype=np.uint8)
            color = (0, 220, 0) if result.passed else (0, 0, 255)
            label = (f"{result.name}: distance={result.score * 1000:.2f}mm "
                     f"< {result.threshold * 1000:g}mm {'PASS' if result.passed else 'fail'}")
            cv2.putText(row, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
            rows.append(row)
            continue
        rgb = np.asarray(images[result.image_key], dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"classifier RGB must be HxWx3, got {rgb.shape}")
        frames = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)]
        if result.depth_key in images:
            depth = np.asarray(images[result.depth_key], dtype=np.uint8)
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            if depth.shape != rgb.shape[:2]:
                raise ValueError(
                    f"classifier RGB/depth mismatch: {rgb.shape[:2]} vs {depth.shape}"
                )
            frames.append(cv2.applyColorMap(depth, cv2.COLORMAP_TURBO))
        row = cv2.hconcat(frames)
        row = cv2.resize(
            row,
            (row.shape[1] * scale, row.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        color = (0, 220, 0) if result.passed else (0, 0, 255)
        label = (
            f"{result.name}={result.score:.3f} threshold={result.threshold:.3f} "
            f"{'PASS' if result.passed else 'fail'}"
        )
        cv2.rectangle(row, (0, 0), (row.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(row, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        rows.append(row)
    max_width = max(row.shape[1] for row in rows)
    padded = [
        cv2.copyMakeBorder(row, 0, 0, 0, max_width - row.shape[1], cv2.BORDER_CONSTANT)
        for row in rows
    ]
    return cv2.vconcat(padded)


class Recorder(ABC):
    def __init__(
        self,
        config: RecorderConfig,
        task: BaseModeling,
        robot: RobotClient,
        operator: OperatorClient,
        success_oracle: Callable[[DictMessage], bool],
        success_evaluator: Callable[[DictMessage], SuccessEvaluation] | None = None,
    ) -> None:
        self.config = config
        self.task = task
        self.robot = robot
        self.operator = operator
        self.success_oracle = success_oracle
        self.success_evaluator = success_evaluator
        self.last_gripper_command = -1.0

        if self.config.max_episodes < 1:
            raise ValueError('max_episodes must be positive')
        validate_demo_destination(self.config.dataset_dir, append=self.config.append)

        self.writer = RawTransitionWriter(
            RawTransitionWriterConfig(
                directory=self.config.dataset_dir,
                chunk_size=int(self.config.dataset_chunk_size),
            )
        )

    def run(self) -> None:
        try:
            print(f'Recording {self.config.max_episodes} NEW successful demos into '
                  f'{self.config.dataset_dir}; next chunk {self.writer.file_index:04d}.', flush=True)
            successful_episodes = 0
            attempt = 0
            while successful_episodes < int(self.config.max_episodes):
                attempt += 1
                print(
                    f"Starting attempt {attempt} for successful episode "
                    f"{successful_episodes + 1}/{int(self.config.max_episodes)}"
                )
                saved = self.run_episode()
                if saved:
                    self.writer.save()
                    successful_episodes += 1
                    print(
                        f"Saved successful episode {successful_episodes}/"
                        f"{int(self.config.max_episodes)}.",
                        flush=True,
                    )
                else:
                    print(
                        f"Episode {attempt} timed out after "
                        f"{int(self.config.max_steps_per_episode)} steps; "
                        "dropping the failed demonstration and retrying.",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("Recording interrupted by user.")
        finally:
            self.task.reset(self.robot.send_action, self.robot.read_observation)

            self.writer.save()
            self.robot.close()
            self.operator.close()
            if self.config.show_classifier_preview:
                cv2.destroyWindow("SRT record-demo classifier")

    def run_episode(self) -> bool:
        """Run one counted attempt and commit it only if it reaches success."""

        self.task.reset(self.robot.send_action, self.robot.read_observation)
        self.operator.reset()
        self.last_gripper_command = -1.0
        smoothness = tracker_for_task(
            getattr(getattr(self.task, "config", None), "task", None),
            initial_command=self.last_gripper_command,
        )

        raw_obs = self.robot.read_observation()
        stored_obs = self.task.prepare_observation(raw_obs)
        if self.success_evaluator is not None:
            self._evaluate_classifier(raw_obs)
        print("Ready: the first SpaceMouse input starts recording immediately.", flush=True)
        pending_action = self._wait_for_operator_start(raw_obs)
        step_idx = 0
        episode_transitions: list[DictMessage] = []
        while True:
            prev_stored_obs = stored_obs
            human_action = pending_action
            pending_action = None
            if human_action is None:
                human_action = self.operator.read_action(
                    self.task.build_operator_request(raw_obs)
                )
            # You have to go through a round trip to align the action with the robot's action.
            resolved_action = self.task.prepare_spacemouse_action(
                human_action, self.last_gripper_command
            )
            action = self.task.build_action_from_spacemouse(
                raw_action=resolved_action, info={}
            )
            raw_action = self.task.parse_action(action)
            self.robot.send_action(raw_action)
            time.sleep(self.config.loop_sleep_s)

            raw_obs = self.robot.read_observation()
            smoothness.observe(float(raw_action["gripper"]))
            stored_obs = self.task.prepare_observation(raw_obs)
            success = self._evaluate_classifier(raw_obs)
            done = bool(success and self.config.early_done_on_success)

            transition = {
                "raw_obs": prev_stored_obs,
                "raw_action": raw_action,
                "reward": float(
                    smoothness.success_reward
                    if success
                    else self.config.not_success_reward
                ),
                "done": done,
                "info": self.task.build_info(
                    is_intervene=True,
                    steps_in_episode=step_idx + 1,
                    max_steps_per_episode=int(self.config.max_steps_per_episode),
                    gripper_direction_reversals=smoothness.reversals,
                ),
            }
            validate_transition(transition)
            episode_transitions.append(transition)
            self.last_gripper_command = float(raw_action["gripper"])

            if done:
                for successful_transition in episode_transitions:
                    self.writer.add(successful_transition)
                return True
            step_idx += 1
            if step_idx >= int(self.config.max_steps_per_episode):
                return False

    def _wait_for_operator_start(self, raw_obs: DictMessage) -> DictMessage:
        """Discard pre-demonstration idle samples without sending them to the robot."""

        while True:
            action = self.operator.read_action(
                self.task.build_operator_request(raw_obs)
            )
            task_intent = getattr(
                self.task, "operator_has_intent", operator_has_intent
            )
            if task_intent(action):
                return action
            time.sleep(self.config.operator_start_poll_s)

    def _evaluate_classifier(self, raw_obs: DictMessage) -> bool:
        if self.success_evaluator is None:
            return bool(self.success_oracle(raw_obs))
        evaluation = self.success_evaluator(raw_obs)
        success = evaluation.success
        if self.config.show_classifier_preview:
            preview = build_classifier_preview(
                raw_obs,
                evaluation=evaluation,
                scale=self.config.classifier_preview_scale,
            )
            cv2.imshow("SRT record-demo classifier", preview)
            cv2.waitKey(1)
        return success
