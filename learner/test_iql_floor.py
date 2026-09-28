from __future__ import annotations

import unittest
import pickle
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from shared.zmq import ZmqEndpointConfig
from actor.runtime import ActorConfig
from actor.test_actor import _Operator, _Policy, _Robot, _Socket, _Task
from actor.runtime import Actor
from .iql_floor_runtime import HumanIQLGroundedLearner, HumanIQLGroundedLearnerConfig
from .iql_modules import (
    BilinearVectorCritic,
    build_vector_bellman_target,
    horizon_extension_violation,
)
from .iql_floor import HumanIQLGroundedPolicy, HumanIQLGroundedPolicyConfig


def _human_batch(batch_size: int, obs_dims: int) -> dict:
    action = torch.rand((batch_size, 4)) * 2.0 - 1.0
    return {
        "observation": torch.randn((batch_size, obs_dims)),
        "next_observation": torch.randn((batch_size, obs_dims)),
        "action": action,
        "reward": torch.full((batch_size,), -0.01),
        "done": torch.zeros(batch_size, dtype=torch.bool),
        "data_info": {
            "is_intervene": torch.ones((batch_size, 1)),
        },
    }


class HumanIQLGroundedMathTest(unittest.TestCase):

    def test_policy_has_no_gripper_reward_penalty(self) -> None:
        config = HumanIQLGroundedPolicyConfig(device="cpu")
        self.assertFalse(hasattr(config, "gripper_penalty"))

    def test_bellman_shift_supervises_all_heads_with_timeout_next(self) -> None:
        next_value = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )
        target, valid = build_vector_bellman_target(
            reward=torch.tensor([-0.01, -0.01, 1.0]),
            success=torch.tensor([False, False, True]),
            timeout=torch.tensor([False, True, False]),
            timeout_next_valid=torch.tensor([False, True, False]),
            next_value=next_value,
            gamma=0.99,
            lower_bound=torch.tensor([-0.06, -0.12, -0.18]),
        )
        self.assertTrue(bool(valid.all()))
        self.assertAlmostEqual(float(target[0, 0]), -0.01, places=6)
        self.assertAlmostEqual(float(target[0, 1]), 0.089, places=6)
        self.assertAlmostEqual(float(target[1, 1]), 0.386, places=6)
        torch.testing.assert_close(target[2], torch.ones(3))

    def test_bellman_shift_is_defined_for_every_one_of_100_heads(self) -> None:
        next_value = torch.linspace(-0.06, 1.0, 100).unsqueeze(0)
        target, valid = build_vector_bellman_target(
            reward=torch.tensor([-0.01]),
            success=torch.tensor([False]),
            timeout=torch.tensor([False]),
            timeout_next_valid=torch.tensor([False]),
            next_value=next_value,
            gamma=0.99,
            lower_bound=torch.linspace(-0.06, -3.0, 100),
        )
        self.assertTrue(bool(valid.all()))
        self.assertAlmostEqual(float(target[0, 0]), -0.01, places=6)
        torch.testing.assert_close(target[0, 1:], -0.01 + 0.99 * next_value[0, :-1])

    def test_success_preserves_graded_reward_on_every_head(self) -> None:
        target, valid = build_vector_bellman_target(
            reward=torch.tensor([0.7]),
            success=torch.tensor([True]),
            timeout=torch.tensor([False]),
            timeout_next_valid=torch.tensor([False]),
            next_value=torch.ones((1, 4)),
            gamma=0.99,
            lower_bound=torch.tensor([-0.06, -0.12, -0.18, -0.24]),
        )
        self.assertTrue(bool(valid.all()))
        torch.testing.assert_close(target, torch.full((1, 4), 0.7))

    def test_missing_legacy_timeout_only_masks_unobservable_heads(self) -> None:
        _, valid = build_vector_bellman_target(
            reward=torch.tensor([-0.01]),
            success=torch.tensor([False]),
            timeout=torch.tensor([True]),
            timeout_next_valid=torch.tensor([False]),
            next_value=torch.ones((1, 4)),
            gamma=0.99,
            lower_bound=torch.tensor([-0.06, -0.12, -0.18, -0.24]),
        )
        self.assertTrue(bool(valid[0, 0]))
        self.assertFalse(bool(valid[0, 1:].any()))

    def test_horizon_violation_matches_soft_inequality(self) -> None:
        value = torch.tensor([[0.5, 0.48, 0.50]])
        violation = horizon_extension_violation(value, 0.01)
        torch.testing.assert_close(violation, torch.tensor([[0.01, 0.0]]))


class ActionSensitivePolicyTest(unittest.TestCase):
    def test_bilinear_critic_has_raw_action_gradient(self) -> None:
        critic = BilinearVectorCritic(
            feature_dim=7,
            action_dim=4,
            hidden_dim=12,
            lower_bound=torch.tensor([-0.06, -0.12, -0.18]),
            rank=3,
        )
        feature = torch.randn((5, 7))
        action = torch.randn((5, 4), requires_grad=True)
        critic(feature, action).sum().backward()
        self.assertIsNotNone(action.grad)
        self.assertGreater(float(action.grad.abs().sum()), 0.0)

    def test_actor_and_critic_parameters_are_disjoint(self) -> None:
        policy = HumanIQLGroundedPolicy(
            HumanIQLGroundedPolicyConfig(
                device="cpu",
                obs_dims=8,
                mechanism_obs_dims=8,
                action_dims=4,
                encoder_dim=8,
                hidden_dim=12,
                horizons=5,
                bilinear_rank=3,
                actor_horizons=(1, 3, 5),
            )
        )
        actor_ids = {
            id(parameter)
            for group in policy.actor_optim.param_groups
            for parameter in group["params"]
        }
        critic_ids = {
            id(parameter)
            for group in policy.critic_optim.param_groups
            for parameter in group["params"]
        }
        human_critic_ids = {
            id(parameter)
            for group in policy.human_critic_optim.param_groups
            for parameter in group["params"]
        }
        self.assertFalse(actor_ids & critic_ids)
        self.assertFalse(critic_ids & human_critic_ids)
        self.assertFalse(actor_ids & human_critic_ids)
        self.assertFalse(policy.human_iql_initialized)
        # There is no reference/BC actor in the algorithm state.
        self.assertFalse(hasattr(policy, "reference_actor"))

    def test_snapshot_roundtrip_preserves_q_only_policy(self) -> None:
        config = HumanIQLGroundedPolicyConfig(
            device="cpu",
            obs_dims=8,
            mechanism_obs_dims=8,
            action_dims=4,
            encoder_dim=8,
            hidden_dim=12,
            horizons=5,
            bilinear_rank=3,
            actor_horizons=(1, 3, 5),
        )
        source = HumanIQLGroundedPolicy(config)
        source.finalize_human_iql()
        observation = torch.randn(8)
        expected = source.sample_action(observation)
        target = HumanIQLGroundedPolicy(config)
        target.load(source.export())
        actual = target.sample_action(observation)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(target.human_iql_initialized)
        self.assertFalse(target.loaded_legacy_iql)

    def test_legacy_snapshot_is_marked_analysis_only(self) -> None:
        config = HumanIQLGroundedPolicyConfig(
            device="cpu",
            obs_dims=8,
            mechanism_obs_dims=8,
            action_dims=4,
            encoder_dim=8,
            hidden_dim=12,
            horizons=5,
            bilinear_rank=3,
        )
        source = HumanIQLGroundedPolicy(config)
        state = source.export()
        state["algorithm"] = "human_iql_grounded_actor_critic_v1"
        for name in (
            "human_critic_encoder",
            "human_critic_1",
            "human_critic_2",
        ):
            del state[name]
        target = HumanIQLGroundedPolicy(config)
        target.load(state)
        self.assertTrue(target.loaded_legacy_iql)
        for legacy, migrated in (
            (target.encoder_critic, target.human_critic_encoder),
            (target.critic_1, target.human_critic_1),
            (target.critic_2, target.human_critic_2),
        ):
            for key, value in legacy.state_dict().items():
                torch.testing.assert_close(value, migrated.state_dict()[key])

    def test_online_update_moves_actor_and_moving_human_value(self) -> None:
        policy = HumanIQLGroundedPolicy(
            HumanIQLGroundedPolicyConfig(
                device="cpu",
                obs_dims=8,
                mechanism_obs_dims=8,
                action_dims=4,
                encoder_dim=8,
                hidden_dim=12,
                horizons=5,
                bilinear_rank=3,
                actor_horizons=(1, 3, 5),
                utd_ratio=1,
            )
        )
        policy.finalize_human_iql()
        batch_size = 8
        action = torch.rand((batch_size, 4)) * 2.0 - 1.0
        batch = {
            "observation": torch.randn((batch_size, 8)),
            "next_observation": torch.randn((batch_size, 8)),
            "action": action,
            "reward": torch.full((batch_size,), -0.01),
            "done": torch.zeros(batch_size, dtype=torch.bool),
            "timeout_next_valid": torch.zeros(batch_size, dtype=torch.bool),
            "data_info": {
                "is_intervene": torch.zeros((batch_size, 1)),
            },
            "human_example": _human_batch(4, 8),
        }
        actor_before = {
            key: value.detach().clone()
            for key, value in policy.actor.state_dict().items()
        }
        value_before = {
            key: value.detach().clone()
            for key, value in policy.human_value.state_dict().items()
        }
        metrics = policy.update(batch)
        self.assertIn("human_floor_loss", metrics)
        self.assertTrue(
            any(
                not torch.equal(actor_before[key], value)
                for key, value in policy.actor.state_dict().items()
            )
        )
        self.assertIn("human_iql_loss", metrics)
        self.assertTrue(
            any(
                not torch.equal(value_before[key], value)
                for key, value in policy.human_value.state_dict().items()
            )
        )

    def test_human_floor_has_no_gradient_path_into_moving_value(self) -> None:
        policy = HumanIQLGroundedPolicy(
            HumanIQLGroundedPolicyConfig(
                device="cpu",
                obs_dims=8,
                mechanism_obs_dims=8,
                action_dims=4,
                encoder_dim=8,
                hidden_dim=12,
                horizons=5,
                bilinear_rank=3,
            )
        )
        policy.finalize_human_iql()
        human_batch = {
            "observation": torch.randn((4, 8)),
            "action": torch.rand((4, 4)) * 2.0 - 1.0,
        }
        floor, _ = policy._human_floor(human_batch)
        floor.backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in policy.human_value.parameters())
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in policy.human_value_encoder.parameters()
            )
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in policy.critic_1.parameters())
        )

    def test_moving_iql_updates_value_without_updating_sac_critic(self) -> None:
        policy = HumanIQLGroundedPolicy(
            HumanIQLGroundedPolicyConfig(
                device="cpu",
                obs_dims=8,
                mechanism_obs_dims=8,
                action_dims=4,
                encoder_dim=8,
                hidden_dim=12,
                horizons=5,
                bilinear_rank=3,
            )
        )
        policy.finalize_human_iql()
        human_batch = _human_batch(8, 8)
        sac_critic_before = {
            key: value.detach().clone() for key, value in policy.critic_1.state_dict().items()
        }
        human_critic_before = {
            key: value.detach().clone()
            for key, value in policy.human_critic_1.state_dict().items()
        }
        value_before = {
            key: value.detach().clone() for key, value in policy.human_value.state_dict().items()
        }
        metrics = policy._update_moving_human_iql(human_batch)
        self.assertIn("human_iql_loss", metrics)
        self.assertEqual(float(metrics["human_iql_valid_pairs"]), 8.0 * 5.0)
        self.assertTrue(
            all(
                torch.equal(sac_critic_before[key], value)
                for key, value in policy.critic_1.state_dict().items()
            )
        )
        self.assertTrue(
            any(
                not torch.equal(human_critic_before[key], value)
                for key, value in policy.human_critic_1.state_dict().items()
            )
        )
        self.assertTrue(
            any(not torch.equal(value_before[key], value) for key, value in policy.human_value.state_dict().items())
        )

    def test_default_actor_objective_means_every_horizon(self) -> None:
        policy = HumanIQLGroundedPolicy(
            HumanIQLGroundedPolicyConfig(
                device="cpu",
                obs_dims=8,
                mechanism_obs_dims=8,
                action_dims=4,
                encoder_dim=8,
                hidden_dim=12,
                horizons=5,
                bilinear_rank=3,
            )
        )
        obs = torch.randn((4, 8))
        _, metrics = policy._actor_loss(obs)
        with torch.no_grad():
            action = policy.actor_actions(obs)
            feature = policy.encoder_critic(obs, detach=True)
            expected = torch.minimum(
                policy.critic_1(feature, action), policy.critic_2(feature, action)
            ).mean()
        torch.testing.assert_close(metrics["actor_q"], expected)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class _TaskConfig:
    obs_dims: int = 4
    action_dims: int = 4


class _VectorTask:
    config = _TaskConfig()

    def __init__(self) -> None:
        self.observation_batch_sizes: list[int] = []

    def build_obs(
        self, raw: dict, _info: dict, augment: bool = False
    ) -> torch.Tensor:
        del augment
        return torch.as_tensor(raw["x"], dtype=torch.float32)

    def build_observations(
        self,
        states: list[dict],
        infos: list[dict],
        *,
        augment: bool = False,
    ) -> torch.Tensor:
        self.observation_batch_sizes.append(len(states))
        return torch.stack(
            [
                self.build_obs(state, info, augment=augment)
                for state, info in zip(states, infos, strict=True)
            ]
        )

    def build_action(
        self, raw: dict, _info: dict, augment: bool = False
    ) -> torch.Tensor:
        del augment
        return torch.as_tensor(raw["a"], dtype=torch.float32)


class LearnerStartupTest(unittest.TestCase):
    def test_first20_no_g_pretrain_initializes_only_q_driven_actor(self) -> None:
        with tempfile.TemporaryDirectory(prefix="srt-human-iql-") as tmp:
            root = Path(tmp)
            buffer_dir = root / "smoke" / "buffer"
            buffer_dir.mkdir(parents=True)
            rows = []
            for episode in range(20):
                for step in range(4):
                    done = step == 3
                    rows.append(
                        {
                            "raw_obs": {
                                "x": np.asarray(
                                    [episode / 20.0, step / 4.0, 0.0, 1.0],
                                    dtype=np.float32,
                                )
                            },
                            "raw_next_obs": {
                                "x": np.asarray(
                                    [episode / 20.0, (step + 1) / 4.0, 0.0, 1.0],
                                    dtype=np.float32,
                                )
                            },
                            "raw_action": {
                                "a": np.asarray([0.1, -0.1, 0.2, 1.0], dtype=np.float32)
                            },
                            "reward": 1.0 if done else -0.01,
                            "done": done,
                            "info": {
                                "is_intervene": True,
                            },
                        }
                    )
            with (buffer_dir / "0000.pkl").open("wb") as handle:
                pickle.dump(rows, handle)
            policy = HumanIQLGroundedPolicy(
                HumanIQLGroundedPolicyConfig(
                    device="cpu",
                    obs_dims=4,
                    mechanism_obs_dims=4,
                    action_dims=4,
                    encoder_dim=8,
                    hidden_dim=12,
                    horizons=4,
                    bilinear_rank=2,
                    actor_horizons=(1, 2, 4),
                )
            )
            task = _VectorTask()
            learner = HumanIQLGroundedLearner(
                config=HumanIQLGroundedLearnerConfig(
                    transitions_endpoint=ZmqEndpointConfig(
                        host="127.0.0.1", port=_free_port()
                    ),
                    actor_parameters_endpoint=ZmqEndpointConfig(
                        host="127.0.0.1", port=_free_port()
                    ),
                    experiment_name="smoke",
                    output_root=root,
                    replay_device="cpu",
                    max_buffer_size=128,
                    batch_size=8,
                    human_example_capacity=128,
                    human_example_batch_size=8,
                    human_iql_updates=2,
                    human_iql_log_every=1,
                    q_actor_warmup_updates=2,
                    save_every_seconds=0,
                ),
                task=task,
                policy=policy,
            )
            try:
                self.assertTrue(policy.human_iql_initialized)
                self.assertEqual(learner.human_example_episodes, 20)
                self.assertEqual(learner.first20_cutoff, 80)
                self.assertEqual(len(learner.human_candidates), 80)
                learner.on_actor_transitions(rows[:4])
                self.assertEqual(learner.human_example_episodes, 21)
                self.assertEqual(len(learner.human_candidates), 84)
                self.assertFalse(hasattr(policy, "reference_actor"))
                self.assertFalse((root / "smoke" / "human_iql" / "reference.pt").exists())
                self.assertTrue(list((root / "smoke" / "checkpoints").glob("*.pkl")))

                timeout_rows = []
                for offset in range(2):
                    timeout_rows.append(
                        {
                            "raw_next_obs": {
                                "x": np.asarray(
                                    [9.0, float(offset), 8.0, 7.0],
                                    dtype=np.float32,
                                )
                            },
                            "reward": 0.0,
                            "done": True,
                            "info": {
                                "is_intervene": False,
                            },
                        }
                    )
                learner._register_timeout_next_observations(timeout_rows, 80)
                self.assertTrue(learner.has_timeout_next_rows)
                self.assertEqual(learner.timeout_next_count, 2)
                self.assertIn(2, task.observation_batch_sizes)
                patch_batch = {"next_observation": torch.zeros((3, 4))}
                learner._patch_timeout_next_observation(
                    patch_batch,
                    torch.tensor([80, 0, 81]),
                    "cpu",
                )
                torch.testing.assert_close(
                    patch_batch["timeout_next_valid"],
                    torch.tensor([True, False, True]),
                )
                torch.testing.assert_close(
                    patch_batch["next_observation"],
                    torch.tensor(
                        [
                            [9.0, 0.0, 8.0, 7.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [9.0, 1.0, 8.0, 7.0],
                        ]
                    ),
                )
            finally:
                learner.transition_receiver.close()
                learner.parameters_sender.close()
                learner.writer.close()


class AlgorithmActorTest(unittest.TestCase):
    def _episode(self, success: bool) -> list[dict]:
        with patch("actor.runtime.Sender", _Socket), patch(
            "actor.runtime.Receiver", _Socket
        ):
            actor = Actor(
                config=ActorConfig(max_steps_per_episode=1, loop_sleep_s=0.0),
                task=_Task(),
                policy=_Policy(),
                robot=_Robot(),
                operator=_Operator(),
                success_oracle=lambda _raw: success,
            )
            return actor.run_episode()

    def test_only_timeout_rows_duplicate_the_terminal_next_observation(self) -> None:
        timeout_episode = self._episode(success=False)
        success_episode = self._episode(success=True)
        self.assertIn("raw_next_obs", timeout_episode[0])
        self.assertNotIn("raw_next_obs", success_episode[0])


if __name__ == "__main__":
    unittest.main()
