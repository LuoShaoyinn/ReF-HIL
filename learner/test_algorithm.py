from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from shared.zmq import ZmqEndpointConfig

from .action_proximity import (
    ACTION_LIKENESS_SIGMA,
    gaussian_density_peak,
    gaussian_density_threshold_for_mass,
    proximity_target,
    proximity_threshold_from_radius,
)
from .policy import LimitActionPolicy, LimitActionPolicyConfig
from .test_iql_floor import _free_port, _VectorTask
from .training import LimitActionLearner, LimitActionLearnerConfig


def small_policy() -> LimitActionPolicy:
    return LimitActionPolicy(
        LimitActionPolicyConfig(
            device="cpu",
            obs_dims=8,
            mechanism_obs_dims=8,
            action_dims=4,
            encoder_dim=8,
            hidden_dim=12,
            horizons=5,
            bilinear_rank=3,
            actor_horizons=None,
            utd_ratio=1,
            action_likeness_threshold=proximity_threshold_from_radius(0.35),
            action_likeness_reference_ratio=0.75,
        )
    )


class ActionLimitPolicyTest(unittest.TestCase):
    def test_relative_fence_accepts_reference_and_detaches_threshold(self):
        policy = small_policy()
        policy.config.action_likeness_reference_ratio = 0.75
        obs = torch.randn(8, policy.config.obs_dims, requires_grad=True)
        ref = policy.recovery_actions(obs)
        threshold = policy._fence_threshold(obs, ref)
        score = policy.action_likeness(obs, ref)
        torch.testing.assert_close(threshold, 0.75 * score.detach())
        self.assertFalse(threshold.requires_grad)
        self.assertTrue(bool((score >= threshold).all()))

    def test_h_sampling_balances_fixed_demos_and_recent_human_actions(self) -> None:
        learner = object.__new__(LimitActionLearner)
        learner.config = LimitActionLearnerConfig(action_likeness_batch_size=256)
        learner.first20_cutoff = 1643
        # Later interventions are deliberately absent from the IQL suffix pool.
        learner.human_example_buffer = SimpleNamespace(raw_ids=torch.arange(1643))
        learner.offline_buffer = SimpleNamespace(raw_ids=torch.arange(6000))
        sampled = learner._sample_action_likeness_raw_ids()
        self.assertEqual(int((sampled < 1643).sum()), 128)
        self.assertEqual(int((sampled >= 6000 - 1643).sum()), 128)
        demo_pool = torch.arange(1643)
        self.assertTrue(torch.isin(sampled[sampled < 1643], demo_pool).all())
        self.assertEqual(len(learner.human_example_buffer.raw_ids), 1643)
        learner.offline_buffer.raw_ids = torch.arange(1643)
        self.assertTrue(
            torch.isin(learner._sample_action_likeness_raw_ids(), demo_pool).all()
        )
        learner.offline_buffer.raw_ids = torch.arange(1650)
        sampled = learner._sample_action_likeness_raw_ids()
        self.assertEqual(int((sampled < 1643).sum()), 128)
        self.assertEqual(int(((sampled >= 1643) & (sampled < 1650)).sum()), 128)

    def test_h_sampling_uses_raw_ids_not_contiguous_replay_positions(self) -> None:
        learner = object.__new__(LimitActionLearner)
        learner.config = LimitActionLearnerConfig(action_likeness_batch_size=256)
        learner.first20_cutoff = 3
        learner.offline_buffer = SimpleNamespace(raw_ids=torch.tensor([0, 1, 2, 40, 60, 91, 110]))
        learner.human_example_buffer = SimpleNamespace(raw_ids=torch.tensor([0, 1, 2, 40]))
        sampled = learner._sample_action_likeness_raw_ids()
        self.assertEqual(int((sampled < 3).sum()), 128)
        self.assertTrue(torch.isin(sampled, torch.tensor([0, 1, 2, 60, 91, 110])).all())

    def test_action_likeness_is_normalized_gaussian_density(self) -> None:
        human = torch.zeros((1, 4))
        action = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        peak = gaussian_density_peak(4)
        expected = peak * torch.exp(
            torch.tensor(-0.25 / (2.0 * ACTION_LIKENESS_SIGMA**2))
        )
        torch.testing.assert_close(proximity_target(action, human), expected[None])
        threshold, radius = gaussian_density_threshold_for_mass(4, retained_mass=0.95)
        self.assertAlmostEqual(threshold, 0.2857663459, places=6)
        self.assertAlmostEqual(radius, 0.5133692789, places=6)

        with self.assertRaisesRegex(ValueError, "fixed at"):
            LimitActionLearnerConfig(action_likeness_sigma=2.0)
        with self.assertRaisesRegex(ValueError, "reference_ratio"):
            LimitActionPolicy(
                LimitActionPolicyConfig(action_likeness_reference_ratio=-1.0)
            )

        config = LimitActionPolicyConfig()
        self.assertIsNone(config.action_likeness_threshold)
        with self.assertRaisesRegex(ValueError, "cv_epochs"):
            LimitActionLearnerConfig(action_radius_cv_epochs=0)
        self.assertAlmostEqual(config.replay_reference_floor_weight, 0.10)
        self.assertFalse(hasattr(config, "action_fence_actor_pull_weight"))
        self.assertAlmostEqual(config.human_iql_expectile, 0.75)
        self.assertAlmostEqual(config.correction_rank_weight, 2.0)
        self.assertEqual(config.correction_noise_samples, 8)
        self.assertAlmostEqual(config.correction_noise_std, 0.15)
        self.assertAlmostEqual(config.correction_reference_exclusion_radius, 0.05)
        self.assertFalse(hasattr(config, "hard_reference_max_distance"))

    def test_action_likeness_importance_loss_is_zero_at_target(self) -> None:
        learner = object.__new__(LimitActionLearner)
        learner.config = LimitActionLearnerConfig()
        learner.policy = small_policy()
        target_density = torch.tensor([0.01, 0.04, 1.0])

        loss, predicted_ratio = learner._action_likeness_loss(
            target_density, target_density, torch.ones_like(target_density)
        )

        torch.testing.assert_close(
            predicted_ratio,
            target_density / gaussian_density_peak(4),
        )
        self.assertAlmostEqual(float(loss), 0.0, places=7)

    def test_state_familiarity_bc_is_not_instantiated(self) -> None:
        policy = small_policy()
        self.assertNotIn("familiarity_checker", policy.networks)
        self.assertFalse(hasattr(policy, "familiarity_checker"))

    def test_inside_sac_bellman_bounds_keep_full_task_range(self) -> None:
        policy = small_policy()
        for critic in (
            policy.critic_1,
            policy.critic_2,
            policy.critic_target_1,
            policy.critic_target_2,
        ):
            torch.testing.assert_close(critic.lower_bound, torch.full((5,), -1.0))
        # The human IQL retains its physical finite-horizon return bounds.
        self.assertFalse(
            torch.equal(
                policy.human_critic_1.lower_bound,
                policy.critic_1.lower_bound,
            )
        )

    def test_replay_state_floors_iql_action_without_actor_gradient(self) -> None:
        policy = small_policy()
        policy.config.hard_iql_reference_floor = False  # legacy soft-floor path
        observation = torch.randn(2, 8)
        target = torch.ones((2, policy.config.horizons))
        with patch.object(policy, "human_critic_min", return_value=target):
            loss, metrics = policy._replay_reference_floor(
                {"observation": observation, "view_count": 1}
            )

        self.assertGreater(float(loss.detach()), 0.0)
        self.assertEqual(float(metrics["replay_reference_applied_fraction"]), 1.0)
        self.assertEqual(
            float(metrics["replay_reference_pairs"]),
            float(2 * policy.config.horizons),
        )
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None for parameter in policy.critic_1.parameters()
            )
        )
        self.assertFalse(
            any(parameter.grad is not None for parameter in policy.actor.parameters())
        )
        self.assertFalse(
            any(
                parameter.grad is not None
                for parameter in policy.recovery_actor.parameters()
            )
        )
        self.assertFalse(
            any(
                parameter.grad is not None
                for parameter in policy.human_critic_1.parameters()
            )
        )

    def test_runtime_policy_never_replaces_sac_action(self) -> None:
        policy = small_policy()
        observation = torch.randn(3, 8)
        sac_action = torch.ones((3, policy.config.action_dims))
        with patch.object(
            policy,
            "actor_actions",
            return_value=sac_action[:1],
        ):
            executed = torch.stack(
                [policy.sample_action(row) for row in observation], dim=0
            )
        torch.testing.assert_close(executed, sac_action)

    def test_recovery_reject_buffer_selects_last_ten_steps_before_takeover(
        self,
    ) -> None:
        transitions = [
            *[{"done": False, "info": {"is_intervene": False}} for _ in range(12)],
            {"done": False, "info": {"is_intervene": True}},
            {"done": False, "info": {"is_intervene": True}},
            {"done": False, "info": {"is_intervene": False}},
            {"done": True, "info": {"is_intervene": True}},
            {"done": True, "info": {"is_intervene": True}},
        ]
        recovery_ids = LimitActionLearner._recovery_reject_raw_ids(
            transitions, list(range(10, 27))
        )
        # The first takeover rejects raw rows 12..21 (the latest ten of twelve).
        # After intervention ends, the one new autonomous row is independently
        # attributed to the next takeover. Human rows are never selected.
        self.assertEqual(recovery_ids, [*range(12, 22), 24])

    def test_snapshot_round_trip_preserves_frozen_action_model(self) -> None:
        source = small_policy()
        source.action_likeness.finalize()
        source.recovery_initialized = True
        snapshot = source.export()
        target = small_policy()
        target.load(snapshot)
        self.assertTrue(bool(target.action_likeness.initialized.item()))
        self.assertTrue(target.recovery_initialized)
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in target.action_likeness.parameters()
            )
        )

    def test_actor_snapshot_contains_no_learner_state(self) -> None:
        from shared.actor_network import ActorInferencePolicy, ActorNetworkConfig

        source = small_policy()
        snapshot = source.export_actor()
        self.assertEqual(
            set(snapshot),
            {
                "format",
                "config",
                "encoder",
                "actor",
                "output_activation",
            },
        )
        self.assertNotIn("training_state", snapshot)
        target = ActorInferencePolicy(
            device="cpu",
            config=ActorNetworkConfig(**snapshot["config"]),
        )
        target.load(snapshot)
        observation = torch.randn(source.config.obs_dims)
        torch.testing.assert_close(
            target.sample_action(observation),
            source.sample_action(observation),
        )


class _BatchVectorTask(_VectorTask):
    def build_actions(
        self,
        raw_actions: list[dict],
        infos: list[dict],
        *,
        augment: bool = False,
    ) -> torch.Tensor:
        return torch.stack(
            [
                self.build_action(action, info, augment=augment)
                for action, info in zip(raw_actions, infos, strict=True)
            ]
        )


class LimitActionStartupTest(unittest.TestCase):
    def test_startup_continues_h_on_all_human_actions_and_can_update(self) -> None:
        with tempfile.TemporaryDirectory(prefix="srt-limit-action-") as tmp:
            root = Path(tmp)
            buffer_dir = root / "smoke" / "buffer"
            buffer_dir.mkdir(parents=True)
            rows = []
            for episode in range(20):
                for step in range(2):
                    done = step == 1
                    rows.append(
                        {
                            "raw_obs": {
                                "x": np.asarray(
                                    [episode / 20.0, float(step), 0.0, 1.0],
                                    dtype=np.float32,
                                )
                            },
                            "raw_action": {
                                "a": np.asarray(
                                    [0.1, -0.1, 0.2, 0.25 - 0.01 * episode],
                                    dtype=np.float32,
                                )
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

            policy = LimitActionPolicy(
                LimitActionPolicyConfig(
                    device="cpu",
                    obs_dims=4,
                    action_likeness_reference_ratio=0.75,
                    mechanism_obs_dims=4,
                    action_dims=4,
                    encoder_dim=8,
                    hidden_dim=12,
                    horizons=4,
                    bilinear_rank=2,
                    actor_horizons=None,
                    utd_ratio=1,
                )
            )
            learner = LimitActionLearner(
                config=LimitActionLearnerConfig(
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
                    human_iql_updates=1,
                    human_iql_log_every=1,
                    q_actor_warmup_updates=1,
                    action_likeness_updates=1,
                    action_radius_cv_epochs=1,
                    action_likeness_batch_size=8,
                    action_likeness_online_update_every=1,
                    recovery_pretrain_updates=1,
                    transitions_before_start=2,
                    save_every_seconds=0,
                ),
                task=_BatchVectorTask(),
                policy=policy,
            )
            try:
                self.assertEqual(policy.config.action_likeness_reference_ratio, 0.75)
                self.assertEqual(policy.actor.output_activation, "algebraic")
                self.assertTrue(bool(policy.action_likeness.initialized.item()))
                self.assertTrue(policy.recovery_initialized)
                self.assertTrue(
                    all(
                        parameter.requires_grad
                        for parameter in policy.action_likeness.parameters()
                    )
                )
                before = {
                    name: parameter.detach().clone()
                    for name, parameter in policy.action_likeness.named_parameters()
                }
                with (
                    patch.object(
                        learner.offline_buffer,
                        "sample",
                        side_effect=AssertionError(
                            "H must not sample all intervention rows"
                        ),
                    ),
                    patch.object(
                        learner.online_buffer,
                        "sample_raw_views",
                        wraps=learner.online_buffer.sample_raw_views,
                    ) as suffix_sample,
                ):
                    learner.step()
                self.assertTrue(suffix_sample.called)
                self.assertEqual(learner.train_steps, 1)
                self.assertIn(
                    "ActionFence/recorded_TD_masked_fraction",
                    learner._train_metric_sums,
                )
                self.assertIn("Loss_SAC/correction_rank", learner._train_metric_sums)
                self.assertIn(
                    "ActionFence/online_importance_MSE",
                    learner._train_metric_sums,
                )
                self.assertTrue(
                    any(
                        not torch.equal(before[name], parameter.detach())
                        for name, parameter in policy.action_likeness.named_parameters()
                    )
                )
                self.assertEqual(learner.offline_buffer.count, 40)
                self.assertEqual(learner.human_example_buffer.count, 40)
                self.assertEqual(learner.recovery_reject_buffer.count, 0)
                self.assertIn(
                    "Continuation/IQL_selected_fraction",
                    learner._train_metric_sums,
                )

                correction_rows = [
                    {
                        "raw_obs": {
                            "x": np.asarray([2.0, 0.0, 0.0, 1.0], dtype=np.float32)
                        },
                        "raw_action": {
                            "a": np.asarray([0.8, 0.0, 0.0, -1.0], dtype=np.float32)
                        },
                        "reward": -0.01,
                        "done": False,
                        "info": {"is_intervene": False},
                    },
                    {
                        "raw_obs": {
                            "x": np.asarray([2.0, 1.0, 0.0, 1.0], dtype=np.float32)
                        },
                        "raw_action": {
                            "a": np.asarray([0.0, 0.8, 0.0, 1.0], dtype=np.float32)
                        },
                        "reward": 1.0,
                        "done": True,
                        "info": {
                            "is_intervene": True,
                            "autonomous_action": {
                                "a": np.asarray([0.8, 0.0, 0.0, -1.0], dtype=np.float32)
                            },
                        },
                    },
                ]
                raw_start = learner.online_buffer.raw_count
                learner.online_buffer.add_many(correction_rows)
                learner._register_recovery_rejects(
                    correction_rows, [raw_start, raw_start + 1]
                )
                self.assertEqual(learner.recovery_reject_buffer.count, 1)
                grouped = learner.online_buffer.sample_raw_views(
                    torch.tensor([0, raw_start]),
                    policy.device,
                    views=1,
                    include_clean=True,
                )
                learner._mark_certified_suffix_next(grouped)
                torch.testing.assert_close(
                    grouped["certified_suffix_next"],
                    torch.tensor([True, False]),
                )
                correction_batch = learner._sample_recovery_reject_batch()
                self.assertIsNotNone(correction_batch)
                assert correction_batch is not None
                self.assertEqual(correction_batch["observation"].shape, (512, 4))
                torch.testing.assert_close(
                    correction_batch["autonomous_action"][0],
                    torch.tensor([0.8, 0.0, 0.0, -1.0]),
                )
                self.assertNotIn("human_action", correction_batch)
                # f is only an actor annotation: no learner sidecar/model.
                bad = dict(correction_rows[-1], reward=-0.01,
                           info={"is_intervene": True, "forced_failure": True},
                           raw_next_obs={"x": np.array([9.,8.,7.,6.],dtype=np.float32)})
                before_suffix = learner.human_example_buffer.count
                with patch.object(learner.transition_receiver, 'recv', return_value=[bad]):
                    learner.step()
                self.assertEqual(learner.human_example_buffer.count, before_suffix)
                self.assertFalse((learner.output_dir/'failure_annotations').exists())
                self.assertFalse(hasattr(learner, 'failure_states'))
                raw_count = learner.online_buffer.raw_count
                event = {'event':'failed_state','raw_obs':bad['raw_next_obs']}
                with patch.object(learner.transition_receiver,'recv',return_value=event):
                    learner.step()
                self.assertEqual(learner.online_buffer.raw_count,raw_count)
                self.assertFalse((learner.output_dir/'failure_annotations').exists())
            finally:
                learner.transition_receiver.close()
                learner.parameters_sender.close()
                learner.writer.close()


if __name__ == "__main__":
    unittest.main()
