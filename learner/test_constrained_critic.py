from __future__ import annotations

import copy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

from shared.actor_network import ActorInferencePolicy, ActorNetworkConfig
from .constrained_critic import factual_td_loss
from .test_algorithm import small_policy


class ConstrainedCriticTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.policy = small_policy()
        # Historical v14 compatibility tests. New semantics are covered in
        # test_suffix_base_floor; old checkpoints retain their original Q.
        self.policy.config.suffix_only_reference_floor = False
        self.policy.config.hard_iql_reference_floor = True
        for critic in (self.policy.critic_1, self.policy.critic_2,
                       self.policy.critic_target_1, self.policy.critic_target_2):
            critic.reset_residual()
        self.obs = torch.randn(12, 8)

    def fence(self):
        stack = ExitStack()
        stack.enter_context(
            patch.object(
                self.policy,
                "recovery_actions",
                side_effect=lambda o: o.new_zeros((len(o), 4)),
            )
        )
        stack.enter_context(
            patch.object(
                self.policy.action_likeness,
                "forward",
                side_effect=lambda o, a: torch.exp(-10 * a.square().sum(-1)),
            )
        )
        return stack

    def test_outside_formula_gradient_and_reference(self):
        p = self.policy
        with self.fence():
            for critic in (
                p.critic_1,
                p.critic_2,
                p.critic_target_1,
                p.critic_target_2,
            ):
                action = torch.full((12, 4), 0.8, requires_grad=True)
                ref = torch.zeros_like(action)
                b = critic(self.obs, ref)
                q = critic(self.obs, action)
                torch.testing.assert_close(
                    q,
                    b
                    - p.config.stay_step_penalty
                    - action.square().sum(-1, keepdim=True),
                )
                gradient = torch.autograd.grad(q.mean(1).sum(), action)[0]
                torch.testing.assert_close(gradient, -2 * action)
                self.assertTrue((q < b).all())
                self.assertFalse(p.rejected_actions(self.obs, ref).any())
                self.assertTrue((q < -1).any())  # no gradient-erasing outside clamp

    def test_noisy_gate_and_reference_are_consistent(self):
        p = self.policy
        with patch.object(p, "recovery_actions", side_effect=lambda o: o[:, :4].tanh()):
            gate = self.obs
            noisy = gate + 0.5
            ref = p.recovery_actions(gate)
            self.assertFalse(p.rejected_actions(gate, ref).any())
            for critic in (p.critic_1, p.critic_target_1):
                if p.config.hard_iql_reference_floor:
                    value = p.human_critic_min(gate, ref)
                    expected = value + (1-value).clamp_min(0) * torch.sigmoid(
                        critic.base.net(critic.base_encoder(noisy))
                    )
                else:
                    expected = (
                        critic.teacher(critic.teacher_encoder(noisy), ref)
                        + critic.base(critic.base_encoder(noisy))
                    ).clamp(-1, 1)
                torch.testing.assert_close(
                    critic(noisy, ref, gate_observation=gate), expected
                )

    def test_factual_td_has_no_rejected_pair_gradient(self):
        q1 = torch.zeros(2, 5, requires_grad=True)
        q2 = torch.zeros(2, 5, requires_grad=True)
        smooth, mse, mask = factual_td_loss(
            q1,
            q2,
            torch.ones_like(q1),
            torch.ones_like(q1, dtype=torch.bool),
            torch.tensor([False, True]),
        )
        (smooth + mse).backward()
        self.assertTrue((q1.grad[0] != 0).all())
        self.assertTrue((q1.grad[1] == 0).all())
        self.assertTrue((q2.grad[1] == 0).all())
        self.assertEqual(mask.sum(), 5)

    def test_rejected_target_cannot_win_against_same_critic_reference(self):
        p = self.policy
        with (
            self.fence(),
            patch.object(p, "actor_actions", return_value=torch.ones(12, 4)),
            patch.object(p, "human_critic_min", return_value=torch.full((12, 5), -1.0)),
        ):
            b = torch.minimum(
                p.critic_target_1(self.obs, torch.zeros(12, 4)),
                p.critic_target_2(self.obs, torch.zeros(12, 4)),
            )
            y, metrics = p._certified_proposal_continuation(self.obs, None, 1)
            torch.testing.assert_close(y, b)
            self.assertEqual(metrics["target_rejected_fraction"], 1)

    def test_actor_gradient_does_not_update_critic_iql_or_h(self):
        p = self.policy
        for critic in (p.critic_1, p.critic_2):
            critic.requires_grad_(False)
        with torch.no_grad():
            p.actor.continuous.bias.fill_(2)
        with self.fence():
            loss, metrics = p._actor_loss(self.obs)
            loss.backward()
        self.assertTrue(
            any(
                x.grad is not None and x.grad.abs().sum() > 0
                for x in p.actor.parameters()
            )
        )
        self.assertEqual(metrics["actor_rejected_fraction"], 1)
        for name in (
            "critic_1",
            "critic_2",
            "recovery_actor",
            "recovery_encoder",
            "human_critic_1",
            "human_value",
            "action_likeness",
        ):
            self.assertTrue(
                all(x.grad is None for x in p.networks[name].parameters()), name
            )

    def test_corrective_ranking_cannot_raise_base(self):
        p = self.policy
        with (
            self.fence(),
            patch.object(
                p,
                "rejected_actions",
                side_effect=lambda o, a, reference=None: torch.zeros(
                    len(o), dtype=torch.bool
                ),
            ),
        ):
            loss, metrics = p._correction_rank_loss(
                dict(
                    observation=self.obs,
                    autonomous_action=torch.full((12, 4), 0.5),
                    view_count=1,
                )
            )
            loss.backward()
        self.assertGreater(float(loss.detach()), 0)
        for critic in (p.critic_1, p.critic_2):
            self.assertTrue(
                all(
                    x.grad is None or x.grad.abs().sum() == 0
                    for x in critic.base.parameters()
                )
            )
            self.assertTrue(
                any(
                    x.grad is not None and x.grad.abs().sum() > 0
                    for x in critic.adv.parameters()
                )
            )

    def test_rejected_advantage_updates_residual_not_base_or_actor(self):
        p = self.policy
        with self.fence(), patch.object(p, "actor_actions", return_value=torch.ones(12, 4)):
            # Hard Q is already correctly ranked; raw residual still needs learning.
            loss, metrics = p._rejected_advantage_loss(
                self.obs, torch.ones(12, 4), self.obs
            )
            self.assertGreater(float(loss), 0)
            self.assertEqual(float(metrics["rejected_advantage_sample_fraction"]), 1)
            loss.backward()
        for critic in (p.critic_1, p.critic_2):
            self.assertTrue(any(x.grad is not None and x.grad.abs().sum() > 0
                                for x in critic.adv.parameters()))
            for module in (critic.base, critic.base_encoder, critic.teacher, critic.teacher_encoder):
                self.assertTrue(all(x.grad is None for x in module.parameters()))
        for module in (p.actor, p.encoder_actor, p.action_likeness):
            self.assertTrue(all(x.grad is None for x in module.parameters()))

    def test_intervention_corrects_raw_advantage_even_when_rejected(self):
        p = self.policy
        p.config.correction_noise_samples = 0
        with self.fence():
            loss, metrics = p._correction_rank_loss(dict(
                observation=self.obs, autonomous_action=torch.ones(12, 4), view_count=1
            ))
            self.assertEqual(float(metrics["correction_cloud_valid_fraction"]), 1)
            self.assertGreater(float(loss), 0)
            loss.backward()
        self.assertTrue(any(x.grad is not None and x.grad.abs().sum() > 0
                            for x in p.critic_1.adv.parameters()))

    def test_accepted_actions_have_no_fence_soft_loss(self):
        p = self.policy
        with self.fence(), patch.object(p, "actor_actions", return_value=torch.zeros(12, 4)):
            loss, metrics = p._rejected_advantage_loss(self.obs, torch.zeros(12, 4), self.obs)
            self.assertEqual(float(loss), 0)
            self.assertEqual(float(metrics["rejected_advantage_sample_fraction"]), 0)

    def test_centered_ranking_has_zero_gradient_at_identical_actions(self):
        critic = self.policy.critic_1
        action = torch.randn(12, 4).tanh().requires_grad_()
        value = critic.ranking_advantage(self.obs, action, action)
        torch.testing.assert_close(value, torch.zeros_like(value))
        value.sum().backward()
        self.assertIsNone(action.grad)
        for name, parameter in critic.named_parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad), msg=name)

    def test_hard_floor_all_states_heads_and_gradient_isolation(self):
        p = self.policy
        for module in p.networks.values(): module.zero_grad(set_to_none=True)
        ref = p.recovery_actions(self.obs).detach()
        value = p.human_critic_min(self.obs, ref).detach()
        q = p.critic_min(self.obs, ref)
        self.assertTrue((q >= value - 1e-6).all())
        self.assertTrue((q <= 1+1e-6).all())
        q.mean().backward()
        for critic in (p.critic_1, p.critic_2):
            self.assertTrue(any(x.grad is not None and x.grad.abs().sum()>0
                                for x in critic.base.parameters()))
        for name in ('human_critic_1','human_critic_2','human_critic_encoder','recovery_actor','recovery_encoder'):
            self.assertTrue(all(x.grad is None for x in p.networks[name].parameters()), name)

    def test_hard_floor_diagnostics_do_not_build_loss_graph(self):
        p = self.policy
        batch = dict(observation=self.obs, view_count=2)
        for fn,key in ((p._human_floor,'human_floor_loss'),
                       (p._replay_reference_floor,'replay_reference_floor_loss')):
            loss, metrics = fn(batch)
            self.assertEqual(float(loss), 0)
            self.assertFalse(loss.requires_grad)
            self.assertNotIn(key, metrics)
            self.assertTrue(all(not v.requires_grad for v in metrics.values()))

    def test_v10_checkpoint_preserves_old_base_semantics(self):
        p = self.policy
        p.config.hard_iql_reference_floor = False
        state = p.export()
        q = p.critic_min(self.obs, p.recovery_actions(self.obs)).detach()
        target = small_policy()
        target.config.suffix_only_reference_floor = False
        target.load(state)
        self.assertFalse(target.config.hard_iql_reference_floor)
        torch.testing.assert_close(target.critic_min(self.obs, target.recovery_actions(self.obs)), q)

    def test_ranking_differentiates_both_residual_evaluations(self):
        critic = self.policy.critic_1
        outputs = []
        def retain(module, inputs, output):
            output.retain_grad()
            outputs.append(output)
        handle = critic.adv.register_forward_hook(retain)
        try:
            critic.ranking_advantage(self.obs, torch.ones(12, 4), torch.zeros(12, 4)).sum().backward()
        finally:
            handle.remove()
        self.assertEqual(len(outputs), 2)
        torch.testing.assert_close(outputs[0].grad, torch.ones_like(outputs[0]))
        torch.testing.assert_close(outputs[1].grad, -torch.ones_like(outputs[1]))

    def test_teacher_freezes_and_actor_is_disjoint(self):
        p = self.policy
        p.finalize_human_iql()
        for critic in (p.critic_1, p.critic_2):
            critic.requires_grad_(False)
            critic.requires_grad_(True)
            self.assertFalse(any(x.requires_grad for x in critic.teacher.parameters()))
            self.assertFalse(
                any(x.requires_grad for x in critic.teacher_encoder.parameters())
            )
        actor_ids = {id(x) for m in (p.actor, p.encoder_actor) for x in m.parameters()}
        critic_ids = {id(x) for m in (p.critic_1, p.critic_2) for x in m.parameters()}
        iql_ids = {
            id(x)
            for m in (p.human_critic_1, p.human_critic_encoder, p.recovery_actor)
            for x in m.parameters()
        }
        self.assertFalse(actor_ids & critic_ids)
        self.assertFalse(iql_ids & critic_ids)
        ref = p.recovery_actions(self.obs).detach()
        value = p.human_critic_min(self.obs, ref)
        torch.testing.assert_close(
            p.critic_1(self.obs, ref),
            value + (1-value).clamp_min(0) * torch.sigmoid(torch.tensor(-4.0)),
        )

    def test_resume_and_published_actor_match_after_update(self):
        p = self.policy
        p.finalize_human_iql()
        batch = dict(
            observation=self.obs,
            next_observation=self.obs + 0.1,
            action=torch.zeros(12, 4),
            reward=torch.full((12,), -0.01),
            done=torch.zeros(12, dtype=torch.bool),
        )
        p.update(batch)
        state = copy.deepcopy(p.export())
        target = small_policy()
        target.load(state)
        torch.testing.assert_close(
            p.actor_actions(self.obs), target.actor_actions(self.obs)
        )
        torch.testing.assert_close(
            p.critic_min(self.obs, batch["action"]),
            target.critic_min(self.obs, batch["action"]),
        )
        self.assertEqual(len(p.actor_optim.state), len(target.actor_optim.state))
        self.assertEqual(len(p.critic_optim.state), len(target.critic_optim.state))
        snapshot = p.export_actor()
        inference = ActorInferencePolicy(
            config=ActorNetworkConfig(**snapshot["config"]), device="cpu"
        )
        inference.load(snapshot)
        torch.testing.assert_close(
            inference.sample_action(self.obs[0]), p.actor_actions(self.obs[:1])[0]
        )
        with self.assertRaisesRegex(ValueError, "start a new experiment"):
            target.load({**state, "algorithm": "limit_action_v9_relative_h_iql_pull"})
        with self.assertRaisesRegex(ValueError, "fallback"):
            inference.load({**snapshot, "reference_max_distance": 0.35})

    def test_output_activation_derivative_and_legacy_inference(self):
        p = self.policy
        logits = torch.tensor([10.0], requires_grad=True)
        algebraic = logits / torch.sqrt(1 + logits.square())
        self.assertGreater(
            float(torch.autograd.grad(algebraic.sum(), logits)[0]), 0.0009
        )
        snapshot = p.export_actor()
        snapshot.pop("output_activation")
        inference = ActorInferencePolicy(
            config=ActorNetworkConfig(**snapshot["config"]), device="cpu"
        )
        inference.load(snapshot)
        self.assertEqual(inference.actor.output_activation, "tanh")


if __name__ == "__main__":
    unittest.main()
