"""Gradient contracts for the isolated suffix-only reference floor."""
import unittest
from unittest.mock import patch
import torch
from .test_algorithm import small_policy


class SuffixBaseFloorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(18)
        self.p = small_policy()
        self.obs = torch.randn(12, 8)

    def test_floor_updates_only_base(self):
        p = self.p
        with patch.object(p, "human_critic_min", return_value=torch.ones(12, 5)):
            loss, _ = p._human_floor(dict(observation=self.obs, view_count=2))
            loss.backward()
        for critic in (p.critic_1, p.critic_2):
            self.assertTrue(any(x.grad is not None and x.grad.abs().sum() > 0 for x in critic.base.parameters()))
            for module in (critic.adv, critic.adv_encoder, critic.teacher, critic.teacher_encoder):
                self.assertTrue(all(x.grad is None for x in module.parameters()))
        for module in (p.human_critic_1, p.human_critic_encoder, p.recovery_actor):
            self.assertTrue(all(x.grad is None for x in module.parameters()))

    def test_td_updates_both_branches_and_reference_is_base(self):
        p = self.p
        with patch.object(p, "rejected_actions", return_value=torch.zeros(12, dtype=torch.bool)):
            ref = p.recovery_actions(self.obs).detach()
            c = p.critic_1
            torch.testing.assert_close(c(self.obs, ref), c.reference_value(self.obs))
            (c(self.obs, torch.ones_like(ref)) - 0.7).square().mean().backward()
        for module in (c.base, c.adv):
            self.assertTrue(any(x.grad is not None and x.grad.abs().sum() > 0 for x in module.parameters()))
        self.assertFalse({id(x) for x in c.base_encoder.parameters()} & {id(x) for x in c.adv_encoder.parameters()})

    def test_no_global_iql_target_floor(self):
        p = self.p
        with patch.object(p, "human_critic_min", return_value=torch.full((12, 5), 100.0)):
            y, metrics = p._certified_proposal_continuation(self.obs, None, 1)
        self.assertTrue((y < 100).all())
        self.assertEqual(float(metrics["target_iql_selected_fraction"]), 0)

    def test_rejections_only_update_advantage(self):
        p = self.p
        with patch.object(p, "rejected_actions", side_effect=lambda o, a, reference=None: torch.ones(len(o), dtype=torch.bool)):
            loss, _ = p._rejected_advantage_loss(self.obs, torch.ones(12, 4), self.obs)
            loss.backward()
        for c in (p.critic_1, p.critic_2):
            self.assertTrue(any(x.grad is not None and x.grad.abs().sum() > 0 for x in c.adv.parameters()))
            for m in (c.base, c.base_encoder):
                self.assertTrue(all(x.grad is None for x in m.parameters()))

    def test_full_update_does_not_query_replay_floor(self):
        p = self.p
        p.finalize_human_iql()
        batch = dict(observation=self.obs, next_observation=self.obs + .1,
                     action=torch.zeros(12, 4), reward=torch.full((12,), -.01),
                     done=torch.zeros(12, dtype=torch.bool))
        with patch.object(p, "_replay_reference_floor", side_effect=AssertionError("global floor called")):
            p.update(batch)

    def test_roundtrip(self):
        p = self.p
        state = p.export()
        self.assertEqual(state['algorithm'], 'limit_action_v15_suffix_base_floor')
        other = small_policy()
        other.load(state)
        ref = p.recovery_actions(self.obs).detach()
        torch.testing.assert_close(p.critic_min(self.obs, ref), other.critic_min(self.obs, ref))

    def test_missing_suffix_has_no_floor(self):
        loss, _ = self.p._human_floor(None)
        self.assertEqual(float(loss), 0)
        self.assertFalse(loss.requires_grad)


if __name__ == '__main__':
    unittest.main()
