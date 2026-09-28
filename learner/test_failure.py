"""Actor-only f annotations: no learned failure behavior in production."""
import copy
import unittest
from unittest.mock import patch
import torch
from actor.runtime import Actor, ActorConfig
from actor.terminal_success import TerminalSuccessKey
from actor.test_actor import _Operator, _Policy, _Robot, _Socket, _Task
from learner.test_algorithm import small_policy

class AnnotationOnlyTest(unittest.TestCase):
    def test_pending_enter_does_not_block_new_failure(self):
        key = TerminalSuccessKey()
        key._fd = 123
        key._success_pending = True
        with (
            patch(
                "actor.terminal_success.select.select",
                side_effect=[([123], [], []), ([], [], [])],
            ),
            patch("actor.terminal_success.os.read", return_value=b"f"),
        ):
            self.assertTrue(key.failure_requested())
        self.assertTrue(key._success_pending)

    def actor(self, requests, success=False):
        return Actor(
            ActorConfig(max_steps_per_episode=20, loop_sleep_s=0),
            _Task(),
            _Policy(),
            _Robot(),
            _Operator(),
            lambda _: success,
            force_failure_requested=lambda: next(requests),
        )

    def test_f_before_first_action_sends_only_state(self):
        with (
            patch("actor.runtime.Sender", _Socket),
            patch("actor.runtime.Receiver", _Socket),
        ):
            actor = self.actor(iter([True]))
            self.assertEqual(actor.run_episode(), [])
            self.assertEqual(actor.robot.actions, [])
            self.assertEqual(actor.learner_sender.sent[0]["event"], "failed_state")
            self.assertTrue(actor.learner_sender.sent[0]['unrecoverable'])

    def test_f_after_action_overrides_success(self):
        with (
            patch("actor.runtime.Sender", _Socket),
            patch("actor.runtime.Receiver", _Socket),
        ):
            actor = self.actor(iter([False, True]), success=True)
            rows = actor.run_episode()
            self.assertEqual(len(actor.robot.actions), 1)
            self.assertEqual(
                rows[-1]["reward"], actor.config.default_not_success_reward
            )
            self.assertTrue(rows[-1]["done"])
            self.assertTrue(rows[-1]["info"]["forced_failure"])
            self.assertTrue(rows[-1]['info']['unrecoverable_next'])
            self.assertEqual(rows[-1]["raw_next_obs"]["frame"], 1)

    def test_f_between_steps_does_not_take_second_action(self):
        with (
            patch("actor.runtime.Sender", _Socket),
            patch("actor.runtime.Receiver", _Socket),
        ):
            actor = self.actor(iter([False, False, True]))
            rows = actor.run_episode()
            self.assertEqual(len(actor.robot.actions), 1)
            self.assertEqual(
                rows[-1]["reward"], actor.config.default_not_success_reward
            )
            self.assertEqual(rows[-1]["raw_obs"]["frame"], 0)
            self.assertEqual(rows[-1]["raw_next_obs"]["frame"], 1)

    def test_keys_preserve_f_and_enter_and_drain(self):
        key = TerminalSuccessKey()
        key._fd = 123
        with (
            patch(
                "actor.terminal_success.select.select",
                side_effect=[([123], [], []), ([], [], []), ([], [], [])],
            ),
            patch("actor.terminal_success.os.read", return_value=b"f\n"),
        ):
            self.assertTrue(key.failure_requested())
            self.assertTrue(key.success_requested())
        key._failure_pending = True
        with patch("actor.terminal_success.termios.tcflush"):
            key.drain()
        self.assertFalse(key._failure_pending)

    def test_policy_has_no_failure_model_or_optimizer(self):
        p = small_policy()
        self.assertFalse(hasattr(p, "failure_classifier"))
        self.assertFalse(hasattr(p, "failed_states"))
        self.assertFalse(any("failure" in k for k in p.networks))
        self.assertFalse(any("failure" in k for k in p.optimizers))
        self.assertEqual(p.export()["algorithm"], "limit_action_v15_suffix_base_floor")

    def test_legacy_checkpoint_failure_fields_are_ignored(self):
        p = small_policy()
        state = copy.deepcopy(p.export())
        state["algorithm"] = "limit_action_v13_failure_vector"
        state["failure_classifier"] = {"obsolete": torch.ones(1)}
        state["training_state"]["failure_optimizer"] = {"obsolete": True}
        q = small_policy()
        q.load(state)
        self.assertFalse(hasattr(q, "failure_classifier"))
        self.assertNotIn("failure_optimizer", q.export()["training_state"])

    def test_failure_flag_does_not_change_update(self):
        p = small_policy()
        p.human_iql_initialized = True
        q = small_policy()
        q.load(copy.deepcopy(p.export()))
        batch = dict(observation=torch.randn(4,8),next_observation=torch.randn(4,8),
                     action=torch.rand(4,4)*2-1,reward=torch.full((4,),-.01),
                     done=torch.ones(4,dtype=torch.bool),timeout_next_valid=torch.ones(4,dtype=torch.bool))
        tagged = dict(batch, failure_next=torch.ones(4,dtype=torch.bool))
        torch.manual_seed(731)
        a = p.update(batch)
        torch.manual_seed(731)
        b = q.update(tagged)
        torch.testing.assert_close(a["critic_loss"], b["critic_loss"])
        for name, model in p.networks.items():
            for k, value in model.state_dict().items():
                torch.testing.assert_close(value, q.networks[name].state_dict()[k])

if __name__ == "__main__":
    unittest.main()
