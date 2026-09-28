import unittest
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from .unrecoverable import (failure_vector, state_supervision, UnrecoverableAnnotations,
                            iql_failure_losses, update_iql_failure)
from .iql_modules import build_vector_bellman_target
from .test_algorithm import small_policy


class UnrecoverableTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_all_horizon_failure_recursion(self):
        for gamma in (1., .99):
            v=failure_vector(200,gamma,.005)
            self.assertAlmostEqual(v[0].item(),-.005)
            torch.testing.assert_close(v[1:],-.005+gamma*v[:-1])
        self.assertAlmostEqual(failure_vector(200,1,.005)[-1].item(),-1.)

    def test_rebuild_and_zero_action_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'buffer').mkdir()
            rows=[dict(done=True,reward=-.01,info={'forced_failure':True},raw_next_obs={'x':[1.,2.]}),
                  dict(done=True,reward=-.01,info={'unrecoverable_next':True},raw_next_obs={'x':[3.,4.]})]
            for row in rows:
                row.update(raw_obs={'x':[0.,0.]},raw_action={'a':[.2]})
            with (root/'buffer'/'0000.pkl').open('wb') as f:pickle.dump(rows,f)
            learner=SimpleNamespace(output_dir=root,policy=SimpleNamespace(device='cpu'),
                online_buffer=SimpleNamespace(config=SimpleNamespace(capacity=20)),
                task=SimpleNamespace(build_obs=lambda state,info,augment=False:torch.tensor(state['x']),
                    build_observations=lambda states,infos,augment=False:torch.tensor([z['x'] for z in states]),
                    build_actions=lambda actions,infos:torch.tensor([z['a'] for z in actions])))
            store=UnrecoverableAnnotations(learner)
            self.assertEqual(len(store.states),1)
            self.assertEqual(store.mask[:2].tolist(),[False,True])
            self.assertEqual(store.trajectory_mask[:2].tolist(),[False,True])
            store.add_state({'x':[5.,6.]},persist=True)
            restored=UnrecoverableAnnotations(learner)
            self.assertEqual(len(restored.states),2)
            self.assertEqual(restored.trajectory_mask[:2].tolist(),[False,True])
            batch={};restored.attach(batch,torch.tensor([0,1]),2)
            self.assertEqual(batch['unrecoverable_next'].tolist(),[False,False,True,True])
            restored.attach_suffix(batch,torch.tensor([0,1]),2)
            self.assertEqual(batch['failure_trajectory'].tolist(),[False,False,True,True])

    def test_whole_failed_trajectory_selected_and_actor_excluded(self):
        from .iql_floor_runtime import HumanIQLGroundedLearner
        from .unrecoverable import successful_only
        rows=[dict(done=False,reward=-.01,info={'is_intervene':False}),
              dict(done=True,reward=-.01,info={'is_intervene':False,'unrecoverable_next':True})]
        self.assertEqual(len(HumanIQLGroundedLearner._complete_human_suffix(rows,10)),2)
        rows[-1]['info'].pop('unrecoverable_next')
        self.assertEqual(HumanIQLGroundedLearner._complete_human_suffix(rows,1),[])
        batch=dict(observation=torch.randn(4,8),failure_trajectory=torch.tensor([True,True,False,False]),view_count=2)
        kept=successful_only(batch)
        self.assertEqual(len(kept['observation']),2)
        self.assertEqual(kept['view_count'],2)
        batch['failure_trajectory'][:]=True
        self.assertIsNone(successful_only(batch))

    def test_failed_suffix_updates_iql_not_imitation_or_floor(self):
        p=small_policy();p.finalize_human_iql()
        obs=torch.randn(8,8)
        batch=dict(observation=obs,next_observation=obs+.1,action=torch.zeros(8,4),
            reward=torch.full((8,),-.01),done=torch.ones(8,dtype=torch.bool),
            failure_trajectory=torch.ones(8,dtype=torch.bool),unrecoverable_next=torch.ones(8,dtype=torch.bool))
        before=[x.detach().clone() for x in p.recovery_actor.parameters()]
        metrics=p._update_moving_human_iql(batch)
        self.assertTrue(torch.isfinite(metrics['human_iql_loss']))
        for a,b in zip(before,p.recovery_actor.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        loss,_=p._human_floor(batch)
        self.assertEqual(loss.item(),0)

    def test_entering_failure_has_one_step_shift(self):
        v=failure_vector(5,.99,.01)
        y,mask=build_vector_bellman_target(torch.tensor([-.02]),torch.tensor([False]),
            torch.tensor([True]),torch.tensor([True]),v[None],gamma=.99,lower_bound=torch.full((5,),-1.))
        self.assertTrue(mask.all())
        self.assertAlmostEqual(y[0,0].item(),-.02)
        torch.testing.assert_close(y[0,1:],-.02+.99*v[:-1])

    def test_state_loss_trains_base_and_adv_not_iql(self):
        p=small_policy();obs=torch.randn(8,8)
        state_supervision(p,obs).backward()
        for m in (p.critic_1.base,p.critic_1.adv):
            self.assertTrue(any(x.grad is not None and x.grad.abs().sum()>0 for x in m.parameters()))
        self.assertTrue(all(x.grad is None for x in p.recovery_actor.parameters()))

    def test_failed_incoming_pair_not_masked_by_fence(self):
        p=small_policy();p.finalize_human_iql();obs=torch.randn(8,8)
        batch=dict(observation=obs,next_observation=obs+.2,action=torch.zeros(8,4),
            reward=torch.full((8,),-.01),done=torch.ones(8,dtype=torch.bool),
            unrecoverable_next=torch.ones(8,dtype=torch.bool),unrecoverable_observation=obs+.2)
        with patch.object(p,'rejected_actions',side_effect=lambda o,a,reference=None: torch.ones(len(o),dtype=torch.bool)):
            metrics=p.update(batch)
        self.assertEqual(metrics['critic_valid_pairs'].item(),40)
        self.assertTrue(torch.isfinite(metrics['unrecoverable_state_loss']))
        self.assertTrue(torch.isfinite(metrics['unrecoverable_iql_q_loss']))
        self.assertTrue(torch.isfinite(metrics['unrecoverable_iql_v_loss']))

    def test_iql_failure_gradients_are_isolated(self):
        p=small_policy();obs=torch.randn(8,8)
        q,v=iql_failure_losses(p,obs)
        (q+v).backward()
        for module in (p.human_critic_1,p.human_value):
            self.assertTrue(any(x.grad is not None and x.grad.abs().sum()>0 for x in module.parameters()))
        for module in (p.actor,p.recovery_actor,p.critic_1,p.critic_2):
            self.assertTrue(all(x.grad is None for x in module.parameters()))

    def test_iql_anchor_fits_without_actor_update(self):
        torch.manual_seed(12)
        p=small_policy();obs=torch.randn(8,8)
        before=[x.detach().clone() for x in p.recovery_actor.parameters()]
        torch.manual_seed(3);q0,v0=iql_failure_losses(p,obs)
        for _ in range(30):update_iql_failure(p,obs)
        torch.manual_seed(3);q1,v1=iql_failure_losses(p,obs)
        self.assertLess((q1+v1).item(),(q0+v0).item())
        for a,b in zip(before,p.recovery_actor.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_iql_incoming_uses_exact_failure_shift(self):
        p=small_policy();obs=torch.randn(8,8)
        transitions=dict(observation=obs,action=torch.zeros(8,4),next_observation=obs+.1,
            reward=torch.full((8,),-.02),unrecoverable_next=torch.ones(8,dtype=torch.bool))
        torch.manual_seed(3);q0,_=iql_failure_losses(p,obs)
        torch.manual_seed(3);q1,_=iql_failure_losses(p,obs,transitions)
        target=failure_vector(5,p.config.gamma,p.config.stay_step_penalty)
        y=torch.full((8,5),-.02);y[:,1:]+=p.config.gamma*target[:-1]
        feat=p.human_critic_encoder(obs)
        expected=sum((c(feat,transitions['action'])-y).square().mean()
                     for c in (p.human_critic_1,p.human_critic_2))/2
        torch.testing.assert_close(q1-q0,expected)

if __name__=='__main__': unittest.main()
