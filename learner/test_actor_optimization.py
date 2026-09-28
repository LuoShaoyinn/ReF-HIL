import copy
import unittest
from unittest.mock import patch

import torch

from .actor_optimization import native_cuda
from .policy import LimitActionPolicy, LimitActionPolicyConfig


def policy(device='cpu'):
    return LimitActionPolicy(LimitActionPolicyConfig(
        device=device, obs_dims=8, mechanism_obs_dims=8, action_dims=4,
        encoder_dim=8, hidden_dim=12, horizons=5, bilinear_rank=3,
    ))


class ActorOptimizationTest(unittest.TestCase):
    def check_parity(self, device, compile_actor):
        torch.manual_seed(42)
        p = policy(device)
        p.config.compile_actor = compile_actor
        for critic in (p.critic_1, p.critic_2):
            with torch.no_grad():
                critic.adv.output.weight.normal_(std=.03)
            critic.requires_grad_(False)
        obs = torch.randn(12, 8, device=device)
        gate = torch.randn_like(obs)
        parameters = [*p.actor.parameters(), *p.encoder_actor.parameters()]
        for threshold in (torch.zeros(12,device=device), torch.full((12,),1e6,device=device),
                          (torch.arange(12,device=device) % 2).float()*1e6):
            with patch.object(p, '_fence_threshold', return_value=threshold):
                p.config.optimize_actor = False
                expected, expected_metrics = p._actor_loss(obs, gate, 2)
                expected_grad = torch.autograd.grad(expected, parameters)
                p.config.optimize_actor = True
                actual, metrics = p._actor_loss(obs, gate, 2)
                actual_grad = torch.autograd.grad(actual, parameters)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            self.assertEqual(metrics.keys(), expected_metrics.keys())
            for key in metrics:
                torch.testing.assert_close(metrics[key], expected_metrics[key], atol=1e-5, rtol=1e-5)
            for a,b in zip(actual_grad, expected_grad):
                torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4)
            for name, network in p.networks.items():
                if name not in ('actor','encoder_actor'):
                    self.assertTrue(all(v.grad is None for v in network.parameters()),name)

    def test_cpu_loss_gradient_and_metrics_parity(self):
        self.check_parity('cpu', False)

    @unittest.skipUnless(torch.cuda.is_available() and native_cuda('cuda'), 'native CUDA required')
    def test_compiled_cuda_loss_gradient_and_metrics_parity(self):
        self.check_parity('cuda', True)

    def test_resume_keeps_moments_but_applies_configured_actor_lr(self):
        p = policy()
        p.actor_optim.param_groups[0]['lr'] = 1e-4
        for group in p.actor_optim.param_groups:
            for parameter in group['params']:
                parameter.grad = torch.ones_like(parameter)
        p.actor_optim.step()
        state = copy.deepcopy(p.export())
        restored = policy()
        restored.load(state)
        self.assertEqual(restored.actor_optim.param_groups[0]['lr'], 5e-4)
        self.assertFalse(restored.actor_optim.param_groups[0]['fused'])
        a = next(iter(p.actor_optim.state.values()))
        b = next(iter(restored.actor_optim.state.values()))
        torch.testing.assert_close(a['exp_avg'], b['exp_avg'])
        self.assertEqual(b['step'].device.type, 'cpu')
        self.assertIsNone(restored._actor_kernel)

    def test_rocm_does_not_enable_cuda_compiler_or_fused_adam(self):
        with patch.object(torch.version, 'hip', '7.2'):
            self.assertFalse(native_cuda('cuda'))


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
