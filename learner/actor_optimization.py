"""Cache action-independent Q/IQL/H work for one actor update only.

Never reuse the cache across critic/IQL/H updates. Compilation is learner-local;
snapshots remain ordinary state dictionaries, including on the ROCm actor.
"""
import torch


def native_cuda(device):
    return torch.device(device).type == 'cuda' and torch.version.hip is None


@torch.no_grad()
def prepare_actor_cache(policy, observation, gate):
    reference = policy.recovery_actions(gate)
    h_state = policy.action_likeness.encode_observation(gate)
    threshold = policy._fence_threshold(gate, reference)
    branches = []
    for critic in (policy.critic_1, policy.critic_2):
        tf = critic.teacher_encoder(observation)
        af = critic.adv_encoder(observation)
        anchor = critic.teacher(tf, reference) + critic.adv(af, reference)
        branches.append((tf, af, critic.reference_value(observation), anchor))
    return reference, h_state, threshold, branches


def cached_actor_objective(policy, observation, view_count, cache):
    reference, h_state, threshold, branches = cache
    action = policy.actor_actions(observation)
    with torch.no_grad():
        rejected = policy.action_likeness.density_from_state(h_state, action) < threshold
    values = []
    for critic, (tf, af, base, anchor) in zip((policy.critic_1, policy.critic_2), branches):
        inside = base + critic.teacher(tf, action) + critic.adv(af, action) - anchor
        outside = (base - policy.config.stay_step_penalty
                   - (action-reference).square().sum(-1, keepdim=True))
        values.append(torch.where(rejected[:, None], outside, inside))
    q = torch.minimum(*values)
    grouped = action.reshape(-1, view_count, action.shape[-1])
    consistency = (grouped-grouped.mean(1, keepdim=True)).square().mean()
    loss = -q.mean() + policy.config.actor_consistency_weight * consistency
    return loss, (q.detach(), action.detach(), rejected, consistency.detach())
