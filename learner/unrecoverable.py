"""Explicit human state annotations, without a learned failure detector."""
import pickle

import torch


def failure_vector(horizons, gamma, step_penalty, *, device=None, dtype=torch.float32):
    return -float(step_penalty) * (float(gamma) ** torch.arange(
        horizons, device=device, dtype=dtype)).cumsum(0)


class UnrecoverableAnnotations:
    def __init__(self, learner):
        self.learner = learner
        self.directory = learner.output_dir / 'unrecoverable_annotations'
        self.states = []
        self.trajectory_mask = torch.zeros(learner.online_buffer.config.capacity,
                                           device=learner.policy.device, dtype=torch.bool)
        self.pending_episode = []
        self.mask = torch.zeros(learner.online_buffer.config.capacity,
                                device=learner.policy.device, dtype=torch.bool)
        # Only new explicit semantics; historical forced_failure is not enough.
        for path in sorted((learner.output_dir / 'buffer').glob('*.pkl')):
            with path.open('rb') as stream:
                rows = pickle.load(stream)
            start = getattr(self, '_loaded_rows', 0)
            self.register(rows, start)
            self._loaded_rows = start + len(rows)
        if self.directory.exists():
            for path in sorted(self.directory.glob('*.pkl')):
                with path.open('rb') as stream:
                    self.add_state(pickle.load(stream)['raw_obs'])

    def add_state(self, state, *, persist=False):
        encoded = self.learner.task.build_obs(state, {}, augment=False).detach()
        self.states.append(encoded.to(self.learner.policy.device).float().reshape(-1))
        if persist:
            self.directory.mkdir(exist_ok=True)
            index = max((int(p.stem) for p in self.directory.glob('*.pkl')), default=-1) + 1
            with (self.directory / f'{index:08d}.pkl').open('xb') as stream:
                pickle.dump({'raw_obs': state}, stream)

    def register(self, rows, start):
        for i, row in enumerate(rows):
            self.pending_episode.append(row)
            if row.get('done', False):
                if row.get('info', {}).get('unrecoverable_next', False):
                    if float(row['reward']) > 0:
                        raise ValueError('unrecoverable episode cannot end in success')
                    self.trajectory_mask[start+i+1-len(self.pending_episode):start+i+1] = True
                self.pending_episode = []
            if not row.get('info', {}).get('unrecoverable_next', False):
                continue
            if not row['done'] or float(row['reward']) > 0:
                raise ValueError('unrecoverable annotation must end a non-success episode')
            self.add_state(row['raw_next_obs'])
            self.mask[start+i] = True

    def attach(self, batch, raw_ids, views):
        if not self.states:
            return
        marked = self.mask[raw_ids.to(self.mask.device)].repeat_interleave(views)
        if bool(marked.any()):
            batch['unrecoverable_next'] = marked
        if self.states:
            ids = torch.randint(len(self.states), (min(256, len(self.states)),)).tolist()
            batch['unrecoverable_observation'] = torch.stack([self.states[i] for i in ids])

    def attach_suffix(self, batch, raw_ids, views):
        ids = raw_ids.to(self.mask.device)
        batch['failure_trajectory'] = self.trajectory_mask[ids].repeat_interleave(views)
        batch['unrecoverable_next'] = self.mask[ids].repeat_interleave(views)


def successful_only(batch):
    """Exclude failure trajectories from human imitation and positive floors."""
    if batch is None or 'failure_trajectory' not in batch:
        return batch
    keep = ~batch['failure_trajectory'].bool()
    if not bool(keep.any()):
        return None
    return {k: v[keep] if isinstance(v, torch.Tensor) and v.ndim and len(v)==len(keep) else v
            for k,v in batch.items()}


def iql_failure_losses(policy, observation, transitions=None):
    """Independent IQL supervision; never differentiate actor or SAC networks."""
    zero = torch.zeros((), device=policy.device)
    if observation is None:
        return zero, zero
    target = failure_vector(policy.config.horizons, policy.config.gamma,
                            policy.config.stay_step_penalty, device=observation.device)
    with torch.no_grad():
        ref = policy.recovery_actions(observation)
        actions = [ref, policy.actor_actions(observation), torch.empty_like(ref).uniform_(-1, 1)]
    features = policy.human_critic_encoder(observation)
    q_loss = torch.stack([(critic(features, action)-target).square().mean()
                         for critic in (policy.human_critic_1, policy.human_critic_2)
                         for action in actions]).mean()
    value = policy.human_value(policy.human_value_encoder(observation))
    v_loss = (value-target).square().mean()
    if transitions is not None:
        obs, nxt, action = (transitions[k] for k in ('observation', 'next_observation', 'action'))
        with torch.no_grad():
            continuation = policy.human_value(policy.human_value_encoder(nxt))
            continuation = torch.where(transitions['unrecoverable_next'][:, None], target, continuation)
            y = transitions['reward'][:, None].expand(-1, policy.config.horizons).clone()
            y[:, 1:] += policy.config.gamma * continuation[:, :-1]
        feat = policy.human_critic_encoder(obs)
        q_loss = q_loss + sum((c(feat, action)-y).square().mean()
                             for c in (policy.human_critic_1, policy.human_critic_2)) / 2
        # Do not force predecessor V to the outcome of a bad action. Its value
        # remains learned from supported human actions/explicit state anchors.
    return q_loss, v_loss


def update_iql_failure(policy, observation, transitions=None):
    if observation is None:
        return {}
    q_loss, v_loss = iql_failure_losses(policy, observation, transitions)
    policy.human_critic_optim.zero_grad(set_to_none=True)
    q_loss.backward()
    torch.nn.utils.clip_grad_norm_([*policy.human_critic_encoder.parameters(),
        *policy.human_critic_1.parameters(), *policy.human_critic_2.parameters()], 10.)
    policy.human_critic_optim.step()
    policy.human_value_optim.zero_grad(set_to_none=True)
    v_loss.backward()
    torch.nn.utils.clip_grad_norm_([*policy.human_value_encoder.parameters(),
                                  *policy.human_value.parameters()], 10.)
    policy.human_value_optim.step()
    return {'unrecoverable_iql_q_loss': q_loss.detach(),
            'unrecoverable_iql_v_loss': v_loss.detach()}


def state_supervision(policy, observation):
    """Fit B and the raw action field; never train against the outside surrogate."""
    if observation is None:
        return torch.zeros((), device=policy.device)
    target = failure_vector(policy.config.horizons, policy.config.gamma,
                            policy.config.stay_step_penalty, device=observation.device)
    with torch.no_grad():
        reference = policy.recovery_actions(observation)
        actions = [reference, policy.actor_actions(observation),
                   torch.empty_like(reference).uniform_(-1, 1)]
    losses = []
    for critic in (policy.critic_1, policy.critic_2):
        base = critic.reference_value(observation)
        af = critic.adv_encoder(observation)
        tf = critic.teacher_encoder(observation)
        anchor = critic.teacher(tf, reference) + critic.adv(af, reference)
        # Separate B fit also gives its encoder a gradient at the failed state.
        losses.append((base-target).square().mean())
        for action in actions:
            q = base + critic.teacher(tf, action) + critic.adv(af, action) - anchor
            losses.append((q-target).square().mean())
    return torch.stack(losses).mean()
