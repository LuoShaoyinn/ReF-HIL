"""Historical offline diagnostic helper; NOT used by the production learner.

Retained for the all-15 audit scripts. Production f handling is annotation-only.
"""

import pickle

import torch
from torch import nn
from torch.nn import functional as F


def unrecoverable_value(
    horizons, gamma, step_penalty, *, device=None, dtype=torch.float32
):
    """No success is reachable: F[0]=0, F[h]=-cost+gamma*F[h-1]."""
    powers = torch.arange(horizons, device=device, dtype=dtype)
    return -float(step_penalty) * (float(gamma) ** powers).cumsum(0)


class FailureClassifier(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dims, 128), nn.SiLU(), nn.Linear(128, 1))
        nn.init.constant_(self.net[-1].bias, -6.0)
        self.register_buffer("mean", torch.zeros(dims))
        self.register_buffer("std", torch.ones(dims))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("has_positive", torch.tensor(False))
        self.register_buffer("threshold", torch.tensor(0.95))
        self.register_buffer("anchors", torch.empty(0, dims))
        generator = torch.Generator().manual_seed(901)
        self.register_buffer(
            "fingerprint", torch.randn(dims, 2, generator=generator) / dims**0.5
        )

    def forward(self, obs):
        return self.net((obs - self.mean) / self.std).squeeze(-1)

    @torch.no_grad()
    def failed(self, obs):
        result = self.has_positive & (self(obs).sigmoid() >= self.threshold)
        # Cheap fingerprint prefilter, followed by actual vector comparison.
        # No nearest-neighbour radius / inferred region is imposed by anchors.
        if len(self.anchors):
            candidate = (
                (obs @ self.fingerprint)[:, None]
                - (self.anchors @ self.fingerprint)[None]
            ).abs().amax(-1) < 1e-4
            rows, cols = candidate.nonzero(as_tuple=True)
            exact = (obs[rows] - self.anchors[cols]).abs().amax(-1) <= 1e-6
            result[rows[exact]] = True
        return result

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "anchors"
        if key in state_dict:
            self.anchors = torch.empty_like(state_dict[key], device=self.mean.device)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class FailureTraining:
    def _initialize_failure_training(self):
        self.failure_dir = self.output_dir / "failure_annotations"
        self.failure_dir.mkdir(exist_ok=True)
        self.failure_ids = set()
        self.failure_id_tensor = torch.empty(
            0, device=self.policy.device, dtype=torch.long
        )
        self.failure_states = []
        for path in sorted(self.failure_dir.glob("*.pkl")):
            with path.open("rb") as f:
                data = pickle.load(f)
            self._register_failure(data["raw_obs"], data.get("raw_id"), persist=False)
        # Recover factual labels even if shutdown occurred before annotation saving.
        raw_id = 0
        successful_ids = []
        episode_ids = []
        for path in sorted((self.output_dir / "buffer").glob("*.pkl")):
            with path.open("rb") as f:
                rows = pickle.load(f)
            for row in rows:
                episode_ids.append(raw_id)
                if row.get("info", {}).get("forced_failure", False):
                    self._register_failure(row["raw_next_obs"], raw_id)
                if row["done"]:
                    if float(row["reward"]) > 0 and not row.get("info", {}).get(
                        "forced_failure", False
                    ):
                        successful_ids.extend(episode_ids)
                    episode_ids = []
                raw_id += 1
        ids = torch.tensor(
            successful_ids, device=self.online_buffer.data_obs.device, dtype=torch.long
        )
        pos = self.online_buffer.raw_positions(ids)
        self.failure_negatives = (
            self.online_buffer.data_obs[pos].detach().to(self.policy.device)
        )
        model = self.policy.failure_classifier
        if not bool(model.initialized):
            demo_ids = self.human_example_buffer.raw_ids[: self.first20_cutoff]
            demo = self.online_buffer.data_obs[
                self.online_buffer.raw_positions(demo_ids)
            ].to(self.policy.device)
            with torch.no_grad():
                model.mean.copy_(demo.mean(0))
                model.std.copy_(demo.std(0, unbiased=False).clamp_min(0.1))
                model.initialized.fill_(True)
            for _ in range(200):
                self._update_failure_classifier()
        # Recompute the gate against rebuilt success labels, including on resume.
        self._refresh_failure_threshold()

    def _register_successful_episode(self, transitions, raw_ids):
        if not transitions or not transitions[-1]["done"]:
            return False
        terminal = transitions[-1]
        if float(terminal["reward"]) <= 0 or terminal.get("info", {}).get(
            "forced_failure", False
        ):
            return False
        ids = torch.tensor(
            raw_ids, device=self.online_buffer.data_obs.device, dtype=torch.long
        )
        states = (
            self.online_buffer.data_obs[self.online_buffer.raw_positions(ids)]
            .detach()
            .to(self.policy.device)
        )
        self.failure_negatives = torch.cat((self.failure_negatives, states))
        self._refresh_failure_threshold()
        return True

    @torch.no_grad()
    def _refresh_failure_threshold(self):
        model = self.policy.failure_classifier
        maximum = model.threshold.new_zeros(())
        for batch in self.failure_negatives.split(1024):
            maximum = torch.maximum(maximum, model(batch).sigmoid().max())
        model.threshold.copy_(torch.maximum(maximum + 0.001, maximum.new_tensor(0.95)))

    def _register_failure(self, raw_obs, raw_id=None, *, persist=True):
        if raw_id is not None and raw_id in self.failure_ids:
            return
        state = self.task.build_obs(raw_obs, {}).detach().to(self.policy.device)
        if persist:
            index = (
                max((int(p.stem) for p in self.failure_dir.glob("*.pkl")), default=-1)
                + 1
            )
            path = self.failure_dir / f"{index:08d}.pkl"
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as f:
                pickle.dump({"raw_obs": raw_obs, "raw_id": raw_id}, f)
            temporary.replace(path)
        self.failure_states.append(state)
        self.policy.failure_classifier.anchors = torch.stack(self.failure_states)
        if raw_id is not None:
            self.failure_ids.add(raw_id)
        self.failure_id_tensor = torch.tensor(
            sorted(self.failure_ids), device=self.policy.device, dtype=torch.long
        )

    def _update_failure_classifier(self):
        model = self.policy.failure_classifier
        model.requires_grad_(True)
        negative = self.failure_negatives
        n = negative[torch.randint(len(negative), (128,), device=negative.device)]
        # Balanced sampling prevents the sparse human labels being drowned out.
        loss = F.softplus(model(n)).mean()
        if self.failure_states:
            positive = torch.stack(self.failure_states)
            p = positive[torch.randint(len(positive), (128,), device=positive.device)]
            loss = 0.5 * (loss + F.softplus(-model(p)).mean())
        self.policy.failure_optimizer.zero_grad()
        loss.backward()
        self.policy.failure_optimizer.step()
        model.requires_grad_(False)
        with torch.no_grad():
            # Include ALL known successful episodes, not just the initial demos.
            scores = model(negative).sigmoid()
            model.threshold.copy_(
                torch.maximum(scores.max() + 0.001, scores.new_tensor(0.95))
            )
            model.has_positive.fill_(bool(self.failure_states))
            recall = (
                (model(torch.stack(self.failure_states)).sigmoid() >= model.threshold)
                .float()
                .mean()
                if self.failure_states
                else scores.new_zeros(())
            )
        return {
            "failure_loss": loss.detach(),
            "failure_nonfailure_fpr": (scores >= model.threshold).float().mean(),
            "failure_negative_count": scores.new_tensor(len(negative)),
            "failure_tag_recall": recall,
            "failure_threshold": model.threshold.detach(),
            "failure_positive_count": scores.new_tensor(len(self.failure_states)),
        }

    def _failure_batch_labels(self, grouped, flat):
        ids = grouped["raw_id"].to(self.policy.device)
        known = self.failure_id_tensor
        flat["failure_next"] = torch.isin(ids, known).repeat_interleave(
            int(flat["view_count"])
        )
