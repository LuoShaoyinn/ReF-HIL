from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

from .action_proximity import (
    ACTION_LIKENESS_QUERY_STD,
    ACTION_LIKENESS_SIGMA,
    action_query_proposal_density,
    gaussian_density_peak,
    proximity_target,
    proximity_threshold_from_radius,
    sample_truncated_action_normal,
)
from .action_radius_calibration import (
    EncodedTeachingData,
    error_summary,
    run_episode_cross_validation,
)
from .iql_floor_runtime import (
    HumanIQLGroundedLearner,
    HumanIQLGroundedLearnerConfig,
)
from .policy import LimitActionPolicy
from .subset_replay import RawIndexBuffer
from .unrecoverable import UnrecoverableAnnotations

RECOVERY_REJECT_WINDOW_STEPS = 10


@dataclass(kw_only=True)
class LimitActionLearnerConfig(HumanIQLGroundedLearnerConfig):
    indexed_replay_views: bool = True
    # Factual TD and its fence mask must refer to the recorded action.
    # Synthetic bad-action noise belongs only in the correction cloud.
    action_noise_std: float = 0.0
    recovery_pretrain_updates: int = 5_000
    augmentation_views: int = 2
    action_likeness_updates: int = 20_000
    action_likeness_batch_size: int = 256
    action_likeness_queries_per_anchor: int = 8
    action_likeness_seed: int = 806
    # H targets a normalized Gaussian; the neural output is not hard-normalized.
    action_likeness_sigma: float = ACTION_LIKENESS_SIGMA
    action_likeness_query_std: float = ACTION_LIKENESS_QUERY_STD
    action_likeness_observation_noise_std: float = 0.01
    action_likeness_online_update_every: int = 50
    action_radius_cv_epochs: int = 100
    correction_batch_size: int = 256

    def __post_init__(self) -> None:
        if self.action_noise_std != 0:
            raise ValueError("constrained Q requires unmodified factual replay actions")
        if float(self.action_likeness_sigma) != ACTION_LIKENESS_SIGMA:
            raise ValueError(
                "action_likeness_sigma is fixed at "
                f"{ACTION_LIKENESS_SIGMA}, got {self.action_likeness_sigma}"
            )
        if self.action_likeness_query_std <= 0.0:
            raise ValueError("action_likeness_query_std must be positive")
        if self.action_likeness_queries_per_anchor < 1:
            raise ValueError("action_likeness_queries_per_anchor must be positive")
        if self.action_radius_cv_epochs < 1:
            raise ValueError("cv_epochs must be positive")


class LimitActionLearner(HumanIQLGroundedLearner):
    """Moving human-IQL floor plus a human-intervention moving action fence."""

    config: LimitActionLearnerConfig
    policy: LimitActionPolicy

    _METRIC_TAGS: ClassVar[dict[str, str]] = {
        "human_iql_loss": "Loss_IQL/value_total",
        "human_iql_critic_loss": "Loss_IQL/critic_total",
        "human_iql_td_smooth_loss": "Loss_IQL/critic_TD_smooth_l1",
        "human_iql_td_mse_loss": "Loss_IQL/critic_TD_MSE",
        "human_iql_expectile_loss": "Loss_IQL/value_expectile",
        "human_iql_rank_loss": "Loss_IQL/value_horizon_rank",
        "human_iql_consistency": "Loss_IQL/value_consistency",
        "recovery_iql_actor_loss": "Loss_IQL/recovery_actor_total",
        "recovery_iql_grounded_loss": "Loss_IQL/recovery_actor_AWR",
        "recovery_iql_consistency": "Loss_IQL/recovery_actor_consistency",
        "critic_loss": "Loss_SAC/critic_total",
        "critic_td_smooth_loss": "Loss_SAC/critic_TD_smooth_l1",
        "critic_td_mse_loss": "Loss_SAC/critic_TD_MSE",
        "critic_horizon_rank_loss": "Loss_SAC/critic_horizon_rank",
        "human_floor_loss": "Loss_SAC/human_floor",
        "unrecoverable_state_loss": "Loss_SAC/unrecoverable_state",
        "unrecoverable_iql_q_loss": "Loss_IQL/unrecoverable_Q",
        "unrecoverable_iql_v_loss": "Loss_IQL/unrecoverable_V",
        "replay_reference_floor_loss": "Loss_SAC/replay_reference_floor",
        "replay_reference_shortfall": "ReplayReferenceFloor/shortfall_mean",
        "replay_reference_violation_fraction": "ReplayReferenceFloor/violation_fraction_gt_0.01",
        "replay_reference_target_q": "ReplayReferenceFloor/IQL_Q_mean",
        "replay_reference_sac_q": "ReplayReferenceFloor/SAC_Q_at_IQL_mean",
        "human_floor_shortfall": "Diagnostics/human_floor_shortfall_mean",
        "human_floor_violation_fraction": "Diagnostics/human_floor_violation_fraction",
        "human_floor_tolerance_violation_fraction": "Diagnostics/human_floor_violation_fraction_gt_0.01",
        "actor_loss": "Loss_SAC/actor_total",
        "actor_standard_loss": "Loss_SAC/actor_standard_Q",
        "actor_raw_q": "SAC_Q_mean/actor_all_h",
        "actor_consistency_loss": "Loss_SAC/actor_consistency",
        "actor_rejected_fraction": "ActionFence/actor_rejected_fraction",
        "actor_reference_l2": "ActionFence/actor_reference_L2",
        "actor_rejected_l2": "ActionFence/rejected_actor_reference_L2",
        "actor_saturation_fraction": "ActorOptimization/saturation_fraction",
        "recorded_rejected_fraction": "ActionFence/recorded_TD_masked_fraction",
        "critic_valid_pairs": "Loss_SAC/valid_TD_heads",
        "target_rejected_fraction": "Continuation/SAC_rejected_fraction",
        "target_iql_selected_fraction": "Continuation/IQL_selected_fraction",
        "action_likeness_online_loss": "ActionFence/online_importance_MSE",
        "action_likeness_online_query_mae": "ActionFence/online_density_ratio_MAE",
        "action_likeness_online_human_score": "ActionFence/online_human_density_mean",
        "action_likeness_online_human_rejected": "ActionFence/online_human_rejected_fraction",
        "correction_rank_loss": "Loss_SAC/correction_rank",
        "rejected_advantage_rank_loss": "Loss_SAC/rejected_advantage_rank",
        "rejected_advantage_violation_fraction": "ActionFence/raw_advantage_violation_fraction",
        "rejected_advantage_sample_fraction": "ActionFence/raw_advantage_sample_fraction",
        "correction_violation_fraction": "Correction/violation_fraction",
        "correction_cloud_valid_fraction": "Correction/cloud_valid_fraction",
    }

    def __init__(self, config, task, policy) -> None:
        # This path has no BC ensemble, state familiarity classifier, OOD
        # index, or state-gated continuation.
        HumanIQLGroundedLearner.__init__(self, config=config, task=task, policy=policy)
        self._initial_teaching_data_cache: EncodedTeachingData | None = None
        if self.config.action_likeness_online_update_every < 1:
            raise ValueError("action_likeness_online_update_every must be positive")
        if self.config.correction_batch_size < 1:
            raise ValueError("correction_batch_size must be positive")
        # Factual recovery-reject evidence is kept separately from the
        # dynamic H-fence proposals sampled from ordinary replay states.
        self.recovery_reject_buffer = RawIndexBuffer(
            self.online_buffer, name="D_recovery_reject"
        )
        self._rebuild_recovery_reject_buffer()
        print(
            "[SAC REFERENCE] "
            + (
                "v15: suffix-only squared floor on isolated B; no global IQL floor"
                if self.policy.config.suffix_only_reference_floor else
                "v11: hard IQL-Q floor on every queried state; nonnegative B"
                if self.policy.config.hard_iql_reference_floor
                else "v10 checkpoint resumed: legacy learned base with soft floors"
            ),
            flush=True,
        )
        if not bool(self.policy.action_likeness.initialized.item()):
            self._pretrain_action_likeness()
        if self.policy.config.action_likeness_reference_ratio is None:
            (
                self.policy.config.action_likeness_threshold,
                self.action_likeness_radius,
            ) = self._load_or_calibrate_action_threshold()
        else:
            print(
                f"[LIMIT ACTION] relative H threshold: {self.policy.config.action_likeness_reference_ratio} * H(s,a_IQL)",
                flush=True,
            )
        self.policy.action_likeness.enable_online_training()
        self.unrecoverable = UnrecoverableAnnotations(self)
        if not self.policy.recovery_initialized:
            self._pretrain_recovery_actor()
        self._train_metric_counts: dict[str, int] = {}
        self._save_model_and_state()

    @staticmethod
    def _recovery_reject_raw_ids(
        transitions: list[dict], raw_ids: list[int]
    ) -> list[int]:
        """Return the last ten executed autonomous rows before each takeover."""

        if len(transitions) != len(raw_ids):
            raise ValueError("transition/raw-ID count mismatch")
        result: list[int] = []
        autonomous_history: list[int] = []
        previous_intervened: bool | None = None
        for transition, raw_id in zip(transitions, raw_ids, strict=True):
            info = transition.get("info", {})
            intervened = bool(info.get("is_intervene", False))
            if previous_intervened is False and intervened:
                result.extend(
                    autonomous_history[-RECOVERY_REJECT_WINDOW_STEPS:]
                )
            if intervened:
                autonomous_history.clear()
            else:
                autonomous_history.append(int(raw_id))
                del autonomous_history[:-RECOVERY_REJECT_WINDOW_STEPS]

            if bool(transition.get("done", False)):
                previous_intervened = None
                autonomous_history.clear()
            else:
                previous_intervened = intervened
        return result

    def _register_recovery_rejects(
        self, transitions: list[dict], raw_ids: list[int]
    ) -> None:
        recovery_ids = self._recovery_reject_raw_ids(transitions, raw_ids)
        if not recovery_ids:
            return
        self.recovery_reject_buffer.add_many(recovery_ids)

    def _rebuild_recovery_reject_buffer(self) -> None:
        rows = self._load_raw_replay(self.output_dir / "buffer")
        if not rows:
            return
        self._register_recovery_rejects(rows, list(range(len(rows))))
        print(
            f"[LIMIT ACTION] rebuilt D_recovery_reject with "
            f"{self.recovery_reject_buffer.count} autonomous predecessors",
            flush=True,
        )

    def _sample_recovery_reject_batch(self) -> dict[str, torch.Tensor] | None:
        if self.recovery_reject_buffer.count == 0:
            return None
        raw_ids = self.recovery_reject_buffer.sample_raw_ids(
            self.config.correction_batch_size
        )
        grouped = self.online_buffer.sample_raw_views(
            raw_ids,
            self.policy.device,
            views=self.config.augmentation_views,
            include_clean=True,
        )
        flat = self._flatten_grouped_batch(grouped)
        views = int(flat["view_count"])
        return {
            "observation": flat["observation"],
            # These are factual actions executed before takeover, rather than
            # post-takeover counterfactual actor proposals.
            "autonomous_action": flat["action"],
            "view_count": views,
        }

    def _flatten_grouped_batch(self, grouped: dict) -> dict:
        _, views, obs_dim = grouped["observation"].shape
        flat = {
            "observation": grouped["observation"].reshape(-1, obs_dim),
            "next_observation": grouped["next_observation"].reshape(-1, obs_dim),
            "gate_observation": grouped["clean_observation"][:, None, :]
            .expand(-1, views, -1)
            .reshape(-1, obs_dim),
            "gate_next_observation": grouped["clean_next_observation"][:, None, :]
            .expand(-1, views, -1)
            .reshape(-1, obs_dim),
            "action": grouped["action"][:, None, :]
            .expand(-1, views, -1)
            .reshape(-1, grouped["action"].shape[1]),
            "reward": grouped["reward"][:, None].expand(-1, views).reshape(-1),
            "done": grouped["done"][:, None].expand(-1, views).reshape(-1),
            "data_info": {
                key: value[:, None, :]
                .expand(-1, views, -1)
                .reshape(-1, value.shape[-1])
                for key, value in grouped["data_info"].items()
            },
            "view_count": views,
        }
        if "certified_suffix_next" in grouped:
            flat["certified_suffix_next"] = (
                grouped["certified_suffix_next"][:, None].expand(-1, views).reshape(-1)
            )
        positions = self.online_buffer.raw_positions(
            grouped["raw_id"][:, None].to(self.online_buffer.config.data_device),
            grouped["next_view_id"].to(self.online_buffer.config.data_device),
        ).reshape(-1)
        self._patch_timeout_next_observation(flat, positions, self.policy.device)
        clean_next = {"next_observation": flat["gate_next_observation"]}
        self._patch_timeout_next_observation(clean_next, positions, self.policy.device)
        flat["gate_next_observation"] = clean_next["next_observation"]
        if self.config.observation_noise_std > 0.0:
            flat["observation"] = flat["observation"] + (
                torch.randn_like(flat["observation"])
                * self.config.observation_noise_std
            )
            flat["next_observation"] = flat["next_observation"] + (
                torch.randn_like(flat["next_observation"])
                * self.config.observation_noise_std
            )
        return flat

    def _mark_certified_suffix_next(self, grouped: dict) -> None:
        """Attach whether each physical transition enters D_human_suffix."""

        replay_device = self.online_buffer.config.data_device
        raw_ids = grouped["raw_id"].to(replay_device).long()
        positions = self.online_buffer.raw_positions(raw_ids)
        done = self.online_buffer.data_done[positions]
        next_raw_ids = torch.where(done, raw_ids, raw_ids + 1)
        suffix_ids = self.human_example_buffer.raw_ids
        grouped["certified_suffix_next"] = torch.isin(
            next_raw_ids, suffix_ids
        ).to(self.policy.device)

    def _initial_teaching_data(self) -> EncodedTeachingData:
        cached = self._initial_teaching_data_cache
        if cached is not None:
            return cached
        rows = self._load_raw_replay(self.output_dir / "buffer")
        selected_episodes: list[list[dict]] = []
        episode: list[dict] = []
        for row in rows:
            episode.append(row)
            if not bool(row.get("done", False)):
                continue
            if float(row.get("reward", 0.0)) <= 0.0:
                raise RuntimeError("an initial demonstration episode did not succeed")
            if not all(
                bool(step.get("info", {}).get("is_intervene", False))
                for step in episode
            ):
                raise RuntimeError("an initial demonstration episode is not fully human")
            selected_episodes.append(episode)
            episode = []
            if len(selected_episodes) == self.config.human_iql_episode_count:
                break
        if len(selected_episodes) != self.config.human_iql_episode_count:
            raise RuntimeError(
                f"need {self.config.human_iql_episode_count} initial human demos, "
                f"found {len(selected_episodes)}"
            )
        selected = [row for episode_rows in selected_episodes for row in episode_rows]
        observations = self.task.build_observations(
            [row["raw_obs"] for row in selected],
            [row["info"] for row in selected],
            augment=False,
        ).detach().float().cpu()
        actions = self.task.build_actions(
            [row["raw_action"] for row in selected],
            [row["info"] for row in selected],
            augment=False,
        ).detach().float().cpu()
        episode_ids = torch.cat(
            [
                torch.full((len(rows),), index, dtype=torch.long)
                for index, rows in enumerate(selected_episodes)
            ]
        )
        step_ids = torch.cat(
            [torch.arange(len(rows), dtype=torch.long) for rows in selected_episodes]
        )
        data = EncodedTeachingData(
            observations=observations,
            actions=actions,
            episode_ids=episode_ids,
            step_ids=step_ids,
            episode_lengths=tuple(map(len, selected_episodes)),
        )
        self._initial_teaching_data_cache = data
        return data

    def _initial_demo_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._initial_teaching_data()
        return (
            data.observations.to(self.policy.device),
            data.actions.to(self.policy.device),
        )

    def _action_threshold_artifact_path(self) -> Path:
        return self.output_dir / "action_fence" / "radius_calibration.json"

    def _log_action_threshold_calibration(
        self, *, threshold: float, radius: float, acceptance: float
    ) -> None:
        self.writer.add_scalar("ActionFence/calibrated_radius", radius, 0)
        self.writer.add_scalar(
            "ActionFence/calibrated_H_threshold", threshold, 0
        )
        self.writer.add_scalar(
            "ActionFence/calibrated_demo_acceptance", acceptance, 0
        )
        self.writer.add_scalar(
            "ActionFence/human_anchor_rejected_fraction", 1.0 - acceptance, 0
        )

    def _load_or_calibrate_action_threshold(self) -> tuple[float, float]:
        """Freeze the density contour at mean episode-held-out BC error."""

        path = self._action_threshold_artifact_path()
        statistic = "episode_cv_mean_action_L2"
        if path.exists():
            with path.open() as handle:
                report = json.load(handle)
            if int(report.get("schema_version", 0)) != 5:
                raise RuntimeError(
                    "saved action-fence calibration uses a previous threshold rule; "
                    "use a fresh experiment directory"
                )
            selected = report["selected"]
            saved_statistic = str(selected.get("statistic", ""))
            saved_radius = float(selected.get("equivalent_radius", float("nan")))
            if saved_statistic != statistic or not math.isfinite(saved_radius) or saved_radius <= 0:
                raise RuntimeError(
                    f"saved action-fence statistic {saved_statistic or 'legacy'!r} "
                    f"does not match configured {statistic!r}; "
                    "use a fresh experiment directory"
                )
            threshold = float(selected["h_score_threshold"])
            radius = float(selected["equivalent_radius"])
            expected = proximity_threshold_from_radius(
                radius, sigma=self.config.action_likeness_sigma,
                action_dim=self.policy.config.action_dims,
            )
            if not math.isclose(threshold, expected, rel_tol=1e-6):
                raise RuntimeError("saved H threshold does not match radius/sigma/action dimensions")
            acceptance = float(selected["demo_acceptance"])
            self._log_action_threshold_calibration(
                threshold=threshold,
                radius=radius,
                acceptance=acceptance,
            )
            print(
                f"[LIMIT ACTION] restored demo {statistic} "
                f"H_threshold={threshold:.6f} acceptance={acceptance:.2%} "
                f"equivalent_radius={radius:.6f}",
                flush=True,
            )
            return threshold, radius

        observations, actions = self._initial_demo_tensors()
        with torch.random.fork_rng(devices=[self.policy.device] if self.policy.device.type == "cuda" else []):
            errors, fold_ids, folds = run_episode_cross_validation(
                self._initial_teaching_data(), folds=5, split_seed=20260904,
                model_seed=4109, mechanism_dim=self.policy.config.mechanism_obs_dims,
                device=self.policy.device, epochs=self.config.action_radius_cv_epochs,
                batch_size=256, learning_rate=3e-4, encoder_dim=256, hidden_dim=256,
            )
        radius = float(errors.mean())
        if not math.isfinite(radius) or radius <= 0:
            raise RuntimeError("demo mean action error must be finite and positive")
        model = self.policy.action_likeness
        model.eval()
        with torch.inference_mode():
            scores = model(observations, actions)
        threshold = proximity_threshold_from_radius(
            radius,
            sigma=self.config.action_likeness_sigma,
            action_dim=actions.shape[-1],
        )
        acceptance = float((scores >= threshold).float().mean())
        density_peak = gaussian_density_peak(
            actions.shape[-1], sigma=self.config.action_likeness_sigma
        )
        report = {
            "schema_version": 5,
            "cross_validation": {"error": error_summary(errors), "folds": folds,
                                 "fold_ids": fold_ids.tolist(), "errors": errors.tolist()},
            "source": "initial successful fully-human demonstrations",
            "episode_count": len(self._initial_teaching_data().episode_lengths),
            "transition_count": len(actions),
            "episode_lengths": list(self._initial_teaching_data().episode_lengths),
            "configuration": {
                "sigma": self.config.action_likeness_sigma,
                "density_domain": "R^D",
                "evaluation_domain": "[-1,1]^D",
                "density_peak": density_peak,
                "query_proposal": "0.5 uniform cube + 0.5 truncated normal",
                "query_std": self.config.action_likeness_query_std,
                "objective": "direct uniform-reference importance-corrected density MSE",
                "configured_radius": radius,
                "cv_epochs": self.config.action_radius_cv_epochs,
                "split_seed": 20260904,
                "model_seed": 4109,
            },
            "demo_H_score": {
                "count": len(scores),
                "mean": float(scores.mean()),
                "median": float(scores.median()),
                "q01": float(torch.quantile(scores, 0.01)),
                "q05": float(torch.quantile(scores, 0.05)),
                "q10": float(torch.quantile(scores, 0.10)),
                "minimum": float(scores.min()),
                "maximum": float(scores.max()),
            },
            "selected": {
                "statistic": statistic,
                "demo_acceptance": acceptance,
                "radius": radius,
                "equivalent_radius": radius,
                "h_score_threshold": threshold,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        temporary.replace(path)
        self._log_action_threshold_calibration(
            threshold=threshold,
            radius=radius,
            acceptance=acceptance,
        )
        print(
            f"[LIMIT ACTION] calibrated demo {statistic} "
            f"H_threshold={threshold:.6f} acceptance={acceptance:.2%} "
            f"equivalent_radius={radius:.6f}",
            flush=True,
        )
        return threshold, radius

    def _pretrain_recovery_actor(self) -> None:
        generator = torch.Generator(
            device=self.human_example_buffer.config.data_device
        ).manual_seed(29_081)
        print("[LIMIT ACTION] pretraining moving IQL actor", flush=True)
        for update in range(1, self.config.recovery_pretrain_updates + 1):
            positions = torch.randint(
                self.first20_cutoff,
                (self.config.human_example_batch_size,),
                generator=generator,
                device=self.human_example_buffer.config.data_device,
            )
            raw_ids = self.human_example_buffer.raw_ids[positions]
            grouped = self.online_buffer.sample_raw_views(
                raw_ids,
                self.policy.device,
                views=self.config.augmentation_views,
                include_clean=True,
            )
            human_batch = self._flatten_grouped_batch(grouped)
            self.unrecoverable.attach_suffix(human_batch, raw_ids, self.config.augmentation_views)
            from .unrecoverable import successful_only
            metrics = self.policy.update_recovery_iql(successful_only(human_batch))
            if update == 1 or update % 500 == 0 or update == self.config.recovery_pretrain_updates:
                self.writer.add_scalar(
                    "Loss_IQL/recovery_actor_total",
                    float(metrics["recovery_iql_actor_loss"]),
                    update,
                )
        self.policy.recovery_initialized = True

    def _sample_replay_raw_ids(
        self, candidates: torch.Tensor, count: int
    ) -> torch.Tensor:
        clean_positions = self.online_buffer.raw_positions(candidates)
        selected_positions = self._exposure_balanced_positions(
            clean_positions, count, self.replay_sample_counts
        )
        return selected_positions

    def _log_episode(self, transitions: list[dict]) -> None:
        self.actor_steps += len(transitions)
        self.actor_episodes += 1
        intervention = sum(
            bool(row["info"].get("is_intervene", False)) for row in transitions
        ) / len(transitions)
        episode_return = sum(float(row["reward"]) for row in transitions)
        success = float(float(transitions[-1]["reward"]) > 0.0)
        self.writer.add_scalar(
            "Actor/intervention_fraction", intervention, self.actor_steps
        )
        self.writer.add_scalar("Actor/episode_return", episode_return, self.actor_steps)
        self.writer.add_scalar("Actor/success", success, self.actor_steps)
        self.writer.add_scalar("Actor/forced_failure", float(transitions[-1]['info'].get('forced_failure', False)), self.actor_steps)

    def _accumulate_train_metrics(self, result) -> None:
        selected = {}
        for key, value in result.items():
            tag = self._METRIC_TAGS.get(key)
            if tag is None:
                for prefix, target in (
                    ("human_v_h", "V_H_mean/h"),
                    ("actor_q_h", "SAC_Q_mean/actor_h"),
                    ("human_q_h", "SAC_Q_mean/reference_h"),
                    ("human_iql_q_h", "IQL_Q_mean/h"),
                    ("human_floor_target_h", "IQL_Qpi_mean/h"),
                    ("human_shortfall_h", "Diagnostics/human_floor_shortfall_h"),
                ):
                    if key.startswith(prefix):
                        tag = target + key.removeprefix(prefix)
                        break
            if tag is not None:
                selected[tag] = value
        for tag, value in selected.items():
            metric = torch.as_tensor(value).detach().float().mean()
            if tag in self._train_metric_sums:
                self._train_metric_sums[tag].add_(metric)
            else:
                self._train_metric_sums[tag] = metric.clone()
            self._train_metric_counts[tag] = self._train_metric_counts.get(tag, 0) + 1
        self._train_metric_count += 1

    def _training_queries(
        self,
        human_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        queries = self.config.action_likeness_queries_per_anchor
        human_action = human_action.repeat_interleave(queries, dim=0)
        batch = len(human_action)
        uniform = torch.empty_like(human_action).uniform_(-1.0, 1.0)
        gaussian = sample_truncated_action_normal(
            human_action, std=self.config.action_likeness_query_std
        )
        choose_uniform = torch.rand(batch, device=self.policy.device) < 0.5
        query_action = torch.where(choose_uniform[:, None], uniform, gaussian)
        proposal_density = action_query_proposal_density(
            query_action,
            human_action,
            local_std=self.config.action_likeness_query_std,
        )
        uniform_density = 2.0 ** (-human_action.shape[-1])
        importance_weight = uniform_density / proposal_density.clamp_min(1e-12)
        target_density = proximity_target(
            query_action, human_action, sigma=self.config.action_likeness_sigma
        )
        return query_action, target_density, importance_weight

    def _action_likeness_loss(
        self,
        predicted_density: torch.Tensor,
        target_density: torch.Tensor,
        importance_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        peak = gaussian_density_peak(
            self.policy.config.action_dims,
            sigma=self.config.action_likeness_sigma,
        )
        # This is the direct uniform-reference estimator of
        # integral_[cube] (H(s,a) - phi(a | a_H))^2 da, up to the fixed cube
        # volume. Do not divide by the Gaussian peak: doing so leaves the same
        # formal minimizer but makes the nearly-zero solution numerically cheap
        # relative to regularization and finite optimization.
        density_error = predicted_density - target_density
        loss = (importance_weight * density_error.square()).mean()
        return loss, predicted_density / peak

    def _pretrain_action_likeness(self) -> None:
        observations, actions = self._initial_demo_tensors()
        model = self.policy.action_likeness
        model.set_normalization(observations, actions)
        model.train()
        model.requires_grad_(True)
        optimizer = self.policy.action_likeness_optim
        generator = torch.Generator(device=self.policy.device).manual_seed(
            self.config.action_likeness_seed
        )
        print(
            "[LIMIT ACTION] training H(s,a) from "
            f"20 episodes / {len(observations)} human anchors",
            flush=True,
        )
        for update in range(1, self.config.action_likeness_updates + 1):
            index = torch.randint(
                len(observations),
                (self.config.action_likeness_batch_size,),
                generator=generator,
                device=self.policy.device,
            )
            human_observation = observations[index]
            if self.config.action_likeness_observation_noise_std > 0.0:
                human_observation = human_observation + torch.randn_like(
                    human_observation
                ) * model.observation_std * self.config.action_likeness_observation_noise_std
            state = model.encode_observation(human_observation).repeat_interleave(
                self.config.action_likeness_queries_per_anchor, dim=0
            )
            action, target_density, importance_weight = self._training_queries(
                actions[index]
            )
            predicted_density = model.density_from_state(state, action)
            loss, _ = self._action_likeness_loss(
                predicted_density, target_density, importance_weight
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            if (
                update == 1
                or update % 500 == 0
                or update == self.config.action_likeness_updates
            ):
                self.writer.add_scalar(
                    "ActionFence/pretrain_importance_MSE",
                    float(loss.detach()),
                    update,
                )
        model.finalize()
        with torch.inference_mode():
            human_score = model(observations, actions)
        self.writer.add_scalar(
            "ActionFence/human_anchor_score_mean", float(human_score.mean()), 0
        )
        print(
            "[LIMIT ACTION] H pretrained "
            f"mean_human={float(human_score.mean()):.4f}; "
            "using the relative IQL-action density fence",
            flush=True,
        )

    def _sample_action_likeness_raw_ids(self) -> torch.Tensor:
        """Equal batches from fixed demos and recent human interventions.

        H models human support regardless of episode outcome. IQL continues
        sampling human_example_buffer, which contains successful suffixes only.
        """
        ids = self.offline_buffer.raw_ids
        # N is the number of steps in the initial teaching episodes. Preserve
        # those N anchors, and slide an equally sized window over all later
        # human steps, including short corrections and unsuccessful recoveries.
        demo = ids[:self.first20_cutoff]
        later = ids[self.first20_cutoff:][-self.first20_cutoff:]
        count = self.config.action_likeness_batch_size
        demo_count = count if len(later) == 0 else (count + 1) // 2
        sampled = demo[torch.randint(len(demo), (demo_count,), device=ids.device)]
        if len(later):
            sampled = torch.cat((sampled, later[torch.randint(len(later), (count-demo_count,), device=ids.device)]))
        return sampled

    def _update_action_likeness(self) -> dict[str, torch.Tensor]:
        """Fit H on N demo steps plus the latest N human intervention steps."""
        sampled = self._sample_action_likeness_raw_ids()
        grouped = self.online_buffer.sample_raw_views(sampled, self.policy.device, views=1, include_clean=True)
        model = self.policy.action_likeness
        clean_observation = grouped["clean_observation"].float()
        observation = clean_observation
        if self.config.action_likeness_observation_noise_std > 0.0:
            observation = observation + torch.randn_like(observation) * (
                model.observation_std
                * self.config.action_likeness_observation_noise_std
            )
        human_action = grouped["action"].float()
        state = model.encode_observation(observation).repeat_interleave(
            self.config.action_likeness_queries_per_anchor, dim=0
        )
        query_action, target_density, importance_weight = self._training_queries(
            human_action
        )
        predicted_density = model.density_from_state(state, query_action)
        loss, predicted_ratio = self._action_likeness_loss(
            predicted_density, target_density, importance_weight
        )
        optimizer = self.policy.action_likeness_optim
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        with torch.no_grad():
            peak = model.density_peak
            target_ratio = target_density / peak
            human_score = model(clean_observation, human_action)
            threshold = self.policy._fence_threshold(
                clean_observation, self.policy.recovery_actions(clean_observation)
            )
        return {
            "action_likeness_online_loss": loss.detach(),
            "action_likeness_online_query_mae": (
                predicted_ratio - target_ratio
            ).abs().mean().detach(),
            "action_likeness_online_human_score": human_score.mean().detach(),
            "action_likeness_online_human_rejected": (
                human_score < threshold
            ).float().mean().detach(),
        }

    def _runtime_policy_export(self):
        return self._attach_runtime_metadata(self.policy.export_actor())

    def _flush_train_metrics(self) -> None:
        if self._train_metric_count:
            for tag, value in self._train_metric_sums.items():
                self.writer.add_scalar(
                    tag,
                    (value / self._train_metric_counts[tag]).item(),
                    self.train_steps,
                )
        self._train_metric_sums.clear()
        self._train_metric_counts.clear()
        self._train_metric_count = 0
        raw_count = self.online_buffer.raw_count
        values = {
            "Buffers/replay_count": raw_count,
            "Buffers/replay_capacity": self.online_buffer.config.capacity,
            "Buffers/human_count": self.offline_buffer.count,
            "Buffers/human_suffix_count": self.human_example_buffer.count,
            "Buffers/H_demo_count": self.first20_cutoff,
            "Buffers/H_recent_human_count": min(
                max(0, self.offline_buffer.count - self.first20_cutoff),
                self.first20_cutoff,
            ),
            "Buffers/correction_count": self.recovery_reject_buffer.count,
        }
        for tag, value in values.items():
            self.writer.add_scalar(tag, value, self.train_steps)

    def step(self) -> None:
        self._maybe_periodic_save()
        transitions = self.transition_receiver.recv(None)
        if isinstance(transitions, dict) and transitions.get("event") == "failed_state":
            if transitions.get('unrecoverable', False):
                self.unrecoverable.add_state(transitions['raw_obs'], persist=True)
            self.actor_episodes += 1
            self.writer.add_scalar('Actor/success', 0., self.actor_steps)
            self.writer.add_scalar('Actor/episode_return', 0., self.actor_steps)
            self.writer.add_scalar('Actor/forced_failure', 1., self.actor_steps)
            # Actor-only abort before an action: outcome logging, no labels/model.
            transitions = None
        if transitions is not None:
            raw_start = self.online_buffer.raw_count
            self._log_episode(transitions)
            self.online_buffer.add_many(transitions)
            self.unrecoverable.register(transitions, raw_start)
            raw_ids = list(range(raw_start, raw_start + len(transitions)))
            self.on_actor_transitions(transitions, raw_ids)
            self._register_recovery_rejects(transitions, raw_ids)
            self._register_timeout_next_observations(transitions, raw_start)
            for offset, transition in enumerate(transitions):
                if bool(transition["info"].get("is_intervene", False)):
                    self.offline_buffer.add(raw_ids[offset])

        if self.online_buffer.raw_count < max(
            2, int(self.config.transitions_before_start)
        ):
            return

        device = self.online_buffer.config.data_device
        batch_size = int(self.config.batch_size)
        human_count = (
            min(int(batch_size * self.config.offline_batch_ratio), batch_size)
            if self.offline_buffer.count
            else 0
        )
        replay_count = batch_size - human_count
        raw_candidates = torch.arange(self.online_buffer.raw_count, device=device)
        clean_done = self.online_buffer.data_done[
            self.online_buffer.raw_positions(raw_candidates)
        ]
        if len(raw_candidates) and not bool(clean_done[-1]):
            raw_candidates = raw_candidates[:-1]
        raw_ids = self._sample_replay_raw_ids(raw_candidates, replay_count)
        if human_count:
            human_candidates = self.offline_buffer.raw_ids
            human_positions = self.online_buffer.raw_positions(human_candidates)
            valid_human = (
                human_positions + 1 < self.online_buffer.tot_transition
            ) | self.online_buffer.data_done[human_positions]
            raw_ids = torch.cat(
                (
                    raw_ids,
                    self._sample_replay_raw_ids(
                        human_candidates[valid_human], human_count
                    ),
                )
            )

        grouped = self.online_buffer.sample_raw_views(
            raw_ids,
            self.policy.device,
            views=self.config.augmentation_views,
            include_clean=True,
        )
        self._mark_certified_suffix_next(grouped)
        batch = self._flatten_grouped_batch(grouped)
        self.unrecoverable.attach(batch, raw_ids, self.config.augmentation_views)
        batch["action"] = (
            batch["action"]
            + torch.randn_like(batch["action"]) * self.config.action_noise_std
        ).clamp_(-1.0, 1.0)

        human_positions = self._sample_human_positions(
            self.config.human_example_batch_size
        )
        human_raw_ids = self.human_example_buffer.raw_ids[human_positions]
        human_grouped = self.online_buffer.sample_raw_views(
            human_raw_ids,
            self.policy.device,
            views=self.config.augmentation_views,
            include_clean=True,
        )
        batch["human_example"] = self._flatten_grouped_batch(human_grouped)
        self.unrecoverable.attach_suffix(batch["human_example"], human_raw_ids,
                                         self.config.augmentation_views)
        batch["correction_example"] = self._sample_recovery_reject_batch()
        result = self.policy.update(batch=batch)
        if (
            self.train_steps + 1
        ) % self.config.action_likeness_online_update_every == 0:
            result.update(self._update_action_likeness())
        self._accumulate_train_metrics(result)
        self.train_steps += 1
        if self.train_steps % self.config.send_parameters_every == 0:
            self.parameters_sender.send(self._runtime_policy_export())
        if self.train_steps % self.config.log_every == 0:
            self._flush_train_metrics()
            print(f"[LIMIT ACTION] step={self.train_steps}", flush=True)

    def run(self) -> None:
        try:
            print("Limit-action learner started", flush=True)
            self.parameters_sender.send(self._runtime_policy_export())
            while True:
                self.step()
        except KeyboardInterrupt:
            print("KeyboardInterrupted", flush=True)
        finally:
            self._flush_train_metrics()
            self._save_model_and_state()
            self._checkpoint_writer.close()
            self.writer.flush()
            self.writer.close()
