# Method and implementation

The supplied `main-6.pdf` is titled *ReF-HIL: Shaping the Critic around Human
Action Neighborhoods for Efficient Human-in-the-Loop Reinforcement Learning*.
This guide describes the current default path in the released source, rather
than the historical v10/v11 variants described by earlier source documents.

## Human-reference-guided value shaping

An independent vector-valued IQL branch learns from successful human suffixes
and explicit unrecoverable-state supervision. Its advantage-weighted reference
actor supplies `a_ref`. The online twin critics split reference values `B(s)`
and centered action advantages `A(s,a) - A(s,a_ref)` into separate trainable
branches. A one-sided squared floor on successful suffix states raises `B`
when it falls below the detached IQL reference. It does not impose a global
hard IQL floor.

Pre-intervention ranking uses up to ten consecutive autonomous steps before
a takeover. It reduces their centered advantages relative to the reference,
with a margin equal to the task step cost. Both advantage evaluations receive
gradients; the reference action and the base branch do not. The implementation
also samples noisy correction actions outside a small reference-exclusion
radius. Explicit unrecoverable states supervise the finite-horizon failure
vector `F_h = -c * sum(gamma**k for k in range(h))`.

## Human action fence

`H(s,a)` regresses normalized Gaussian kernel targets around human actions
with sigma `1/6`. The default accepted region is
`H(s,a) >= 0.9 * H(s,a_ref)`. The score and reference threshold are detached
when choosing the critic branch. The reference therefore remains accepted.

Inside the fence, the critic evaluates `B(s) + A(s,a) - A(s,a_ref)`.
Outside, it evaluates `B(s) - c - ||a-a_ref||²`. This outside field is an
optimization surrogate, not a calibrated environmental return. Its gradient
pulls the actor toward the reference. Inside the fence, the actor follows
value gradients without a reference-action imitation loss. Observation-view
consistency regularization remains enabled.

Online H updates mix initial demonstration steps with an equally sized recent
window of later human intervention steps, including unsuccessful recoveries
and brief corrections. This differs from the IQL successful-suffix dataset.

## Bellman updates and runtime

Each critic predicts every horizon from one step to the task episode limit.
The continuation is the per-head maximum of target twin-min values at the
online policy proposal and reference action. In the current default path,
there is no additional global maximum with independent IQL Q. Ordinary
factual TD uses fence-accepted recorded actions; transitions into explicitly
unrecoverable states retain TD supervision even when rejected. Rejected
actions additionally train a latent-advantage ranking loss.

The actor maximizes the mean twin-min value over all horizons and uses
`z / sqrt(1 + z²)` outputs. Runtime executes its action directly: there is no
IQL substitution, fence projection, or reference-policy fallback on the robot.

## Code map

| Paper concept | Implementation |
| --- | --- |
| Reference IQL and horizon targets | `learner/iql_floor.py`, `learner/iql_modules.py` |
| Human suffixes and independent reference updates | `learner/iql_floor_runtime.py` |
| B / centered A critic and analytical outside field | `learner/constrained_critic.py` |
| TD, correction ranking, actor losses | `learner/constrained_learning.py` |
| One-sided reference floor and policy configuration | `learner/policy.py` |
| H score / Gaussian targets | `learner/action_proximity.py` |
| H sampling, pre-intervention replay, startup | `learner/training.py` |
| Failure-state supervision | `learner/unrecoverable.py`, `learner/failure.py` |
| Deployment network | `shared/actor_network.py` |

## Important defaults

| Setting | Value |
| --- | --- |
| Discount / target update coefficient | 0.99 / 0.01 |
| Relative fence ratio / Gaussian sigma | 0.9 / 1/6 |
| IQL expectile / reference actor beta | 0.75 / 200 |
| Actor / critic learning rates | 5e-4 / 1e-4 |
| Successful-suffix floor weight | 1.0 |
| Correction / rejected-advantage rank weights | 2.0 / 2.0 |
| Learner batch / human replay fraction | 4096 / 0.5 |
| Initial successful human episodes | 20 |
| IQL / reference actor / H pretraining updates | 20000 / 5000 / 20000 |

`suffix_only_reference_floor=True` and `hard_iql_reference_floor=False` select
the current method. Legacy checkpoint loading retains historical settings;
it is not an automatic conversion into this method. The exposed
`replay_reference_floor_weight` is inactive on the default suffix-only path.

Task rewards retain the source behavior: negative task-specific step costs,
and a positive completion reward reduced by significant gripper-direction
reversals (floored at 0.5). See `shared/task/gripper_smoothness.py` and
`tasks/<name>/config.py` for exact values. The paper's compact reward description
does not enumerate all of these implementation details.
