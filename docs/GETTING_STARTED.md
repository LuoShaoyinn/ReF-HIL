# Setup, training, and evaluation

Run all Python modules from the repository root. The examples use a separate
AMD actor machine and NVIDIA learner, matching the source's network topology.
Install the corresponding extras as described in the root README.

## Configure the physical setup

The robot driver reads these files:

- `actor/robot/ur5e/ur_impedance_control.py`: robot RTDE IP and controller settings.
- `actor/robot/ur5e/robotiq_gripper.py`: gripper hostname and port.
- `actor/robot/ur5e/ur5e_base.py`: global and two wrist camera serial numbers.
- `tasks/<task>/config.py`: workspace, action scales, reset trajectory, crops,
  classifier checkpoints, and episode limit.

Camera serials and physical coordinates are laboratory calibration, not portable
defaults. Inspect each task's reset implementation as well as its bounds.
`insert_gear` has an interactive startup pickup and a TCP-position success gate.
The `fake_mode` field on the hardware configuration does not bypass its hardware
constructors; use focused software tests for hardware-free validation.

Default global video recording requires FFmpeg and a supported hardware encoder
in `actor/robot/ur5e/global_video.py`. Use `--no-global-video` to run without this
recording dependency.

## Collect success-classifier data

For example, use `push_t` and experiment `push_t-refhil-1`. On the actor machine,
start these in separate terminals after calibration:

```bash
uv run --locked --extra rocm --extra actor python -m actor.robot.run --task push_t --no-global-video
uv run --locked --extra rocm --extra actor python -m actor.operator.run
```

Collect and label RGB-D observations, then train the success model:

```bash
uv run --locked --extra rocm --extra actor python -m tools.collect_images_teleop \
  --task push_t --output outputs/classifier_samples/push_t
uv run --locked --extra rocm python -m tools.tag_images \
  outputs/classifier_samples/push_t --condition success \
  --output outputs/classifier_samples/push_t/labels.jsonl
uv run --locked --extra rocm python -m tools.train_classifier \
  --task push_t --condition success \
  --labels outputs/classifier_samples/push_t/labels.jsonl \
  --output outputs/push_t-refhil-1/classifier.pt --device cuda
```

For tasks with several success conditions, label/train each condition and use
the exact checkpoint paths in its `TaskConfig.success_conditions`. All conditions
must pass. Validate the classifier on the actual scene before collecting demos.

## Record twenty demonstrations

With the robot and SpaceMouse servers running, record:

```bash
uv run --locked --extra rocm --extra actor python -m actor.record_demo \
  --task push_t --experiment-name push_t-refhil-1 --episodes 20
```

Only successful episodes count. Timed-out attempts are retried. Raw episode
chunks appear under `outputs/push_t-refhil-1/buffer/`. Stop the recorder before
starting the online actor; they consume the same robot/operator responses.
`--append` explicitly adds new successful demonstrations to an existing buffer.

Copy the initial `buffer/` directory to the same experiment path on the learner
machine. Keep the classifier files on the actor machine. The learner does not
need physical robot SDKs or success-classifier weights.

## Train

On the learner machine, replace `ACTOR_IP` with the actor machine's address:

```bash
uv run --locked --extra cuda python -m learner.learner \
  --task push_t --experiment-name push_t-refhil-1 --actor-host ACTOR_IP
```

The learner loads demos, initializes IQL, the reference actor, and H, then
publishes the initial actor. On the actor machine, replace `LEARNER_IP`:

```bash
uv run --locked --extra rocm --extra actor python -m actor.actor \
  --task push_t --experiment-name push_t-refhil-1 --learner-host LEARNER_IP
```

The actor waits for parameters before real execution. Human SpaceMouse motion
or gripper buttons override autonomous actions. Enter marks success in the
interactive actor; `f` explicitly annotates an unrecoverable state. These are
human labels, not automatic outcome verification.

The learner stores replay, TensorBoard logs, and elapsed-time checkpoints in
the experiment directory. Restarting with the same name resumes the latest
complete checkpoint; use a new experiment directory for a fresh run. To seed
a fresh run, copy only the desired raw demonstrations, not old checkpoints.

The defaults expect two hosts: actor-side SpaceMouse responses and learner-side
transitions both use port 7003. For a single-host setup, use e.g.
`--transitions-port 7013 --actor-host 127.0.0.1` on the learner and
`--learner-port 7013 --learner-host 127.0.0.1` on the actor.

## Evaluate

Stop the online actor and copy learner checkpoints to the actor machine.
Retain the task's classifier files, then run:

```bash
uv run --locked --extra rocm --extra actor python -m actor.validate \
  --task push_t --experiment-name push_t-refhil-1 --episodes-per-checkpoint 20
```

Validation is autonomous, does not connect to the learner, and never adds its
episodes to training replay. Results go to newly numbered subdirectories under
`outputs/validate-push_t-refhil-1/`. `--checkpoint-dir` selects copied checkpoints;
`--start-checkpoint` starts inclusively from an elapsed-time checkpoint. `n` marks
an unsafe attempt failed. Interrupted attempts remain incomplete.

Use `--actor-name actor_episode_000062.pkl` to evaluate one exact saved training
episode policy, or `--start-actor-episode 1` for a sequence. These snapshots live
on the actor machine under `outputs/<experiment>/episode_actors/` by default.

```bash
uv run --locked --extra rocm python -m tools.plot_experiment_performance --help
uv run --locked --extra rocm python -m tools.plot_episode_actor_validation --help
```

Full CLI options are available through `python -m <module> --help`.
