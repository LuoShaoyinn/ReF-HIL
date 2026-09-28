# Operator and dataset tools

The tools fall into two groups:

- task-neutral hardware viewers communicate with the already-running robot
  driver and do not reproduce task control logic;
- task-aware collection, training, and demo commands load components through
  `--task NAME`, using the same crops, action adapter, reset, reward, and
  classifier preprocessing as the actor.

## Inspect the assemble hardware

Start the robot process first:

```bash
uv run --extra rocm --extra actor python -m actor.robot.run --task assemble
```

Then inspect the raw global camera with the Assemble classifier/policy crops
overlaid, or print the current TCP pose:

```bash
uv run --extra rocm --extra actor python tools/show_global_camera.py --task assemble
uv run --extra rocm --extra actor python tools/show_robot_position.py
```

`show_global_camera.py` remains usable without `--task`. Mouse selections print
both exclusive corner coordinates and the exact `(x, y, width, height)` tuple
used by task configuration.

Run the task reset procedure directly with:

```bash
uv run --extra rocm --extra actor python tools/reset_assemble.py
```

## Record and label success/failure images

Start the robot and SpaceMouse processes:

```bash
uv run --extra rocm --extra actor python -m actor.robot.run --task assemble
uv run --extra rocm --extra actor python -m actor.operator.run
```

Then collect task-cropped images:

```bash
uv run --extra rocm --extra actor python tools/collect_images_teleop.py \
  --task assemble --output outputs/classifier_samples/assemble
```

Each episode is a directory of PNGs. Classifier RGB and depth pairs use names
ending in `_classifier_rgb.png` and `_classifier_depth_depth.png`; no replay
transitions are written by this utility.

## Tag and train the classifier

```bash
uv run --extra rocm python tools/tag_images.py \
  outputs/classifier_samples/assemble \
  --condition success \
  --output outputs/classifier_samples/assemble/labels.jsonl

uv run --extra rocm python tools/train_classifier.py \
  --task assemble \
  --condition success \
  --labels outputs/classifier_samples/assemble/labels.jsonl \
  --output outputs/ASSEMBLE_EXPERIMENT/classifier.pt \
  --device cuda
```

The trainer pairs each tagged classifier RGB PNG with its depth PNG, holds out
whole episode directories for validation, and uses the same RGB-D normalization
as runtime inference. `actor.record_demo` and `actor.actor` load
`outputs/<experiment-name>/classifier.pt`.

For a task with another condition named `released`, collect once (the collector
saves every configured crop), then tag and train that condition independently:

```bash
uv run --extra rocm python tools/tag_images.py \
  outputs/classifier_samples/TASK \
  --condition released \
  --output outputs/classifier_samples/TASK/released_labels.jsonl

uv run --extra rocm python tools/train_classifier.py \
  --task TASK --condition released \
  --labels outputs/classifier_samples/TASK/released_labels.jsonl \
  --output outputs/EXPERIMENT/classifiers/released.pt \
  --device cuda
```

The output path must match that condition's `checkpoint` setting. Runtime
loads every configured checkpoint and combines decisions with logical AND.

## Record human demonstrations

With the task robot and SpaceMouse processes running, record replay-compatible
human demonstrations with:

```bash
uv run --extra rocm --extra actor python -m actor.record_demo \
  --task assemble \
  --experiment-name assemble-base \
  --episodes 20
```

This command loads the task's action mapping, reset, step penalty, classifier,
and classifier preview dynamically. It writes raw transitions to
`outputs/<experiment-name>/buffer`. `--episodes` counts successful
demonstrations: an attempt that reaches the task horizon without classifier
success is discarded and automatically retried.

## Evaluation and publication utilities

- `list_successful_episode_actors.py`: match saved actors to successful training episodes.
- `plot_episode_actor_validation.py`: plot per-episode actor validation results.
- `plot_experiment_performance.py` and `plot_intervention_rate.py`: plot a selected run.
- `export_iql_actor.py`: export an IQL reference actor for autonomous validation.
- `train_bc_offline.py`: fit an offline behavior-cloning reference from a selected dataset.
- `calibrate_limit_action_threshold.py`: diagnostic held-out action-radius calibration;
  the default ReF-HIL learner uses a relative H threshold and does not require it.
- `plot_task_tcp_ranges.py`: render task workspace/reset summaries.
- `inventory_media.py`: inventory source video metadata using ffprobe.
- `prepare_web_media.py`: make stripped-metadata H.264/AAC videos and posters in
  a separate webpage worktree, with full decode checks and provenance hashes.

Run these as `python -m tools.<name> --help` from the repository root. See
[the setup guide](../docs/GETTING_STARTED.md) for the supported end-to-end workflow.
