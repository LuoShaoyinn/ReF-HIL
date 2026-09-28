# ReF-HIL

**Shaping the Critic around Human Action Neighborhoods for Efficient Human-in-the-Loop Reinforcement Learning**

ReF-HIL learns real-world manipulation from initial demonstrations, autonomous
experience, and human corrections. An independent IQL reference guides online
value learning. A learned human-action fence concentrates exploration around
human behavior while allowing value-driven improvement without imitation
penalties inside the fence.

This repository contains the real-robot actor–learner implementation. The code
lives on `main`; project-page content and media live on `webpages`.

## Paper tasks

| Task | Task package | Policy action | Episode limit |
| --- | --- | --- | --- |
| Push-T | `push_t` | XYZ | 150 |
| Cap unscrewing | `rotate_knob` | XYZ, local yaw, gripper | 200 |
| Gear assembly | `insert_gear` | XYZ, local yaw | 150 |
| Plug insertion | `plug_power_socket` | XYZ, gripper | 200 |
| Dual-branch cable routing | `hang_double_strings_2` | XYZ, gripper | 200 |

Task names retain the source implementation's identifiers for dataset and
checkpoint compatibility. `plug_in` is an additional, differently calibrated
plug profile; it is not an alias for `plug_power_socket`. All ten source task
profiles are preserved; see [task configuration](tasks/README.md).

The supplied manuscript reports 90% autonomous success after 18–63 minutes of
active training and final mean success rates of 91.7–100%. These are paper
results, not results reproduced by installing this repository. See
[results and reproduction scope](docs/REPRODUCIBILITY.md).

## Installation

Use Linux, Python 3.14, and `uv`. Run commands from the repository root. Select
one accelerator extra on each machine:

```bash
# NVIDIA learner
uv sync --locked --extra cuda

# AMD actor and physical robot peripherals (on the actor machine)
uv sync --locked --extra rocm --extra actor
```

Use `--extra cuda --extra actor` for an NVIDIA actor. The two accelerator
extras are mutually exclusive. The committed lockfile preserves the source
environment's dependency versions and PyTorch indexes; it does not establish
compatibility with every driver or platform. PyTorch uses the `cuda` device
name for both CUDA and ROCm.

The real setup uses a UR5e, Robotiq Hand-E, three RealSense D405 cameras, and
a SpaceMouse. Configure robot addresses, camera serials, workspace bounds,
camera crops, resets, and success classifiers for the actual setup before
starting a driver. The defaults describe the original laboratory setup.

Follow the [setup and training guide](docs/GETTING_STARTED.md) to collect
classifier data, record 20 successful demonstrations, start the learner and
actor, and evaluate checkpoints. Demonstration data, trained success
classifiers, and policy checkpoints are not bundled.

## Repository guide

| Directory | Purpose |
| --- | --- |
| `actor/` | Robot and operator servers, inference, demonstration recording, validation |
| `learner/` | IQL reference, human-action model, online actor–critic, replay, checkpoints |
| `shared/` | Observation/action contracts, inference network, shared task and robot utilities |
| `tasks/` | Physical task configuration, modeling, reset, and success detection |
| `tools/` | Camera/data tools, classifier training, evaluation plots, media preparation |
| `docs/` | Method mapping, runtime contracts, provenance, paper results, media inventory |

- [Method and implementation](docs/METHOD.md)
- [Runtime and data contracts](docs/ARCHITECTURE.md)
- [Source provenance and release scope](docs/RELEASE.md)
- [Video review and webpage preparation](docs/MEDIA.md)
- [Software validation](docs/VALIDATION.md)

Internal names such as `LimitActionPolicy`, `iql_floor`, and `SAC` are retained
to avoid breaking checkpoint loading and imports. The deployed actor is
deterministic; it receives only its inference network, not the learner's
IQL reference or fence.

## License and review status

Code is released under [GNU GPLv3](LICENSE) (`GPL-3.0-only`). Authors are
withheld while the paper is under review. The project page is prepared for
anonymous hosting; it carries no personal profile or identifiable source link.
Videos on the webpage branch are stored with Git LFS.
