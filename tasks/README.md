# Task Packages

This repository currently contains ten runnable task profiles:

| Task | Learned action | Distinct behavior |
| --- | --- | --- |
| `hang_double_strings_2` | `[dx, dy, dz, gripper]` | canonical paper two-clip task using a native 1280x720 global stream |
| `assemble` | `[dx, dy, dz, gripper]` | restricted gripper travel and custom reset |
| `insert_usb` | `[dx, dy, dz, gripper]` | fixed Hang String orientation and USB insertion workspace |
| `insert_rj45` | `[dx, dy, dz, gripper]` | parameter-identical RJ45 copy of `insert_usb` |
| `plug_in` | `[dx, dy, dz, gripper]` | plug-in workspace with minimum-speed, maximum-force gripper |
| `gear` | `[dx, dy, dz, drz, gripper]` | local-yaw control and staged reset |
| `insert_gear` | `[dx, dy, dz, drz]` | local-yaw insertion, fixed closed gripper, two visual success conditions, and a TCP-distance gate |
| `rotate_knob` | `[dx, dy, dz, drz, gripper]` | cap/knob rotation profile with local-yaw control |
| `plug_power_socket` | `[dx, dy, dz, gripper]` | power-strip scene at native 1280x720 |
| `push_t` | `[dx, dy, dz]` | fixed orientation and fixed closed gripper |

`--task NAME` dynamically imports components from `tasks/NAME/`. There is no
task registry or central name-to-class router.

Each task package provides the same modules and exported names:

- `config.py`: `TaskConfig` containing physical action scales, observation
  normalization, reset parameters, safety fences, policy crops, and one or
  more independent success-classifier conditions.
- `modeling.py`: `Modeling` and `ModelingConfig` for actor/learner observation
  and action conversion. The task reset function lives here.
- `robot.py`: `Robot` and `RobotConfig` for task-specific safety enforcement,
  action application, and camera crops. Generic UR5e impedance control remains
  under `actor/robot/ur5e/`.
- `classifier.py`: the thin task binding for the shared RGB-D classifier.
- `learner.py`: `LearnerConfig`, which may override the branch-local learner
  defaults. Leave it as an empty subclass when no override is needed.

## Shared manipulation stack

Reusable implementation is under `shared/task/`, not inside any concrete task:

- `base.py`: the minimal modeling interface used by actor and replay;
- `manipulation_config.py`: common camera, normalization, Cartesian-control,
  safety, and reset defaults;
- `manipulation_modeling.py`: frozen ResNet observation encoding, normalized
  proprioception, Cartesian-plus-gripper adapters, intervention selection, and
  the default reset procedure;
- `manipulation_robot.py`: normalized action application, workspace clipping,
  rotation recovery, camera acquisition/cropping, and continuous gripper I/O;
- `manipulation_classifier.py`: common RGB-D classifier architecture and
  preprocessing.

Concrete task packages subclass these components only where their action space,
reset, gripper behavior, orientation control, crops, or safety ranges differ.
No production task uses another concrete task as its framework base.

## Multiple success conditions

Success detection is task configuration, not task-specific runtime code. Each
task declares a tuple of `SuccessCondition` values. Every condition has its own
global-camera crop, model checkpoint, threshold, and observation keys. The
actor, demo recorder, and checkpoint validator evaluate all models independently
and report success only when every condition passes.

Existing tasks use one condition and retain the historical `classifier` /
`classifier_depth` image keys and `classifier.pt` checkpoint. A two-condition
task can add a second entry without changing actor or recorder code:

```python
success_conditions = (
    SuccessCondition(name="inserted", clip=(100, 100, 112, 112)),
    SuccessCondition(
        name="released",
        clip=(240, 100, 112, 112),
        image_key="classifier_released",
        depth_key="classifier_released_depth",
        checkpoint="classifiers/released.pt",
    ),
)
```

Condition names, RGB keys, depth keys, and checkpoints must be unique. The
first entry is not otherwise privileged; the legacy names are merely retained
for old single-classifier datasets and checkpoints.

## Cartesian axis control

Every task declares one `movable_axes` tuple in XYZ/RX/RY/RZ order. There are
only two controller modes:

- a movable axis keeps the normal compliant PID and receives policy/operator
  target updates;
- a fixed axis uses the shared strong PID while far from its target and falls
  back to the normal soft PID inside 5 mm for translation or 5 degrees for
  rotation.

The PID values and mode application live in the shared UR5e controller. Tasks
declare only the mask and their fixed targets; they do not mutate PID arrays.

`push_t` uses a three-dimensional learned action `[dx, dy, dz]`. Its robot task
holds roll, pitch, and yaw at the canonical straight-up orientation through
compliant impedance and injects a fixed closed-gripper hardware command outside
the policy/replay action tensor.

`gear` uses a five-dimensional learned action
`[dx, dy, dz, local_drz, gripper]`. Roll and pitch remain at the canonical
straight-up orientation, yaw is constrained to `[-180, 0]` degrees, and
the continuous gripper remains part of the policy and replay action.

`insert_usb` uses `[dx, dy, dz, gripper]` with the shared translation scale,
full gripper travel, and fixed downward orientation. Its CLI
package name is lowercase snake case; experiment directories may use the
human-facing `insert_USB-*` spelling.

Entrypoints resolve only the component they need with
`load_component(task_name, component)`. Consequently learner-only processes do
not import robot hardware dependencies.
