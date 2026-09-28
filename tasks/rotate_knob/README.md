# Rotate knob

Task name: `rotate_knob`. Uses the branch's existing learner unchanged.

- Learned action: `[dx, dy, dz, local_drz, gripper]` (5 outputs).
- Four Cartesian DoFs: XYZ and rotation about the tool's nominal local Z axis.
  RX/RY are held at `zero_point_rot`; RZ is **not** the third component of the
  UR rotation vector. Rotation is composed using `R_nominal * Rz(yaw)`.
- XYZ target offset: normalized command times `linear_speed` (0.05 m).
  Local yaw target increment: normalized command times `angular_speed` (0.1571 rad, 9.0 deg).
  Yaw commands accumulate from the previous target and are clipped to the yaw
  range. Roll/pitch operator inputs are ignored.
  Controller RZ speed cap is 1.5708 rad/s (90.0 deg/s); other speed caps and
  torque limits are unchanged. This is separate from the action increment.
- Gripper matches `hang_double_strings_2`: continuous [-1, 1], -1=open, +1=closed,
  calibrated full travel, speed 128, force setting 200, and the same reversal
  penalty.
- One visual success classifier (`classifier.pt`), not the hang task's two
  classifiers. Camera preprocessing/recording/training use the shared stack.
- 200 steps per episode. Reset opens the gripper once, moves only Z to 160 mm,
  rotates in place to the configured reset yaw through numerically increasing
  local-yaw waypoints of at most 30 degrees, then moves only XY to the randomized
  reset point while holding Z/yaw. This makes a -100 to +100 degree reset traverse
  through zero instead of taking SO(3)'s opposite 160-degree shortest path. A
  failed reset stage (5-second timeout per stage) sends a hold-current-pose
  command to cancel
  the stale target and raises an error. This needs the updated robot driver and
  working communication; it is not an emergency stop and safety recovery remains
  enabled. Between episodes, robot motion and the subsequent manual countdown
  together target 7 seconds (outside episode steps). At least 5 seconds of manual
  time are retained: robot motion exceeding 5 seconds extends the total duration.
  Startup skips this extra wait. The robot reset does NOT reset the knob's angle automatically.

Every reset (including startup) independently samples X and Y uniformly over
the full workspace: X [-204, -44] mm and Y [-601, -451] mm. Z stays at the
maximum 160 mm. The knob target (-124, -526, 109) mm is centered in XY and lies on the lower Z
boundary. The local reset yaw remains 100 degrees.

## Calibration required before hardware use

The global camera is acquired at 1280x720. Crop coordinates use
`(x, y, width, height)` on that native frame; no robot motion was tested.
User-selected workspace: X [-204, -44] mm, Y [-601, -451] mm. Nominal orientation
is still provisional. The global policy crop `(618, 1, 300, 300)` covers the
knob fixture and nearby gripper approach area in the current frame. The classifier
crop `(700, 110, 100, 100)` surrounds the knob and position marker in the current
global frame. Collect and label knob success/failure states and retrain the classifier;
the previous indicator classifier must not be reused. Z is limited to [109, 160] mm, with reset
Z and reset-clearance Z at 160 mm. These command-target limits do not
guarantee the measured pose cannot overshoot under compliant control.

Before running, set `config.py` for the actual fixture:

1. `zero_point_rot`: align the tool Z axis with the knob shaft.
2. XYZ safety limits, `reset_pos`, and a collision-free `reset_clearance_z`
   path (this task moves Z to 160 mm, rotates, then moves XY).
3. `yaw_range_rad` and `reset_yaw_rad`: user-selected range is -100 to 100 degrees,
   relative to the nominal tool orientation; reset yaw is set to 100 degrees.
   This implementation is bounded-angle control, not continuous multi-turn.
4. `global_policy_clip` and the success condition's `clip`, then collect labels
   and train a new knob classifier; do not reuse a hang-string classifier.

Entry points discover `--task rotate_knob` automatically. Five-dimensional demo
buffers and checkpoints must be recorded/trained for this task; the double-string task's
four-dimensional actions/checkpoints are not interchangeable.

## Record classifier demonstrations with task resets

With the `rotate_knob` robot driver and SpaceMouse process already running:

```bash
uv run --extra rocm --extra actor python -m tasks.rotate_knob.record_classifier_demo
```

This task-specific collector creates one directory per episode under
`outputs/classifier_samples/rotate_knob`, runs the same randomized reset pose and
reset countdown as task execution before every episode, and records 200 frames by
default. Use `--append` to preserve existing samples, `n` to finish the current
episode early, and `q` or Escape to exit.
