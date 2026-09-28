# Insert-gear reset

Task success requires both image classifiers AND TCP position within a strict
10 mm Euclidean distance of (-118, -460, 35) mm. This gate does not check
orientation. Missing/non-finite TCP observations fail the gate. Existing image
classifier slots remain in place; no third image classifier is introduced. The
demo preview shows the TCP distance alongside both image decisions.

Startup still uses the robot driver's three Enter confirmations: approach and
descend to the pickup point, close, then lift. Complete startup before running
actor/recorder commands.

Each subsequent episode reset follows this sequence (coordinates are world
TCP positions in mm):

1. Read the actual height. If Z > 48, descend in place to Z=48, retaining
   current XY, orientation and closed gripper. At Z <= 48, release in place.
2. Open the gripper, wait the configured 1 second mechanical settling time,
   and lift at the actual release XY/orientation to Z=121.
3. Restore the complete task nominal orientation, not a zero rotation vector.
4. Print the manual reposition prompt and wait 5 seconds.
5. With the gripper open, move at Z=121 to pickup XY (-118, -460), then
   descend to the startup pickup height Z=35.
6. Close once, wait 2 seconds, and lift to Z=121.
7. Uniformly sample the complete safety workspace in XY (X=-175.5..24.5,
   Y=-525.7..-350.7) at fixed Z=121, align RZ to 0 degrees, and settle for
   1 second.

The gripper waits do not verify successful object capture. Opening uses
`reset_gripper_wait_s` (1 second); closing uses `reset_gripper_close_wait_s`
(2 seconds) before the lift.
Movement stages use the existing 5 mm position and 1 degree orientation
tolerances, with a 10-second timeout; a failed stage stops later stages.

The workspace is X=-175.5..24.5, Y=-525.7..-350.7, Z=33..121 mm, including reset
commands. Explicit `insert_gear_reset` commands carry open/close
commands; ordinary actions retain the fixed closed gripper. The prior RX -5
degree and Y+ release motions have been removed. Local RZ is constrained to
-90..20 degrees; reset and startup use 0 degrees.

The global camera is acquired at 1280x720. Its calibrated raw crops are
`global_policy=(663,138,294,294)`, `classifier_1=(685,299,80,82)`, and
`classifier_2=(760,304,67,67)`, all in `(x,y,width,height)` format.

Restart both the robot driver and the actor/recorder after updating: both sides
must understand the reset command marker. Tests use simulated poses/gripper
commands; no physical robot movement is performed by the tests.
