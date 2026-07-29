# Official IsaacLab OSC Wrist-Advantage Design

## Objective

Demonstrate, in a side-by-side Isaac Sim scene, why the Flexiv Rizon 4s
benefits from the additional two-degree-of-freedom wrist during ultrasound
scanning and fixed-point probe reorientation.

The red robot uses the original seven Rizon joints with the supplemental
wrist locked. The green robot uses all nine joints. Both robots receive the
same operational-space pose, wrench, and stiffness commands. During each
fixed-point reorientation, the green robot's first seven joints must travel
at least 50 percent less than the red robot's first seven joints.

## Controller Boundary

All task-space torque generation comes directly from
`isaaclab.controllers.OperationalSpaceController`. The project must not
implement a pseudoinverse, operational-space inertia matrix, Jacobian
transpose force term, null-space projector, or an alternative OSC solver.

The controller data flow follows IsaacLab's
`scripts/tutorials/05_controllers/run_osc.py`:

1. Read the PhysX end-effector Jacobian, generalized mass matrix, gravity
   compensation vector, pose, velocity, contact force, joint position, and
   joint velocity.
2. Express the Jacobian, pose, velocity, task frame, and force in the robot
   root frame.
3. Construct a `pose_abs + wrench_abs + variable_kp` command.
4. Call `OperationalSpaceController.set_command()`.
5. Call `OperationalSpaceController.compute()` with the complete dynamics
   state and the null-space joint-position target.
6. Apply only the returned joint efforts, subject to the robot's physical
   effort and effort-rate limits.

The hybrid controller configuration uses:

- `target_types=["pose_abs", "wrench_abs"]`
- `impedance_mode="variable_kp"`
- `inertial_dynamics_decoupling=True`
- `partial_inertial_dynamics_decoupling=False`
- `motion_damping_ratio_task=1.0`
- `motion_control_axes_task=[1, 1, 0, 1, 1, 1]`
- `contact_wrench_control_axes_task=[0, 0, 1, 0, 0, 0]`
- `nullspace_control="position"`

The patient-normal task axis remains force-controlled. The two surface
tangent axes and all three orientation axes remain motion-controlled.
Robot gravity remains enabled, so the controller's built-in gravity
compensation is enabled and receives the PhysX gravity vector. This is the
only deliberate configuration difference from the tutorial, whose Franka
example disables robot gravity.

The approach phase uses a second official
`OperationalSpaceController` configured for six-axis pose control. It uses
the same full inertial decoupling, variable stiffness, and built-in
null-space implementation. Switching controllers does not introduce any
locally computed task-space torque.

## Redundancy Policy

The red controller receives only Rizon joints 1 through 7. Its supplemental
wrist is locked at zero with the existing position drive.

The green controller receives all nine joints. At the start of each
reorientation:

- joints 1 through 7 in the null-space target are frozen to their measured
  phase-start positions;
- wrist pitch is assigned the negative requested task-frame pitch during a
  pitch-only phase, matching the authored positive-Y joint frame;
- wrist yaw is assigned the positive requested task-frame yaw during a
  yaw-only phase, matching the authored positive-Z joint frame;
- the unused supplemental wrist axis is held at zero.

The wrist-axis signs are verified from the simulated Jacobian columns and a
small positive joint-displacement test before the challenge sequence is
accepted. The final signs are constants in the checked-in configuration;
there is no run-time trial-and-error controller.

The null-space stiffness is IsaacLab's default value of 10.0 for both
robots. A separate high-gain controller for the green robot is not allowed.

## Demonstration Sequence

The default GUI run contains two examples after approach and force ramp.
The probe remains in contact and the target normal wrench remains 15 N
throughout both examples.

### Example 1: Equal-accuracy motion comparison

1. Scan 160 mm along the upper torso in 4 seconds while keeping the probe
   acoustic axis normal to the surface.
2. At the scan endpoint, rotate only task-frame pitch from 0 to 35 degrees.
3. Return pitch to zero.
4. Rotate only task-frame yaw from 0 to 45 degrees.
5. Return yaw to zero.

Both robots must complete this example without non-probe patient collision.
For both the pitch-only and yaw-only phases, the green robot's cumulative
absolute travel over joints 1 through 7 must be no more than 50 percent of
the red robot's corresponding travel. End-effector pose and force accuracy
must be comparable before the reduction is reported.

### Example 2: Near-patient collision challenge

The common probe target moves to Assembly3 torso surface coordinates
`x=0.10 m, y=1.32 m`, near the shoulder-side boundary of the validated
surface map. At that fixed point, pitch alone changes from 0 to 50 degrees
and returns to zero.

The green robot must complete the pose while maintaining contact through
the distal wrist. The red robot is expected either to:

- trigger a non-probe robot-to-patient collision, or
- exceed twice the green robot's joints-1-through-7 travel.

If the red robot triggers a collision, its challenge trajectory freezes at
the last safe reference and its status becomes `COLLISION_STOP`. The green
robot continues to the common requested goal. Physics collision response
stays enabled; visual interpenetration is not used as evidence.

The challenge point and 50-degree angle are fixed test inputs. A result that
does not meet the stated collision-or-motion criterion is a failed
demonstration, not a reason to silently move the target during the run.

## Collision Classification

Probe-to-patient contact is intentional and is measured by the existing
probe contact sensors.

Additional contact sensors cover every robot rigid body except
`linear_probe`. Their patient-filtered contact histories classify unwanted
arm contact. A non-probe collision is latched when patient contact exceeds
2 N. Contacts with the pedestal, ground, or the robot itself do not count as
patient collisions.

The GUI displays each side independently:

- `CONTACT OK`
- `NEAR COLLISION` for a non-probe patient contact between 0.5 N and 2 N
- `COLLISION STOP` after the 2 N latch

## Metrics and Visual Evidence

Metrics are accumulated separately for every phase instead of relying only
on whole-run cumulative travel.

Each phase records:

- cumulative absolute travel of joints 1 through 7 for red and green;
- cumulative absolute travel of green wrist joints 8 and 9;
- green-to-red main-arm travel ratio and reduction percentage;
- maximum end-effector position and orientation error per side;
- maximum and root-mean-square 15 N force error per side after settling;
- maximum contact-loss duration per side;
- maximum non-probe patient-contact force per side;
- whether each side completed the requested phase;
- whether both sides received identical pose, wrench, stiffness, and task
  frame commands before a collision stop.

The screen overlays the current phase, force command and measurement,
phase-local arm travel, green wrist travel, validated reduction percentage,
and collision state. Reduction text remains hidden until both sides satisfy
the phase's accuracy gates.

The JSON report contains separate `equal_accuracy_comparison` and
`collision_challenge` sections. Whole-run cumulative travel remains
diagnostic only.

## Acceptance Criteria

The equal-accuracy comparison passes only when:

- both sides receive identical operational-space commands;
- both sides complete pitch-only and yaw-only phases;
- green joints 1 through 7 travel at least 50 percent less in each phase;
- force commands remain 15 N;
- post-settling measured force stays within 1.5 N of target;
- probe normal error during surface scanning stays below 3 degrees;
- task orientation error stays below 5 degrees;
- tangent position error stays below 10 mm;
- contact loss does not exceed 0.1 seconds;
- neither robot has a non-probe patient collision.

The collision challenge passes only when:

- green completes the 50-degree pitch goal;
- green maintains the same force, pose, and contact tolerances;
- green has no non-probe patient collision;
- red either enters `COLLISION_STOP` or its main-arm travel exceeds twice
  green's main-arm travel.

The complete demonstration passes only when both sections pass. The report
must preserve failed measurements and must not claim a wrist advantage when
the accuracy gates fail.

## Testing and Verification

Pure Python tests cover:

- official-controller source-contract checks;
- the exact tutorial-derived configuration;
- wrist-target sign and phase-start arm-hold behavior;
- phase-local travel aggregation and the 50 percent gate;
- reduction hiding when accuracy differs;
- probe versus non-probe collision classification;
- collision-stop state transitions;
- the two fixed demonstration sequences.

A full GPU validation run produces the JSON report and exits nonzero unless
all acceptance criteria pass. After headless validation, a GUI run is
started without `--max_steps`; it remains open until the user closes the
Isaac Sim window.

## Non-Goals

- No custom operational-space solver or weighted pseudoinverse.
- No artificial reduction of the green robot's arm effort limits.
- No separate position controller that directly drives wrist joints 8 and
  9 during the comparison.
- No deformable-patient simulation.
- No claim of hardware or clinical validation.
