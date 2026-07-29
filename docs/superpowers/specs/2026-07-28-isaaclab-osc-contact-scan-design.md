# Isaac Lab OSC contact-scan design

## Objective

Compare the same Flexiv Rizon 4s ultrasound task with seven controlled joints
and with a two-DoF distal wrist, while keeping the clinical reference and
controller implementation identical.

## Control boundary

Isaac Lab `OperationalSpaceController` owns all task-to-joint torque
calculation. Local modules may produce geometry, trajectory, measured-force
filtering, safety state, null-space joint targets, and metrics, but may not
produce OSC torque.

The approach/acquisition mode uses the built-in pose OSC. Contact mode uses
the built-in hybrid pose/wrench OSC with task Z selected for force and the
remaining axes selected for motion.

## Comparison validity

- Red and green receive one shared surface pose and 15 N task wrench.
- Red locks joints 8–9.
- Green exposes all nine joints and supplies an Isaac Lab null-space posture
  target that preserves joints 1–7 and assigns reorientation to the wrist.
- Main-arm reduction is reported separately from distal-wrist travel.
- A reduction should only be presented as a task-equivalent advantage when
  pose/force accuracy is also shown.

## Assets and collision

The original working Rizon USD is referenced rather than reconstructed. The
Onshape-derived wrist geometry is wrapped with a corrected terminal revolute
joint. The Assembly3 STL is rendered in full, while a reproducible smoothed
upper-torso triangle mesh provides stable contact and normals. All support
objects and roots are static/fixed.

## Task sequence

The deterministic state sequence is approach, force/contact ramp, 16-cm
surface scan, pitch-only reorientation, neutral, yaw-only reorientation, and
neutral. The scan uses quintic progress. This is analytical task-space path
generation, not search-based motion planning.

## Safety and evidence

The runtime monitors surface validity, contact loss, overload, torque
magnitude/rate, fixed-transform drift, force, pose error, normal angle, and
joint travel. Magenta command and cyan measurement arrows must never be
described as the same quantity.
