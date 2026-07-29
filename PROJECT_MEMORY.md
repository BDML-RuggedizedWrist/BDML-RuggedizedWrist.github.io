# Project memory: Flexiv 9-DoF ultrasound OSC comparison

Last updated: 2026-07-28

## Professor's goal

The professor asked for a MuJoCo or Isaac simulation that explains why a
two-DoF distal wrist remains valuable even when the base robot already has
seven joints. The preferred deliverable is one or two left/right comparisons
of the same tool-reorientation task, without and with the wrist. The clinical
tool may be an ultrasound probe or VisionFT. The important visual result is:

- the 7-DoF arm must make awkward, larger proximal-joint motions;
- the 9-DoF system uses the two distal joints and moves joints 1–7 less;
- operational-space control keeps the user-facing task definition simple.

The current implementation uses Isaac Sim/Isaac Lab and a linear ultrasound
probe.

## Final user requirements collected across sessions

1. Use the first known-working Flexiv Rizon 4s asset.
2. Connect the corrected Onshape wrist with two real revolute joints:
   pitch about Y and terminal roll about Z.
3. Center the linear ultrasound probe on the terminal output and mount it
   axially.
4. Use the downloaded Assembly3 patient model, not a box phantom.
5. Fix the patient, beds, pedestals, and robot bases.
6. Enable collision/contact sensing and prevent the probe from silently
   passing through the patient.
7. Scan the upper torso quickly, with the acoustic axis following the local
   patient normal.
8. Stop on the body and adjust only one orientation axis at a time.
9. Command 15 N in the surface-normal direction.
10. Show force vectors in distinct colors.
11. Compare red 7-DoF and green 9-DoF robots under the same clinical
    pose/wrench reference.
12. Prefer joints 8–9 whenever they can achieve the reorientation, minimizing
    motion of green joints 1–7.
13. Most importantly, use Isaac Lab's own
    `OperationalSpaceController`, not a custom OSC implementation.
14. Keep the Isaac Sim window open after launch.
15. Put the reproducible source on
    `universeleaf/Isaacsim-9DoF-flexiv`.

## Implemented architecture

### Controller

`scripts/run_osc_comparison.py` imports and constructs:

```python
from isaaclab.controllers import (
    OperationalSpaceController,
    OperationalSpaceControllerCfg,
)
```

The runner directly calls `set_command()` and `compute()`. Hybrid contact
control uses:

```text
motion_control_axes_task         = (1, 1, 0, 1, 1, 1)
contact_wrench_control_axes_task = (0, 0, 1, 0, 0, 0)
contact_wrench_stiffness_task    = (0, 0, 0.8, 0, 0, 0)
nullspace_control                = "position"
```

The project does not calculate OSC torque, operational-space inertia, or a
Jacobian pseudoinverse. Project code only supplies task references, measured
force, null-space posture targets, and final safety saturation/rate limits.

### Force feedback

The probe has an Isaac Lab `ContactSensor`. Feedback follows the official
Isaac Lab OSC tutorial by averaging
`ContactSensor.data.net_forces_w_history`, then rotating it into the robot
root frame and passing it through `current_ee_force_b`.

An earlier implementation used `force_matrix_w_history`. That was incorrect
for this static patient under GPU PhysX: the filtered matrix remained zero
and emitted an unsupported-filter warning even while the unfiltered net
sensor measured contact. This root cause was diagnosed and removed.

Magenta is the commanded inward force. Cyan is the measured outward patient
reaction. Arrow lengths scale with force magnitude.

### Path generation

This is not sampling-based motion planning. The code preprocesses the
Assembly3 upper torso into a regular height/normal map. A quintic analytical
trajectory moves 16 cm along this surface in 4 seconds. At each sample it
queries surface position and normal, constructs the desired acoustic frame,
and sends that pose to Isaac Lab OSC.

The original STL contained millimeter-scale facet/normal jumps that destabilized
fast scanning. The retained contact surface uses Gaussian smoothing with
sigma eight 4-mm grid cells. The full patient mesh remains unchanged for
visual rendering.

### 9-DoF preference

Both robots get the same clinical task pose and wrench. Red controls only
joints 1–7 and locks the supplemental wrist. Green controls all nine.

`RedundancyPolicy` supplies only a built-in null-space position target:

- joints 1–7 target their values at the start of each reorientation phase;
- pitch is assigned to wrist joint 8;
- yaw is assigned to wrist joint 9.

The reorientation OSC instance uses stronger built-in null-space stiffness
for green. This is how the implementation prefers the distal wrist without
adding a custom weighted OSC solver.

## Current progress and evidence

Completed:

- corrected two-revolute-joint wrist/probe USD generation;
- Assembly3 full-body visual and derived torso collider;
- fixed bases, patient, beds, and pedestals;
- collision/contact sensing;
- no initial impact in the retained registered scene;
- official Isaac Lab pose and hybrid force/motion OSC;
- official net-force history feedback;
- 15 N command and colored/scaled force arrows;
- 16-cm/4-s torso scan;
- pitch-only, neutral, yaw-only phase sequence;
- final neutral contact hold (the sequence does not restart);
- contact-loss safety and invalid-surface hold;
- actual red/green OSC command auditing after safety overrides;
- monotonic contact-loss peak recording and complete-phase acceptance gating;
- phase-local main-arm and distal-wrist travel logging;
- pure unit/source-contract tests.

Representative retained GPU run:

- static transform drift: `0`;
- scan measured force usually `14.8–15.2 N`;
- no threshold-triggering contact loss during the scan;
- normal alignment usually below `3 deg`, with brief peaks around `3.7 deg`;
- 20-degree pitch phase: approximately `1.86 rad` red versus `1.40 rad`
  green main-arm travel near the end of the phase, about 25% less;
- green distal wrist travel in that phase: approximately `0.44 rad`.

The final 2300-step headless regression included the complete scan, pitch,
yaw, and final neutral hold. Its last sample was:

- cumulative joints 1–7 travel: `6.278 rad` red versus `5.367 rad` green,
  about 14.5% lower for green;
- measured force: `14.949 N` red and `14.876 N` green;
- tangential error: `2.84 mm` red and `1.64 mm` green;
- maximum contact-loss episode: `0.040 s` red and `0.044 s` green, below
  the `0.1 s` limit;
- fixed-object drift: `0`.

The strict aggregate report is intentionally `overall_pass=false`. Its
maximums include the pitch transient and the middle of the fast scan:

- force error: `2.64 N` red and `2.27 N` green;
- scan normal angle: `3.38 deg` red and `3.69 deg` green;
- orientation tracking error: `10.41 deg` red and `7.84 deg` green during
  the pitch transient;
- tangential error: `33.3 mm` red and `28.6 mm` green.

Therefore the cumulative reduction is useful engineering evidence, but is
not yet an accuracy-gated publication result.

The numbers vary slightly with render/headless timing. Do not describe a
15 N command as if it were a measured result.

## Known limitations and future work

1. **Reorientation force transient:** on the retained rigid patient and flat
   probe face, pitch reorientation varies approximately `12.4–17.2 N`.
   Scan force regulation is much tighter. A compliant gel/patient contact
   model or a calibrated rounded acoustic contact proxy is the next physics
   improvement.
2. **Rigid flat-face geometry:** a finite flat probe cannot tilt about its
   center while remaining fully flush with a rigid curved surface. PhysX
   naturally transitions to edge contact. Do not add a discontinuous
   center-to-edge target jump; that approach was tested and rejected.
3. **Peak scan error:** fast 4-second scanning can briefly exceed the desired
   3-degree/10-mm accuracy gates even though it settles near the end.
4. **Presentation evidence:** the latest run covers the entire yaw phase, but
   publication-quality paired GIF capture and a phase-by-phase plotted metric
   report are still needed.
5. **On-screen HUD:** current quantitative metrics are printed in the launch
   terminal. A polished in-viewport overlay and automated paired GIF capture
   remain future presentation work.
6. **Acceptance report:** strict maximum-error gates should remain strict;
   do not loosen them merely to display PASS.
7. **Hardware calibration:** wrist effort/inertia, probe mass, contact
   material, and patient compliance need real measurements before claiming
   hardware fidelity.

## Reproduce in a new session

```bash
cd /home/bdml-sim/Isaacsim-9DoF-flexiv

/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  tools/prepare_local_assets.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python -m pytest -q

./launch_osc_comparison.sh
```

Close the GUI manually. With no `--max_steps`, the launcher does not close it
automatically.

For a finite evidence run:

```bash
./launch_osc_comparison.sh --viz none --max_steps 2300 \
  --validation_report generated/validation_report.json
```
