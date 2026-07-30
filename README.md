# Isaac Lab OSC: 7-DoF vs 9-DoF Flexiv ultrasound

This project runs a side-by-side Isaac Sim comparison:

- red: Flexiv Rizon 4s joints 1–7, with the supplemental wrist locked;
- green: the same Rizon plus the corrected two-revolute-joint wrist;
- both: the same linear ultrasound probe, Assembly3 patient, surface path,
  15 N normal-force command, and Isaac Lab controller implementation.

The torque controller is **Isaac Lab's unmodified**
`isaaclab.controllers.OperationalSpaceController`. The runner calls
`set_command()` and `compute()` directly. There is no project-local OSC,
Jacobian inverse, or secondary torque optimizer.

## What the demo does

1. Approaches the upper torso without an initial collision.
2. Acquires contact and ramps the task-normal wrench to 15 N.
3. Scans 16 cm along the upper torso in 4 seconds while following the
   smoothed Assembly3 surface normal.
4. Stops at the final patient point.
5. Performs a 20-degree pitch-only reorientation, returns to neutral, then
   performs yaw-only reorientation.
6. Returns to neutral and keeps the probe on the patient until the window is
   closed.

The path is an analytical task-space curve sampled from the patient surface;
it is not RRT, A*, OMPL, or another obstacle-search planner.

## Prepare local assets

The downloaded/generated CAD is intentionally not committed. Defaults can be
overridden with `RIZON_BASE_USD`, `RIZON_WRIST_GEOMETRY_USD`,
`ASSEMBLY3_STL`, `ASSEMBLY3_URDF`, and `RIZON_GROUND_USD`.

```bash
cd /home/bdml-sim/Isaacsim-9DoF-flexiv
/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  tools/prepare_local_assets.py
```

This creates:

```text
generated/rizon4s_exact_wrist_probe.usda
generated/assembly3_patient.usd
generated/assembly3_torso_surface.npz
```

The full Assembly3 mesh is used visually. A smoothed, static upper-torso
triangle surface derived from the same STL is used for contact and normals.
The patient, beds, pedestals, and both robot roots are fixed.

## Open the visual simulation

```bash
cd /home/bdml-sim/Isaacsim-9DoF-flexiv
./launch_osc_comparison.sh
```

With no `--max_steps`, the window remains open until you close it. The
launcher supports `ISAACLAB_ROOT` and `ISAACSIM_PYTHON` overrides.

Visual legend:

- red robot: 7 controlled joints;
- green robot: 9 controlled joints;
- magenta arrow: commanded probe force, length proportional to the command;
- cyan arrow: measured patient reaction, length proportional to measurement;
- frames: current and target acoustic poses.

Headless finite run:

```bash
./launch_osc_comparison.sh --viz none --max_steps 2100
```

Write a JSON validation report:

```bash
./launch_osc_comparison.sh --viz none --max_steps 2300 \
  --validation_report generated/validation_report.json
```

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python -m pytest -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids unrelated ROS pytest plugins in this
environment.

## Current measured behavior

The stabilized GPU runs show:

- static scene transform drift: `0`;
- surface scan force: generally about `14.8–15.2 N`;
- no contact-loss threshold violation in the retained configuration
  (tracked peaks `0.040/0.044 s`, below the `0.1 s` limit);
- scan normal alignment: usually below `3 deg`, with short peaks around
  `3.7 deg`;
- pitch reorientation: green uses the distal wrist and reduces phase-local
  motion of joints 1–7 by roughly 20–25% in representative runs;
- full 2300-step run: cumulative joints 1–7 travel is `6.278 rad` red versus
  `5.367 rad` green, about 14.5% lower for green;
- rigid-face reorientation still causes transient force variation, roughly
  `12.4–17.2 N`, and is the main remaining physics/control limitation.

The strict validation report remains `overall_pass=false`: it deliberately
fails when any sampled force, scan-normal, or tangential-error maximum exceeds
its gate. It also requires full scan/pitch/yaw/final-hold phase coverage and
audits the actual per-side OSC commands after safety overrides. It is
evidence, not a tuned-to-pass score.

See [PROJECT_MEMORY.md](PROJECT_MEMORY.md) for the full requirement history,
architecture, evidence, and remaining work.
