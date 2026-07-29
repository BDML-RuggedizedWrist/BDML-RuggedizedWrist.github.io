# Weighted OSC Contact Scan Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a reproducible Isaac Sim comparison in which a red 7-DoF Flexiv Rizon 4s and a green 9-DoF Rizon-plus-wrist follow the same Assembly3 torso task with torque-level operational-space control, measured 15 N normal-force regulation, fixed scene assets, and an accuracy-gated proof that the extra wrist reduces motion of the seven main-arm joints.

**Architecture:** Keep all controller mathematics independent of Isaac in the `rizon_osc` package. Preprocess the downloaded Assembly3 STL into a compact height/normal map, generate a differentiable surface-constrained task trajectory, solve a weighted damped operational-space acceleration problem, and close the normal-force loop with filtered contact measurements. The Isaac runner owns scene composition, contact sensors, robot state extraction, torque/rate limits, markers, metrics, static-transform assertions, and GUI lifetime.

**Tech Stack:** Python 3.11+, NumPy, PyTorch, pytest, trimesh, Isaac Sim 5.x, Isaac Lab 2.x, USD/PhysX, Git.

---

## Task 1: Create the testable package skeleton

**Files:**

- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/rizon_osc/__init__.py`
- Create: `tests/conftest.py`
- Modify: `README.md`

**Step 1: Add a failing package smoke test**

Create `tests/test_package.py`:

```python
def test_public_package_version_exists():
    import rizon_osc

    assert rizon_osc.__version__
```

**Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_package.py -q
```

Expected: `ModuleNotFoundError: No module named 'rizon_osc'`.

**Step 3: Add the minimum package metadata**

Configure setuptools with `src/` layout, install-free pytest path configuration, and a nonempty `__version__`. Ignore caches, generated torso artifacts, generated USD assets, validation reports, and Kit logs.

**Step 4: Run the test to verify it passes**

Run:

```bash
python -m pytest tests/test_package.py -q
```

Expected: one passing test.

**Step 5: Commit**

```bash
git add pyproject.toml .gitignore src tests README.md
git commit -m "chore: scaffold testable weighted OSC package"
```

## Task 2: Implement the Assembly3 surface model

**Files:**

- Create: `src/rizon_osc/surface_model.py`
- Create: `tests/test_surface_model.py`
- Create: `tools/preprocess_assembly3_surface.py`

**Step 1: Write failing interpolation and validity tests**

Cover:

- bilinear height interpolation on a known sloped plane;
- interpolated normals are unit length;
- invalid-mask cells and out-of-bounds queries return `valid=False`;
- local-map points and normals transform correctly into robot-base coordinates;
- saved metadata preserves source path, SHA-256, and coordinate convention.

**Step 2: Run the focused tests and verify failure**

Run:

```bash
python -m pytest tests/test_surface_model.py -q
```

Expected: import failure because `rizon_osc.surface_model` does not exist.

**Step 3: Implement `SurfaceMap`**

Implement:

- validated ascending `x_grid` and `y_grid`;
- `height`, `normal`, and boolean `valid_mask` arrays;
- bilinear query returning a typed `SurfaceSample`;
- no extrapolation outside the grid or across invalid cells;
- normalized interpolated normals;
- load/save support for `.npz` plus JSON metadata;
- rigid point/normal transform helper.

**Step 4: Implement the STL preprocessing CLI**

The CLI must:

- default to `/home/bdml-sim/Downloads/Assembly 3/assembly_3/meshes/Part_2.stl`;
- print mesh bounds and require explicit ROI limits or accept documented Assembly3 defaults;
- ray-sample the upper torso onto a regular grid;
- fill only small interior holes;
- estimate smoothed gradients and outward unit normals;
- save under `generated/assembly3_torso_surface.npz`;
- include source SHA-256 and preprocessing parameters;
- never modify files in Downloads.

**Step 5: Run focused tests**

Run:

```bash
python -m pytest tests/test_surface_model.py -q
```

Expected: all surface-model tests pass.

**Step 6: Commit**

```bash
git add src/rizon_osc/surface_model.py tests/test_surface_model.py tools/preprocess_assembly3_surface.py
git commit -m "feat: add Assembly3 torso surface map"
```

## Task 3: Implement the continuous surface-constrained trajectory

**Files:**

- Create: `src/rizon_osc/trajectory.py`
- Create: `tests/test_trajectory.py`

**Step 1: Write failing trajectory tests**

Cover:

- quintic progress equals 0/1 at endpoints;
- endpoint velocity and acceleration are zero;
- scan position lies on the queried surface;
- task +Z follows outward surface normal and probe acoustic -Z points into the body;
- tangent frame remains orthonormal and continuous;
- trajectory returns finite position, quaternion, linear/angular velocity, and acceleration;
- pitch-only and yaw-only phases change exactly one relative orientation axis;
- scan duration is 4 seconds over 0.16 m.

**Step 2: Run focused tests and verify failure**

Run:

```bash
python -m pytest tests/test_trajectory.py -q
```

Expected: import failure because `rizon_osc.trajectory` does not exist.

**Step 3: Implement the trajectory generator**

Implement an immutable `TaskReference` containing phase, pose, twist, acceleration, surface normal, contact point, and force target. Use quintic timing, finite-difference frame derivatives with sign-continuous quaternions, and the approved phases:

1. `APPROACH`;
2. `CONTACT_RAMP`;
3. `SURFACE_SCAN`;
4. `PITCH_ONLY`;
5. `RETURN_NEUTRAL`;
6. `YAW_ONLY`.

The scan occurs once; the three fixed-contact orientation phases repeat. Invalid surface queries hold the last safe reference and pause path time.

**Step 4: Run focused tests**

Run:

```bash
python -m pytest tests/test_trajectory.py -q
```

Expected: all trajectory tests pass.

**Step 5: Commit**

```bash
git add src/rizon_osc/trajectory.py tests/test_trajectory.py
git commit -m "feat: add surface constrained ultrasound trajectory"
```

## Task 4: Implement measured-force feedback and safety state transitions

**Files:**

- Create: `src/rizon_osc/force_control.py`
- Create: `src/rizon_osc/state_machine.py`
- Create: `tests/test_force_control.py`
- Create: `tests/test_state_machine.py`

**Step 1: Write failing force-controller tests**

Cover:

- positive force error raises commanded compression;
- command is bounded to `[0, max_force]`;
- the integral does not wind up while saturated;
- reset clears filter and integrator;
- force projection uses the current surface normal and clamps tensile force to zero;
- history filtering rejects a one-sample impulse.

**Step 2: Write failing safety-transition tests**

Cover:

- contact loss below 0.1 s does not leave tracking;
- contact loss above 0.1 s enters `REACQUIRE`;
- invalid surface and hard-force violation freeze path time;
- stable reacquired contact resumes the previous task;
- non-contact phases reset the force integrator.

**Step 3: Verify both suites fail**

Run:

```bash
python -m pytest tests/test_force_control.py tests/test_state_machine.py -q
```

Expected: import failures for both modules.

**Step 4: Implement the bounded PI force loop**

Implement `NormalForceController` with:

- configurable target 15 N, proportional/integral gains, maximum command, history length, and low-pass coefficient;
- conditional-integration anti-windup;
- reset outside force-controlled phases;
- raw, averaged, filtered, commanded, error, and saturation outputs.

**Step 5: Implement the phase/safety supervisor**

Implement deterministic state updates using explicit elapsed durations. Separate the nominal trajectory phase from safety mode so `REACQUIRE` and `FORCE_HOLD` cannot be mistaken for task phases.

**Step 6: Run focused tests**

Run:

```bash
python -m pytest tests/test_force_control.py tests/test_state_machine.py -q
```

Expected: all tests pass.

**Step 7: Commit**

```bash
git add src/rizon_osc/force_control.py src/rizon_osc/state_machine.py tests/test_force_control.py tests/test_state_machine.py
git commit -m "feat: add closed loop normal force control"
```

## Task 5: Implement weighted dynamic operational-space control

**Files:**

- Create: `src/rizon_osc/weighted_osc.py`
- Create: `tests/test_weighted_osc.py`

**Step 1: Write failing weighted-solve tests**

Use synthetic Jacobians with redundant compatible wrist columns. Cover:

- output shape and finite values;
- damped task residual stays within tolerance;
- lowering the final two weights reduces the norm of the first seven joint accelerations;
- increasing a near-limit joint weight reduces its motion;
- motion axes contain two tangential translations plus three rotations;
- normal-force torque is exactly `J_n.T @ f_command`;
- full torque includes `M @ qdd`, gravity, damping, and bounded nullspace terms;
- singular matrices increase damping and never return NaN.

**Step 2: Verify focused tests fail**

Run:

```bash
python -m pytest tests/test_weighted_osc.py -q
```

Expected: import failure because `rizon_osc.weighted_osc` does not exist.

**Step 3: Implement the controller**

Implement batched PyTorch operations:

```text
qdd = W^-1 J_m^T (J_m W^-1 J_m^T + lambda^2 I)^-1 a_task
tau = M qdd + gravity + J_n^T f_normal + tau_damping + tau_null
```

Requirements:

- construct the 5-D motion Jacobian in the moving task frame;
- include feed-forward task acceleration and PD pose/twist error;
- use high configurable weights on green joints 1–7 and low weights on joints 8–9;
- red joints use comparable weights;
- add a projected joint-limit/neutral secondary objective without corrupting the primary task;
- apply torque magnitude and torque-rate limits;
- return diagnostic residual, damping, task error, and split joint motion;
- do not call inverse kinematics or generate a joint trajectory.

**Step 4: Run focused tests**

Run:

```bash
python -m pytest tests/test_weighted_osc.py -q
```

Expected: all controller tests pass.

**Step 5: Commit**

```bash
git add src/rizon_osc/weighted_osc.py tests/test_weighted_osc.py
git commit -m "feat: implement weighted dynamic OSC"
```

## Task 6: Define fixed Isaac assets and preprocess local geometry

**Files:**

- Create: `src/rizon_osc/scene_assets.py`
- Create: `tests/test_scene_assets.py`
- Create: `tools/build_exact_rizon_wrist_asset.py`
- Create: `assets/rizon4s_exact_wrist_probe.usda`
- Create: `tools/prepare_local_assets.py`

**Step 1: Write failing static-configuration tests**

Without importing Isaac, parse declarative asset metadata and verify:

- patient, both beds, both pedestals, and both robot roots are marked fixed/static;
- collision is enabled only for the torso contact proxy and required supports;
- robot path resolves to the exact first-project Rizon wrapper;
- probe body and patient torso body names match contact-sensor filters;
- no generated asset points into `/tmp`;
- downloaded source paths are configurable.

**Step 2: Verify test failure**

Run:

```bash
python -m pytest tests/test_scene_assets.py -q
```

Expected: import failure because `rizon_osc.scene_assets` does not exist.

**Step 3: Add declarative asset specifications**

Keep pure dataclasses at module import time. Put Isaac-specific imports inside factory functions so unit tests do not require Kit. Beds and pedestals are collision-enabled static prims without dynamic rigid-body APIs. Patient and robot wrappers receive explicit fixed root joints.

**Step 4: Add reproducible local asset preparation**

Migrate the existing exact Rizon/wrist/probe composition source. Add a CLI that:

- validates the external base Rizon USD and Assembly3 STL/URDF;
- builds the robot wrapper;
- invokes the surface preprocessing tool;
- converts the patient visual and torso collision proxy to project-local generated USD;
- reports missing local prerequisites clearly;
- leaves Downloads untouched.

Do not commit downloaded CAD or generated binaries.

**Step 5: Run tests**

Run:

```bash
python -m pytest tests/test_scene_assets.py -q
```

Expected: all static-configuration tests pass.

**Step 6: Commit**

```bash
git add src/rizon_osc/scene_assets.py tests/test_scene_assets.py tools assets
git commit -m "feat: add fixed Isaac scene asset pipeline"
```

## Task 7: Integrate controllers, contact sensors, metrics, and markers in Isaac

**Files:**

- Create: `scripts/run_osc_comparison.py`
- Create: `src/rizon_osc/metrics.py`
- Create: `tests/test_metrics.py`
- Create: `launch_osc_comparison.sh`

**Step 1: Write failing metric/acceptance tests**

Cover:

- force pass only after settling and within `15 ± 1.5 N`;
- normal angle pass below 3 degrees;
- contact loss pass below 0.1 seconds;
- main-arm reduction is hidden when either robot fails the accuracy gate;
- green wrist motion is reported separately;
- static transform drift above tolerance fails;
- both robots must share identical task references.

**Step 2: Verify focused tests fail**

Run:

```bash
python -m pytest tests/test_metrics.py -q
```

Expected: import failure because `rizon_osc.metrics` does not exist.

**Step 3: Implement acceptance aggregation**

Provide per-side rolling metrics, per-phase travel counters, worst-case violations, a JSON-serializable report, and an overall pass/fail status.

**Step 4: Build the Isaac runner**

Refactor the current working comparison to:

- spawn the exact red/green Rizon wrapper;
- lock the red supplemental wrist and effort-control all nine green joints;
- spawn project-local Assembly3 visual/torso collision and explicit static supports;
- attach `ContactSensorCfg` to each `linear_probe`, filtered to its own torso body;
- transform the measured contact force into each robot base and project onto the current surface normal;
- feed measured force to independent 15 N PI loops;
- send identical task-space pose, twist, acceleration, and force references to both weighted OSC instances;
- command torques only, except the red wrist lock;
- record and assert initial transforms for patient, beds, pedestals, and robot roots;
- draw target/current frames, real normal, probe axis, magenta commanded-force arrows, contrasting cyan measured-force arrows, contact points, and scan trace;
- show HUD/terminal values for phase, controller mode, force command/measurement, normal angle, tangential error, contact loss, seven-arm travel, wrist travel, and accuracy-gated reduction;
- keep the GUI open when `--max_steps=0`;
- support `--headless --max_steps N --validation_report path.json` and return nonzero on failed required acceptance checks after the full scenario.

**Step 5: Run metric tests and import/compile checks**

Run:

```bash
python -m pytest tests/test_metrics.py -q
python -m compileall -q src scripts tools
bash -n launch_osc_comparison.sh
```

Expected: all commands succeed.

**Step 6: Commit**

```bash
git add src/rizon_osc/metrics.py tests/test_metrics.py scripts launch_osc_comparison.sh
git commit -m "feat: integrate measured contact weighted OSC demo"
```

## Task 8: Run local preprocessing and headless Isaac validation

**Files:**

- Generate, do not commit: `generated/assembly3_torso_surface.npz`
- Generate, do not commit: `generated/assembly3_patient.usd`
- Generate, do not commit: `generated/validation_report.json`
- Modify as failures demand: controller, scene, and runner files above

**Step 1: Run the complete fast test suite**

Run:

```bash
python -m pytest -q
```

Expected: all unit tests pass.

**Step 2: Generate local assets**

Run:

```bash
/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python tools/prepare_local_assets.py
```

Expected: surface map, patient USD/collision proxy, and exact robot wrapper exist under `generated/` or `assets/`, with no modification to Downloads.

**Step 3: Run a short headless smoke test**

Run:

```bash
./launch_osc_comparison.sh --headless --max_steps 250
```

Expected: scene initializes, contact-sensor filters resolve, commands remain finite, and the process exits cleanly.

**Step 4: Run the full headless scenario**

Run enough steps to include approach, ramp, 4-second scan, pitch-only, return-neutral, and yaw-only:

```bash
./launch_osc_comparison.sh \
  --headless \
  --max_steps 4500 \
  --validation_report generated/validation_report.json
```

Expected:

- measured settled force for both sides stays in `15 ± 1.5 N`;
- surface-normal angle stays below 3 degrees;
- no contact loss exceeds 0.1 seconds;
- static transform drift stays below tolerance;
- references are identical;
- green main-arm movement is lower during reorientation and the accuracy gate passes.

**Step 5: Diagnose failures systematically**

For every failure, preserve the failing report, identify whether the cause is geometry alignment, sensing sign/filtering, trajectory feed-forward, weighted solve, saturation, or physics contact tuning, add the smallest regression test possible, then change one cause at a time. Do not relax acceptance thresholds to obtain a pass.

**Step 6: Commit validated tuning**

```bash
git add src scripts tools tests
git commit -m "fix: validate 15N surface contact behavior"
```

## Task 9: Document operation, evidence, and limitations

**Files:**

- Modify: `README.md`
- Create: `PROJECT_MEMORY.md`
- Copy: `docs/superpowers/specs/2026-07-28-weighted-osc-contact-scan-design.md`
- Modify: this plan with execution notes if implementation differs

**Step 1: Update README**

Document:

- prerequisites and external local asset paths;
- exact setup, asset generation, unit-test, headless-validation, and GUI launch commands;
- how to open the demo and close it manually;
- why the implementation qualifies as operational-space control;
- the analytical surface path answer: no RRT/A*/OMPL;
- visual legend and acceptance meanings;
- command force versus measured force;
- generated assets excluded from Git.

**Step 2: Write durable project memory**

Include the professor's original goal, all user refinements, original shortcomings, implemented architecture, validation results, exact paths, branch/commit, unresolved items, and future extensions. Emphasize that 9-DoF advantage is accepted only under comparable task accuracy.

**Step 3: Run documentation consistency checks**

Run:

```bash
rg -n "open.loop|convex hull|commanded.*measured|RRT|15.?N|weighted" README.md PROJECT_MEMORY.md docs
git diff --check
```

Expected: no stale claim that commanded force is measured, no statement that the old whole-body convex hull remains, and no whitespace errors.

**Step 4: Commit**

```bash
git add README.md PROJECT_MEMORY.md docs
git commit -m "docs: explain weighted OSC contact comparison"
```

## Task 10: Final verification, self-review, GUI launch, and push

**Files:**

- Review: all changed files
- Generate, do not commit: `generated/final_validation_report.json`

**Step 1: Run final verification from a clean shell**

Run:

```bash
python -m pytest -q
python -m compileall -q src scripts tools
bash -n launch_osc_comparison.sh
./launch_osc_comparison.sh \
  --headless \
  --max_steps 4500 \
  --validation_report generated/final_validation_report.json
git diff --check
git status --short
```

Expected: tests pass, report passes all required gates, and only intended files are tracked.

**Step 2: Perform structured code review**

Because the current execution policy does not permit spawning a review subagent, perform the `superpowers:requesting-code-review` checklist as a documented self-review:

- compare the entire diff against the approved design;
- search for IK/joint-trajectory shortcuts;
- inspect force sign, frame transforms, sensor filters, saturation, and static fixation;
- verify no absolute developer-only paths are silently required without CLI overrides;
- verify no generated/downloaded binary or secret is staged;
- fix every critical/important issue before proceeding.

**Step 3: Launch the final GUI and leave it open**

Run:

```bash
./launch_osc_comparison.sh
```

Expected: Isaac Sim opens the two-arm comparison and remains running until the user closes it.

**Step 4: Push the feature branch**

Run:

```bash
git push -u origin feature/weighted-osc-contact
```

Expected: the branch is available at `universeleaf/Isaacsim-9DoF-flexiv`.

**Step 5: Report exact evidence**

Return:

- pushed branch and commit;
- unit-test count;
- headless report path and measured force/normal/contact/travel results;
- GUI process status;
- any acceptance item that remains unverified or failed, without overstating completion.
