# Isaac Lab OSC Contact Scan Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a reproducible Isaac Sim comparison in which a red 7-DoF Flexiv Rizon 4s and a green 9-DoF Rizon-plus-wrist follow the same Assembly3 torso task using Isaac Lab's unmodified `OperationalSpaceController`, measured closed-loop 15 N normal-force regulation, fixed scene assets, and an accuracy-gated demonstration that the extra wrist reduces motion of the seven main-arm joints.

**Architecture:** Geometry, trajectory, measurement filtering, safety, redundancy targets, and metrics live in small pure-Python modules. The torque controller itself is not reimplemented: the runner constructs `isaaclab.controllers.OperationalSpaceController`, calls `set_command()`, passes measured force through `current_ee_force_b`, and applies the torques returned by `compute()`. The green system uses the controller's built-in `nullspace_control="position"` interface to hold the first seven joints near each reorientation phase's starting posture while targeting pitch/yaw with joints 8–9 whenever the task permits.

**Tech Stack:** Python 3.12, NumPy, PyTorch, pytest, trimesh, Isaac Sim 5.x, Isaac Lab 2.x, USD/PhysX, Git.

---

## Task 1: Testable package skeleton

**Files:**

- `pyproject.toml`
- `.gitignore`
- `src/rizon_osc/__init__.py`
- `tests/test_package.py`
- `tests/conftest.py`
- `README.md`

**Procedure:**

1. Write `tests/test_package.py` and confirm `ModuleNotFoundError`.
2. Add the `src/` package and pytest configuration.
3. Run:

   ```bash
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
     /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
     -m pytest tests/test_package.py -q
   ```

4. Commit `chore: scaffold testable Isaac Lab OSC package`.

## Task 2: Assembly3 surface map

**Files:**

- Create `src/rizon_osc/surface_model.py`
- Create `tests/test_surface_model.py`
- Create `tools/preprocess_assembly3_surface.py`

**Procedure:**

1. First write failing tests for bilinear height/normal interpolation, normalized normals, invalid cells, rigid transforms, shape validation, and source metadata.
2. Implement `SurfaceMap`, `SurfaceSample`, `.npz` load/save, and no-extrapolation queries.
3. Implement an STL preprocessing CLI that reads the default Assembly3 STL, extracts a configurable upper-torso ROI, ray-samples the upper surface, estimates smooth outward normals, saves source SHA-256 and settings, and never modifies Downloads.
4. Run:

   ```bash
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
     /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
     -m pytest tests/test_surface_model.py -q
   ```

5. Commit `feat: add Assembly3 torso surface map`.

## Task 3: Surface-constrained task trajectory

**Files:**

- Create `src/rizon_osc/trajectory.py`
- Create `tests/test_trajectory.py`

**Procedure:**

1. First write failing tests for quintic endpoint derivatives, a 0.16 m scan in 4 seconds, normal-aligned probe frame, continuous finite twist/acceleration, pitch-only/yaw-only references, and invalid-query hold.
2. Implement the phases `APPROACH`, `CONTACT_RAMP`, `SURFACE_SCAN`, `PITCH_ONLY`, `RETURN_NEUTRAL`, and `YAW_ONLY`.
3. Return pose plus derivative data for metrics and future controller versions. The current Isaac Lab OSC consumes the moving pose target directly and retains its own velocity-damping implementation; no custom feed-forward torque term is added.
4. Run the focused tests and commit `feat: add surface constrained ultrasound trajectory`.

## Task 4: Contact measurement filtering and safety

**Files:**

- Create `src/rizon_osc/force_control.py`
- Create `src/rizon_osc/state_machine.py`
- Create `tests/test_force_control.py`
- Create `tests/test_state_machine.py`

**Procedure:**

1. First write failing tests for force projection/sign, history filtering, reset, 0.1 s contact-loss threshold, invalid-surface hold, overload hold, and reacquisition.
2. Implement only measurement filtering and supervision. Do not implement a second force controller.
3. The filtered measured root-frame force is passed to Isaac Lab OSC through `current_ee_force_b`. Closed-loop force gain is configured with `contact_wrench_stiffness_task`; the target wrench remains 15 N.
4. Run focused tests and commit `feat: add contact measurement safety pipeline`.

## Task 5: Isaac Lab null-space redundancy policy

**Files:**

- Create `src/rizon_osc/redundancy_policy.py`
- Create `tests/test_redundancy_policy.py`

**Procedure:**

1. First write failing tests that the phase-start values of joints 1–7 remain the null-space target, pitch maps to wrist joint 8 with the imported wrist's sign, yaw maps to joint 9, and scan returns the distal wrist toward neutral.
2. Implement only null-space joint target selection.
3. Do not compute a Jacobian inverse, operational-space inertia, or torque in this module. Pass its output to Isaac Lab OSC as `nullspace_joint_pos_target`.
4. Run focused tests and commit `feat: add distal wrist nullspace policy`.

## Task 6: Fixed asset configuration and local preprocessing

**Files:**

- Create `src/rizon_osc/scene_assets.py`
- Create `tests/test_scene_assets.py`
- Migrate `tools/build_exact_rizon_wrist_asset.py`
- Migrate `assets/rizon4s_exact_wrist_probe.usda`
- Create `tools/prepare_local_assets.py`

**Procedure:**

1. Write failing pure tests for fixed patient/robot roots, static beds/pedestals, collision/contact names, configurable source paths, and no `/tmp` generated asset.
2. Keep declarative paths/specifications importable without Kit; put Isaac imports inside factory functions.
3. Preserve the exact first-project Rizon asset and the corrected 2-DoF wrist/probe chain.
4. Generate a project-local patient USD and torso contact proxy; downloaded CAD and generated binaries remain ignored.
5. Run focused tests and commit `feat: add fixed Isaac scene asset pipeline`.

## Task 7: Integrate Isaac Lab OperationalSpaceController

**Files:**

- Create `scripts/run_osc_comparison.py`
- Create `tests/test_isaaclab_osc_contract.py`
- Create `launch_osc_comparison.sh`

**Required controller contract:**

```python
from isaaclab.controllers import (
    OperationalSpaceController,
    OperationalSpaceControllerCfg,
)

cfg = OperationalSpaceControllerCfg(
    target_types=["pose_abs", "wrench_abs"],
    motion_control_axes_task=(1, 1, 0, 1, 1, 1),
    contact_wrench_control_axes_task=(0, 0, 1, 0, 0, 0),
    contact_wrench_stiffness_task=(0, 0, K_FORCE, 0, 0, 0),
    nullspace_control="position",
    ...
)
osc = OperationalSpaceController(cfg, num_envs=1, device=sim.device)
osc.set_command(...)
torque = osc.compute(
    jacobian_b=...,
    current_ee_pose_b=...,
    current_ee_vel_b=...,
    current_ee_force_b=measured_force_b,
    mass_matrix=...,
    gravity=...,
    current_joint_pos=...,
    current_joint_vel=...,
    nullspace_joint_pos_target=...,
)
```

**Procedure:**

1. First add a source-contract test that fails until the runner directly imports/constructs/calls Isaac Lab OSC and uses its force/null-space arguments. The test forbids `WeightedOSC`.
2. Migrate the current working two-robot scene.
3. Attach `ContactSensorCfg` to each `linear_probe`, filtered to the matching Assembly3 torso collider.
4. Convert/average sensor force exactly as the official Isaac Lab OSC tutorial does, then rotate it from world to the robot root. Verify sign against the 15 N task-frame command during headless tuning.
5. Send identical task pose/wrench commands to red and green OSC instances.
6. Red controls Rizon joints 1–7 and locks supplemental joints 8–9. Green controls all nine. Both use the built-in null-space position control; green receives the distal-wrist policy target.
7. Apply only magnitude/rate safety limits after `osc.compute()`. Do not add a custom torque objective.
8. Keep GUI open when `--max_steps=0`; support finite headless runs and JSON validation output.
9. Run contract test, compile check, and shell syntax check.
10. Commit `feat: integrate Isaac Lab closed loop OSC`.

## Task 8: Metrics, markers, and acceptance gates

**Files:**

- Create `src/rizon_osc/metrics.py`
- Create `tests/test_metrics.py`
- Modify `scripts/run_osc_comparison.py`

**Procedure:**

1. First write failing tests for `15 ± 1.5 N`, normal angle below 3 degrees, contact loss below 0.1 s, static transform tolerance, identical references, separate main/wrist travel, and hiding reduction when task accuracy is unequal.
2. Record task/measurement values after contact settling.
3. Draw target/current frames, real normal, probe axis, magenta commanded force, cyan measured force, contact point, and scan trace.
4. Report phase, built-in OSC mode, commanded/measured force, normal angle, tangential error, contact loss, main-arm travel, wrist travel, and accuracy-gated reduction.
5. Commit `feat: add accuracy gated OSC comparison metrics`.

## Task 9: Unit and Isaac validation

**Generated, not committed:**

- `generated/assembly3_torso_surface.npz`
- `generated/assembly3_patient.usd`
- `generated/validation_report.json`

**Procedure:**

1. Run all pure tests:

   ```bash
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
     /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python -m pytest -q
   ```

2. Prepare local assets:

   ```bash
   /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
     tools/prepare_local_assets.py
   ```

3. Run a 250-step headless smoke test.
4. Run a full headless scenario covering approach, ramp, 4-second scan, pitch, neutral, and yaw.
5. Diagnose failures one cause at a time under `superpowers:systematic-debugging`. Do not loosen acceptance gates to manufacture a pass.
6. Record actual results; never label the 15 N command as a measurement.

## Task 10: Documentation, review, GUI, and push

**Files:**

- Update `README.md`
- Create `PROJECT_MEMORY.md`
- Create `docs/superpowers/specs/2026-07-28-isaaclab-osc-contact-scan-design.md`

**Procedure:**

1. Document the professor's goal, every user refinement, exact local asset preparation, tests, headless validation, GUI launch, visual legend, measured results, and limitations.
2. Explicitly answer the path-planning question: the known clinical path is an analytical task-space curve, not RRT/A*/OMPL.
3. Explain that all task torques come from Isaac Lab `OperationalSpaceController`; the helper modules do not implement OSC.
4. Run:

   ```bash
   git diff --check
   rg -n "WeightedOSC|open.loop|commanded.*measured|convex hull" .
   ```

5. Perform the code-review checklist as a structured self-review because the current policy forbids spawning a review subagent.
6. Run final pure tests, compile checks, and full headless acceptance.
7. Launch `./launch_osc_comparison.sh` and leave the GUI open.
8. Push `feature/isaaclab-osc-contact` to `universeleaf/Isaacsim-9DoF-flexiv`.
9. Report exact commit, test count, report results, and any failed/unverified acceptance item.
