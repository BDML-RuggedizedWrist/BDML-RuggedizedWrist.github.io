# Scheme A Run-Initial Wrist Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Scheme A so Isaac Lab's official operational-space controller receives an immutable run-initial J8/J9 posture baseline with exact signed pitch/yaw offsets, then admit the change only through deterministic watchdog-gated 2,300-step and 3,500-step GPU validation.

**Architecture:** `RedundancyPolicy` remains a pure provider of `nullspace_joint_pos_target`; it captures the run-initial wrist once, captures only J1-J7 at each phase boundary, and never computes torque or kinematics. The runner performs that one-time initialization after reset/sign verification and continues to send all pose/wrench/null-space state through `isaaclab.controllers.OperationalSpaceController`. A separate pure `ValidationWatchdog` owns continuous-window safety decisions and serializable evidence, while the runner only adapts Isaac tensors into watchdog samples.

**Tech Stack:** Python 3.11, NumPy, Isaac Sim 5.x, IsaacLab main checkout, PyTorch, PhysX articulation/contact sensors, pytest, Python `ast`, JSON.

## Global Constraints

- Work only in `/home/bdml-sim/Isaacsim-9DoF-flexiv-weighted-osc` on branch `feature/isaaclab-osc-contact`.
- Treat confirmed design commit `78376fa` as the immutable Scheme A specification; do not amend, recommit, or rewrite that specification.
- All task-space efforts must come from `isaaclab.controllers.OperationalSpaceController`; do not add a pseudoinverse, operational-space inertia implementation, Jacobian-transpose force term, null-space projector, weighted solver, direct green-wrist position drive, or alternative OSC solver.
- The green policy may only supply `nullspace_joint_pos_target` to `OperationalSpaceController.compute()`. It must not add to, replace, clamp, or post-process the official controller's torque except for the already-authored physical effort and effort-rate limits.
- Both robots receive identical pose, wrench, stiffness, and task-frame commands until a latched red collision stop.
- Keep `target_types=["pose_abs", "wrench_abs"]`, `impedance_mode="variable_kp"`, full inertial dynamics decoupling, no partial decoupling, built-in gravity compensation, task damping ratio `1.0`, and built-in null-space position control with stiffness `10.0`.
- Keep `motion_control_axes_task=[1, 1, 0, 1, 1, 1]` and `contact_wrench_control_axes_task=[0, 0, 1, 0, 0, 0]`.
- Keep all active effort-controlled joints at implicit stiffness `0` and damping `0`; only the red supplemental wrist retains the existing position lock with stiffness `45` and damping `7`.
- Keep the existing trajectory, all phase durations, 15 N force command, gains, physical limits, patient/surface assets, collision threshold `2 N`, challenge coordinate, challenge angle, and acceptance thresholds unchanged.
- The user's later near-to-far scan, fixed-point 90-degree acoustic-axis rotation, heart examination, and approximately 2.5-times speedup are explicitly deferred to a separate design and implementation plan. Do not change trajectory points, timings, phase sequence, angles, force references, or clinical behavior. Task 2's already-reviewed algebraic `split_task_frame_rotation()` helper and exact recomposition test are the only permitted `src/rizon_osc/trajectory.py` / `tests/test_trajectory.py` changes.
- Task 3 may commit only the already-reviewed direct-successor completion inference `PITCH_ONLY -> RETURN_PITCH` and `YAW_ONLY -> RETURN_YAW`. Do not change challenge completion, red post-collision-latch handling, numerical thresholds, phase coverage, or any other metric/report semantics even if validation exposes a failure.
- Preserve every pre-existing tracked and untracked worktree change. At the start and end of every task run `git status --short`; never use `git reset`, `git checkout --`, `git clean`, or `git stash`.
- The worktree begins with reviewed but uncommitted runtime prerequisites in `tools/`, `src/`, `scripts/`, and `tests/`. Tasks 1-3 must independently reverify and commit every such prerequisite before Scheme A implementation starts. The obsolete position-unbiased `RedundancyPolicy` hunk is not a prerequisite and must be replaced directly by Scheme A in Task 4.
- Never run a GPU validation against uncommitted runtime code. Before the 2,300-step run, `git diff --name-only -- scripts src tests tools`, `git diff --cached --name-only -- scripts src tests tools`, and `git ls-files --others --exclude-standard -- scripts src tests tools` must all produce no path.
- Stage only the named reviewed hunks with `git add -p`, inspect `git diff --cached`, and never use `git add -A`. README and PROJECT_MEMORY remain unstaged until passing-evidence documentation; ignored `generated/` reports and `.superpowers/` scratch may remain local.
- Do not close, interrupt, or send input to the existing Isaac Sim GUI process. Start a new persistent GUI only after the complete 3,500-step report has `overall_pass=true`; launch it without `--max_steps` and leave it open until the user closes it.
- A 2,300-step report is intentionally incomplete and may exit with status `2`; its gate is `validation_watchdog.passed=true`, no watchdog reasons, and a passed authored wrist-axis check. A 3,500-step report must exit `0` and contain `overall_pass=true`.
- Run the system-independent suite with `--ignore=tests/test_wrist_asset_contact_reporting.py`; run that generated-USD file separately inside a headless `SimulationApp` and require its two Kit tests to pass.

## Files and Interfaces

- Modify `src/rizon_osc/redundancy_policy.py`: lifecycle and exact Scheme A null-space target.
- Modify `tests/test_redundancy_policy.py`: pure lifecycle, immutability, signs, validation, and aliasing tests.
- Modify `tools/build_exact_rizon_wrist_asset.py` and commit `tests/test_wrist_asset_contact_reporting.py`: authored J9 sign and nested rigid-body contact reporting.
- Modify `src/rizon_osc/trajectory.py` and `tests/test_trajectory.py`: reviewed task-frame/relative-pose decomposition.
- Modify `src/rizon_osc/metrics.py` and `tests/test_metrics.py`: reviewed pitch/yaw successor-phase completion only.
- Modify `scripts/run_osc_comparison.py`: one `initialize_run()` call and pure-watchdog adaptation/reporting.
- Modify `tests/test_isaaclab_osc_contract.py`: link-frame velocity, active-drive, lifecycle, and watchdog AST/source contracts without importing Isaac Sim.
- Modify `tests/test_osc_profile.py`: explicit official null-space damping ratio `1.0` contract.
- Create `src/rizon_osc/validation_watchdog.py`: reusable, simulator-independent continuous-window safety monitor.
- Create `tests/test_validation_watchdog.py`: deterministic unit tests for every stop rule.
- Modify `README.md` and `PROJECT_MEMORY.md` only after a passing full report: exact measured evidence and reproduction commands.
- Produce untracked runtime evidence under `generated/scheme_a_short_2300.json` and `generated/scheme_a_full_3500.json`; do not commit generated local assets or reports.

The policy interface is fixed as
`RedundancyPolicy.initialize_run(joint_position: np.ndarray) -> None`,
`RedundancyPolicy.begin_phase(phase: str, joint_position: np.ndarray) -> None`,
and `RedundancyPolicy.target(current_joint_position: np.ndarray, *,
relative_pitch: float, relative_yaw: float) -> np.ndarray`.

The watchdog interface is fixed as immutable `WatchdogSample` and
`WatchdogSnapshot` dataclasses plus
`ValidationWatchdog.update(sample: WatchdogSample) -> WatchdogSnapshot`,
`ValidationWatchdog.snapshot() -> WatchdogSnapshot`, and
`WatchdogSnapshot.as_dict() -> dict[str, object]`.

---

### Task 1: Commit the Reviewed Wrist USD Physics Prerequisite

**Files:**
- Modify: `tools/build_exact_rizon_wrist_asset.py`
- Create: `tests/test_wrist_asset_contact_reporting.py`

**Interfaces:**
- Consumes: the exact Rizon/wrist/probe USD builder and its existing zero-pose geometry.
- Produces: a generated wrist whose positive J9 motion is positive task-frame yaw while preserving the zero pose, plus `PhysxContactReportAPI` with threshold `0.0` on `wrist_base`, `wrist_pitch_link`, `probe_roll_output`, and `linear_probe`.
- Review status: these existing hunks were already implemented and exercised during the earlier torque-chain diagnosis. This task re-runs their exact Kit test and obtains an independent staged-diff review before committing; it must not redesign wrist geometry.

- [ ] **Step 1: Isolate the exact existing asset diff**

Run:

```bash
cd /home/bdml-sim/Isaacsim-9DoF-flexiv-weighted-osc
git status --short
git diff -- tools/build_exact_rizon_wrist_asset.py
sed -n '1,260p' tests/test_wrist_asset_contact_reporting.py
```

Expected implementation content:

```python
ROLL_JOINT_ROT0 = (0.0, 0.9988782, 0.0, 0.04735377)
ROLL_JOINT_ROT1 = (0.0, 1.0, 0.0, 0.0)

for body_path in (WRIST_BASE, PITCH_LINK, TOOL_ROLL_LINK, PROBE):
    body = stage.GetPrimAtPath(body_path)
    body.AddAppliedSchema("PhysxContactReportAPI")
    body.CreateAttribute(
        "physxContactReport:threshold", Sdf.ValueTypeNames.Float
    ).Set(0.0)
```

The roll joint passes `ROLL_JOINT_ROT0` as local frame 0 and
`ROLL_JOINT_ROT1` as local frame 1. No probe transform, mass, collision
proxy, joint limit, or visible CAD placement changes in this task.

- [ ] **Step 2: Verify both generated-USD contracts inside Kit**

Run:

```bash
/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python -c '
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import pytest
code = pytest.main([
    "-q",
    "tests/test_wrist_asset_contact_reporting.py",
])
app.close()
raise SystemExit(code)
'
```

Expected: `2 passed`. The two tests must assert all four contact-report
schemas/thresholds and these exact joint-frame quaternion components:

```python
assert _quat_components(
    joint.GetAttribute("physics:localRot0").Get()
) == pytest.approx((0.0, 0.9988782, 0.0, 0.04735377))
assert _quat_components(
    joint.GetAttribute("physics:localRot1").Get()
) == pytest.approx((0.0, 1.0, 0.0, 0.0))
```

- [ ] **Step 3: Obtain independent staged-diff review**

Stage only these files:

```bash
git add tools/build_exact_rizon_wrist_asset.py \
  tests/test_wrist_asset_contact_reporting.py
git diff --cached --check
git diff --cached -- tools/build_exact_rizon_wrist_asset.py \
  tests/test_wrist_asset_contact_reporting.py
```

Give a fresh reviewer this exact read-only scope:

```text
Verify the staged USD builder preserves q=0 geometry, reverses only J9's
authored local +Z direction, authors PhysxContactReportAPI threshold 0 on
exactly the four wrist/probe rigid bodies used by sensors, and has generated
USD regression tests. Report critical/important findings with line evidence;
do not edit files.
```

Expected: no critical or important finding. If one exists, unstage only the
affected hunk with `git restore --staged` for the named file, return it to
the original implementer, rerun the Kit test, and repeat review.

- [ ] **Step 4: Commit the reproducible asset prerequisite**

Run:

```bash
git diff --cached --check
git commit -m "fix: author wrist yaw and contact reporting"
git status --short
```

Expected: the builder and its test are committed; README, PROJECT_MEMORY,
the Scheme A plan, and every other existing hunk remain untouched.

---

### Task 2: Commit Task-Frame and Link-State Consistency Prerequisites

**Files:**
- Modify: `src/rizon_osc/trajectory.py`
- Modify: `tests/test_trajectory.py`
- Modify: `scripts/run_osc_comparison.py`
- Modify: `tests/test_isaaclab_osc_contract.py`

**Interfaces:**
- Consumes: a trajectory reference quaternion/relative RPY and Isaac Lab's link Jacobian/link pose data.
- Produces: `split_task_frame_rotation(target_quaternion, relative_rpy) -> tuple[np.ndarray, np.ndarray]`; runner task tensors that exactly recompose the target rotation; `robot_state()` velocity from `body_link_vel_w` in the same link frame as pose and Jacobian.
- Excludes: actuator `0/0`, policy, metric, force, gain, asset, and trajectory-timing changes.

- [ ] **Step 1: Verify the existing task-frame helper and regression**

The exact retained helper is:

```python
def split_task_frame_rotation(
    target_quaternion: np.ndarray, relative_rpy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    target_rotation = rotation_matrix_from_quaternion(target_quaternion)
    relative_rotation = _rotation_from_rpy(
        *np.asarray(relative_rpy, dtype=np.float64)
    )
    return target_rotation @ relative_rotation.T, relative_rotation
```

The runner must import that helper and use:

```python
neutral_rotation, relative_rotation = split_task_frame_rotation(
    trajectory_reference.quaternion,
    trajectory_reference.relative_rpy,
)
```

The parameterized test at times `(0.0, 1.25, 4.0, 6.1, 7.8, 10.5)` must
assert:

```python
assert task_frame_rotation @ relative_rotation == pytest.approx(
    rotation_matrix_from_quaternion(reference.quaternion), abs=1e-7
)
```

- [ ] **Step 2: Verify the exact link-frame velocity correction and source contract**

The only velocity-source change in `robot_state()` is:

```python
relative_velocity_w = (
    robot.data.body_link_vel_w.torch[:, ee_body_idx]
    - robot.data.root_vel_w.torch
)
```

The retained contract is:

```python
def test_robot_state_velocity_uses_the_same_link_frame_as_pose_and_jacobian():
    source = RUNNER.read_text()
    robot_state = source.split("def robot_state(", maxsplit=1)[1]
    robot_state = robot_state.split(
        "def verify_wrist_axis_signs(", maxsplit=1
    )[0]

    assert "body_link_jacobian_w" in robot_state
    assert "body_link_vel_w" in robot_state
    assert "body_vel_w" not in robot_state
```

Do not stage the adjacent actuator-drive or position-unbiased policy
contracts in this task.

- [ ] **Step 3: Re-run focused tests and compile**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q \
  tests/test_trajectory.py \
  tests/test_isaaclab_osc_contract.py::test_robot_state_velocity_uses_the_same_link_frame_as_pose_and_jacobian
/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m py_compile scripts/run_osc_comparison.py src/rizon_osc/trajectory.py
```

Expected: all selected tests and compilation pass without launching Isaac
Sim.

- [ ] **Step 4: Stage only the frame/state hunks and obtain independent review**

Run:

```bash
git add src/rizon_osc/trajectory.py tests/test_trajectory.py
git add -p scripts/run_osc_comparison.py tests/test_isaaclab_osc_contract.py
git diff --cached --check
git diff --cached
```

Select only the task-frame import/use/helper removal, `body_link_vel_w`, and
the link-velocity source contract. Reject `import ast`, actuator, policy,
watchdog, and metric hunks; the structured actuator task introduces `ast`.

Reviewer scope:

```text
Verify the staged task-frame split recomposes every target rotation without
changing trajectory values/timing, and robot_state uses link velocity in the
same frame as its link pose/Jacobian. Verify no controller, gain, actuator,
policy, or metric behavior is staged. Report critical/important findings;
do not edit files.
```

Expected: no critical or important finding.

- [ ] **Step 5: Commit the reproducible kinematic-state prerequisite**

Run:

```bash
git commit -m "fix: align OSC task and link state frames"
git status --short
```

Expected: only the reviewed frame/state hunks and tests are committed.

---

### Task 3: Commit the Official Actuation and Metric-Cadence Prerequisites

**Files:**
- Modify: `scripts/run_osc_comparison.py`
- Modify: `tests/test_isaaclab_osc_contract.py`
- Modify: `tests/test_osc_profile.py`
- Modify: `src/rizon_osc/metrics.py`
- Modify: `tests/test_metrics.py`

**Interfaces:**
- Produces: active red J1-J7 and green J1-J9 implicit drives exactly `stiffness=0.0`, `damping=0.0`; red supplemental wrist lock exactly `45.0/7.0`; explicit official null-space damping-ratio test `1.0`; pitch/yaw completion inferred only from their direct successor phase.
- Excludes: challenge completion, red post-latch semantics, acceptance thresholds, policy behavior, trajectory behavior, and any direct wrist effort/position controller.

- [ ] **Step 1: Verify and retain the exact official actuation boundary**

The runner's actuator groups must resolve to:

```python
"shoulder_effort": {
    "joint_names_expr": ["joint[1-2]"],
    "stiffness": 0.0,
    "damping": 0.0,
}
"elbow_effort": {
    "joint_names_expr": ["joint[3-4]"],
    "stiffness": 0.0,
    "damping": 0.0,
}
"arm_wrist_effort": {
    "joint_names_expr": ["joint[5-7]"],
    "stiffness": 0.0,
    "damping": 0.0,
}
"supplemental_wrist": {
    "joint_names_expr": ["wrist_.*_joint"],
    "stiffness": 0.0 if wrist_active else 45.0,
    "damping": 0.0 if wrist_active else 7.0,
}
```

Retain the structured AST helper `_resolved_robot_actuator_drives()` and
`test_effort_controlled_osc_joints_disable_implicit_drives()`. The test must
evaluate both `wrist_active=True` and `False`; a text-only substring test is
not an acceptable substitute.

In `tests/test_osc_profile.py`, retain for both pose and hybrid profiles:

```python
assert cfg["nullspace_damping_ratio"] == 1.0
```

The implementation already sets that exact value in
`src/rizon_osc/osc_profile.py`; do not create a redundant implementation
hunk.

- [ ] **Step 2: Verify and retain only direct-successor phase completion**

The exact metrics change is:

```python
next_phase = {
    "PITCH_ONLY": "RETURN_PITCH",
    "YAW_ONLY": "RETURN_YAW",
}[phase]
completed = (
    last.completed_7 and last.completed_9
) or next_phase in self._seen_phases
```

Retain these three tests:

```text
test_pitch_completion_is_inferred_when_the_next_phase_was_observed
test_yaw_completion_is_inferred_when_the_next_phase_was_observed
test_unrelated_successor_does_not_mark_pitch_complete
```

Do not modify `collision_challenge`, challenge completion, force
post-latch handling, or any numerical gate.

- [ ] **Step 3: Run the exact prerequisite regression set**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q \
  tests/test_osc_profile.py \
  tests/test_metrics.py \
  tests/test_isaaclab_osc_contract.py
```

Expected: all selected tests pass, including the structured `0/0` and
`45/7` checks.

- [ ] **Step 4: Stage only these prerequisite hunks and obtain independent review**

Run:

```bash
git add src/rizon_osc/metrics.py tests/test_metrics.py \
  tests/test_osc_profile.py
git add -p scripts/run_osc_comparison.py tests/test_isaaclab_osc_contract.py
git diff --cached --check
git diff --cached
```

Select only active-drive `0/0`, `import ast`, and the AST resolver/test. Do
not stage the position-unbiased redundancy-policy source
contract; Task 4 replaces it.

Reviewer scope:

```text
Verify staged active OSC joints are 0/0, only the red inactive wrist is
45/7, nullspace_damping_ratio is explicitly tested as 1.0, and phase
completion is inferred only for PITCH_ONLY->RETURN_PITCH and
YAW_ONLY->RETURN_YAW. Reject challenge/red-latch/threshold/policy changes.
Report critical/important findings; do not edit files.
```

Expected: no critical or important finding.

- [ ] **Step 5: Commit and prove no reviewed prerequisite remains uncommitted**

Run:

```bash
git commit -m "fix: make official OSC acceptance reproducible"
git status --short
git diff --name-only -- scripts src tests tools
git ls-files --others --exclude-standard -- scripts src tests tools
```

Expected after this commit: the only runtime hunk still listed is the
obsolete position-unbiased policy/test/source-contract work that Task 4
will replace. No asset, frame, link-velocity, drive, profile-test, or
metrics prerequisite remains outside Git history.

---

### Task 4: Capture an Immutable Run-Initial Wrist Baseline

**Files:**
- Modify: `src/rizon_osc/redundancy_policy.py`
- Modify: `tests/test_redundancy_policy.py`
- Modify: `tests/test_isaaclab_osc_contract.py`

**Interfaces:**
- Consumes: one nine-element reset-synchronized position in `initialize_run()`, a nine-element phase-start position in `begin_phase()`, and finite signed task-frame `relative_pitch`/`relative_yaw` radians in `target()`.
- Produces: a new finite `(9,)` `float64` array with J1-J7 equal to the current phase's start, J8 equal to `q8_initial - relative_pitch`, and J9 equal to `q9_initial + relative_yaw`.
- Lifecycle: `initialize_run()` succeeds exactly once per policy instance and copies its input; every second call raises `RuntimeError`. `begin_phase()` and `target()` both raise `RuntimeError` before run initialization. `target()` also raises until a phase has begun. No method aliases a caller-owned array.

- [ ] **Step 1: Record and protect the dirty baseline**

Run:

```bash
cd /home/bdml-sim/Isaacsim-9DoF-flexiv-weighted-osc
git status --short
git diff -- src/rizon_osc/redundancy_policy.py tests/test_redundancy_policy.py \
  tests/test_isaaclab_osc_contract.py \
  > /tmp/scheme_a_task4_preexisting.patch
```

Expected: Tasks 1-3 have committed every other runtime prerequisite. The
remaining diff contains the obsolete position-unbiased policy/tests/source
contract that this task replaces, plus permitted README/PROJECT_MEMORY
documentation. The patch file is evidence only; do not apply, delete, or
stage it.

- [ ] **Step 2: Replace the position-unbiased tests with failing Scheme A lifecycle tests**

Add these exact behaviors to `tests/test_redundancy_policy.py`:

```python
def test_target_requires_explicit_run_initialization():
    policy = RedundancyPolicy()

    with pytest.raises(RuntimeError, match="initialize_run"):
        policy.target(
            np.zeros(9),
            relative_pitch=np.deg2rad(35.0),
            relative_yaw=0.0,
        )


def test_begin_phase_requires_explicit_run_initialization():
    policy = RedundancyPolicy()

    with pytest.raises(RuntimeError, match="initialize_run"):
        policy.begin_phase("PITCH_ONLY", np.zeros(9))


def test_run_initial_wrist_is_immutable_across_phase_changes():
    policy = RedundancyPolicy()
    run_initial = np.array(
        [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.31, -0.27]
    )
    policy.initialize_run(run_initial)
    run_initial[7:] = 9.0

    first_phase = np.linspace(-0.7, 0.7, 9)
    policy.begin_phase("PITCH_ONLY", first_phase)
    pitch_target = policy.target(
        first_phase + 0.2,
        relative_pitch=np.deg2rad(35.0),
        relative_yaw=0.0,
    )

    second_phase = np.linspace(0.6, -0.6, 9)
    policy.begin_phase("YAW_ONLY", second_phase)
    yaw_target = policy.target(
        second_phase - 0.2,
        relative_pitch=0.0,
        relative_yaw=np.deg2rad(45.0),
    )

    assert pitch_target[:7] == pytest.approx(first_phase[:7])
    assert pitch_target[7] == pytest.approx(0.31 - np.deg2rad(35.0))
    assert pitch_target[8] == pytest.approx(-0.27)
    assert yaw_target[:7] == pytest.approx(second_phase[:7])
    assert yaw_target[7] == pytest.approx(0.31)
    assert yaw_target[8] == pytest.approx(-0.27 + np.deg2rad(45.0))


def test_zero_relative_angles_return_wrist_to_run_initial_baseline():
    policy = RedundancyPolicy()
    policy.initialize_run(np.array([0, 0, 0, 0, 0, 0, 0, 0.22, -0.18]))
    phase_start = np.linspace(-0.4, 0.4, 9)
    policy.begin_phase("SURFACE_SCAN", phase_start)

    target = policy.target(
        np.linspace(1.0, 1.8, 9),
        relative_pitch=0.0,
        relative_yaw=0.0,
    )

    assert target[:7] == pytest.approx(phase_start[:7])
    assert target[7:] == pytest.approx((0.22, -0.18))


def test_initialize_run_rejects_every_second_capture():
    policy = RedundancyPolicy()
    policy.initialize_run(np.zeros(9))

    with pytest.raises(RuntimeError, match="already initialized"):
        policy.initialize_run(np.ones(9))


@pytest.mark.parametrize("method", ("initialize_run", "begin_phase", "target"))
def test_policy_rejects_nonfinite_joint_positions(method):
    policy = RedundancyPolicy()
    bad = np.zeros(9)
    bad[7] = np.nan
    if method != "initialize_run":
        policy.initialize_run(np.zeros(9))

    with pytest.raises(ValueError, match="finite"):
        if method == "initialize_run":
            policy.initialize_run(bad)
        elif method == "begin_phase":
            policy.begin_phase("PITCH_ONLY", bad)
        else:
            policy.begin_phase("PITCH_ONLY", np.zeros(9))
            policy.target(bad, relative_pitch=0.0, relative_yaw=0.0)


@pytest.mark.parametrize(
    "relative_pitch,relative_yaw",
    ((np.nan, 0.0), (0.0, np.inf), (-np.inf, 0.0)),
)
def test_target_rejects_nonfinite_relative_angles(relative_pitch, relative_yaw):
    policy = RedundancyPolicy()
    policy.initialize_run(np.zeros(9))
    policy.begin_phase("PITCH_ONLY", np.zeros(9))

    with pytest.raises(ValueError, match="finite"):
        policy.target(
            np.zeros(9),
            relative_pitch=relative_pitch,
            relative_yaw=relative_yaw,
        )
```

Retain and adapt the existing `(8,)`, `(10,)`, and `(1, 9)` shape tests so all three public methods require exactly `(9,)`.

Replace the obsolete `target = current.copy()` source contract in
`tests/test_isaaclab_osc_contract.py` with:

```python
def test_redundancy_policy_is_scheme_a_target_without_runtime_kinematics():
    source = REDUNDANCY_POLICY.read_text()

    assert "def initialize_run(" in source
    assert "self._run_initial_wrist" in source
    assert (
        "target[: self.num_arm_joints] = self._phase_start_arm"
        in source
    )
    assert (
        "self._run_initial_wrist[0] - pitch"
        in source
    )
    assert (
        "self._run_initial_wrist[1] + yaw"
        in source
    )
    assert "pinv(" not in source
    assert "lstsq(" not in source
    assert "jacobian" not in source.lower()
    assert "projector" not in source.lower()
```

- [ ] **Step 3: Run the focused test and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q \
  tests/test_redundancy_policy.py \
  tests/test_isaaclab_osc_contract.py::test_redundancy_policy_is_scheme_a_target_without_runtime_kinematics
```

Expected: failures report that `RedundancyPolicy` has no `initialize_run()`,
the old target uses the current wrist position, and the old source contract
does not contain Scheme A formulas.

- [ ] **Step 4: Implement the exact lifecycle and signed target**

Replace the policy implementation with:

```python
"""Null-space targets supplied to Isaac Lab OperationalSpaceController."""

from __future__ import annotations

import numpy as np


class RedundancyPolicy:
    """Provide Scheme A posture targets without computing joint efforts."""

    def __init__(
        self, num_arm_joints: int = 7, num_wrist_joints: int = 2
    ) -> None:
        if num_arm_joints < 1 or num_wrist_joints != 2:
            raise ValueError(
                "the Rizon comparison requires arm joints plus a 2-DoF wrist"
            )
        self.num_arm_joints = int(num_arm_joints)
        self.num_wrist_joints = int(num_wrist_joints)
        self._phase = ""
        self._run_initial_wrist: np.ndarray | None = None
        self._phase_start_arm: np.ndarray | None = None

    @property
    def _expected_joint_count(self) -> int:
        return self.num_arm_joints + self.num_wrist_joints

    def _position(self, joint_position: np.ndarray) -> np.ndarray:
        position = np.asarray(joint_position, dtype=np.float64)
        if position.shape != (self._expected_joint_count,):
            raise ValueError(
                f"expected {self._expected_joint_count} joint positions, "
                f"got {position.shape}"
            )
        if not np.isfinite(position).all():
            raise ValueError("joint positions must be finite")
        return position

    def initialize_run(self, joint_position: np.ndarray) -> None:
        if self._run_initial_wrist is not None:
            raise RuntimeError("run initial wrist baseline is already initialized")
        position = self._position(joint_position)
        self._run_initial_wrist = position[self.num_arm_joints :].copy()

    def begin_phase(self, phase: str, joint_position: np.ndarray) -> None:
        if self._run_initial_wrist is None:
            raise RuntimeError("initialize_run must be called before begin_phase")
        position = self._position(joint_position)
        self._phase = str(phase)
        self._phase_start_arm = position[: self.num_arm_joints].copy()

    def target(
        self,
        current_joint_position: np.ndarray,
        *,
        relative_pitch: float,
        relative_yaw: float,
    ) -> np.ndarray:
        current = self._position(current_joint_position)
        del current
        if self._run_initial_wrist is None:
            raise RuntimeError("initialize_run must be called before target")
        if self._phase_start_arm is None:
            raise RuntimeError("begin_phase must be called before target")
        pitch = float(relative_pitch)
        yaw = float(relative_yaw)
        if not np.isfinite([pitch, yaw]).all():
            raise ValueError("relative pitch and yaw must be finite")

        target = np.empty(self._expected_joint_count, dtype=np.float64)
        target[: self.num_arm_joints] = self._phase_start_arm
        target[self.num_arm_joints] = self._run_initial_wrist[0] - pitch
        target[self.num_arm_joints + 1] = self._run_initial_wrist[1] + yaw
        return target
```

The unused `current_joint_position` remains in the public interface so the runner and official-controller call site remain stable; it is validated but must never become a wrist baseline.

- [ ] **Step 5: Run focused and neighboring pure tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q \
  tests/test_redundancy_policy.py \
  tests/test_osc_profile.py \
  tests/test_isaaclab_osc_contract.py
```

Expected: all selected tests pass. No test may expect J8/J9 to equal the current position.

- [ ] **Step 6: Stage only Scheme A hunks, inspect, and commit**

Run:

```bash
git add -p src/rizon_osc/redundancy_policy.py \
  tests/test_redundancy_policy.py tests/test_isaaclab_osc_contract.py
git diff --cached --check
git diff --cached -- src/rizon_osc/redundancy_policy.py \
  tests/test_redundancy_policy.py tests/test_isaaclab_osc_contract.py
git commit -m "feat: anchor wrist preference to run initial state"
git status --short
```

Expected staged content: only the lifecycle, immutable baseline, signed
target, pure tests, and Scheme A source contract. After commit,
`git diff --name-only -- scripts src tests tools` produces no path.

---

### Task 5: Initialize Scheme A Exactly Once Before the Control Loop

**Files:**
- Modify: `scripts/run_osc_comparison.py`
- Modify: `tests/test_isaaclab_osc_contract.py`

**Interfaces:**
- Consumes: the refreshed `state_9[6][0]` captured after `verify_wrist_axis_signs()` has restored the articulation and after `robot_state()` has been called again.
- Produces: exactly one call to `policy_9.initialize_run(state_9[6][0].detach().cpu().numpy())` before `while simulation_app.is_running()`.
- Preserves: phase changes call only `policy_9.begin_phase(...)`; the official OSC receives `green_null_target` only through `nullspace_joint_pos_target=green_null_target`.

- [ ] **Step 1: Record the runner/test pre-task diff**

Run:

```bash
git status --short
git diff -- scripts/run_osc_comparison.py tests/test_isaaclab_osc_contract.py \
  > /tmp/scheme_a_task5_preexisting.patch
```

Expected: the saved patch is empty before this task. Every prerequisite and
Task 4 runtime hunk is already committed; only README/PROJECT_MEMORY and
local scratch/evidence may remain dirty.

- [ ] **Step 2: Add a failing AST lifecycle contract**

Append this test to `tests/test_isaaclab_osc_contract.py`:

```python
def test_runner_initializes_green_policy_once_before_control_loop():
    module = ast.parse(RUNNER.read_text())
    run_simulator = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_simulator"
    )
    initialize_calls = [
        node
        for node in ast.walk(run_simulator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "initialize_run"
    ]
    control_loop = next(
        node for node in ast.walk(run_simulator) if isinstance(node, ast.While)
    )

    assert len(initialize_calls) == 1
    initialize_call = initialize_calls[0]
    assert initialize_call.lineno < control_loop.lineno
    assert initialize_call not in set(ast.walk(control_loop))
    assert ast.unparse(initialize_call.func.value) == "policy_9"
    assert ast.unparse(initialize_call.args[0]) == (
        "state_9[6][0].detach().cpu().numpy()"
    )


def test_phase_changes_do_not_recapture_the_run_initial_wrist():
    source = RUNNER.read_text()
    phase_block = source.split(
        "if reference.phase.value != previous_phase:", maxsplit=1
    )[1].split("green_null_target_np =", maxsplit=1)[0]

    assert "policy_9.begin_phase(" in phase_block
    assert "initialize_run(" not in phase_block
```

- [ ] **Step 3: Run the lifecycle contract and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q \
  tests/test_isaaclab_osc_contract.py::test_runner_initializes_green_policy_once_before_control_loop \
  tests/test_isaaclab_osc_contract.py::test_phase_changes_do_not_recapture_the_run_initial_wrist
```

Expected: the first test fails with `len(initialize_calls) == 0`.

- [ ] **Step 4: Add the single reset-synchronized initialization**

Immediately after the post-sign-check refresh:

```python
state_7 = robot_state(robot_7, ee_7_idx, joints_7)
state_9 = robot_state(robot_9, ee_9_idx, joints_9)
```

and before `while simulation_app.is_running():`, create and initialize the policy exactly once:

```python
policy_9 = RedundancyPolicy()
policy_9.initialize_run(state_9[6][0].detach().cpu().numpy())
```

Remove the later duplicate `policy_9 = RedundancyPolicy()` construction. Keep the existing phase transition call exactly as a phase-start J1-J7 capture:

```python
policy_9.begin_phase(
    reference.phase.value, state_9[6][0].detach().cpu().numpy()
)
```

- [ ] **Step 5: Verify lifecycle and the complete official-controller boundary**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q \
  tests/test_redundancy_policy.py \
  tests/test_osc_profile.py \
  tests/test_isaaclab_osc_contract.py
```

Expected: all selected tests pass, including the existing AST assertions that active drives are `0/0`, the red locked wrist is `45/7`, the runner directly constructs `OperationalSpaceController`, and no local pseudoinverse/projector code exists.

- [ ] **Step 6: Compile without launching Isaac Sim**

Run:

```bash
/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m py_compile scripts/run_osc_comparison.py \
  src/rizon_osc/redundancy_policy.py
git diff --check
```

Expected: both commands exit `0`; this step must not create an Isaac Sim process.

- [ ] **Step 7: Stage only lifecycle integration hunks and commit**

Run:

```bash
git add -p scripts/run_osc_comparison.py tests/test_isaaclab_osc_contract.py
git diff --cached --check
git diff --cached -- scripts/run_osc_comparison.py tests/test_isaaclab_osc_contract.py
git commit -m "feat: initialize Scheme A wrist baseline once"
git status --short
```

Expected staged content: one runner initialization, removal of only the duplicate policy construction, and the two lifecycle contracts. The pre-existing active-drive/velocity/frame fixes remain intact.

---

### Task 6: Build the Pure Continuous-Window Validation Watchdog

**Files:**
- Create: `src/rizon_osc/validation_watchdog.py`
- Create: `tests/test_validation_watchdog.py`

**Interfaces:**
- Consumes: one `WatchdogSample` per physics step. Robot-axis arrays always use side order `(red_7dof, green_9dof)`; wrist arrays use `(J8, J9)`.
- Produces: a latched `WatchdogSnapshot` with stable reason strings and JSON-safe `as_dict()`.
- Applies exact stop rules: wrist-limit margin `0.02 rad` for at least `0.10 s`; wrist speed at or above `1.99 rad/s` for at least `0.10 s`; required-contact loss longer than `0.10 s`; filtered normal-force magnitude above `30 N`; green non-probe force at or above `2 N`, or either side at or above `2 N` before `CHALLENGE_TRANSIT`; a `0.25 s` task-progress window with the exact translation/rotation excitation and response thresholds; any non-finite input.
- The red freeze detector is disabled after intentional red `COLLISION_STOP`; all endpoint dwells remain safe because neither command channel reaches its excitation threshold.

- [ ] **Step 1: Create failing tests for limit, speed, contact, force, and collision**

Create `tests/test_validation_watchdog.py` with:

```python
from dataclasses import replace

import numpy as np
import pytest

from rizon_osc.validation_watchdog import (
    ValidationWatchdog,
    WatchdogSample,
)


def sample(**overrides) -> WatchdogSample:
    values = {
        "step": 1,
        "dt_s": 0.004,
        "phase": "SURFACE_SCAN",
        "wrist_position_rad": np.zeros(2),
        "wrist_velocity_rad_s": np.zeros(2),
        "wrist_limits_rad": np.array([[-1.57, 1.57], [-1.57, 1.57]]),
        "contact_required": True,
        "contact_present": np.ones(2, dtype=bool),
        "measured_normal_force_n": np.array([15.0, 15.0]),
        "nonprobe_force_n": np.zeros(2),
        "red_collision_stop": False,
        "target_position_m": np.zeros((2, 3)),
        "measured_position_m": np.zeros((2, 3)),
        "target_quaternion_wxyz": np.array(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        "measured_quaternion_wxyz": np.array(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        "finite_payloads": (),
    }
    values.update(overrides)
    return WatchdogSample(**values)


def advance(
    watchdog: ValidationWatchdog, current: WatchdogSample, count: int
):
    snapshot = watchdog.snapshot()
    for index in range(count):
        snapshot = watchdog.update(replace(current, step=index + 1))
    return snapshot


def test_near_limit_requires_a_continuous_tenth_second():
    watchdog = ValidationWatchdog()
    near = sample(wrist_position_rad=np.array([1.56, 0.0]))

    assert advance(watchdog, near, 24).passed
    stopped = watchdog.update(replace(near, step=25))

    assert not stopped.passed
    assert "green_wrist_limit_j8" in stopped.reasons


def test_wrist_speed_requires_a_continuous_tenth_second():
    watchdog = ValidationWatchdog()
    fast = sample(wrist_velocity_rad_s=np.array([0.0, -1.99]))

    assert advance(watchdog, fast, 24).passed
    stopped = watchdog.update(replace(fast, step=25))

    assert "green_wrist_speed_j9" in stopped.reasons


def test_wrist_timer_resets_when_condition_clears():
    watchdog = ValidationWatchdog()
    near = sample(wrist_position_rad=np.array([1.56, 0.0]))
    advance(watchdog, near, 20)
    watchdog.update(sample(step=21))

    assert advance(watchdog, replace(near, step=22), 20).passed


def test_required_contact_loss_must_exceed_point_one_seconds():
    watchdog = ValidationWatchdog()
    lost = sample(contact_present=np.array([True, False]))

    assert advance(watchdog, lost, 24).passed
    advance(watchdog, lost, 2)
    stopped = watchdog.snapshot()

    assert "probe_contact_loss_9" in stopped.reasons


def test_contact_loss_is_disabled_when_contact_is_not_required():
    watchdog = ValidationWatchdog()
    lost = sample(
        phase="APPROACH",
        contact_required=False,
        contact_present=np.array([False, False]),
    )

    assert advance(watchdog, lost, 40).passed


def test_force_above_thirty_newtons_stops_immediately():
    snapshot = ValidationWatchdog().update(
        sample(measured_normal_force_n=np.array([15.0, 30.01]))
    )

    assert "normal_force_overload_9" in snapshot.reasons


def test_red_collision_is_rejected_before_challenge():
    snapshot = ValidationWatchdog().update(
        sample(nonprobe_force_n=np.array([2.0, 0.0]))
    )

    assert "pre_challenge_nonprobe_collision_7" in snapshot.reasons


def test_red_collision_is_allowed_after_challenge_starts_but_green_is_not():
    red_watchdog = ValidationWatchdog()
    red = red_watchdog.update(
        sample(
            phase="CHALLENGE_TRANSIT",
            nonprobe_force_n=np.array([2.2, 0.0]),
            red_collision_stop=True,
        )
    )
    green = ValidationWatchdog().update(
        sample(
            phase="CHALLENGE_TRANSIT",
            nonprobe_force_n=np.array([0.0, 2.0]),
        )
    )

    assert red.passed
    assert "green_nonprobe_collision" in green.reasons
```

- [ ] **Step 2: Add failing task-freeze, non-finite, serialization, and shape tests**

Continue the same file:

```python
def z_rotation(angle_rad: float) -> np.ndarray:
    return np.array(
        [np.cos(0.5 * angle_rad), 0.0, 0.0, np.sin(0.5 * angle_rad)]
    )


def test_translation_freeze_uses_a_quarter_second_progress_window():
    watchdog = ValidationWatchdog()
    snapshot = watchdog.snapshot()
    for index in range(66):
        target = np.zeros((2, 3))
        target[1, 0] = 0.002 * index / 65.0
        snapshot = watchdog.update(
            sample(
                step=index,
                target_position_m=target,
                measured_position_m=np.zeros((2, 3)),
            )
        )

    assert "task_freeze_translation_9" in snapshot.reasons


def test_rotation_freeze_uses_geodesic_quaternion_progress():
    watchdog = ValidationWatchdog()
    snapshot = watchdog.snapshot()
    for index in range(66):
        target = sample().target_quaternion_wxyz.copy()
        target[0] = z_rotation(np.deg2rad(1.0) * index / 65.0)
        snapshot = watchdog.update(
            sample(step=index, target_quaternion_wxyz=target)
        )

    assert "task_freeze_rotation_7" in snapshot.reasons


def test_endpoint_dwell_does_not_trigger_freeze():
    dwell = sample(
        target_position_m=np.full((2, 3), 0.25),
        measured_position_m=np.zeros((2, 3)),
    )

    assert advance(ValidationWatchdog(), dwell, 100).passed


def test_red_freeze_detector_is_disabled_after_collision_stop():
    watchdog = ValidationWatchdog()
    snapshot = watchdog.snapshot()
    for index in range(66):
        target = np.zeros((2, 3))
        target[0, 0] = 0.002 * index / 65.0
        snapshot = watchdog.update(
            sample(
                step=index,
                phase="CHALLENGE_PITCH_ONLY",
                red_collision_stop=True,
                target_position_m=target,
            )
        )

    assert snapshot.passed


def test_any_nonfinite_payload_stops_and_latches():
    watchdog = ValidationWatchdog()
    first = watchdog.update(
        sample(finite_payloads=(np.array([1.0, np.nan]),))
    )
    second = watchdog.update(sample(step=2))

    assert first.reasons == ("nonfinite",)
    assert second.reasons == ("nonfinite",)
    assert second.first_failure_step == 1


def test_snapshot_serializes_stable_evidence():
    snapshot = ValidationWatchdog().update(
        sample(measured_normal_force_n=np.array([31.0, 15.0]))
    )
    report = snapshot.as_dict()

    assert report["passed"] is False
    assert report["stop_requested"] is True
    assert report["reasons"] == ["normal_force_overload_7"]
    assert report["first_failure_step"] == 1
    assert report["max_measured_normal_force_n"] == pytest.approx([31.0, 15.0])


@pytest.mark.parametrize(
    "field,value",
    (
        ("wrist_position_rad", np.zeros(3)),
        ("wrist_limits_rad", np.zeros((2, 3))),
        ("contact_present", np.zeros(3, dtype=bool)),
        ("target_position_m", np.zeros((3, 3))),
        ("target_quaternion_wxyz", np.zeros((2, 3))),
    ),
)
def test_watchdog_rejects_wrong_shapes(field, value):
    watchdog = ValidationWatchdog()

    with pytest.raises(ValueError, match=field):
        watchdog.update(sample(**{field: value}))
```

- [ ] **Step 3: Run the new test file and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/test_validation_watchdog.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'rizon_osc.validation_watchdog'`.

- [ ] **Step 4: Implement immutable input/output types and exact thresholds**

Create `src/rizon_osc/validation_watchdog.py` with these types and constants:

```python
"""Pure continuous-window watchdog for finite Isaac Lab validation runs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


CHALLENGE_PHASES = {
    "CHALLENGE_TRANSIT",
    "CHALLENGE_PITCH_ONLY",
    "RETURN_NEUTRAL",
}


@dataclass(frozen=True)
class WatchdogSample:
    step: int
    dt_s: float
    phase: str
    wrist_position_rad: np.ndarray
    wrist_velocity_rad_s: np.ndarray
    wrist_limits_rad: np.ndarray
    contact_required: bool
    contact_present: np.ndarray
    measured_normal_force_n: np.ndarray
    nonprobe_force_n: np.ndarray
    red_collision_stop: bool
    target_position_m: np.ndarray
    measured_position_m: np.ndarray
    target_quaternion_wxyz: np.ndarray
    measured_quaternion_wxyz: np.ndarray
    finite_payloads: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True)
class WatchdogSnapshot:
    passed: bool
    stop_requested: bool
    reasons: tuple[str, ...]
    first_failure_step: int | None
    max_measured_normal_force_n: tuple[float, float]
    max_nonprobe_force_n: tuple[float, float]
    max_near_limit_duration_s: tuple[float, float]
    max_overspeed_duration_s: tuple[float, float]
    max_contact_loss_duration_s: tuple[float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "stop_requested": self.stop_requested,
            "reasons": list(self.reasons),
            "first_failure_step": self.first_failure_step,
            "max_measured_normal_force_n": list(
                self.max_measured_normal_force_n
            ),
            "max_nonprobe_force_n": list(self.max_nonprobe_force_n),
            "max_near_limit_duration_s": list(
                self.max_near_limit_duration_s
            ),
            "max_overspeed_duration_s": list(
                self.max_overspeed_duration_s
            ),
            "max_contact_loss_duration_s": list(
                self.max_contact_loss_duration_s
            ),
        }


@dataclass(frozen=True)
class _PoseRecord:
    time_s: float
    target_position_m: np.ndarray
    measured_position_m: np.ndarray
    target_quaternion_wxyz: np.ndarray
    measured_quaternion_wxyz: np.ndarray


def _quaternion_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm <= 1.0e-12 or second_norm <= 1.0e-12:
        return math.inf
    dot = abs(float(np.dot(first / first_norm, second / second_norm)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))
```

- [ ] **Step 5: Implement validation, timers, collision rules, and rolling task progress**

Continue the module with the complete monitor:

```python
class ValidationWatchdog:
    LIMIT_MARGIN_RAD = 0.02
    LIMIT_DURATION_S = 0.10
    SPEED_THRESHOLD_RAD_S = 1.99
    SPEED_DURATION_S = 0.10
    CONTACT_LOSS_DURATION_S = 0.10
    FORCE_LIMIT_N = 30.0
    NONPROBE_COLLISION_N = 2.0
    FREEZE_WINDOW_S = 0.25
    TRANSLATION_COMMAND_M = 0.001
    TRANSLATION_RESPONSE_M = 0.0001
    ROTATION_COMMAND_RAD = math.radians(0.5)
    ROTATION_RESPONSE_RAD = math.radians(0.05)

    def __init__(self) -> None:
        self._time_s = 0.0
        self._reasons: list[str] = []
        self._first_failure_step: int | None = None
        self._challenge_started = False
        self._near_limit_s = np.zeros(2)
        self._overspeed_s = np.zeros(2)
        self._contact_loss_s = np.zeros(2)
        self._max_near_limit_s = np.zeros(2)
        self._max_overspeed_s = np.zeros(2)
        self._max_contact_loss_s = np.zeros(2)
        self._max_force_n = np.zeros(2)
        self._max_nonprobe_n = np.zeros(2)
        self._pose_history = [deque(), deque()]

    @staticmethod
    def _array(
        sample: WatchdogSample, field: str, shape: tuple[int, ...]
    ) -> np.ndarray:
        value = np.asarray(getattr(sample, field))
        if value.shape != shape:
            raise ValueError(f"{field} must have shape {shape}, got {value.shape}")
        return value

    def _fail(self, reason: str, step: int) -> None:
        if reason not in self._reasons:
            self._reasons.append(reason)
        if self._first_failure_step is None:
            self._first_failure_step = int(step)

    def snapshot(self) -> WatchdogSnapshot:
        passed = not self._reasons
        return WatchdogSnapshot(
            passed=passed,
            stop_requested=not passed,
            reasons=tuple(self._reasons),
            first_failure_step=self._first_failure_step,
            max_measured_normal_force_n=tuple(self._max_force_n.tolist()),
            max_nonprobe_force_n=tuple(self._max_nonprobe_n.tolist()),
            max_near_limit_duration_s=tuple(
                self._max_near_limit_s.tolist()
            ),
            max_overspeed_duration_s=tuple(
                self._max_overspeed_s.tolist()
            ),
            max_contact_loss_duration_s=tuple(
                self._max_contact_loss_s.tolist()
            ),
        )

    def update(self, sample: WatchdogSample) -> WatchdogSnapshot:
        wrist_position = self._array(sample, "wrist_position_rad", (2,))
        wrist_velocity = self._array(sample, "wrist_velocity_rad_s", (2,))
        wrist_limits = self._array(sample, "wrist_limits_rad", (2, 2))
        contact_present = self._array(sample, "contact_present", (2,))
        measured_force = self._array(
            sample, "measured_normal_force_n", (2,)
        )
        nonprobe_force = self._array(sample, "nonprobe_force_n", (2,))
        target_position = self._array(sample, "target_position_m", (2, 3))
        measured_position = self._array(
            sample, "measured_position_m", (2, 3)
        )
        target_quaternion = self._array(
            sample, "target_quaternion_wxyz", (2, 4)
        )
        measured_quaternion = self._array(
            sample, "measured_quaternion_wxyz", (2, 4)
        )
        dt = float(sample.dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")

        finite_arrays = (
            wrist_position,
            wrist_velocity,
            wrist_limits,
            measured_force,
            nonprobe_force,
            target_position,
            measured_position,
            target_quaternion,
            measured_quaternion,
            *(np.asarray(value) for value in sample.finite_payloads),
        )
        if not all(np.isfinite(value).all() for value in finite_arrays):
            self._fail("nonfinite", sample.step)
            return self.snapshot()

        self._time_s += dt
        self._challenge_started = (
            self._challenge_started or sample.phase in CHALLENGE_PHASES
        )
        self._max_force_n = np.maximum(
            self._max_force_n, np.abs(measured_force)
        )
        self._max_nonprobe_n = np.maximum(
            self._max_nonprobe_n, nonprobe_force
        )

        lower_distance = wrist_position - wrist_limits[:, 0]
        upper_distance = wrist_limits[:, 1] - wrist_position
        near_limit = (
            np.minimum(lower_distance, upper_distance)
            <= self.LIMIT_MARGIN_RAD
        )
        overspeed = np.abs(wrist_velocity) >= self.SPEED_THRESHOLD_RAD_S
        self._near_limit_s = np.where(
            near_limit, self._near_limit_s + dt, 0.0
        )
        self._overspeed_s = np.where(
            overspeed, self._overspeed_s + dt, 0.0
        )
        self._max_near_limit_s = np.maximum(
            self._max_near_limit_s, self._near_limit_s
        )
        self._max_overspeed_s = np.maximum(
            self._max_overspeed_s, self._overspeed_s
        )
        for joint_index, joint_name in enumerate(("j8", "j9")):
            if self._near_limit_s[joint_index] >= self.LIMIT_DURATION_S:
                self._fail(
                    f"green_wrist_limit_{joint_name}", sample.step
                )
            if self._overspeed_s[joint_index] >= self.SPEED_DURATION_S:
                self._fail(
                    f"green_wrist_speed_{joint_name}", sample.step
                )

        if sample.contact_required:
            self._contact_loss_s = np.where(
                contact_present.astype(bool),
                0.0,
                self._contact_loss_s + dt,
            )
        else:
            self._contact_loss_s.fill(0.0)
        self._max_contact_loss_s = np.maximum(
            self._max_contact_loss_s, self._contact_loss_s
        )
        for side_index, side_name in enumerate(("7", "9")):
            if (
                self._contact_loss_s[side_index]
                > self.CONTACT_LOSS_DURATION_S
            ):
                self._fail(
                    f"probe_contact_loss_{side_name}", sample.step
                )
            if abs(measured_force[side_index]) > self.FORCE_LIMIT_N:
                self._fail(
                    f"normal_force_overload_{side_name}", sample.step
                )

        if nonprobe_force[1] >= self.NONPROBE_COLLISION_N:
            self._fail("green_nonprobe_collision", sample.step)
        if not self._challenge_started:
            for side_index, side_name in enumerate(("7", "9")):
                if nonprobe_force[side_index] >= self.NONPROBE_COLLISION_N:
                    self._fail(
                        f"pre_challenge_nonprobe_collision_{side_name}",
                        sample.step,
                    )

        for side_index, side_name in enumerate(("7", "9")):
            history = self._pose_history[side_index]
            if side_index == 0 and sample.red_collision_stop:
                history.clear()
                continue
            history.append(
                _PoseRecord(
                    time_s=self._time_s,
                    target_position_m=target_position[side_index].copy(),
                    measured_position_m=measured_position[side_index].copy(),
                    target_quaternion_wxyz=target_quaternion[
                        side_index
                    ].copy(),
                    measured_quaternion_wxyz=measured_quaternion[
                        side_index
                    ].copy(),
                )
            )
            cutoff = self._time_s - self.FREEZE_WINDOW_S
            while len(history) >= 2 and history[1].time_s <= cutoff:
                history.popleft()
            oldest = history[0]
            if self._time_s - oldest.time_s <= self.FREEZE_WINDOW_S:
                continue
            target_translation = float(
                np.linalg.norm(
                    target_position[side_index]
                    - oldest.target_position_m
                )
            )
            measured_translation = float(
                np.linalg.norm(
                    measured_position[side_index]
                    - oldest.measured_position_m
                )
            )
            target_rotation = _quaternion_distance_rad(
                target_quaternion[side_index],
                oldest.target_quaternion_wxyz,
            )
            measured_rotation = _quaternion_distance_rad(
                measured_quaternion[side_index],
                oldest.measured_quaternion_wxyz,
            )
            if (
                target_translation >= self.TRANSLATION_COMMAND_M
                and measured_translation < self.TRANSLATION_RESPONSE_M
            ):
                self._fail(
                    f"task_freeze_translation_{side_name}", sample.step
                )
            if (
                target_rotation >= self.ROTATION_COMMAND_RAD
                and measured_rotation < self.ROTATION_RESPONSE_RAD
            ):
                self._fail(
                    f"task_freeze_rotation_{side_name}", sample.step
                )

        return self.snapshot()
```

- [ ] **Step 6: Run watchdog tests and the full pure suite**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/test_validation_watchdog.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q --ignore=tests/test_wrist_asset_contact_reporting.py
```

Expected: every watchdog test passes, followed by the complete system-independent suite passing.

- [ ] **Step 7: Commit the isolated pure module**

Run:

```bash
git add src/rizon_osc/validation_watchdog.py \
  tests/test_validation_watchdog.py
git diff --cached --check
git diff --cached -- src/rizon_osc/validation_watchdog.py \
  tests/test_validation_watchdog.py
git commit -m "feat: add Scheme A validation watchdog"
git status --short
```

Expected staged content: exactly two new files. No runner, metric, trajectory, documentation, or generated evidence file is staged.

---

### Task 7: Integrate Watchdog Evidence Without Changing OSC

**Files:**
- Modify: `scripts/run_osc_comparison.py`
- Modify: `tests/test_isaaclab_osc_contract.py`

**Interfaces:**
- Consumes: post-step J8/J9 position and velocity, authored joint limits, contact/filter/collision states, actual per-side task target, measured end-effector pose, and controller/state arrays.
- Produces: `report["validation_watchdog"]` with `enabled`, exact thresholds, and `WatchdogSnapshot.as_dict()` fields. A latched watchdog failure makes `report["overall_pass"]` false and ends a finite validation run.
- The watchdog is enabled only when `--validation_report` is supplied. A GUI run without a validation report retains the current infinite lifetime and is never auto-closed by watchdog logic.

- [ ] **Step 1: Add failing source/AST integration contracts**

Append:

```python
def test_runner_integrates_pure_validation_watchdog_after_physics_step():
    source = RUNNER.read_text()

    assert "from rizon_osc.validation_watchdog import (" in source
    assert "ValidationWatchdog" in source
    assert "WatchdogSample" in source
    assert "validation_watchdog.update(" in source
    assert source.rindex("post_state_9 = robot_state(") < source.index(
        "validation_watchdog.update("
    )
    assert 'report["validation_watchdog"] =' in source
    assert 'report["overall_pass"] = bool(' in source


def test_watchdog_does_not_change_official_osc_torque_boundary():
    module = ast.parse(RUNNER.read_text())
    run_simulator = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_simulator"
    )
    watchdog_calls = [
        node
        for node in ast.walk(run_simulator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
        and ast.unparse(node.func.value) == "validation_watchdog"
    ]

    assert len(watchdog_calls) == 1
    assert "torque_7 =" not in ast.unparse(watchdog_calls[0])
    assert "torque_9 =" not in ast.unparse(watchdog_calls[0])
    assert "set_joint_effort_target" not in ast.unparse(watchdog_calls[0])
```

- [ ] **Step 2: Run both contracts and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q \
  tests/test_isaaclab_osc_contract.py::test_runner_integrates_pure_validation_watchdog_after_physics_step \
  tests/test_isaaclab_osc_contract.py::test_watchdog_does_not_change_official_osc_torque_boundary
```

Expected: both tests fail because the runner has no watchdog import or update.

- [ ] **Step 3: Import and initialize the validation adapter**

Add:

```python
from rizon_osc.validation_watchdog import (
    ValidationWatchdog,
    WatchdogSample,
)
```

After joint discovery, capture the authored J8/J9 limits:

```python
wrist_limits_9 = (
    robot_9.data.joint_pos_limits.torch[0, joints_9[-2:], :]
    .detach()
    .cpu()
    .numpy()
)
```

Alongside the other pure monitors initialize:

```python
watchdog_enabled = args_cli.validation_report is not None
validation_watchdog = ValidationWatchdog()
watchdog_snapshot = validation_watchdog.snapshot()
latest_metric_values = np.empty(0, dtype=np.float64)
```

- [ ] **Step 4: Adapt post-step state into one pure sample per physics step**

At the end of the existing `0.1 s` metric block, after every displayed and
reported metric has been calculated, refresh:

```python
latest_metric_values = np.asarray(
    [
        tangent_error_7,
        tangent_error_9,
        normal_position_error_7,
        normal_position_error_9,
        normal_velocity_7,
        normal_velocity_9,
        orientation_error_7_deg,
        orientation_error_9_deg,
        angle_7,
        angle_9,
        drift,
        travel.arm_7_rad,
        travel.arm_9_rad,
        travel.wrist_9_rad,
    ],
    dtype=np.float64,
)
```

After that metric block and immediately before the existing `max_steps`
check, add one watchdog update per physics step:

```python
if watchdog_enabled:
    finite_payloads = tuple(
        tensor.detach().cpu().numpy()
        for tensor in (
            *state_7,
            *state_9,
            *post_state_7,
            *post_state_9,
            command_7,
            command_9,
            torque_7,
            torque_9,
            applied_7,
            applied_9,
        )
    )
    watchdog_snapshot = validation_watchdog.update(
        WatchdogSample(
            step=step_count,
            dt_s=dt,
            phase=reference.phase.value,
            wrist_position_rad=post_state_9[6][0, 7:]
            .detach()
            .cpu()
            .numpy(),
            wrist_velocity_rad_s=post_state_9[7][0, 7:]
            .detach()
            .cpu()
            .numpy(),
            wrist_limits_rad=wrist_limits_9,
            contact_required=phase_requires_contact(
                reference.phase.value
            ),
            contact_present=np.array(
                [filtered_7 > 0.5, filtered_9 > 0.5],
                dtype=bool,
            ),
            measured_normal_force_n=np.array(
                [filtered_7, filtered_9], dtype=np.float64
            ),
            nonprobe_force_n=np.array(
                [
                    collision_7.current_force_n,
                    collision_9.current_force_n,
                ],
                dtype=np.float64,
            ),
            red_collision_stop=collision_7.freeze_path,
            target_position_m=np.stack(
                (
                    actual_target_7_b[0, :3].detach().cpu().numpy(),
                    actual_target_9_b[0, :3].detach().cpu().numpy(),
                )
            ),
            measured_position_m=np.stack(
                (
                    post_state_7[3][0, :3].detach().cpu().numpy(),
                    post_state_9[3][0, :3].detach().cpu().numpy(),
                )
            ),
            target_quaternion_wxyz=np.stack(
                (
                    actual_target_7_b[0, 3:7].detach().cpu().numpy(),
                    actual_target_9_b[0, 3:7].detach().cpu().numpy(),
                )
            ),
            measured_quaternion_wxyz=np.stack(
                (
                    post_state_7[3][0, 3:7].detach().cpu().numpy(),
                    post_state_9[3][0, 3:7].detach().cpu().numpy(),
                )
            ),
            finite_payloads=finite_payloads
            + (
                np.asarray(
                    [
                        filtered_7,
                        filtered_9,
                        collision_7.current_force_n,
                        collision_9.current_force_n,
                    ]
                ),
                latest_metric_values,
            ),
        )
    )
    if watchdog_snapshot.stop_requested:
        print(
            "[WATCHDOG] validation stopped: "
            + ", ".join(watchdog_snapshot.reasons)
        )
        break
```

Do not move, wrap, replace, or modify either `OperationalSpaceController.compute()` call or either joint-effort application call.

- [ ] **Step 5: Add exact report fields and make failures disqualifying**

Immediately before returning the report:

```python
watchdog_report = {
    "enabled": watchdog_enabled,
    "thresholds": {
        "wrist_limit_margin_rad": ValidationWatchdog.LIMIT_MARGIN_RAD,
        "wrist_limit_duration_s": ValidationWatchdog.LIMIT_DURATION_S,
        "wrist_speed_rad_s": ValidationWatchdog.SPEED_THRESHOLD_RAD_S,
        "wrist_speed_duration_s": ValidationWatchdog.SPEED_DURATION_S,
        "contact_loss_duration_s": (
            ValidationWatchdog.CONTACT_LOSS_DURATION_S
        ),
        "normal_force_limit_n": ValidationWatchdog.FORCE_LIMIT_N,
        "nonprobe_collision_n": (
            ValidationWatchdog.NONPROBE_COLLISION_N
        ),
        "freeze_window_s": ValidationWatchdog.FREEZE_WINDOW_S,
        "translation_command_m": (
            ValidationWatchdog.TRANSLATION_COMMAND_M
        ),
        "translation_response_m": (
            ValidationWatchdog.TRANSLATION_RESPONSE_M
        ),
        "rotation_command_deg": 0.5,
        "rotation_response_deg": 0.05,
    },
    **watchdog_snapshot.as_dict(),
}
report["validation_watchdog"] = watchdog_report
report["overall_pass"] = bool(
    report.get("overall_pass", False)
    and (not watchdog_enabled or watchdog_snapshot.passed)
)
```

For a non-validation GUI run, `enabled` is false and the untouched acceptance result remains authoritative.

- [ ] **Step 6: Run focused tests, full pure tests, compile, and boundary scan**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q \
  tests/test_validation_watchdog.py \
  tests/test_redundancy_policy.py \
  tests/test_osc_profile.py \
  tests/test_isaaclab_osc_contract.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q --ignore=tests/test_wrist_asset_contact_reporting.py
/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m py_compile scripts/run_osc_comparison.py \
  src/rizon_osc/validation_watchdog.py
rg -n "pinv|pseudoinverse|projector|jacobian_b\\.mT @" \
  scripts/run_osc_comparison.py src/rizon_osc
git diff --check
```

Expected: all tests and compilation pass. The search returns no executable custom OSC/null-space solver; references in explanatory tests/docstrings may be reviewed manually.

- [ ] **Step 7: Stage only watchdog integration hunks and commit**

Run:

```bash
git add -p scripts/run_osc_comparison.py tests/test_isaaclab_osc_contract.py
git diff --cached --check
git diff --cached -- scripts/run_osc_comparison.py tests/test_isaaclab_osc_contract.py
git commit -m "feat: gate Scheme A validation with watchdog evidence"
git status --short
```

Expected staged content: import, state adaptation, one watchdog update, report fields, disqualifying gate, and two integration contracts. No gain, actuator, trajectory, metric-threshold, patient, or controller-configuration change is staged.

---

### Task 8: Run the 2,300-Step GPU Admission Gate

**Files:**
- Runtime output only: `generated/scheme_a_short_2300.json`
- No source, test, documentation, asset, or configuration file changes.

**Interfaces:**
- Consumes: the reviewed and committed prerequisite/Scheme A work from Tasks 1-7 and the existing local Isaac assets.
- Produces: a short-run JSON report whose watchdog and wrist-axis evidence alone determine whether Task 9 may start.

- [ ] **Step 1: Verify the code state before using the GPU**

Run:

```bash
cd /home/bdml-sim/Isaacsim-9DoF-flexiv-weighted-osc
git status --short
git log --oneline -6
git diff --name-only -- scripts src tests tools
git diff --cached --name-only -- scripts src tests tools
git ls-files --others --exclude-standard -- scripts src tests tools
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q --ignore=tests/test_wrist_asset_contact_reporting.py
git diff --check
git rev-parse HEAD | tee generated/scheme_a_short_2300.head
```

Expected: all three runtime path-listing commands print nothing, tests pass,
and the sidecar contains the exact tested HEAD. README, PROJECT_MEMORY,
ignored `generated/`, and `.superpowers/` may remain local; no modified,
staged, or untracked `scripts/`, `src/`, `tests/`, or `tools/` path is
allowed. If any runtime path is printed, stop before GPU and return it to
Tasks 1-7 for review and commit.

- [ ] **Step 2: Run exactly 2,300 headless physics steps**

Run from the project root:

```bash
set +e
./launch_osc_comparison.sh --viz none --max_steps 2300 \
  --validation_report generated/scheme_a_short_2300.json
short_status=$?
set -e
printf '%s\n' "$short_status"
```

Expected: the process finishes without touching the existing GUI. Status `2` is acceptable because 2,300 steps do not complete the full scenario; a simulator crash, missing report, traceback, or signal exit is not acceptable.

- [ ] **Step 3: Enforce the JSON admission gate**

Run:

```bash
/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python -c '
import json
from pathlib import Path
p = Path("generated/scheme_a_short_2300.json")
d = json.loads(p.read_text())
w = d["validation_watchdog"]
assert d["physics_steps"] == 2300, d["physics_steps"]
assert d["wrist_axis_check"]["passed"] is True, d["wrist_axis_check"]
assert w["enabled"] is True
assert w["passed"] is True, w
assert w["stop_requested"] is False, w
assert w["reasons"] == [], w
print(json.dumps({
    "physics_steps": d["physics_steps"],
    "wrist_axis_check": d["wrist_axis_check"],
    "validation_watchdog": w,
}, indent=2, sort_keys=True))
'
```

Expected: every assertion passes. `overall_pass=false` is expected for an incomplete short scenario and is not the short-run gate.

- [ ] **Step 4: Inspect exact Scheme A wrist behavior without relaxing a threshold**

From the report and terminal log confirm:

- J8 and J9 did not stay within `0.02 rad` of a limit for `0.10 s`.
- Neither wrist stayed at or above `1.99 rad/s` for `0.10 s`.
- Required contact loss never exceeded `0.10 s`.
- Filtered force never exceeded `30 N`.
- There was no green non-probe collision and no pre-challenge collision.
- No excited task channel met the freeze criterion.
- Every controller/state/metric payload remained finite.

If any assertion or item fails, stop here. Preserve the JSON and terminal output, report the exact reason and first failure step, and do not run 3,500 steps. Do not change trajectory, gains, limits, controller type, acceptance thresholds, or metric semantics to force admission. Any diagnostic edit must be reverted with a reviewed inverse patch or a dedicated `git revert` commit; never use a destructive reset.

- [ ] **Step 5: Record rollback evidence without committing runtime files**

Run:

```bash
git status --short
git diff --check
git rev-parse HEAD
cat generated/scheme_a_short_2300.head
sha256sum generated/scheme_a_short_2300.json
```

Expected: the runtime report is untracked/ignored and no source diff was created by validation. The printed commit and report digest uniquely identify the tested state and evidence.

---

### Task 9: Independent Compliance Review and Full 3,500-Step Acceptance

**Files:**
- Runtime output only: `generated/scheme_a_full_3500.json`
- Read-only review: confirmed spec, Scheme A commits, policy, watchdog, runner, and tests.

**Interfaces:**
- Consumes: a passing Task 8 watchdog gate.
- Produces: an independent review with no critical/important findings and a complete JSON report with `overall_pass=true`.

- [ ] **Step 1: Dispatch an independent spec-compliance reviewer**

The reviewer must not be a Task 4-7 implementer. Give the reviewer this exact scope:

```text
Review Scheme A against commit 78376fa and the implementation plan.
Verify: immutable one-time q8/q9 baseline; phase changes capture only J1-J7;
exact q8_initial-relative_pitch and q9_initial+relative_yaw signs; zero-angle
return to run baseline; explicit lifecycle/finite/shape failures; exactly one
runner initialization after restored robot_state and before the loop; official
OperationalSpaceController remains the only torque source; active joints remain
0/0 and red locked wrist 45/7; watchdog implements every exact consecutive
window and exception; no trajectory, gain, limit, patient, metric-semantic, or
acceptance-threshold change; dirty user work is preserved. Report findings by
severity with file and line evidence. Do not edit files.
```

Expected: no critical or important finding. If the reviewer finds one, return the issue to the original task implementer, add a failing test, fix only that issue, rerun the affected suite, recommit, and repeat independent review before continuing.

Any implementation/test/tool fix changes the tested HEAD and invalidates the
existing 2,300-step admission. After such a fix, return to Task 8, recreate
`scheme_a_short_2300.head`, rerun all 2,300 steps, and pass the JSON gate
again before continuing here.

- [ ] **Step 2: Rerun pre-GPU verification after review**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q --ignore=tests/test_wrist_asset_contact_reporting.py
/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m py_compile scripts/run_osc_comparison.py src/rizon_osc/*.py
git diff --check
git status --short
test "$(git rev-parse HEAD)" = "$(
  cat generated/scheme_a_short_2300.head
)"
git diff --quiet -- scripts src tests tools
git diff --cached --quiet -- scripts src tests tools
test -z "$(
  git ls-files --others --exclude-standard -- scripts src tests tools
)"
git rev-parse HEAD | tee generated/scheme_a_full_3500.head
```

Expected: tests and compilation pass; the current commit exactly matches the
short-run tested HEAD; runtime code/tests/tools have no tracked, staged, or
untracked difference; and the full-run sidecar records that same commit.

- [ ] **Step 3: Run the full headless acceptance only after Task 8 and review pass**

Run:

```bash
./launch_osc_comparison.sh --viz none --max_steps 3500 \
  --validation_report generated/scheme_a_full_3500.json
```

Expected: exit status `0`. The existing GUI remains untouched.

- [ ] **Step 4: Assert exact full-run gates from JSON**

Run:

```bash
/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python -c '
import json
from pathlib import Path
p = Path("generated/scheme_a_full_3500.json")
d = json.loads(p.read_text())
w = d["validation_watchdog"]
assert d["physics_steps"] == 3500, d["physics_steps"]
assert d["wrist_axis_check"]["passed"] is True
assert w["enabled"] is True and w["passed"] is True, w
assert w["reasons"] == [], w
assert d["scenario_complete"]["pass"] is True
assert d["equal_accuracy_comparison"]["pass"] is True
assert d["collision_challenge"]["pass"] is True
assert d["overall_pass"] is True
print(json.dumps(d, indent=2, sort_keys=True))
'
```

Expected: every assertion passes. The printed JSON is the sole source for documentation measurements.

If the command exits `2` or an assertion fails, do not launch a replacement GUI and do not write a PASS claim. Preserve the complete JSON, identify the exact failed keys, and stop Scheme A acceptance. In particular, do not alter successor-phase completion or red post-latch metric semantics under this plan; if those keys alone block acceptance, create a separate reviewed specification and plan.

- [ ] **Step 5: Preserve full-run evidence identity**

Run:

```bash
git rev-parse HEAD
cat generated/scheme_a_short_2300.head
cat generated/scheme_a_full_3500.head
sha256sum generated/scheme_a_short_2300.json \
  generated/scheme_a_full_3500.json
git status --short
```

Expected: the tested commit and both report digests are recorded for the documentation task; neither report is staged.

---

### Task 10: Publish Exact Passing Evidence and Start a Persistent GUI

**Files:**
- Modify: `README.md`
- Modify: `PROJECT_MEMORY.md`
- Runtime only: a new Isaac Sim GUI process without a finite step limit.

**Interfaces:**
- Consumes: only a Task 9 report that passed every JSON assertion.
- Produces: reproducible documentation containing exact report values and a visible GUI that remains open until the user closes it.

- [ ] **Step 1: Extract documentation values directly from the passing report**

Run:

```bash
/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python -c '
import json
from pathlib import Path
d = json.loads(Path("generated/scheme_a_full_3500.json").read_text())
assert d["overall_pass"] is True
keys = {
    "physics_steps": d["physics_steps"],
    "task_time_s": d["task_time_s"],
    "normal_force_target_n": d["normal_force_target_n"],
    "watchdog": d["validation_watchdog"],
    "equal_accuracy_comparison": d["equal_accuracy_comparison"],
    "collision_challenge": d["collision_challenge"],
    "force_7": d["force_7"],
    "force_9": d["force_9"],
    "normal_7": d["normal_7"],
    "normal_9": d["normal_9"],
    "orientation_7": d["orientation_7"],
    "orientation_9": d["orientation_9"],
    "contact_7": d["contact_7"],
    "contact_9": d["contact_9"],
}
print(json.dumps(keys, indent=2, sort_keys=True))
'
```

Expected: extraction aborts unless `overall_pass` is exactly true.

- [ ] **Step 2: Update README with the exact Scheme A lifecycle and measured evidence**

In `README.md`:

- State that J8/J9 use one immutable post-reset run baseline.
- Include the exact formulas `J8 = q8_initial - relative_pitch` and `J9 = q9_initial + relative_yaw`.
- State that J1-J7 use the phase-start target through Isaac Lab's official null-space position interface.
- State that zero-relative-angle phases return J8/J9 targets to their run-initial values.
- Replace the previous “latest validation failed” section only with the exact Task 10 Step 1 values, including both phase-local reductions, force/orientation/contact results, challenge outcome, watchdog maxima, tested commit, and report digest.
- Retain the exact 2,300-step and 3,500-step reproduction commands.
- Retain the statement that later 90-degree slice rotation, heart task, and speedup are separate future trajectory work.

Do not round a failed measurement into a pass and do not omit any report field whose gate is false.

- [ ] **Step 3: Update PROJECT_MEMORY with requirements, architecture, and raw evidence**

In `PROJECT_MEMORY.md`:

- Record confirmed spec commit `78376fa` and the Scheme A implementation commit list.
- Record the policy lifecycle and exact signed target equations.
- Record that `OperationalSpaceController.compute()` remains the only source of green active-joint efforts.
- Record the complete `validation_watchdog` object and the equal-accuracy/challenge report objects exactly as emitted.
- Record the tested commit hash and both SHA-256 report digests.
- Preserve the professor's broader objective and list the deferred near-to-far scan, 90-degree acoustic-axis rotation, heart examination, and speedup as unimplemented future work.
- Preserve any older failed evidence as historical evidence rather than rewriting it as success.

- [ ] **Step 4: Review and commit only documentation hunks**

Run:

```bash
git diff --check
git diff -- README.md PROJECT_MEMORY.md
git add -p README.md PROJECT_MEMORY.md
git diff --cached --check
git diff --cached -- README.md PROJECT_MEMORY.md
git commit -m "docs: record passing Scheme A validation"
git status --short
```

Expected staged content: the already-reviewed project requirement/history update
plus exact passing report evidence and Scheme A reproduction/lifecycle text.
After commit, tracked documentation must be clean. If a genuinely unrelated
user documentation change remains, stop and ask the user how to preserve it;
do not hide it, overwrite it, or proceed to push with a dirty tracked file.

- [ ] **Step 5: Start the replacement visual run without closing any existing GUI**

Run with GUI permission from the project root:

```bash
./launch_osc_comparison.sh
```

Expected: a new Isaac Sim window displays the red/green comparison. There is no `--max_steps`, `--validation_report`, `--headless`, or `--viz none`, so the window remains open until the user closes it. Do not send Ctrl-C, terminate the process, close the old GUI, or call any cleanup command after launch.

---

### Task 11: Final Verification, Commit Audit, and Push

**Files:**
- Read-only verification of the repository and generated reports.
- Remote branch: `origin/feature/isaaclab-osc-contact`.

**Interfaces:**
- Consumes: passing full report, independently reviewed commits, exact documentation, and the persistent GUI.
- Produces: a pushed branch at `https://github.com/universeleaf/Isaacsim-9DoF-flexiv.git` without disturbing local user changes or the GUI.

- [ ] **Step 1: Run final system-independent verification**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q --ignore=tests/test_wrist_asset_contact_reporting.py
/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python -c '
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import pytest
code = pytest.main(["-q", "tests/test_wrist_asset_contact_reporting.py"])
app.close()
raise SystemExit(code)
'
/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m py_compile scripts/run_osc_comparison.py src/rizon_osc/*.py
git diff --check
```

Expected: the system-independent suite passes, the two Kit/USD tests pass,
compilation exits `0`, and no whitespace error is reported.

- [ ] **Step 2: Audit the official-controller boundary and commit contents**

Run:

```bash
rg -n "OperationalSpaceController|initialize_run|nullspace_joint_pos_target|ValidationWatchdog" \
  scripts/run_osc_comparison.py src/rizon_osc tests
rg -n "torch\\.pinverse|torch\\.linalg\\.pinv|jacobian_b\\.mT @|null.?space projector" \
  scripts/run_osc_comparison.py src/rizon_osc
git log --oneline --decorate -12
git show --stat --oneline HEAD
git status --short
git diff --quiet
git diff --cached --quiet
test -z "$(
  git ls-files --others --exclude-standard -- scripts src tests tools
)"
```

Expected: the official controller calls, one Scheme A initialization,
official null-space target, and watchdog are visible; no executable custom
solver is found. Every tracked file and index entry is clean, and no
untracked runtime/source/test/tool file exists. Ignored `generated/` reports
and untracked `.superpowers/` scratch may remain, but no dirty tracked source
or documentation is acceptable before push.

- [ ] **Step 3: Reassert report identity immediately before push**

Run:

```bash
/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python -c '
import json
from pathlib import Path
for name in ("scheme_a_short_2300.json", "scheme_a_full_3500.json"):
    data = json.loads((Path("generated") / name).read_text())
    assert data["validation_watchdog"]["passed"] is True, name
full = json.loads(Path("generated/scheme_a_full_3500.json").read_text())
assert full["overall_pass"] is True
short_head = Path("generated/scheme_a_short_2300.head").read_text().strip()
full_head = Path("generated/scheme_a_full_3500.head").read_text().strip()
assert short_head == full_head
assert len(full_head) == 40
print("Scheme A reports verified")
'
tested_head="$(cat generated/scheme_a_full_3500.head)"
git merge-base --is-ancestor "$tested_head" HEAD
git diff --quiet "$tested_head" HEAD -- scripts src tests tools
```

Expected: `Scheme A reports verified`; short and full GPU runs used the same
40-character commit; that tested commit is an ancestor of documented HEAD;
and every post-validation commit changes documentation only, not runtime
code, tests, or tools.

- [ ] **Step 4: Push the reviewed branch**

Run:

```bash
git remote -v
git push -u origin feature/isaaclab-osc-contact
```

Expected: `origin` resolves to `https://github.com/universeleaf/Isaacsim-9DoF-flexiv.git` and the push succeeds. If authentication or network access fails, preserve all local commits and report the exact remote error; do not rewrite history or change credentials.

- [ ] **Step 5: Hand off exact outcome**

Report:

- pushed commit hash and branch;
- total pure-test result;
- 2,300-step watchdog status and report path;
- 3,500-step `overall_pass` and report path;
- exact pitch/yaw reductions and collision-challenge result;
- that all active efforts still come from Isaac Lab `OperationalSpaceController`;
- the persistent GUI state and that it was not auto-closed;
- ignored/untracked evidence or scratch paths, if `git status --short`
  still lists them; tracked files must be clean;
- the explicitly deferred trajectory/speed work that requires the next design.

Do not describe Scheme A as complete if Task 8, Task 9, independent review, documentation, or push remains unresolved.
