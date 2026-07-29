# Official IsaacLab OSC Wrist-Advantage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current coupled OSC configuration with the tutorial-derived IsaacLab controller flow and produce two measured side-by-side demonstrations in which the 9-DoF Rizon uses its distal wrist to reduce joints-1-through-7 motion by at least 50 percent and avoids a near-patient collision challenge.

**Architecture:** The simulation runner remains responsible for Isaac Sim/IsaacLab objects and calls only `isaaclab.controllers.OperationalSpaceController` for task-space torque. Pure Python modules define the tutorial-derived configuration, trajectory segments, redundancy targets, collision state, phase-local travel, acceptance reporting, and HUD text so they can be developed with fast unit tests before GPU validation.

**Tech Stack:** Python 3.11, Isaac Sim 5.x, IsaacLab main checkout, PyTorch, NumPy, PhysX articulation/contact sensors, pytest, USD.

## Global Constraints

- Work only in `/home/bdml-sim/Isaacsim-9DoF-flexiv-weighted-osc` on branch `feature/isaaclab-osc-contact`.
- All task-space efforts must come from `isaaclab.controllers.OperationalSpaceController`; do not add a pseudoinverse, operational-space inertia implementation, Jacobian-transpose force term, null-space projector, or alternative OSC solver.
- Both robots receive identical pose, wrench, stiffness, and task-frame commands until a latched red collision stop.
- The hybrid command is `pose_abs + wrench_abs + variable_kp`.
- Use full inertial dynamics decoupling, no partial decoupling, built-in gravity compensation, and built-in null-space position control.
- Keep the robot physical effort and effort-rate limits identical between the red and green Rizon joints 1 through 7.
- Keep the target patient-normal force at 15 N.
- The equal-accuracy pitch-only and yaw-only phases each require at least 50 percent less joints-1-through-7 travel for green.
- Probe-patient contact is intentional; non-probe robot-patient force above 2 N latches a collision.
- Do not close the existing GUI process while implementing. Start the replacement GUI only after headless validation and leave it open until the user closes it.

---

### Task 1: Encode the tutorial-derived official OSC profiles

**Files:**
- Create: `src/rizon_osc/osc_profile.py`
- Create: `tests/test_osc_profile.py`
- Modify: `scripts/run_osc_comparison.py:328-385`
- Modify: `tests/test_isaaclab_osc_contract.py`

**Interfaces:**
- Consumes: `force_gain: float` from the runner CLI.
- Produces: `pose_osc_kwargs() -> dict[str, object]`, `hybrid_osc_kwargs(force_gain: float) -> dict[str, object]`, and `variable_kp_command_parts(*, hybrid: bool) -> tuple[str, ...]`.

- [ ] **Step 1: Write failing profile tests**

```python
from rizon_osc.osc_profile import (
    hybrid_osc_kwargs,
    pose_osc_kwargs,
    variable_kp_command_parts,
)


def test_hybrid_profile_matches_isaaclab_run_osc_structure():
    cfg = hybrid_osc_kwargs(force_gain=0.1)

    assert cfg["target_types"] == ["pose_abs", "wrench_abs"]
    assert cfg["impedance_mode"] == "variable_kp"
    assert cfg["inertial_dynamics_decoupling"] is True
    assert cfg["partial_inertial_dynamics_decoupling"] is False
    assert cfg["gravity_compensation"] is True
    assert cfg["motion_damping_ratio_task"] == 1.0
    assert cfg["motion_control_axes_task"] == [1, 1, 0, 1, 1, 1]
    assert cfg["contact_wrench_control_axes_task"] == [0, 0, 1, 0, 0, 0]
    assert cfg["contact_wrench_stiffness_task"] == [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
    assert cfg["nullspace_control"] == "position"
    assert cfg["nullspace_stiffness"] == 10.0


def test_pose_profile_is_official_six_axis_variable_kp_osc():
    cfg = pose_osc_kwargs()

    assert cfg["target_types"] == ["pose_abs"]
    assert cfg["impedance_mode"] == "variable_kp"
    assert cfg["inertial_dynamics_decoupling"] is True
    assert cfg["partial_inertial_dynamics_decoupling"] is False
    assert cfg["motion_control_axes_task"] == [1, 1, 1, 1, 1, 1]
    assert cfg["nullspace_stiffness"] == 10.0


def test_variable_kp_command_layout_is_unambiguous():
    assert variable_kp_command_parts(hybrid=False) == ("pose", "kp")
    assert variable_kp_command_parts(hybrid=True) == ("pose", "wrench", "kp")
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/test_osc_profile.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'rizon_osc.osc_profile'`.

- [ ] **Step 3: Implement the pure profile module**

```python
"""Configuration values passed directly to IsaacLab's official OSC."""

from __future__ import annotations


def _common_kwargs() -> dict[str, object]:
    return {
        "impedance_mode": "variable_kp",
        "inertial_dynamics_decoupling": True,
        "partial_inertial_dynamics_decoupling": False,
        "gravity_compensation": True,
        "motion_damping_ratio_task": 1.0,
        "nullspace_control": "position",
        "nullspace_stiffness": 10.0,
        "nullspace_damping_ratio": 1.0,
    }


def pose_osc_kwargs() -> dict[str, object]:
    return {
        **_common_kwargs(),
        "target_types": ["pose_abs"],
        "motion_control_axes_task": [1, 1, 1, 1, 1, 1],
    }


def hybrid_osc_kwargs(force_gain: float) -> dict[str, object]:
    return {
        **_common_kwargs(),
        "target_types": ["pose_abs", "wrench_abs"],
        "motion_control_axes_task": [1, 1, 0, 1, 1, 1],
        "contact_wrench_control_axes_task": [0, 0, 1, 0, 0, 0],
        "contact_wrench_stiffness_task": [0.0, 0.0, float(force_gain), 0.0, 0.0, 0.0],
    }


def variable_kp_command_parts(*, hybrid: bool) -> tuple[str, ...]:
    return ("pose", "wrench", "kp") if hybrid else ("pose", "kp")
```

- [ ] **Step 4: Verify the profile tests are GREEN**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/test_osc_profile.py
```

Expected: 3 passed.

- [ ] **Step 5: Write and verify the failing runner source contract**

Add assertions to `tests/test_isaaclab_osc_contract.py`:

```python
def test_runner_uses_tutorial_profile_without_local_osc_math():
    source = RUNNER.read_text()

    assert "hybrid_osc_kwargs" in source
    assert "pose_osc_kwargs" in source
    assert "torch.cat((pose_task, wrench_task, pose_kp_task)" in source
    assert "torch.pinverse" not in source
    assert "torch.linalg.pinv" not in source
    assert "jacobian_b.mT @" not in source
    assert "nullspace_stiffness = 55.0" not in source
```

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q \
  tests/test_isaaclab_osc_contract.py::test_runner_uses_tutorial_profile_without_local_osc_math
```

Expected: FAIL because the runner has not imported the profile, has no
variable-KP command, and still contains the 55.0 gain.

- [ ] **Step 6: Make the runner instantiate the official configs directly**

Replace the hand-authored values in `make_pose_osc()` and
`make_hybrid_osc()` with:

```python
from rizon_osc.osc_profile import hybrid_osc_kwargs, pose_osc_kwargs


def make_pose_osc(device: str) -> OperationalSpaceController:
    cfg = OperationalSpaceControllerCfg(**pose_osc_kwargs())
    return OperationalSpaceController(cfg, num_envs=1, device=device)


def make_hybrid_osc(device: str) -> OperationalSpaceController:
    cfg = OperationalSpaceControllerCfg(
        **hybrid_osc_kwargs(force_gain=args_cli.force_gain)
    )
    return OperationalSpaceController(cfg, num_envs=1, device=device)
```

Create one pose and one hybrid official controller for each side. Remove
`reorientation=True`, the 55.0 green null-space gain, all fixed-impedance
controller variants, and the phase-dependent controller selection.

Append variable stiffness to every command:

```python
pose_kp_task = torch.tensor(
    [[360.0, 360.0, 360.0, 120.0, 120.0, 120.0]],
    dtype=torch.float32,
    device=sim.device,
)
pose_command = torch.cat((pose_task, pose_kp_task), dim=-1)
hybrid_command = torch.cat((pose_task, wrench_task, pose_kp_task), dim=-1)
```

Call `osc.reset()` whenever entering a new phase, then use the same
`set_command()` and `compute()` interface already present.

- [ ] **Step 7: Run controller tests and commit**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/test_osc_profile.py tests/test_isaaclab_osc_contract.py
```

Expected: all focused tests pass.

Commit:

```bash
git add src/rizon_osc/osc_profile.py tests/test_osc_profile.py \
  tests/test_isaaclab_osc_contract.py scripts/run_osc_comparison.py
git commit -m "refactor: match IsaacLab official OSC tutorial"
```

---

### Task 2: Make wrist preference and phase-local travel explicit

**Files:**
- Modify: `src/rizon_osc/redundancy_policy.py`
- Create: `src/rizon_osc/joint_travel.py`
- Modify: `tests/test_redundancy_policy.py`
- Create: `tests/test_joint_travel.py`

**Interfaces:**
- Consumes: phase name, current nine-joint position, requested task-frame pitch and yaw.
- Produces: `RedundancyPolicy.target(...) -> np.ndarray`, `JointTravelTracker.begin_phase(...)`, `JointTravelTracker.update(...)`, and `JointTravelTracker.snapshot() -> JointTravel`.

- [ ] **Step 1: Add failing wrist-target tests**

```python
def test_challenge_pitch_holds_main_arm_and_uses_negative_pitch_axis():
    policy = RedundancyPolicy()
    phase_start = np.linspace(-0.4, 0.4, 9)
    policy.begin_phase("CHALLENGE_PITCH_ONLY", phase_start)

    target = policy.target(
        phase_start + 0.05,
        relative_pitch=np.deg2rad(50.0),
        relative_yaw=0.0,
    )

    assert target[:7] == pytest.approx(phase_start[:7])
    assert target[7] == pytest.approx(-np.deg2rad(50.0))
    assert target[8] == pytest.approx(0.0)


def test_yaw_target_is_positive_and_pitch_axis_is_zero():
    policy = RedundancyPolicy()
    policy.begin_phase("YAW_ONLY", np.zeros(9))

    target = policy.target(
        np.zeros(9),
        relative_pitch=0.0,
        relative_yaw=np.deg2rad(45.0),
    )

    assert target[:7] == pytest.approx(np.zeros(7))
    assert target[7] == pytest.approx(0.0)
    assert target[8] == pytest.approx(np.deg2rad(45.0))
```

- [ ] **Step 2: Add a failing travel-tracker test**

```python
import numpy as np
import pytest

from rizon_osc.joint_travel import JointTravelTracker


def test_phase_travel_counts_only_absolute_incremental_motion():
    tracker = JointTravelTracker()
    tracker.begin_phase("PITCH_ONLY", np.zeros(7), np.zeros(9))
    tracker.update(
        np.array([0.1, 0, 0, 0, 0, 0, 0]),
        np.array([0.02, 0, 0, 0, 0, 0, 0, 0.2, 0]),
    )
    tracker.update(
        np.array([0.05, 0, 0, 0, 0, 0, 0]),
        np.array([0.01, 0, 0, 0, 0, 0, 0, 0.15, 0]),
    )

    result = tracker.snapshot()
    assert result.phase == "PITCH_ONLY"
    assert result.arm_7_rad == pytest.approx(0.15)
    assert result.arm_9_rad == pytest.approx(0.03)
    assert result.wrist_9_rad == pytest.approx(0.25)
    assert result.reduction_percent == pytest.approx(80.0)
```

- [ ] **Step 3: Run the tests and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/test_redundancy_policy.py tests/test_joint_travel.py
```

Expected: the challenge policy test fails because the phase is unsupported,
and test collection fails because `rizon_osc.joint_travel` does not exist.

- [ ] **Step 4: Extend the redundancy policy**

Use exact phase sets:

```python
PITCH_PHASES = {"PITCH_ONLY", "CHALLENGE_PITCH_ONLY"}
YAW_PHASES = {"YAW_ONLY"}

if self._phase in PITCH_PHASES:
    target[-2] = -float(relative_pitch)
    target[-1] = 0.0
elif self._phase in YAW_PHASES:
    target[-2] = 0.0
    target[-1] = float(relative_yaw)
else:
    target[-2:] = 0.0
```

Do not change the first seven entries after `begin_phase()`.

- [ ] **Step 5: Implement the pure phase-local tracker**

```python
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointTravel:
    phase: str
    arm_7_rad: float
    arm_9_rad: float
    wrist_9_rad: float

    @property
    def reduction_percent(self) -> float | None:
        if self.arm_7_rad <= 1.0e-9:
            return None
        return 100.0 * (1.0 - self.arm_9_rad / self.arm_7_rad)


class JointTravelTracker:
    def begin_phase(
        self, phase: str, joint_7: np.ndarray, joint_9: np.ndarray
    ) -> None:
        self._phase = str(phase)
        self._previous_7 = np.asarray(joint_7, dtype=float).copy()
        self._previous_9 = np.asarray(joint_9, dtype=float).copy()
        self._arm_7 = self._arm_9 = self._wrist_9 = 0.0

    def update(self, joint_7: np.ndarray, joint_9: np.ndarray) -> None:
        current_7 = np.asarray(joint_7, dtype=float)
        current_9 = np.asarray(joint_9, dtype=float)
        delta_7 = np.abs(current_7 - self._previous_7)
        delta_9 = np.abs(current_9 - self._previous_9)
        self._arm_7 += float(delta_7[:7].sum())
        self._arm_9 += float(delta_9[:7].sum())
        self._wrist_9 += float(delta_9[7:].sum())
        self._previous_7 = current_7.copy()
        self._previous_9 = current_9.copy()

    def snapshot(self) -> JointTravel:
        return JointTravel(
            self._phase, self._arm_7, self._arm_9, self._wrist_9
        )
```

Add shape validation for `(7,)` and `(9,)` arrays before storing or
updating them.

- [ ] **Step 6: Verify GREEN and commit**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/test_redundancy_policy.py tests/test_joint_travel.py
```

Expected: all focused tests pass.

Commit:

```bash
git add src/rizon_osc/redundancy_policy.py src/rizon_osc/joint_travel.py \
  tests/test_redundancy_policy.py tests/test_joint_travel.py
git commit -m "feat: track phase-local wrist advantage"
```

---

### Task 3: Add the equal-accuracy and shoulder challenge trajectories

**Files:**
- Modify: `src/rizon_osc/trajectory.py`
- Modify: `tests/test_trajectory.py`

**Interfaces:**
- Consumes: `SurfaceMap`, fixed scan endpoints, fixed challenge point `(0.10, 1.32)`, and time.
- Produces: `TaskReference` values for `PITCH_ONLY`, `YAW_ONLY`, `CHALLENGE_TRANSIT`, `CHALLENGE_PITCH_ONLY`, and final `RETURN_NEUTRAL`.

- [ ] **Step 1: Write failing sequence tests**

```python
def test_default_sequence_uses_35_pitch_45_yaw_and_50_challenge(curved_surface):
    trajectory = SurfaceTrajectory(
        curved_surface,
        scan_start_xy=(0.0, 0.02),
        scan_end_xy=(0.0, 0.18),
        challenge_xy=(0.10, 0.17),
        approach_duration=1.0,
        contact_ramp_duration=0.5,
        scan_duration=4.0,
        pitch_duration=1.2,
        neutral_duration=0.5,
        yaw_duration=1.2,
        challenge_transit_duration=2.0,
        challenge_pitch_duration=1.5,
        challenge_return_duration=0.6,
        pitch_angle=np.deg2rad(35.0),
        yaw_angle=np.deg2rad(45.0),
        challenge_pitch_angle=np.deg2rad(50.0),
    )

    pitch = trajectory.reference(1.0 + 0.5 + 4.0 + 1.2)
    yaw = trajectory.reference(1.0 + 0.5 + 4.0 + 1.2 + 0.5 + 1.2)
    challenge = trajectory.reference(
        1.0 + 0.5 + 4.0 + 1.2 + 0.5 + 1.2 + 0.5 + 2.0 + 1.5
    )

    assert pitch.relative_rpy[1] == pytest.approx(np.deg2rad(35.0))
    assert yaw.relative_rpy[2] == pytest.approx(np.deg2rad(45.0))
    assert challenge.phase is Phase.CHALLENGE_PITCH_ONLY
    assert challenge.contact_point[:2] == pytest.approx((0.10, 0.17))
    assert challenge.relative_rpy[1] == pytest.approx(np.deg2rad(50.0))
    assert pitch.normal_force == yaw.normal_force == challenge.normal_force == 15.0


def test_challenge_transit_stays_on_surface_and_changes_no_orientation(curved_surface):
    trajectory = make_full_trajectory(curved_surface)
    transit = trajectory.reference(trajectory.challenge_transit_mid_time)

    assert transit.phase is Phase.CHALLENGE_TRANSIT
    assert transit.relative_rpy == pytest.approx(np.zeros(3))
    sample = curved_surface.query(*transit.contact_point[:2])
    assert transit.contact_point == pytest.approx(sample.point)
    assert transit.probe_acoustic_axis == pytest.approx(-sample.normal)


def test_full_sequence_finishes_in_neutral_hold_at_challenge_point(curved_surface):
    trajectory = make_full_trajectory(curved_surface)
    final = trajectory.reference(100.0)

    assert final.phase is Phase.RETURN_NEUTRAL
    assert final.contact_point[:2] == pytest.approx(trajectory.challenge_xy)
    assert final.relative_rpy == pytest.approx(np.zeros(3))
```

In the test file, implement `make_full_trajectory()` with the exact
durations and angles shown in the first test so every assertion is
deterministic.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/test_trajectory.py
```

Expected: `SurfaceTrajectory.__init__()` rejects the new challenge
arguments and `Phase.CHALLENGE_PITCH_ONLY` is missing.

- [ ] **Step 3: Extend phases and constructor**

Add:

```python
class Phase(str, Enum):
    APPROACH = "APPROACH"
    CONTACT_RAMP = "CONTACT_RAMP"
    SURFACE_SCAN = "SURFACE_SCAN"
    PITCH_ONLY = "PITCH_ONLY"
    RETURN_PITCH = "RETURN_PITCH"
    YAW_ONLY = "YAW_ONLY"
    RETURN_YAW = "RETURN_YAW"
    CHALLENGE_TRANSIT = "CHALLENGE_TRANSIT"
    CHALLENGE_PITCH_ONLY = "CHALLENGE_PITCH_ONLY"
    RETURN_NEUTRAL = "RETURN_NEUTRAL"
```

Store these exact defaults:

```python
challenge_xy=(0.10, 1.32)
pitch_angle=math.radians(35.0)
yaw_angle=math.radians(45.0)
challenge_pitch_angle=math.radians(50.0)
challenge_transit_duration=2.0
challenge_pitch_duration=1.5
challenge_return_duration=0.6
```

- [ ] **Step 4: Implement the complete phase timeline**

Use `quintic_progress()` for every pose transition. Interpolate torso
surface `x/y` during `CHALLENGE_TRANSIT`, call `SurfaceMap.query()` at the
interpolated coordinates, and reconstruct the normal-aligned probe frame at
each sample. During `CHALLENGE_PITCH_ONLY`, hold the challenge contact point
fixed and set only `relative_rpy[1]`. During `RETURN_NEUTRAL`, hold the same
contact point and quintically return pitch to zero, then hold indefinitely.
Treat each rising reorientation endpoint as inclusive so a reference
queried exactly at the pitch, yaw, or challenge end time contains the full
35-, 45-, or 50-degree target before the following return phase begins.

Expose:

```python
@property
def challenge_transit_mid_time(self) -> float:
    return self._challenge_transit_start + 0.5 * self.challenge_transit_duration
```

Implement the timeline with explicit boundaries:

```python
def _phase_values(
    self, time_seconds: float
) -> tuple[Phase, float, np.ndarray, float]:
    approach_end = self.approach_duration
    ramp_end = approach_end + self.contact_ramp_duration
    scan_end = ramp_end + self.scan_duration
    pitch_end = scan_end + self.pitch_duration
    return_pitch_end = pitch_end + self.neutral_duration
    yaw_end = return_pitch_end + self.yaw_duration
    return_yaw_end = yaw_end + self.neutral_duration
    challenge_transit_end = (
        return_yaw_end + self.challenge_transit_duration
    )
    challenge_pitch_end = (
        challenge_transit_end + self.challenge_pitch_duration
    )
    challenge_return_end = (
        challenge_pitch_end + self.challenge_return_duration
    )

    if time_seconds < approach_end:
        return Phase.APPROACH, 0.0, np.zeros(3), 0.0
    if time_seconds < ramp_end:
        u = (time_seconds - approach_end) / max(
            self.contact_ramp_duration, 1.0e-12
        )
        force_progress, _, _ = quintic_progress(u)
        return (
            Phase.CONTACT_RAMP,
            u,
            np.zeros(3),
            self.target_force * force_progress,
        )
    if time_seconds <= scan_end:
        u = (time_seconds - ramp_end) / self.scan_duration
        return Phase.SURFACE_SCAN, u, np.zeros(3), self.target_force
    if self.pitch_duration > 0.0 and time_seconds <= pitch_end:
        u = (time_seconds - scan_end) / self.pitch_duration
        progress, _, _ = quintic_progress(u)
        return (
            Phase.PITCH_ONLY,
            u,
            np.array([0.0, self.pitch_angle * progress, 0.0]),
            self.target_force,
        )
    if self.neutral_duration > 0.0 and time_seconds < return_pitch_end:
        u = (time_seconds - pitch_end) / self.neutral_duration
        progress, _, _ = quintic_progress(u)
        return (
            Phase.RETURN_PITCH,
            u,
            np.array([0.0, self.pitch_angle * (1.0 - progress), 0.0]),
            self.target_force,
        )
    if self.yaw_duration > 0.0 and time_seconds <= yaw_end:
        u = (time_seconds - return_pitch_end) / self.yaw_duration
        progress, _, _ = quintic_progress(u)
        return (
            Phase.YAW_ONLY,
            u,
            np.array([0.0, 0.0, self.yaw_angle * progress]),
            self.target_force,
        )
    if self.neutral_duration > 0.0 and time_seconds < return_yaw_end:
        u = (time_seconds - yaw_end) / self.neutral_duration
        progress, _, _ = quintic_progress(u)
        return (
            Phase.RETURN_YAW,
            u,
            np.array([0.0, 0.0, self.yaw_angle * (1.0 - progress)]),
            self.target_force,
        )
    if (
        self.challenge_transit_duration > 0.0
        and time_seconds < challenge_transit_end
    ):
        u = (time_seconds - return_yaw_end) / self.challenge_transit_duration
        return (
            Phase.CHALLENGE_TRANSIT,
            u,
            np.zeros(3),
            self.target_force,
        )
    if (
        self.challenge_pitch_duration > 0.0
        and time_seconds <= challenge_pitch_end
    ):
        u = (
            time_seconds - challenge_transit_end
        ) / self.challenge_pitch_duration
        progress, _, _ = quintic_progress(u)
        return (
            Phase.CHALLENGE_PITCH_ONLY,
            u,
            np.array(
                [0.0, self.challenge_pitch_angle * progress, 0.0]
            ),
            self.target_force,
        )
    if (
        self.challenge_return_duration > 0.0
        and time_seconds < challenge_return_end
    ):
        u = (
            time_seconds - challenge_pitch_end
        ) / self.challenge_return_duration
        progress, _, _ = quintic_progress(u)
        return (
            Phase.RETURN_NEUTRAL,
            u,
            np.array(
                [
                    0.0,
                    self.challenge_pitch_angle * (1.0 - progress),
                    0.0,
                ]
            ),
            self.target_force,
        )
    return Phase.RETURN_NEUTRAL, 1.0, np.zeros(3), self.target_force
```

Update `_raw_reference()` point selection exactly as follows:

```python
if phase in (Phase.APPROACH, Phase.CONTACT_RAMP):
    xy = self.scan_start_xy
elif phase is Phase.SURFACE_SCAN:
    scan_progress, _, _ = quintic_progress(progress)
    xy = self.scan_start_xy + scan_progress * (
        self.scan_end_xy - self.scan_start_xy
    )
elif phase is Phase.CHALLENGE_TRANSIT:
    transit_progress, _, _ = quintic_progress(progress)
    xy = self.scan_end_xy + transit_progress * (
        self.challenge_xy - self.scan_end_xy
    )
elif phase in (Phase.CHALLENGE_PITCH_ONLY, Phase.RETURN_NEUTRAL):
    xy = self.challenge_xy
else:
    xy = self.scan_end_xy
```

Update `total_duration` to include both moderate returns, challenge transit,
challenge pitch, and challenge return exactly once.

- [ ] **Step 5: Verify all trajectory tests and commit**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/test_trajectory.py
```

Expected: all trajectory tests pass.

Commit:

```bash
git add src/rizon_osc/trajectory.py tests/test_trajectory.py
git commit -m "feat: add wrist advantage challenge trajectory"
```

---

### Task 4: Detect and latch non-probe patient collision

**Files:**
- Create: `src/rizon_osc/collision.py`
- Create: `tests/test_collision.py`
- Modify: `scripts/run_osc_comparison.py:250-325`
- Modify: `scripts/run_osc_comparison.py:644-980`

**Interfaces:**
- Consumes: patient-filtered non-probe contact-force magnitudes from manually instantiated IsaacLab `ContactSensor` objects.
- Produces: `CollisionMonitor.update(force_n: float) -> CollisionSnapshot`, `CollisionSnapshot.freeze_path`, and per-side sensor lists.

- [ ] **Step 1: Write failing collision-state tests**

```python
import pytest

from rizon_osc.collision import CollisionLevel, CollisionMonitor


def test_collision_levels_and_latch():
    monitor = CollisionMonitor(near_threshold_n=0.5, stop_threshold_n=2.0)

    assert monitor.update(0.2).level is CollisionLevel.CONTACT_OK
    assert monitor.update(0.8).level is CollisionLevel.NEAR_COLLISION
    stopped = monitor.update(2.1)
    assert stopped.level is CollisionLevel.COLLISION_STOP
    assert stopped.freeze_path
    assert stopped.peak_force_n == pytest.approx(2.1)

    still_stopped = monitor.update(0.0)
    assert still_stopped.level is CollisionLevel.COLLISION_STOP
    assert still_stopped.freeze_path


def test_collision_monitor_rejects_negative_force():
    monitor = CollisionMonitor()

    with pytest.raises(ValueError, match="nonnegative"):
        monitor.update(-0.1)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/test_collision.py
```

Expected: collection fails because `rizon_osc.collision` does not exist.

- [ ] **Step 3: Implement the pure collision latch**

```python
from dataclasses import dataclass
from enum import Enum


class CollisionLevel(str, Enum):
    CONTACT_OK = "CONTACT OK"
    NEAR_COLLISION = "NEAR COLLISION"
    COLLISION_STOP = "COLLISION STOP"


@dataclass(frozen=True)
class CollisionSnapshot:
    level: CollisionLevel
    current_force_n: float
    peak_force_n: float
    freeze_path: bool


class CollisionMonitor:
    def __init__(
        self, near_threshold_n: float = 0.5, stop_threshold_n: float = 2.0
    ) -> None:
        if not 0.0 <= near_threshold_n < stop_threshold_n:
            raise ValueError("collision thresholds must satisfy 0 <= near < stop")
        self.near_threshold_n = float(near_threshold_n)
        self.stop_threshold_n = float(stop_threshold_n)
        self._latched = False
        self._peak = 0.0

    def update(self, force_n: float) -> CollisionSnapshot:
        force = float(force_n)
        if force < 0.0:
            raise ValueError("collision force must be nonnegative")
        self._peak = max(self._peak, force)
        self._latched = self._latched or force >= self.stop_threshold_n
        if self._latched:
            level = CollisionLevel.COLLISION_STOP
        elif force >= self.near_threshold_n:
            level = CollisionLevel.NEAR_COLLISION
        else:
            level = CollisionLevel.CONTACT_OK
        return CollisionSnapshot(level, force, self._peak, self._latched)
```

- [ ] **Step 4: Verify the pure collision tests are GREEN**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/test_collision.py
```

Expected: 2 passed.

- [ ] **Step 5: Write and verify the failing collision-sensor source contract**

Append to `tests/test_isaaclab_osc_contract.py`:

```python
def test_runner_separates_probe_contact_from_patient_collision():
    source = RUNNER.read_text()

    assert "NON_PROBE_BODY_SUFFIXES" in source
    assert '"linear_probe"' not in source.split(
        "NON_PROBE_BODY_SUFFIXES =", maxsplit=1
    )[1].split(")", maxsplit=1)[0]
    assert "force_matrix_w_history" in source
    assert "maximum_patient_contact_force" in source
```

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q \
  tests/test_isaaclab_osc_contract.py::test_runner_separates_probe_contact_from_patient_collision
```

Expected: FAIL because the runner has no non-probe sensor list or
patient-filtered collision-force helper.

- [ ] **Step 6: Instantiate patient-filtered sensors for each non-probe body**

Add this exact body suffix list to the runner:

```python
NON_PROBE_BODY_SUFFIXES = (
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "flange",
    "flange/wrist_base",
    "flange/wrist_base/wrist_pitch_link",
    "flange/wrist_base/wrist_pitch_link/probe_roll_output",
)
```

Before `sim.reset()`, manually instantiate one IsaacLab `ContactSensor` per
suffix and side. Each sensor must match one rigid body per environment and
filter only its side's static patient surface:

```python
def make_patient_collision_sensors(side: str) -> list[ContactSensor]:
    robot_name = f"Robot{side}DoF"
    patient_name = f"Patient{side}DoF"
    patient_surface = (
        f"/World/envs/env_.*/{patient_name}/torso_collider/"
        "upper_torso_contact_surface"
    )
    return [
        ContactSensor(
            ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/{robot_name}/{suffix}",
                update_period=0.0,
                history_length=2,
                filter_prim_paths_expr=[patient_surface],
            )
        )
        for suffix in NON_PROBE_BODY_SUFFIXES
    ]
```

Use `"7"` and `"9"` as the side argument. Do not include `base_link` or
`linear_probe`. In `main()`, create both lists after `InteractiveScene(...)`
and before `sim.reset()`, then pass them explicitly to:

```python
run_simulator(
    sim,
    scene,
    surface,
    collision_sensors_7=collision_sensors_7,
    collision_sensors_9=collision_sensors_9,
)
```

- [ ] **Step 7: Read only patient-filtered contact histories**

Add:

```python
def maximum_patient_contact_force(
    sensors: list[ContactSensor], dt: float
) -> float:
    maximum = 0.0
    for sensor in sensors:
        sensor.update(dt, force_recompute=True)
        history = _tensor_data(sensor.data.force_matrix_w_history)
        if history is None:
            raise RuntimeError("patient-filtered force history is unavailable")
        maximum = max(
            maximum,
            float(torch.linalg.vector_norm(history, dim=-1).max().item()),
        )
    return maximum
```

Feed each maximum into an independent `CollisionMonitor`. Preserve the last
safe red pose/task frame. Once red latches, stop advancing its challenge
reference, set its force command to zero, and keep computing its holding
torque with the official pose OSC. Do not freeze the green reference.

- [ ] **Step 8: Run focused tests and commit**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/test_collision.py tests/test_isaaclab_osc_contract.py
```

Expected: all focused tests pass.

Commit:

```bash
git add src/rizon_osc/collision.py tests/test_collision.py \
  tests/test_isaaclab_osc_contract.py scripts/run_osc_comparison.py
git commit -m "feat: stop on non-probe patient collision"
```

---

### Task 5: Report phase-local accuracy, reduction, and challenge outcomes

**Files:**
- Modify: `src/rizon_osc/metrics.py`
- Modify: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `MetricSample` values containing phase-local travel, tracking accuracy, contact state, collision force, completion flags, and command identity.
- Produces: `AcceptanceMetrics.report()` with `equal_accuracy_comparison`, `collision_challenge`, and honest `overall_pass`.

- [ ] **Step 1: Extend the test sample factory**

Update the existing `passing_sample()` helper in `tests/test_metrics.py` so every
sample explicitly supplies:

```python
phase_arm_travel_7_rad=1.0,
phase_arm_travel_9_rad=0.4,
phase_wrist_travel_9_rad=0.6,
nonprobe_force_7_n=0.0,
nonprobe_force_9_n=0.0,
collision_stop_7=False,
collision_stop_9=False,
completed_7=True,
completed_9=True,
```

- [ ] **Step 2: Write failing equal-accuracy report tests**

```python
def test_equal_accuracy_requires_fifty_percent_reduction_in_both_axes():
    metrics = AcceptanceMetrics(settling_samples=0)
    metrics.add(passing_sample(phase="PITCH_ONLY", phase_arm_travel_7_rad=1.0,
                               phase_arm_travel_9_rad=0.49))
    metrics.add(passing_sample(phase="YAW_ONLY", phase_arm_travel_7_rad=0.8,
                               phase_arm_travel_9_rad=0.39))
    metrics.add(passing_sample(phase="CHALLENGE_PITCH_ONLY",
                               phase_arm_travel_7_rad=1.2,
                               phase_arm_travel_9_rad=0.4))

    report = metrics.report(scenario_complete=True)

    assert report["equal_accuracy_comparison"]["pitch"]["reduction_percent"] == pytest.approx(51.0)
    assert report["equal_accuracy_comparison"]["yaw"]["reduction_percent"] == pytest.approx(51.25)
    assert report["equal_accuracy_comparison"]["pass"]


def test_reduction_is_hidden_when_accuracy_is_unequal():
    metrics = AcceptanceMetrics(settling_samples=0)
    metrics.add(passing_sample(phase="PITCH_ONLY", orientation_error_9_deg=6.0))
    metrics.add(passing_sample(phase="YAW_ONLY"))
    metrics.add(passing_sample(phase="CHALLENGE_PITCH_ONLY"))

    report = metrics.report(scenario_complete=True)

    assert not report["equal_accuracy_comparison"]["pitch"]["accuracy_gate"]
    assert report["equal_accuracy_comparison"]["pitch"]["reduction_percent"] is None
    assert not report["equal_accuracy_comparison"]["pass"]
```

- [ ] **Step 3: Write failing collision-challenge tests**

```python
def test_challenge_passes_when_green_completes_and_red_collision_stops():
    metrics = AcceptanceMetrics(settling_samples=0)
    metrics.add(passing_sample(phase="PITCH_ONLY"))
    metrics.add(passing_sample(phase="YAW_ONLY"))
    metrics.add(passing_sample(
        phase="CHALLENGE_PITCH_ONLY",
        nonprobe_force_7_n=2.4,
        collision_stop_7=True,
        completed_7=False,
        completed_9=True,
        phase_arm_travel_7_rad=0.3,
        phase_arm_travel_9_rad=0.2,
    ))

    report = metrics.report(scenario_complete=True)

    assert report["collision_challenge"]["red_collision_stop"]
    assert report["collision_challenge"]["green_completed"]
    assert report["collision_challenge"]["pass"]


def test_challenge_fails_if_green_collides():
    metrics = AcceptanceMetrics(settling_samples=0)
    metrics.add(passing_sample(phase="PITCH_ONLY"))
    metrics.add(passing_sample(phase="YAW_ONLY"))
    metrics.add(passing_sample(
        phase="CHALLENGE_PITCH_ONLY",
        nonprobe_force_9_n=2.1,
        collision_stop_9=True,
    ))

    report = metrics.report(scenario_complete=True)

    assert not report["collision_challenge"]["green_collision_free"]
    assert not report["collision_challenge"]["pass"]
    assert not report["overall_pass"]
```

- [ ] **Step 4: Run the metrics tests and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/test_metrics.py
```

Expected: `MetricSample` rejects the new fields and the new report sections
are absent.

- [ ] **Step 5: Extend `MetricSample` and group post-settling samples**

Add the ten fields from Step 1 to `MetricSample`. In `report()`, group
samples by phase and calculate the following for each phase:

```python
def phase_accuracy(samples: list[MetricSample]) -> bool:
    return (
        max(abs(s.measured_force_7 - self.force_target) for s in samples)
        <= self.force_tolerance
        and max(abs(s.measured_force_9 - self.force_target) for s in samples)
        <= self.force_tolerance
        and max(s.orientation_error_7_deg for s in samples)
        <= self.orientation_error_limit_deg
        and max(s.orientation_error_9_deg for s in samples)
        <= self.orientation_error_limit_deg
        and max(s.tangent_error_7_m for s in samples)
        <= self.tangent_error_limit_m
        and max(s.tangent_error_9_m for s in samples)
        <= self.tangent_error_limit_m
        and max(s.contact_loss_7_s for s in samples)
        <= self.contact_loss_limit_s
        and max(s.contact_loss_9_s for s in samples)
        <= self.contact_loss_limit_s
        and max(s.nonprobe_force_7_n for s in samples) < 2.0
        and max(s.nonprobe_force_9_n for s in samples) < 2.0
    )
```

Use the last sample in each phase for travel. Publish a reduction only when
`phase_accuracy()` is true and red travel is nonzero.

- [ ] **Step 6: Build the two required report sections**

`equal_accuracy_comparison` contains `pitch`, `yaw`, and `pass`. Each axis
passes only when its reduction is at least 50.0 percent, both sides
completed, commands were identical, and its accuracy gate is true.

`collision_challenge` contains:

```python
{
    "green_completed": bool,
    "green_collision_free": bool,
    "red_collision_stop": bool,
    "red_to_green_arm_travel_ratio": float | None,
    "green_accuracy_gate": bool,
    "pass": bool,
}
```

It passes when green completed, green is collision-free, green meets force,
pose, tangent, and contact tolerances, and red either collision-stopped or
its travel ratio is at least 2.0.

Keep the existing static drift, phase coverage, exact force-command, and
scenario-complete gates. Evaluate both sides' exact 15 N command before a
red collision latch; after the latch, exclude red holding samples from the
red force-command gate and continue requiring the exact 15 N green command.
Likewise, evaluate `references_identical` over samples with
`collision_stop_7=False`; post-latch safe holding samples are recorded as
different but do not retroactively invalidate the pre-collision comparison.
Set `overall_pass` only when both new sections and all retained safety gates
pass.

- [ ] **Step 7: Verify metrics tests and commit**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/test_metrics.py
```

Expected: all metrics tests pass.

Commit:

```bash
git add src/rizon_osc/metrics.py tests/test_metrics.py
git commit -m "feat: validate phase-local wrist advantage"
```

---

### Task 6: Integrate the controller, collision stop, metrics, and GUI HUD

**Files:**
- Create: `src/rizon_osc/hud.py`
- Create: `tests/test_hud.py`
- Modify: `scripts/run_osc_comparison.py:473-509`
- Modify: `scripts/run_osc_comparison.py:601-1183`

**Interfaces:**
- Consumes: official OSC profiles, `SurfaceTrajectory`, `RedundancyPolicy`, `JointTravelTracker`, `CollisionMonitor`, and `AcceptanceMetrics`.
- Produces: a complete headless report and a GUI overlay formatted by `format_hud(snapshot: HudSnapshot) -> str`.

- [ ] **Step 1: Write a failing HUD formatter test**

```python
from rizon_osc.hud import HudSnapshot, format_hud


def test_hud_shows_force_phase_travel_reduction_and_collision():
    text = format_hud(HudSnapshot(
        phase="PITCH_ONLY",
        force_7_n=14.8,
        force_9_n=15.1,
        arm_7_rad=1.0,
        arm_9_rad=0.4,
        wrist_9_rad=0.6,
        reduction_percent=60.0,
        collision_7="NEAR COLLISION",
        collision_9="CONTACT OK",
    ))

    assert "PITCH_ONLY" in text
    assert "14.8 / 15.1 N" in text
    assert "1.000 / 0.400 rad" in text
    assert "60.0%" in text
    assert "NEAR COLLISION / CONTACT OK" in text
```

- [ ] **Step 2: Run the HUD test and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/test_hud.py
```

Expected: collection fails because `rizon_osc.hud` does not exist.

- [ ] **Step 3: Implement the pure HUD formatter**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class HudSnapshot:
    phase: str
    force_7_n: float
    force_9_n: float
    arm_7_rad: float
    arm_9_rad: float
    wrist_9_rad: float
    reduction_percent: float | None
    collision_7: str
    collision_9: str


def format_hud(snapshot: HudSnapshot) -> str:
    reduction = (
        f"{snapshot.reduction_percent:.1f}%"
        if snapshot.reduction_percent is not None
        else "hidden until equal-accuracy gate"
    )
    return (
        f"Phase: {snapshot.phase}\n"
        f"Measured normal force 7 / 9: "
        f"{snapshot.force_7_n:.1f} / {snapshot.force_9_n:.1f} N\n"
        f"Phase arm travel 7 / 9: "
        f"{snapshot.arm_7_rad:.3f} / {snapshot.arm_9_rad:.3f} rad\n"
        f"9-DoF distal wrist travel: {snapshot.wrist_9_rad:.3f} rad\n"
        f"Validated main-arm reduction: {reduction}\n"
        f"Collision 7 / 9: {snapshot.collision_7} / "
        f"{snapshot.collision_9}"
    )
```

- [ ] **Step 4: Verify the HUD test is GREEN**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q tests/test_hud.py
```

Expected: 1 passed.

- [ ] **Step 5: Write and verify the failing wrist-axis source contract**

Add to `tests/test_isaaclab_osc_contract.py`:

```python
def test_runner_verifies_authored_wrist_axis_signs_before_control():
    source = RUNNER.read_text()

    assert "def verify_wrist_axis_signs(" in source
    assert "pitch_delta_task_rad" in source
    assert "yaw_delta_task_rad" in source
    assert "wrist_axis_check" in source
```

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q \
  tests/test_isaaclab_osc_contract.py::test_runner_verifies_authored_wrist_axis_signs_before_control
```

Expected: FAIL because the runner has no simulated wrist-axis check.

- [ ] **Step 6: Verify the authored wrist-axis signs in simulation**

Before the control timeline begins, save the green default joint state and
probe pose. Displace only wrist joint 8 by positive one degree, take an
unrendered physics step, express the measured probe rotation delta in the
neutral task frame, and require a negative task-Y component greater than
0.5 degree in magnitude. Restore the state, repeat with positive one degree
on joint 9, and require a positive task-Z component greater than 0.5 degree.
Restore the complete default joint state and zero velocity before starting
the task.

Expose the check as:

```python
@dataclass(frozen=True)
class WristAxisCheck:
    pitch_delta_task_rad: tuple[float, float, float]
    yaw_delta_task_rad: tuple[float, float, float]
    passed: bool


def verify_wrist_axis_signs(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    ee_body_idx: int,
    controlled_joint_ids: list[int],
    task_frame_quat_b: torch.Tensor,
) -> WristAxisCheck:
    dt = sim.get_physics_dt()
    baseline_joint_pos = robot.data.joint_pos.torch.clone()
    zero_joint_vel = torch.zeros_like(robot.data.joint_vel.torch)
    baseline_pose_b = robot_state(
        robot, ee_body_idx, controlled_joint_ids
    )[3].clone()

    def write_and_measure(position: torch.Tensor) -> torch.Tensor:
        robot.write_joint_position_to_sim_index(position=position)
        robot.write_joint_velocity_to_sim_index(velocity=zero_joint_vel)
        robot.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)
        return robot_state(robot, ee_body_idx, controlled_joint_ids)[3].clone()

    pitch_trial = baseline_joint_pos.clone()
    pitch_trial[:, controlled_joint_ids[-2]] += math.radians(1.0)
    pitch_pose_b = write_and_measure(pitch_trial)
    write_and_measure(baseline_joint_pos)

    yaw_trial = baseline_joint_pos.clone()
    yaw_trial[:, controlled_joint_ids[-1]] += math.radians(1.0)
    yaw_pose_b = write_and_measure(yaw_trial)
    write_and_measure(baseline_joint_pos)

    _, pitch_delta_b = compute_pose_error(
        baseline_pose_b[:, :3],
        baseline_pose_b[:, 3:],
        pitch_pose_b[:, :3],
        pitch_pose_b[:, 3:],
    )
    _, yaw_delta_b = compute_pose_error(
        baseline_pose_b[:, :3],
        baseline_pose_b[:, 3:],
        yaw_pose_b[:, :3],
        yaw_pose_b[:, 3:],
    )
    pitch_delta_task = quat_apply_inverse(
        task_frame_quat_b, pitch_delta_b
    )
    yaw_delta_task = quat_apply_inverse(task_frame_quat_b, yaw_delta_b)
    passed = bool(
        pitch_delta_task[0, 1] < -math.radians(0.5)
        and yaw_delta_task[0, 2] > math.radians(0.5)
    )
    return WristAxisCheck(
        tuple(float(value) for value in pitch_delta_task[0].tolist()),
        tuple(float(value) for value in yaw_delta_task[0].tolist()),
        passed,
    )
```

The function uses `write_joint_position_to_sim_index()`,
`write_joint_velocity_to_sim_index()`, `compute_pose_error()`, and
`quat_apply_inverse()` only. It must not calculate or apply a controller
torque. Add `wrist_axis_check` to the validation JSON and abort before the
task if `passed` is false.

- [ ] **Step 7: Replace cumulative-only travel in the loop**

Instantiate `JointTravelTracker`. On every phase change:

```python
travel_tracker.begin_phase(
    reference.phase.value,
    state_7[6][0].detach().cpu().numpy(),
    state_9[6][0].detach().cpu().numpy(),
)
policy_9.begin_phase(
    reference.phase.value,
    state_9[6][0].detach().cpu().numpy(),
)
pose_osc_7.reset()
pose_osc_9.reset()
hybrid_osc_7.reset()
hybrid_osc_9.reset()
```

After each physics step, call `travel_tracker.update()` with the current
seven- and nine-joint positions. Store its snapshot in every metric sample.
Whole-run travel may remain in console diagnostics but must not drive the
advantage gate.

- [ ] **Step 8: Use identical tutorial-layout commands**

For approach, concatenate `pose_task` and `pose_kp_task`. For every contact
phase, concatenate `pose_task`, `wrench_task`, and `pose_kp_task`. Verify
identity with all three tensors and the task frame:

```python
references_identical = (
    torch.allclose(command_7, command_9)
    and torch.allclose(task_frame_7, task_frame_9)
    and not collision_7.freeze_path
)
```

Before a red collision stop, `command_7` and `command_9` must be the exact
same tensor value. After the latch, preserve the last safe red target,
switch red to the official pose controller, set its normal-force command to
zero, and continue the green common task.

- [ ] **Step 9: Feed the extended metric sample**

At the existing 0.1-second metric interval, populate:

```python
travel = travel_tracker.snapshot()
MetricSample(
    phase=reference.phase.value,
    commanded_force_7=commanded_force_7,
    commanded_force_9=commanded_force_9,
    measured_force_7=filtered_7,
    measured_force_9=filtered_9,
    normal_angle_7_deg=angle_7,
    normal_angle_9_deg=angle_9,
    orientation_error_7_deg=orientation_error_7_deg,
    orientation_error_9_deg=orientation_error_9_deg,
    tangent_error_7_m=tangent_error_7,
    tangent_error_9_m=tangent_error_9,
    contact_loss_7_s=last_supervisor_7.max_contact_loss_duration,
    contact_loss_9_s=last_supervisor_9.max_contact_loss_duration,
    arm_travel_7_rad=arm_travel_7,
    arm_travel_9_rad=arm_travel_9,
    wrist_travel_9_rad=wrist_travel_9,
    static_drift_m=drift,
    references_identical=references_identical,
    phase_arm_travel_7_rad=travel.arm_7_rad,
    phase_arm_travel_9_rad=travel.arm_9_rad,
    phase_wrist_travel_9_rad=travel.wrist_9_rad,
    nonprobe_force_7_n=collision_7.current_force_n,
    nonprobe_force_9_n=collision_9.current_force_n,
    collision_stop_7=collision_7.freeze_path,
    collision_stop_9=collision_9.freeze_path,
    completed_7=phase_completed_7,
    completed_9=phase_completed_9,
)
```

`phase_completed_9` becomes true when green reaches the end of the current
phase. `phase_completed_7` follows the same clock unless red collision
latched during that phase.

- [ ] **Step 10: Add the GUI overlay**

Import `omni.ui as ui` only after `AppLauncher` starts. In non-headless mode,
create one persistent `ui.Window("OSC 7-DoF vs 9-DoF", width=520,
height=220)`, one `ui.Label`, and update its `.text` from `format_hud()` at
the metric interval. Do not create or destroy a window every frame.

Keep the final neutral pose and 15 N contact after
`trajectory.total_duration`. With `--max_steps 0`, continue rendering until
the user closes the Isaac Sim window.

- [ ] **Step 11: Run the complete pure test suite and commit**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q
```

Expected: all tests pass.

Commit:

```bash
git add src/rizon_osc/hud.py tests/test_hud.py \
  scripts/run_osc_comparison.py
git commit -m "feat: integrate official OSC comparison HUD"
```

---

### Task 7: Validate on GPU, document evidence, and launch the persistent GUI

**Files:**
- Modify: `README.md`
- Modify: `PROJECT_MEMORY.md`
- Generated but not committed: `generated/validation_report.json`

**Interfaces:**
- Consumes: the completed runner and local assets.
- Produces: a passing headless validation report, updated operator documentation, and a GUI process that remains open.

- [ ] **Step 1: Run all fast verification commands**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q
/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m compileall -q scripts src tests tools
bash -n launch_osc_comparison.sh
git diff --check
```

Expected: pytest has zero failures, compileall and shell syntax exit 0, and
`git diff --check` prints nothing.

- [ ] **Step 2: Rebuild and verify local assets**

Run:

```bash
/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  tools/build_exact_rizon_wrist_asset.py
/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  tools/preprocess_assembly3_surface.py
```

Expected: both commands exit 0 and preserve the fixed challenge point
inside the valid surface map.

- [ ] **Step 3: Run the full headless GPU acceptance**

The full timeline is 13.0 seconds at 0.004-second physics time steps. Run
3500 steps to include final settling:

```bash
./launch_osc_comparison.sh \
  --headless \
  --max_steps 3500 \
  --validation_report generated/validation_report.json
```

Expected: exit 0. Inspect the JSON and require:

```text
controller = isaaclab.controllers.OperationalSpaceController
overall_pass = true
equal_accuracy_comparison.pitch.reduction_percent >= 50
equal_accuracy_comparison.yaw.reduction_percent >= 50
collision_challenge.pass = true
collision_challenge.green_completed = true
collision_challenge.green_collision_free = true
```

If the command exits nonzero, use `superpowers:systematic-debugging` and
change one evidenced variable at a time. Do not weaken an acceptance
threshold, alter the fixed challenge point, or publish a failed reduction
as an advantage.

- [ ] **Step 4: Update operator and memory documentation**

In `README.md`, document:

```bash
cd /home/bdml-sim/Isaacsim-9DoF-flexiv-weighted-osc
./launch_osc_comparison.sh
```

Explain the two examples, HUD fields, magenta command-force arrow, cyan
measured-force arrow, collision states, and manual window close.

In `PROJECT_MEMORY.md`, record:

- the official tutorial-derived OSC configuration;
- the per-phase 50 percent acceptance rule;
- the fixed challenge point and angle;
- the exact final JSON measurements;
- the fact that the GUI intentionally remains open;
- every failed acceptance measurement if the final run did not pass.

- [ ] **Step 5: Re-run verification after documentation**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python \
  -m pytest -q
git diff --check
git status --short
```

Expected: tests pass, no whitespace errors, and only intended source,
test, and documentation files are modified.

- [ ] **Step 6: Commit the validated implementation**

Do not add `generated/validation_report.json`, which remains ignored.

```bash
git add README.md PROJECT_MEMORY.md
git commit -m "docs: record official OSC wrist advantage evidence"
```

- [ ] **Step 7: Start the replacement GUI without auto-close**

Do not terminate the existing GUI automatically. Ask the user to close the
old window, then run:

```bash
./launch_osc_comparison.sh
```

Expected: the new Isaac Sim window displays both robots, the HUD, force
arrows, the equal-accuracy example, and the collision challenge. The
process remains active after the final neutral pose until the user manually
closes the window.

- [ ] **Step 8: Perform final branch verification**

Run:

```bash
git status --short --branch
git log --oneline --decorate -8
git diff --check origin/main..HEAD
```

Expected: the worktree is clean, commits are on
`feature/isaaclab-osc-contact`, and the branch remains ready for an
authenticated push.
