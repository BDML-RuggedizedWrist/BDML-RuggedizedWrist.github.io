from pathlib import Path

import numpy as np
import pytest

from rizon_osc.near_far_policy import NearFarRedundancyPolicy
from rizon_osc.near_far_trajectory import NearFarPhase, NearFarTrajectory
from rizon_osc.surface_model import SurfaceMap


PROJECT_ROOT = Path(__file__).parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_osc_near_far_comparison.py"
LAUNCHER = PROJECT_ROOT / "launch_osc_near_far_comparison.sh"
HEART_RUNNER = PROJECT_ROOT / "scripts" / "run_osc_comparison.py"


@pytest.fixture
def torso_surface() -> SurfaceMap:
    x = np.linspace(-0.1, 0.1, 21)
    y = np.linspace(1.1, 1.4, 31)
    xx, yy = np.meshgrid(x, y)
    height = 0.42 - 0.5 * xx**2 + 0.03 * (yy - 1.25) ** 2
    dzdx = -xx
    dzdy = 0.06 * (yy - 1.25)
    normals = np.stack((-dzdx, -dzdy, np.ones_like(height)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return SurfaceMap(
        x,
        y,
        height,
        normals,
        np.ones_like(height, dtype=bool),
        metadata={
            "scan_start_xy": [0.0, 1.18],
            "scan_end_xy": [0.0, 1.34],
        },
    )


def make_trajectory(surface: SurfaceMap) -> NearFarTrajectory:
    return NearFarTrajectory(
        surface,
        near_xy=(0.0, 1.18),
        far_xy=(0.0, 1.34),
        approach_duration=1.0,
        contact_ramp_duration=1.0,
        scan_duration=2.0,
        settle_duration=0.4,
        pitch_duration=1.5,
        return_pitch_duration=0.8,
        axial_slice_duration=2.0,
        pitch_angle=np.deg2rad(-35.0),
        axial_slice_angle=np.deg2rad(90.0),
        target_force=15.0,
    )


def test_near_to_far_scan_reaches_far_endpoint_with_constant_force(torso_surface):
    trajectory = make_trajectory(torso_surface)
    scan_start = trajectory.reference(trajectory.scan_start_time)
    scan_middle = trajectory.reference(trajectory.scan_start_time + 1.0)
    far = trajectory.reference(trajectory.far_hold_start_time + 0.2)

    assert scan_start.phase is NearFarPhase.SCAN_NEAR_TO_FAR
    assert scan_start.contact_point[:2] == pytest.approx((0.0, 1.18))
    assert scan_middle.contact_point[1] == pytest.approx(1.26)
    assert far.phase is NearFarPhase.SETTLE_FAR
    assert far.contact_point[:2] == pytest.approx((0.0, 1.34))
    assert scan_start.normal_force == pytest.approx(15.0)
    assert scan_middle.normal_force == pytest.approx(15.0)
    assert far.normal_force == pytest.approx(15.0)


def test_far_point_orientation_changes_only_one_coordinate_at_a_time(torso_surface):
    trajectory = make_trajectory(torso_surface)
    settle_end = trajectory.far_hold_start_time + trajectory.settle_duration
    pitch = trajectory.reference(settle_end + 0.75)
    return_pitch = trajectory.reference(
        settle_end + trajectory.pitch_duration + 0.4
    )
    axial = trajectory.reference(
        settle_end
        + trajectory.pitch_duration
        + trajectory.return_pitch_duration
        + 1.0
    )
    final = trajectory.reference(100.0)

    assert pitch.phase is NearFarPhase.PITCH_ONLY
    assert pitch.relative_rpy[[0, 2]] == pytest.approx((0.0, 0.0))
    assert abs(pitch.relative_rpy[1]) > 0.1
    assert return_pitch.phase is NearFarPhase.RETURN_PITCH
    assert return_pitch.relative_rpy[[0, 2]] == pytest.approx((0.0, 0.0))
    assert axial.phase is NearFarPhase.AXIAL_SLICE_90
    assert axial.relative_rpy[:2] == pytest.approx((0.0, 0.0))
    assert axial.relative_rpy[2] == pytest.approx(np.deg2rad(45.0))
    assert final.phase is NearFarPhase.HOLD_FINAL_SLICE
    assert final.relative_rpy == pytest.approx((0.0, 0.0, np.deg2rad(90.0)))
    assert pitch.contact_point == pytest.approx(axial.contact_point)
    assert axial.contact_point == pytest.approx(final.contact_point)
    assert final.normal_force == pytest.approx(15.0)


def test_green_policy_freezes_far_arm_and_assigns_pitch_and_axial_to_wrist():
    initial = np.array([0.0, -0.7, 0.0, 1.57, 0.0, 0.7, 0.0, 0.1, -0.2])
    far_arrival = initial + np.array(
        [0.1, 0.2, -0.1, 0.15, 0.05, -0.08, 0.03, 0.0, 0.0]
    )
    later = far_arrival + np.array(
        [0.5, -0.4, 0.3, -0.2, 0.4, 0.1, -0.3, 0.2, -0.1]
    )
    policy = NearFarRedundancyPolicy()
    policy.initialize(initial)
    policy.begin_phase(NearFarPhase.SETTLE_FAR, far_arrival)
    policy.begin_phase(NearFarPhase.PITCH_ONLY, later)

    pitch = np.deg2rad(-35.0)
    pitch_target = policy.target(
        later,
        phase=NearFarPhase.PITCH_ONLY,
        relative_pitch=pitch,
        relative_axial=0.0,
    )
    axial_target = policy.target(
        later,
        phase=NearFarPhase.AXIAL_SLICE_90,
        relative_pitch=0.0,
        relative_axial=np.deg2rad(90.0),
    )

    assert pitch_target[:7] == pytest.approx(far_arrival[:7])
    assert axial_target[:7] == pytest.approx(far_arrival[:7])
    assert pitch_target[7] == pytest.approx(initial[7] - pitch)
    assert pitch_target[8] == pytest.approx(initial[8])
    assert axial_target[7] == pytest.approx(initial[7])
    assert axial_target[8] == pytest.approx(initial[8] + np.deg2rad(90.0))


def test_independent_runner_uses_only_official_isaaclab_osc_for_active_torques():
    source = RUNNER.read_text()

    assert RUNNER.is_file()
    assert RUNNER != HEART_RUNNER
    assert "from isaaclab.controllers import (" in source
    assert "OperationalSpaceController" in source
    assert "OperationalSpaceControllerCfg" in source
    assert "osc_7.compute(" in source
    assert "osc_9.compute(" in source
    assert "nullspace_joint_pos_target=green_null_target" in source
    assert "torch.pinverse" not in source
    assert "torch.linalg.pinv" not in source
    assert "pseudoinverse" not in source.lower()
    assert "projector" not in source.lower()


def test_runner_does_not_mutate_isaac_state_views_when_shifting_to_probe_tip():
    source = RUNNER.read_text()
    robot_state_source = source.split("def robot_state(", maxsplit=1)[1]
    robot_state_source = robot_state_source.split(
        "def verify_wrist_axis_signs(", maxsplit=1
    )[0]

    assert "ee_pos_b = ee_pos_b + tip_offset_b" in robot_state_source
    assert "ee_pos_w = ee_pos_w + tip_offset_w" in robot_state_source
    assert "ee_pos_b += tip_offset_b" not in robot_state_source
    assert "ee_pos_w += tip_offset_w" not in robot_state_source


def test_runner_has_fixed_assets_collision_latches_and_force_visualization():
    source = RUNNER.read_text()

    assert "fix_root_link=True" in source
    assert "collision_enabled=True" in source
    assert "collision_monitor_7 = CollisionMonitor()" in source
    assert "collision_monitor_9 = CollisionMonitor()" in source
    assert "collision_7.freeze_path" in source
    assert "collision_9.freeze_path" in source
    assert "collision_hold_frame_7 = state_7[3].clone()" in source
    assert "collision_hold_frame_9 = state_9[3].clone()" in source
    assert "def verify_wrist_axis_signs(" in source
    assert "stabilize_precheck_pair(" in source
    assert "if not wrist_axis_check.passed:" in source
    assert "NearFarCommandForce" in source
    assert "NearFarMeasuredForce" in source
    assert 'for side in ("7", "9")' in source
    assert "args_cli.normal_force" in source
    assert "default=15.0" in source
    assert '"green_max_contact_loss_s"' in source
    assert '"green_nonprobe_collision_peak_n"' in source
    assert '"green_final_orientation_error_deg"' in source
    assert '"phase_summaries"' in source


def test_independent_launcher_defaults_to_persistent_gui():
    runner_source = RUNNER.read_text()
    launcher_source = LAUNCHER.read_text()

    assert 'parser.add_argument(\n    "--max_steps"' in runner_source
    assert "default=0" in runner_source
    assert "run_osc_near_far_comparison.py" in launcher_source
    assert "--viz kit" in launcher_source
    assert "run_osc_comparison.py" not in launcher_source
