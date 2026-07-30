from pathlib import Path

import numpy as np
import pytest

from rizon_osc.near_far_trajectory import NearFarPhase, NearFarTrajectory
from rizon_osc.scan_profiles import cross_waist_profile, scan_profile
from rizon_osc.surface_model import SurfaceMap


PROJECT_ROOT = Path(__file__).parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_osc_cross_waist_comparison.py"
SHARED_RUNTIME = PROJECT_ROOT / "scripts" / "run_osc_near_far_comparison.py"
LAUNCHER = PROJECT_ROOT / "launch_osc_cross_waist_comparison.sh"


@pytest.fixture
def torso_surface() -> SurfaceMap:
    x = np.linspace(-0.12, 0.12, 61)
    y = np.linspace(1.12, 1.40, 71)
    xx, yy = np.meshgrid(x, y)
    height = 0.10 - 0.15 * xx**2 - 0.02 * (yy - 1.18) ** 2
    normals = np.zeros((*height.shape, 3), dtype=np.float64)
    normals[..., 2] = 1.0
    return SurfaceMap(
        x,
        y,
        height,
        normals,
        np.ones_like(height, dtype=bool),
        metadata={"scan_start_xy": [0.0, 1.18]},
    )


def test_cross_waist_orders_endpoints_by_robot_base_distance(torso_surface):
    profile = cross_waist_profile(
        torso_surface,
        surface_translation_xy=(0.70, -1.29),
    )
    start = np.asarray(profile.start_xy)
    end = np.asarray(profile.end_xy)
    translation = np.array([0.70, -1.29])

    assert profile.name == "cross_waist"
    assert start[1] == pytest.approx(end[1])
    assert start[0] < end[0]
    assert np.linalg.norm(start + translation) < np.linalg.norm(end + translation)
    assert torso_surface.query(*profile.start_xy).valid
    assert torso_surface.query(*profile.end_xy).valid


def test_cross_waist_uses_large_far_side_orientation_commands(torso_surface):
    profile = scan_profile(
        torso_surface,
        task_variant="cross_waist",
        surface_translation_xy=(0.70, -1.29),
    )

    assert profile.orientation_direction_xy == pytest.approx((0.0, 1.0))
    assert profile.bend_axis == "pitch"
    assert profile.wrist_bend_sign == pytest.approx(-1.0)
    assert np.degrees(profile.bend_angle) == pytest.approx(-40.0)
    assert np.degrees(profile.axial_slice_angle) == pytest.approx(90.0)
    assert profile.scan_duration == pytest.approx(2.4)

    trajectory = NearFarTrajectory(
        torso_surface,
        near_xy=profile.start_xy,
        far_xy=profile.end_xy,
        orientation_direction_xy=profile.orientation_direction_xy,
        pitch_angle=profile.bend_angle,
        bend_axis=profile.bend_axis,
    )
    settle_end = trajectory.far_hold_start_time + trajectory.settle_duration
    bend = trajectory.reference(settle_end + 0.5 * trajectory.pitch_duration)
    assert bend.phase is NearFarPhase.PITCH_ONLY
    assert bend.relative_rpy[1] < -0.1
    assert bend.relative_rpy[[0, 2]] == pytest.approx((0.0, 0.0))


def test_cross_waist_has_independent_persistent_gui_entry():
    runner_source = RUNNER.read_text()
    launcher_source = LAUNCHER.read_text()
    shared_source = SHARED_RUNTIME.read_text()

    assert RUNNER.is_file()
    assert LAUNCHER.is_file()
    assert '["--task_variant", "cross_waist"]' in runner_source
    assert "run_osc_cross_waist_comparison.py" in launcher_source
    assert "--viz kit" in launcher_source
    assert 'default="near_to_far"' in shared_source
    assert '"task_variant": args_cli.task_variant' in shared_source


def test_shared_runtime_keeps_all_active_torques_in_official_isaaclab_osc():
    source = SHARED_RUNTIME.read_text()

    assert "OperationalSpaceController" in source
    assert "osc_7.compute(" in source
    assert "osc_9.compute(" in source
    assert "torch.linalg.pinv" not in source
    assert "torch.pinverse" not in source
