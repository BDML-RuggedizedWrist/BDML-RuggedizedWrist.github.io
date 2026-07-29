import numpy as np
import pytest

from rizon_osc.surface_model import SurfaceMap
from rizon_osc.trajectory import (
    Phase,
    SurfaceTrajectory,
    quintic_progress,
    rotation_matrix_from_quaternion,
)


@pytest.fixture
def curved_surface() -> SurfaceMap:
    x = np.linspace(-0.1, 0.1, 21)
    y = np.linspace(0.0, 0.2, 21)
    xx, yy = np.meshgrid(x, y)
    height = 0.4 - 0.4 * xx**2 + 0.05 * yy
    dzdx = -0.8 * xx
    dzdy = np.full_like(yy, 0.05)
    normals = np.stack((-dzdx, -dzdy, np.ones_like(height)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return SurfaceMap(x, y, height, normals, np.ones_like(height, dtype=bool))


def test_quintic_progress_has_zero_endpoint_velocity_and_acceleration():
    for u, expected in ((0.0, 0.0), (1.0, 1.0)):
        position, velocity, acceleration = quintic_progress(u)
        assert position == pytest.approx(expected)
        assert velocity == pytest.approx(0.0)
        assert acceleration == pytest.approx(0.0)


def test_surface_scan_uses_16cm_in_4_seconds(curved_surface):
    trajectory = SurfaceTrajectory(
        curved_surface,
        scan_start_xy=(0.0, 0.01),
        scan_end_xy=(0.0, 0.17),
        approach_duration=0.0,
        contact_ramp_duration=0.0,
        scan_duration=4.0,
    )
    start = trajectory.reference(0.0)
    middle = trajectory.reference(2.0)
    end = trajectory.reference(4.0)

    assert start.phase is Phase.SURFACE_SCAN
    assert end.contact_point[1] - start.contact_point[1] == pytest.approx(0.16)
    assert middle.contact_point[1] == pytest.approx(0.09)
    assert np.linalg.norm(start.linear_velocity) == pytest.approx(0.0, abs=1e-8)
    assert np.linalg.norm(end.linear_velocity) == pytest.approx(0.0, abs=1e-8)


def test_probe_axis_is_opposite_surface_normal_and_frame_is_orthonormal(curved_surface):
    trajectory = SurfaceTrajectory(
        curved_surface,
        scan_start_xy=(-0.04, 0.02),
        scan_end_xy=(0.04, 0.18),
        approach_duration=0.0,
        contact_ramp_duration=0.0,
    )
    reference = trajectory.reference(1.7)
    rotation = rotation_matrix_from_quaternion(reference.quaternion)

    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-7)
    assert rotation[:, 2] == pytest.approx(reference.surface_normal, abs=1e-7)
    assert reference.probe_acoustic_axis == pytest.approx(-reference.surface_normal, abs=1e-7)
    assert np.isfinite(reference.twist).all()
    assert np.isfinite(reference.acceleration).all()


def test_fixed_contact_orientation_changes_one_axis_at_a_time(curved_surface):
    trajectory = SurfaceTrajectory(
        curved_surface,
        approach_duration=0.2,
        contact_ramp_duration=0.5,
        scan_duration=4.0,
        pitch_duration=1.0,
        neutral_duration=0.5,
        yaw_duration=1.0,
    )
    pitch = trajectory.reference(0.2 + 0.5 + 4.0 + 0.5)
    neutral = trajectory.reference(0.2 + 0.5 + 4.0 + 1.0 + 0.25)
    yaw = trajectory.reference(0.2 + 0.5 + 4.0 + 1.0 + 0.5 + 0.5)

    assert pitch.phase is Phase.PITCH_ONLY
    assert pitch.relative_rpy[0] == pytest.approx(0.0)
    assert abs(pitch.relative_rpy[1]) > 0.1
    assert pitch.relative_rpy[2] == pytest.approx(0.0)
    assert neutral.phase is Phase.RETURN_NEUTRAL
    assert yaw.phase is Phase.YAW_ONLY
    assert yaw.relative_rpy[0] == pytest.approx(0.0)
    assert yaw.relative_rpy[1] == pytest.approx(0.0)
    assert abs(yaw.relative_rpy[2]) > 0.1
    assert pitch.contact_point == pytest.approx(yaw.contact_point)


def test_invalid_surface_query_holds_last_safe_reference(curved_surface):
    trajectory = SurfaceTrajectory(
        curved_surface,
        scan_start_xy=(0.0, 0.02),
        scan_end_xy=(0.0, 0.30),
        approach_duration=0.0,
        contact_ramp_duration=0.0,
    )
    safe = trajectory.reference(0.1)
    held = trajectory.reference(4.0)

    assert safe.valid
    assert not held.valid
    assert held.position == pytest.approx(safe.position)

