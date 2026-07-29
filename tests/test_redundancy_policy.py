import numpy as np
import pytest

from rizon_osc.redundancy_policy import RedundancyPolicy


def test_green_policy_holds_main_arm_at_phase_start():
    policy = RedundancyPolicy(num_arm_joints=7, num_wrist_joints=2)
    phase_start = np.linspace(-0.3, 0.3, 9)
    policy.begin_phase("PITCH_ONLY", phase_start)

    target = policy.target(
        current_joint_position=phase_start + 0.1,
        relative_pitch=0.35,
        relative_yaw=0.0,
    )

    assert target[:7] == pytest.approx(phase_start[:7])
    assert target[7] == pytest.approx(-0.35)
    assert target[8] == pytest.approx(0.0)


def test_yaw_phase_uses_only_distal_yaw_joint_target():
    policy = RedundancyPolicy(num_arm_joints=7, num_wrist_joints=2)
    start = np.zeros(9)
    policy.begin_phase("YAW_ONLY", start)

    target = policy.target(start, relative_pitch=0.0, relative_yaw=0.25)

    assert target[:8] == pytest.approx(np.zeros(8))
    assert target[8] == pytest.approx(0.25)


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


def test_scan_returns_wrist_to_neutral_without_changing_arm_reference():
    policy = RedundancyPolicy(num_arm_joints=7, num_wrist_joints=2)
    start = np.arange(9, dtype=float) / 10.0
    policy.begin_phase("SURFACE_SCAN", start)

    target = policy.target(start, relative_pitch=0.2, relative_yaw=0.2)

    assert target[:7] == pytest.approx(start[:7])
    assert target[7:] == pytest.approx(np.zeros(2))
