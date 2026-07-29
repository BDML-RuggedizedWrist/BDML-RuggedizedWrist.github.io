import numpy as np
import pytest

from rizon_osc.redundancy_policy import RedundancyPolicy


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


@pytest.mark.parametrize("bad_shape", ((8,), (10,), (1, 9)))
@pytest.mark.parametrize("method", ("initialize_run", "begin_phase", "target"))
def test_policy_rejects_unexpected_joint_position_shape(method, bad_shape):
    policy = RedundancyPolicy()
    bad = np.zeros(bad_shape)

    with pytest.raises(ValueError, match="expected 9 joint positions"):
        if method == "initialize_run":
            policy.initialize_run(bad)
        else:
            policy.initialize_run(np.zeros(9))
            if method == "begin_phase":
                policy.begin_phase("PITCH_ONLY", bad)
            else:
                policy.begin_phase("PITCH_ONLY", np.zeros(9))
                policy.target(bad, relative_pitch=0.0, relative_yaw=0.0)
