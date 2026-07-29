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


@pytest.mark.parametrize("dt_s", (np.nan, np.inf, -np.inf))
def test_nonfinite_dt_latches_instead_of_raising(dt_s):
    snapshot = ValidationWatchdog().update(sample(dt_s=dt_s))

    assert snapshot.reasons == ("nonfinite",)
    assert snapshot.first_failure_step == 1


def test_nonpositive_finite_dt_remains_an_input_error():
    with pytest.raises(ValueError, match="dt_s"):
        ValidationWatchdog().update(sample(dt_s=0.0))


def test_nonfinite_contact_payload_latches():
    snapshot = ValidationWatchdog().update(
        sample(contact_present=np.array([True, np.nan]))
    )

    assert snapshot.reasons == ("nonfinite",)


def test_nonnumeric_object_payload_latches_instead_of_raising():
    snapshot = ValidationWatchdog().update(
        sample(finite_payloads=(np.array(["not-a-number"], dtype=object),))
    )

    assert snapshot.reasons == ("nonfinite",)


@pytest.mark.parametrize(
    "field,active,reason",
    (
        (
            "wrist_position_rad",
            np.array([1.56, 0.0]),
            "green_wrist_limit_j8",
        ),
        (
            "wrist_velocity_rad_s",
            np.array([0.0, -1.99]),
            "green_wrist_speed_j9",
        ),
    ),
)
@pytest.mark.parametrize("dt_s,count", ((0.01, 10), (0.004, 25)))
def test_inclusive_wrist_timers_fire_at_nominal_tenth_second(
    field, active, reason, dt_s, count
):
    watchdog = ValidationWatchdog()
    current = sample(dt_s=dt_s, **{field: active})

    assert advance(watchdog, current, count - 1).passed
    assert reason in watchdog.update(replace(current, step=count)).reasons


@pytest.mark.parametrize("dt_s,count", ((0.01, 10), (0.004, 25)))
def test_strict_contact_timer_is_safe_at_nominal_tenth_second(dt_s, count):
    watchdog = ValidationWatchdog()
    lost = sample(dt_s=dt_s, contact_present=np.array([True, False]))

    assert advance(watchdog, lost, count).passed
    stopped = watchdog.update(replace(lost, step=count + 1))

    assert "probe_contact_loss_9" in stopped.reasons


def _translation_freeze_after_exact_window(dt_s: float, intervals: int):
    watchdog = ValidationWatchdog()
    snapshot = watchdog.snapshot()
    for index in range(intervals + 1):
        target = np.zeros((2, 3))
        target[1, 0] = 0.002 * index / intervals
        snapshot = watchdog.update(
            sample(
                step=index,
                dt_s=dt_s,
                target_position_m=target,
            )
        )
    return snapshot


@pytest.mark.parametrize("dt_s,intervals", ((0.05, 5), (0.01, 25)))
def test_translation_freeze_checks_the_exact_quarter_second_window(
    dt_s, intervals
):
    snapshot = _translation_freeze_after_exact_window(dt_s, intervals)

    assert "task_freeze_translation_9" in snapshot.reasons


def test_rotation_freeze_accepts_exact_half_degree_command_boundary():
    watchdog = ValidationWatchdog()
    snapshot = watchdog.snapshot()
    for index in range(6):
        target = sample().target_quaternion_wxyz.copy()
        target[0] = z_rotation(np.deg2rad(0.5) * index / 5)
        snapshot = watchdog.update(
            sample(step=index, dt_s=0.05, target_quaternion_wxyz=target)
        )

    assert "task_freeze_rotation_7" in snapshot.reasons


def test_red_freeze_remains_disabled_after_a_one_shot_collision_stop():
    watchdog = ValidationWatchdog()
    watchdog.update(sample(red_collision_stop=True))
    snapshot = watchdog.snapshot()
    for index in range(1, 28):
        target = np.zeros((2, 3))
        target[0, 0] = 0.002 * index / 27
        snapshot = watchdog.update(
            sample(
                step=index,
                dt_s=0.01,
                phase="CHALLENGE_PITCH_ONLY",
                target_position_m=target,
            )
        )

    assert snapshot.passed


def test_watchdog_sample_defensively_copies_and_freezes_array_payloads():
    wrist_position = np.zeros(2)
    finite_payload = np.array([1.0])
    watchdog_sample = sample(
        wrist_position_rad=wrist_position,
        finite_payloads=(finite_payload,),
    )
    wrist_position[0] = 1.56
    finite_payload[0] = np.nan

    assert watchdog_sample.wrist_position_rad.tolist() == [0.0, 0.0]
    assert watchdog_sample.finite_payloads[0].tolist() == [1.0]
    with pytest.raises(ValueError):
        watchdog_sample.wrist_position_rad[0] = 1.56
    with pytest.raises(ValueError):
        watchdog_sample.finite_payloads[0][0] = np.nan
