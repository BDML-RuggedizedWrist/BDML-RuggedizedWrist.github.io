import pytest

from rizon_osc.metrics import AcceptanceMetrics, MetricSample


def passing_sample(**overrides) -> MetricSample:
    values = {
        "phase": "SURFACE_SCAN",
        "commanded_force_7": 15.0,
        "commanded_force_9": 15.0,
        "measured_force_7": 15.2,
        "measured_force_9": 14.9,
        "normal_angle_7_deg": 1.2,
        "normal_angle_9_deg": 1.0,
        "orientation_error_7_deg": 1.0,
        "orientation_error_9_deg": 1.0,
        "tangent_error_7_m": 0.002,
        "tangent_error_9_m": 0.002,
        "contact_loss_7_s": 0.0,
        "contact_loss_9_s": 0.0,
        "arm_travel_7_rad": 1.0,
        "arm_travel_9_rad": 0.6,
        "wrist_travel_9_rad": 0.5,
        "static_drift_m": 0.0,
        "references_identical": True,
        "phase_arm_travel_7_rad": 1.0,
        "phase_arm_travel_9_rad": 0.4,
        "phase_wrist_travel_9_rad": 0.6,
        "nonprobe_force_7_n": 0.0,
        "nonprobe_force_9_n": 0.0,
        "collision_stop_7": False,
        "collision_stop_9": False,
        "completed_7": True,
        "completed_9": True,
    }
    values.update(overrides)
    return MetricSample(**values)


def add_passing_scenario(metrics: AcceptanceMetrics, **first_overrides) -> None:
    for phase in (
        "SURFACE_SCAN",
        "PITCH_ONLY",
        "RETURN_NEUTRAL",
        "YAW_ONLY",
        "CHALLENGE_PITCH_ONLY",
    ):
        metrics.add(passing_sample(phase=phase, **first_overrides))


def test_passing_metrics_accept_15n_normal_contact_and_reduction():
    metrics = AcceptanceMetrics(settling_samples=0)
    add_passing_scenario(metrics)

    report = metrics.report(scenario_complete=True)

    assert report["force_7"]["pass"]
    assert report["force_9"]["pass"]
    assert report["force_command_7"]["pass"]
    assert report["force_command_9"]["pass"]
    assert report["normal_7"]["pass"]
    assert report["normal_9"]["pass"]
    assert report["orientation_7"]["pass"]
    assert report["orientation_9"]["pass"]
    assert report["contact_7"]["pass"]
    assert report["contact_9"]["pass"]
    assert report["static_assets"]["pass"]
    assert report["phase_coverage"]["pass"]
    assert report["scenario_complete"]["pass"]
    assert report["main_arm_reduction"]["visible"]
    assert report["main_arm_reduction"]["percent"] == pytest.approx(40.0)
    assert report["overall_pass"]


def test_force_outside_15_plus_minus_1_5_fails():
    metrics = AcceptanceMetrics(settling_samples=0)
    metrics.add(passing_sample(measured_force_9=13.49))

    assert not metrics.report(scenario_complete=True)["force_9"]["pass"]


def test_reduction_is_hidden_when_accuracy_gate_fails():
    metrics = AcceptanceMetrics(settling_samples=0)
    add_passing_scenario(metrics, tangent_error_9_m=0.02)

    reduction = metrics.report(scenario_complete=True)["main_arm_reduction"]
    assert not reduction["visible"]
    assert reduction["percent"] is None


def test_reduction_is_hidden_when_reorientation_tracking_fails():
    metrics = AcceptanceMetrics(settling_samples=0)
    add_passing_scenario(metrics, orientation_error_9_deg=5.01)

    report = metrics.report(scenario_complete=True)

    assert not report["orientation_9"]["pass"]
    assert not report["main_arm_reduction"]["visible"]
    assert not report["overall_pass"]


def test_static_drift_contact_loss_reference_and_normal_thresholds():
    metrics = AcceptanceMetrics(settling_samples=0)
    add_passing_scenario(
        metrics,
        normal_angle_7_deg=3.01,
        contact_loss_9_s=0.101,
        static_drift_m=2.0e-5,
        references_identical=False,
    )
    report = metrics.report(scenario_complete=True)

    assert not report["normal_7"]["pass"]
    assert not report["contact_9"]["pass"]
    assert not report["static_assets"]["pass"]
    assert not report["identical_references"]["pass"]
    assert not report["main_arm_reduction"]["visible"]


def test_wrist_motion_is_reported_separately():
    metrics = AcceptanceMetrics(settling_samples=0)
    metrics.add(passing_sample(wrist_travel_9_rad=0.75))

    assert metrics.report(scenario_complete=True)["wrist_motion_9_rad"] == pytest.approx(0.75)


def test_partial_scenario_cannot_pass_or_expose_reduction():
    metrics = AcceptanceMetrics(settling_samples=0)
    metrics.add(passing_sample())

    report = metrics.report(scenario_complete=False)

    assert not report["phase_coverage"]["pass"]
    assert not report["scenario_complete"]["pass"]
    assert not report["main_arm_reduction"]["visible"]
    assert not report["overall_pass"]


def test_force_command_must_match_configured_target():
    metrics = AcceptanceMetrics(settling_samples=0)
    add_passing_scenario(metrics, commanded_force_9=0.0)

    report = metrics.report(scenario_complete=True)

    assert not report["force_command_9"]["pass"]
    assert not report["main_arm_reduction"]["visible"]


def test_configurable_force_target_is_used_for_command_and_measurement():
    metrics = AcceptanceMetrics(settling_samples=0, force_target=12.0)
    add_passing_scenario(
        metrics,
        commanded_force_7=12.0,
        commanded_force_9=12.0,
        measured_force_7=12.2,
        measured_force_9=11.9,
    )

    report = metrics.report(scenario_complete=True)

    assert report["force_7"]["pass"]
    assert report["force_9"]["pass"]
    assert report["force_command_7"]["pass"]
    assert report["force_command_9"]["pass"]


def test_first_sample_of_each_phase_is_excluded_for_settling():
    metrics = AcceptanceMetrics(settling_samples=1)
    metrics.add(passing_sample(measured_force_7=30.0))
    metrics.add(passing_sample(measured_force_7=15.0))

    report = metrics.report(scenario_complete=False)

    assert report["force_7"]["pass"]
    assert report["sample_count"] == 1


def test_settling_restarts_when_same_phase_name_returns_later():
    metrics = AcceptanceMetrics(settling_samples=1)
    metrics.add(passing_sample(phase="RETURN_NEUTRAL", measured_force_7=30.0))
    metrics.add(passing_sample(phase="RETURN_NEUTRAL", measured_force_7=15.0))
    metrics.add(passing_sample(phase="YAW_ONLY", measured_force_7=15.0))
    metrics.add(passing_sample(phase="RETURN_NEUTRAL", measured_force_7=30.0))

    assert len(metrics.samples) == 1
    assert metrics.samples[0].measured_force_7 == 15.0


def test_equal_accuracy_requires_fifty_percent_reduction_in_both_axes():
    metrics = AcceptanceMetrics(settling_samples=0)
    metrics.add(
        passing_sample(
            phase="PITCH_ONLY",
            phase_arm_travel_7_rad=1.0,
            phase_arm_travel_9_rad=0.49,
        )
    )
    metrics.add(
        passing_sample(
            phase="YAW_ONLY",
            phase_arm_travel_7_rad=0.8,
            phase_arm_travel_9_rad=0.39,
        )
    )
    metrics.add(
        passing_sample(
            phase="CHALLENGE_PITCH_ONLY",
            phase_arm_travel_7_rad=1.2,
            phase_arm_travel_9_rad=0.4,
        )
    )

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


def test_challenge_passes_when_green_completes_and_red_collision_stops():
    metrics = AcceptanceMetrics(settling_samples=0)
    metrics.add(passing_sample(phase="PITCH_ONLY"))
    metrics.add(passing_sample(phase="YAW_ONLY"))
    metrics.add(
        passing_sample(
            phase="CHALLENGE_PITCH_ONLY",
            nonprobe_force_7_n=2.4,
            collision_stop_7=True,
            completed_7=False,
            completed_9=True,
            phase_arm_travel_7_rad=0.3,
            phase_arm_travel_9_rad=0.2,
        )
    )

    report = metrics.report(scenario_complete=True)

    assert report["collision_challenge"]["red_collision_stop"]
    assert report["collision_challenge"]["green_completed"]
    assert report["collision_challenge"]["pass"]


def test_challenge_fails_if_green_collides():
    metrics = AcceptanceMetrics(settling_samples=0)
    metrics.add(passing_sample(phase="PITCH_ONLY"))
    metrics.add(passing_sample(phase="YAW_ONLY"))
    metrics.add(
        passing_sample(
            phase="CHALLENGE_PITCH_ONLY",
            nonprobe_force_9_n=2.1,
            collision_stop_9=True,
        )
    )

    report = metrics.report(scenario_complete=True)

    assert not report["collision_challenge"]["green_collision_free"]
    assert not report["collision_challenge"]["pass"]
    assert not report["overall_pass"]


def test_post_latch_red_hold_is_excluded_only_from_command_and_identity_gates():
    metrics = AcceptanceMetrics(settling_samples=0)
    add_passing_scenario(metrics)
    metrics.add(
        passing_sample(
            phase="CHALLENGE_PITCH_ONLY",
            commanded_force_7=0.0,
            measured_force_7=0.0,
            references_identical=False,
            collision_stop_7=True,
        )
    )

    report = metrics.report(scenario_complete=True)

    assert report["force_command_7"]["pass"]
    assert report["identical_references"]["pass"]
    assert not report["force_7"]["pass"]
    assert not report["overall_pass"]


def test_green_force_command_remains_exact_after_red_collision_latch():
    metrics = AcceptanceMetrics(settling_samples=0)
    add_passing_scenario(metrics)
    metrics.add(
        passing_sample(
            phase="CHALLENGE_PITCH_ONLY",
            commanded_force_7=0.0,
            commanded_force_9=0.0,
            references_identical=False,
            collision_stop_7=True,
        )
    )

    report = metrics.report(scenario_complete=True)

    assert report["force_command_7"]["pass"]
    assert not report["force_command_9"]["pass"]
    assert not report["overall_pass"]


def test_pre_latch_reference_difference_remains_disqualifying():
    metrics = AcceptanceMetrics(settling_samples=0)
    add_passing_scenario(metrics)
    metrics.add(
        passing_sample(
            phase="CHALLENGE_PITCH_ONLY",
            references_identical=False,
            collision_stop_7=False,
        )
    )

    report = metrics.report(scenario_complete=True)

    assert not report["identical_references"]["pass"]
    assert not report["overall_pass"]


def test_missing_pre_latch_sample_cannot_satisfy_command_or_identity_gates():
    metrics = AcceptanceMetrics(settling_samples=0)
    metrics.add(
        passing_sample(
            phase="CHALLENGE_PITCH_ONLY",
            commanded_force_7=0.0,
            references_identical=False,
            collision_stop_7=True,
        )
    )

    report = metrics.report(scenario_complete=True)

    assert not report["force_command_7"]["pass"]
    assert not report["identical_references"]["pass"]
