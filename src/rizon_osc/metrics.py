"""Acceptance metrics for the 7-DoF/9-DoF comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MetricSample:
    phase: str
    commanded_force_7: float
    commanded_force_9: float
    measured_force_7: float
    measured_force_9: float
    normal_angle_7_deg: float
    normal_angle_9_deg: float
    orientation_error_7_deg: float
    orientation_error_9_deg: float
    tangent_error_7_m: float
    tangent_error_9_m: float
    contact_loss_7_s: float
    contact_loss_9_s: float
    arm_travel_7_rad: float
    arm_travel_9_rad: float
    wrist_travel_9_rad: float
    static_drift_m: float
    references_identical: bool


class AcceptanceMetrics:
    """Aggregate evidence without claiming an advantage at unequal accuracy."""

    def __init__(
        self,
        *,
        settling_samples: int = 1,
        force_target: float = 15.0,
        force_tolerance: float = 1.5,
        normal_angle_limit_deg: float = 3.0,
        orientation_error_limit_deg: float = 5.0,
        contact_loss_limit_s: float = 0.1,
        tangent_error_limit_m: float = 0.01,
        static_drift_limit_m: float = 1.0e-5,
    ) -> None:
        self.settling_samples = max(0, int(settling_samples))
        self.force_target = float(force_target)
        self.force_tolerance = float(force_tolerance)
        self.normal_angle_limit_deg = float(normal_angle_limit_deg)
        self.orientation_error_limit_deg = float(orientation_error_limit_deg)
        self.contact_loss_limit_s = float(contact_loss_limit_s)
        self.tangent_error_limit_m = float(tangent_error_limit_m)
        self.static_drift_limit_m = float(static_drift_limit_m)
        self.samples: list[MetricSample] = []
        self._seen_phases: set[str] = set()
        self._active_phase: str | None = None
        self._active_phase_sample_count = 0

    def add(self, sample: MetricSample) -> None:
        self._seen_phases.add(sample.phase)
        if sample.phase != self._active_phase:
            self._active_phase = sample.phase
            self._active_phase_sample_count = 0
        phase_count = self._active_phase_sample_count
        self._active_phase_sample_count += 1
        if phase_count < self.settling_samples:
            return
        self.samples.append(sample)

    def report(self, *, scenario_complete: bool = False) -> dict[str, Any]:
        required_phases = {
            "SURFACE_SCAN",
            "PITCH_ONLY",
            "RETURN_NEUTRAL",
            "YAW_ONLY",
        }
        missing_phases = sorted(required_phases - self._seen_phases)
        phase_coverage = not missing_phases
        if not self.samples:
            return {
                "overall_pass": False,
                "reason": "no post-settling metric samples",
                "scenario_complete": {"pass": bool(scenario_complete)},
                "phase_coverage": {
                    "pass": phase_coverage,
                    "missing": missing_phases,
                },
            }

        force_error_7 = max(
            abs(sample.measured_force_7 - self.force_target) for sample in self.samples
        )
        force_error_9 = max(
            abs(sample.measured_force_9 - self.force_target) for sample in self.samples
        )
        command_error_7 = max(
            abs(sample.commanded_force_7 - self.force_target)
            for sample in self.samples
        )
        command_error_9 = max(
            abs(sample.commanded_force_9 - self.force_target)
            for sample in self.samples
        )
        # Reorientation deliberately tilts the probe. The <3 degree gate is
        # therefore evaluated over normal-aligned surface scanning only.
        normal_samples = [
            sample
            for sample in self.samples
            if sample.phase in ("CONTACT_RAMP", "SURFACE_SCAN")
        ] or self.samples
        max_normal_7 = max(sample.normal_angle_7_deg for sample in normal_samples)
        max_normal_9 = max(sample.normal_angle_9_deg for sample in normal_samples)
        max_orientation_error_7 = max(
            sample.orientation_error_7_deg for sample in self.samples
        )
        max_orientation_error_9 = max(
            sample.orientation_error_9_deg for sample in self.samples
        )
        max_tangent_7 = max(sample.tangent_error_7_m for sample in self.samples)
        max_tangent_9 = max(sample.tangent_error_9_m for sample in self.samples)
        max_loss_7 = max(sample.contact_loss_7_s for sample in self.samples)
        max_loss_9 = max(sample.contact_loss_9_s for sample in self.samples)
        max_static_drift = max(sample.static_drift_m for sample in self.samples)
        references_identical = all(sample.references_identical for sample in self.samples)
        final = self.samples[-1]

        checks: dict[str, Any] = {
            "force_command_7": {
                "pass": command_error_7 <= 1.0e-6,
                "max_abs_error_n": command_error_7,
            },
            "force_command_9": {
                "pass": command_error_9 <= 1.0e-6,
                "max_abs_error_n": command_error_9,
            },
            "force_7": {
                "pass": force_error_7 <= self.force_tolerance,
                "max_abs_error_n": force_error_7,
            },
            "force_9": {
                "pass": force_error_9 <= self.force_tolerance,
                "max_abs_error_n": force_error_9,
            },
            "normal_7": {
                "pass": max_normal_7 < self.normal_angle_limit_deg,
                "max_angle_deg": max_normal_7,
            },
            "normal_9": {
                "pass": max_normal_9 < self.normal_angle_limit_deg,
                "max_angle_deg": max_normal_9,
            },
            "orientation_7": {
                "pass": max_orientation_error_7 <= self.orientation_error_limit_deg,
                "max_error_deg": max_orientation_error_7,
            },
            "orientation_9": {
                "pass": max_orientation_error_9 <= self.orientation_error_limit_deg,
                "max_error_deg": max_orientation_error_9,
            },
            "contact_7": {
                "pass": max_loss_7 <= self.contact_loss_limit_s,
                "max_loss_s": max_loss_7,
            },
            "contact_9": {
                "pass": max_loss_9 <= self.contact_loss_limit_s,
                "max_loss_s": max_loss_9,
            },
            "static_assets": {
                "pass": max_static_drift <= self.static_drift_limit_m,
                "max_drift_m": max_static_drift,
            },
            "identical_references": {"pass": references_identical},
            "phase_coverage": {
                "pass": phase_coverage,
                "missing": missing_phases,
            },
            "scenario_complete": {"pass": bool(scenario_complete)},
        }
        accuracy_gate = (
            max_tangent_7 <= self.tangent_error_limit_m
            and max_tangent_9 <= self.tangent_error_limit_m
            and checks["force_command_7"]["pass"]
            and checks["force_command_9"]["pass"]
            and checks["force_7"]["pass"]
            and checks["force_9"]["pass"]
            and checks["normal_7"]["pass"]
            and checks["normal_9"]["pass"]
            and checks["orientation_7"]["pass"]
            and checks["orientation_9"]["pass"]
            and checks["contact_7"]["pass"]
            and checks["contact_9"]["pass"]
            and checks["static_assets"]["pass"]
            and references_identical
            and phase_coverage
            and scenario_complete
        )
        if accuracy_gate and final.arm_travel_7_rad > 1.0e-9:
            reduction_percent = 100.0 * (
                1.0 - final.arm_travel_9_rad / final.arm_travel_7_rad
            )
            reduction_visible = True
        else:
            reduction_percent = None
            reduction_visible = False
        checks["main_arm_reduction"] = {
            "visible": reduction_visible,
            "percent": reduction_percent,
            "arm_travel_7_rad": final.arm_travel_7_rad,
            "arm_travel_9_rad": final.arm_travel_9_rad,
            "accuracy_gate": accuracy_gate,
        }
        checks["wrist_motion_9_rad"] = final.wrist_travel_9_rad
        checks["tangent_error_7_max_m"] = max_tangent_7
        checks["tangent_error_9_max_m"] = max_tangent_9
        required = [
            checks["force_command_7"]["pass"],
            checks["force_command_9"]["pass"],
            checks["force_7"]["pass"],
            checks["force_9"]["pass"],
            checks["normal_7"]["pass"],
            checks["normal_9"]["pass"],
            checks["orientation_7"]["pass"],
            checks["orientation_9"]["pass"],
            checks["contact_7"]["pass"],
            checks["contact_9"]["pass"],
            checks["static_assets"]["pass"],
            checks["identical_references"]["pass"],
            checks["phase_coverage"]["pass"],
            checks["scenario_complete"]["pass"],
            accuracy_gate,
            reduction_visible and reduction_percent is not None and reduction_percent > 0.0,
        ]
        checks["overall_pass"] = all(required)
        checks["sample_count"] = len(self.samples)
        checks["last_sample"] = asdict(final)
        return checks
