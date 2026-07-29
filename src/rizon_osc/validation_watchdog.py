"""Pure continuous-window watchdog for finite Isaac Lab validation runs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
import math

import numpy as np


CHALLENGE_PHASES = {
    "CHALLENGE_TRANSIT",
    "CHALLENGE_PITCH_ONLY",
    "RETURN_NEUTRAL",
}

_DECIMAL_ZERO = Decimal("0")
_LIMIT_DURATION = Decimal("0.10")
_SPEED_DURATION = Decimal("0.10")
_CONTACT_LOSS_DURATION = Decimal("0.10")
_FREEZE_WINDOW = Decimal("0.25")


def _readonly_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        array = np.full(array.shape, "<object>", dtype="<U8")
    data = array.tobytes(order="C")
    return np.frombuffer(data, dtype=array.dtype, count=array.size).reshape(
        array.shape
    )


def _is_finite(value: object) -> bool:
    try:
        return bool(np.isfinite(np.asarray(value)).all())
    except (TypeError, ValueError):
        return False


def _geometrically_at_least(value: float, threshold: float) -> bool:
    return value > threshold or math.isclose(
        value,
        threshold,
        rel_tol=0.0,
        abs_tol=8 * math.ulp(threshold),
    )


def _geometrically_less_than(value: float, threshold: float) -> bool:
    return value < threshold and not math.isclose(
        value,
        threshold,
        rel_tol=0.0,
        abs_tol=8 * math.ulp(threshold),
    )


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

    def __post_init__(self) -> None:
        for field in (
            "wrist_position_rad",
            "wrist_velocity_rad_s",
            "wrist_limits_rad",
            "contact_present",
            "measured_normal_force_n",
            "nonprobe_force_n",
            "target_position_m",
            "measured_position_m",
            "target_quaternion_wxyz",
            "measured_quaternion_wxyz",
        ):
            object.__setattr__(self, field, _readonly_array(getattr(self, field)))
        object.__setattr__(
            self,
            "finite_payloads",
            tuple(_readonly_array(value) for value in self.finite_payloads),
        )


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
            "max_overspeed_duration_s": list(self.max_overspeed_duration_s),
            "max_contact_loss_duration_s": list(
                self.max_contact_loss_duration_s
            ),
        }


@dataclass(frozen=True)
class _PoseRecord:
    time_s: Decimal
    target_position_m: np.ndarray
    measured_position_m: np.ndarray
    target_quaternion_wxyz: np.ndarray
    measured_quaternion_wxyz: np.ndarray


def _quaternion_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm <= 1.0e-12 or second_norm <= 1.0e-12:
        return math.inf
    unit_first = first / first_norm
    unit_second = second / second_norm
    if float(np.dot(unit_first, unit_second)) < 0.0:
        unit_second = -unit_second
    chord = float(np.linalg.norm(unit_first - unit_second))
    return 4.0 * math.asin(min(1.0, 0.5 * chord))


def _interpolate_quaternion(
    first: np.ndarray, second: np.ndarray, fraction: float
) -> np.ndarray:
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm <= 1.0e-12 or second_norm <= 1.0e-12:
        return np.zeros(4)
    unit_first = first / first_norm
    unit_second = second / second_norm
    dot = float(np.dot(unit_first, unit_second))
    if dot < 0.0:
        unit_second = -unit_second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        blended = unit_first + fraction * (unit_second - unit_first)
        return blended / np.linalg.norm(blended)
    angle = math.acos(dot)
    sine = math.sin(angle)
    return (
        math.sin((1.0 - fraction) * angle) / sine * unit_first
        + math.sin(fraction * angle) / sine * unit_second
    )


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
        self._time_s = _DECIMAL_ZERO
        self._reasons: list[str] = []
        self._first_failure_step: int | None = None
        self._challenge_started = False
        self._red_collision_stopped = False
        self._near_limit_s = [_DECIMAL_ZERO, _DECIMAL_ZERO]
        self._overspeed_s = [_DECIMAL_ZERO, _DECIMAL_ZERO]
        self._contact_loss_s = [_DECIMAL_ZERO, _DECIMAL_ZERO]
        self._max_near_limit_s = [_DECIMAL_ZERO, _DECIMAL_ZERO]
        self._max_overspeed_s = [_DECIMAL_ZERO, _DECIMAL_ZERO]
        self._max_contact_loss_s = [_DECIMAL_ZERO, _DECIMAL_ZERO]
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
                float(value) for value in self._max_near_limit_s
            ),
            max_overspeed_duration_s=tuple(
                float(value) for value in self._max_overspeed_s
            ),
            max_contact_loss_duration_s=tuple(
                float(value) for value in self._max_contact_loss_s
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
        try:
            dt = float(sample.dt_s)
        except (TypeError, ValueError):
            self._fail("nonfinite", sample.step)
            return self.snapshot()
        if not math.isfinite(dt):
            self._fail("nonfinite", sample.step)
            return self.snapshot()
        if dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        dt_decimal = Decimal(str(sample.dt_s))

        finite_arrays = (
            wrist_position,
            wrist_velocity,
            wrist_limits,
            contact_present,
            measured_force,
            nonprobe_force,
            target_position,
            measured_position,
            target_quaternion,
            measured_quaternion,
            *(np.asarray(value) for value in sample.finite_payloads),
        )
        if not all(_is_finite(value) for value in finite_arrays):
            self._fail("nonfinite", sample.step)
            return self.snapshot()

        self._time_s += dt_decimal
        self._challenge_started = (
            self._challenge_started or sample.phase in CHALLENGE_PHASES
        )
        self._red_collision_stopped = (
            self._red_collision_stopped or sample.red_collision_stop
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
        for joint_index, joint_name in enumerate(("j8", "j9")):
            if near_limit[joint_index]:
                self._near_limit_s[joint_index] += dt_decimal
            else:
                self._near_limit_s[joint_index] = _DECIMAL_ZERO
            if overspeed[joint_index]:
                self._overspeed_s[joint_index] += dt_decimal
            else:
                self._overspeed_s[joint_index] = _DECIMAL_ZERO
            self._max_near_limit_s[joint_index] = max(
                self._max_near_limit_s[joint_index],
                self._near_limit_s[joint_index],
            )
            self._max_overspeed_s[joint_index] = max(
                self._max_overspeed_s[joint_index],
                self._overspeed_s[joint_index],
            )
            if self._near_limit_s[joint_index] >= _LIMIT_DURATION:
                self._fail(
                    f"green_wrist_limit_{joint_name}", sample.step
                )
            if self._overspeed_s[joint_index] >= _SPEED_DURATION:
                self._fail(
                    f"green_wrist_speed_{joint_name}", sample.step
                )

        if sample.contact_required:
            for side_index in range(2):
                if contact_present[side_index]:
                    self._contact_loss_s[side_index] = _DECIMAL_ZERO
                else:
                    self._contact_loss_s[side_index] += dt_decimal
        else:
            self._contact_loss_s = [_DECIMAL_ZERO, _DECIMAL_ZERO]
        for side_index, side_name in enumerate(("7", "9")):
            self._max_contact_loss_s[side_index] = max(
                self._max_contact_loss_s[side_index],
                self._contact_loss_s[side_index],
            )
            if self._contact_loss_s[side_index] > _CONTACT_LOSS_DURATION:
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
            if side_index == 0 and self._red_collision_stopped:
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
            cutoff = self._time_s - _FREEZE_WINDOW
            while len(history) >= 2 and history[1].time_s <= cutoff:
                history.popleft()
            oldest = history[0]
            if oldest.time_s > cutoff:
                continue
            if oldest.time_s == cutoff:
                reference_target_position = oldest.target_position_m
                reference_measured_position = oldest.measured_position_m
                reference_target_quaternion = oldest.target_quaternion_wxyz
                reference_measured_quaternion = oldest.measured_quaternion_wxyz
            else:
                following = history[1]
                fraction = float(
                    (cutoff - oldest.time_s)
                    / (following.time_s - oldest.time_s)
                )
                reference_target_position = (
                    oldest.target_position_m
                    + fraction
                    * (following.target_position_m - oldest.target_position_m)
                )
                reference_measured_position = (
                    oldest.measured_position_m
                    + fraction
                    * (
                        following.measured_position_m
                        - oldest.measured_position_m
                    )
                )
                reference_target_quaternion = _interpolate_quaternion(
                    oldest.target_quaternion_wxyz,
                    following.target_quaternion_wxyz,
                    fraction,
                )
                reference_measured_quaternion = _interpolate_quaternion(
                    oldest.measured_quaternion_wxyz,
                    following.measured_quaternion_wxyz,
                    fraction,
                )
            target_translation = float(
                np.linalg.norm(
                    target_position[side_index]
                    - reference_target_position
                )
            )
            measured_translation = float(
                np.linalg.norm(
                    measured_position[side_index]
                    - reference_measured_position
                )
            )
            target_rotation = _quaternion_distance_rad(
                target_quaternion[side_index],
                reference_target_quaternion,
            )
            measured_rotation = _quaternion_distance_rad(
                measured_quaternion[side_index],
                reference_measured_quaternion,
            )
            if (
                _geometrically_at_least(
                    target_translation, self.TRANSLATION_COMMAND_M
                )
                and _geometrically_less_than(
                    measured_translation, self.TRANSLATION_RESPONSE_M
                )
            ):
                self._fail(
                    f"task_freeze_translation_{side_name}", sample.step
                )
            if (
                _geometrically_at_least(
                    target_rotation, self.ROTATION_COMMAND_RAD
                )
                and _geometrically_less_than(
                    measured_rotation, self.ROTATION_RESPONSE_RAD
                )
            ):
                self._fail(
                    f"task_freeze_rotation_{side_name}", sample.step
                )

        return self.snapshot()
