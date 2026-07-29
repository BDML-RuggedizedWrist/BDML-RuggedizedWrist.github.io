"""Analytical task-space trajectory constrained to an ultrasound surface."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

import numpy as np

from .surface_model import SurfaceMap


class Phase(str, Enum):
    APPROACH = "APPROACH"
    CONTACT_RAMP = "CONTACT_RAMP"
    SURFACE_SCAN = "SURFACE_SCAN"
    PITCH_ONLY = "PITCH_ONLY"
    RETURN_NEUTRAL = "RETURN_NEUTRAL"
    YAW_ONLY = "YAW_ONLY"


@dataclass(frozen=True)
class TaskReference:
    phase: Phase
    position: np.ndarray
    quaternion: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray
    angular_acceleration: np.ndarray
    contact_point: np.ndarray
    surface_normal: np.ndarray
    probe_acoustic_axis: np.ndarray
    relative_rpy: np.ndarray
    normal_force: float
    valid: bool = True

    @property
    def pose(self) -> np.ndarray:
        return np.concatenate((self.position, self.quaternion))

    @property
    def twist(self) -> np.ndarray:
        return np.concatenate((self.linear_velocity, self.angular_velocity))

    @property
    def acceleration(self) -> np.ndarray:
        return np.concatenate((self.linear_acceleration, self.angular_acceleration))


def quintic_progress(u: float) -> tuple[float, float, float]:
    """Return quintic position and derivatives with respect to normalized time."""
    u = float(np.clip(u, 0.0, 1.0))
    position = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    velocity = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    acceleration = 60.0 * u - 180.0 * u**2 + 120.0 * u**3
    return position, velocity, acceleration


def quaternion_from_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a sign-canonical XYZW quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
                0.25 * s,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    0.25 * s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                    (matrix[2, 1] - matrix[1, 2]) / s,
                ]
            )
        elif index == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    0.25 * s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                    (matrix[0, 2] - matrix[2, 0]) / s,
                ]
            )
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                    0.25 * s,
                    (matrix[1, 0] - matrix[0, 1]) / s,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def rotation_matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Convert an XYZW quaternion to a rotation matrix."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _angular_velocity(rotation_minus: np.ndarray, rotation_plus: np.ndarray, dt: float) -> np.ndarray:
    rotation_rate = (rotation_plus - rotation_minus) / (2.0 * dt)
    omega_skew = rotation_rate @ ((rotation_plus + rotation_minus) * 0.5).T
    return np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])


class SurfaceTrajectory:
    """One upper-torso scan followed by repeating fixed-point reorientation."""

    def __init__(
        self,
        surface: SurfaceMap,
        *,
        scan_start_xy: tuple[float, float] | None = None,
        scan_end_xy: tuple[float, float] | None = None,
        approach_duration: float = 1.0,
        contact_ramp_duration: float = 0.5,
        scan_duration: float = 4.0,
        pitch_duration: float = 1.0,
        neutral_duration: float = 0.5,
        yaw_duration: float = 1.0,
        approach_clearance: float = 0.005,
        target_force: float = 15.0,
        reorientation_angle: float = math.radians(25.0),
        derivative_dt: float = 1.0e-3,
    ) -> None:
        durations = (
            approach_duration,
            contact_ramp_duration,
            scan_duration,
            pitch_duration,
            neutral_duration,
            yaw_duration,
        )
        if any(value < 0.0 for value in durations) or scan_duration <= 0.0:
            raise ValueError("trajectory durations must be nonnegative and scan duration positive")
        self.surface = surface
        if scan_start_xy is None or scan_end_xy is None:
            center_x = 0.5 * (surface.x_grid[0] + surface.x_grid[-1])
            y_span = surface.y_grid[-1] - surface.y_grid[0]
            scan_start_xy = (
                center_x,
                surface.y_grid[0] + 0.1 * y_span,
            ) if scan_start_xy is None else scan_start_xy
            scan_end_xy = (
                center_x,
                surface.y_grid[-1] - 0.1 * y_span,
            ) if scan_end_xy is None else scan_end_xy
        self.scan_start_xy = np.asarray(scan_start_xy, dtype=np.float64)
        self.scan_end_xy = np.asarray(scan_end_xy, dtype=np.float64)
        self.approach_duration = float(approach_duration)
        self.contact_ramp_duration = float(contact_ramp_duration)
        self.scan_duration = float(scan_duration)
        self.pitch_duration = float(pitch_duration)
        self.neutral_duration = float(neutral_duration)
        self.yaw_duration = float(yaw_duration)
        self.approach_clearance = float(approach_clearance)
        self.target_force = float(target_force)
        self.reorientation_angle = float(reorientation_angle)
        self.derivative_dt = float(derivative_dt)
        self._last_safe: TaskReference | None = None

    def reference(self, time_seconds: float) -> TaskReference:
        """Return a reference, holding the most recent safe one if geometry is invalid."""
        time_seconds = max(0.0, float(time_seconds))
        raw = self._raw_reference(time_seconds)
        if raw is None:
            if self._last_safe is None:
                raise ValueError("first trajectory reference lies outside the valid surface")
            return replace(self._last_safe, valid=False)

        velocity, acceleration, angular_velocity, angular_acceleration = self._derivatives(time_seconds, raw)
        reference = replace(
            raw,
            linear_velocity=velocity,
            linear_acceleration=acceleration,
            angular_velocity=angular_velocity,
            angular_acceleration=angular_acceleration,
        )
        self._last_safe = reference
        return reference

    def _raw_reference(self, time_seconds: float) -> TaskReference | None:
        phase, progress, relative_rpy, normal_force = self._phase_values(time_seconds)
        if phase in (Phase.APPROACH, Phase.CONTACT_RAMP):
            xy = self.scan_start_xy
        elif phase is Phase.SURFACE_SCAN:
            scan_progress, _, _ = quintic_progress(progress)
            xy = self.scan_start_xy + scan_progress * (self.scan_end_xy - self.scan_start_xy)
        else:
            xy = self.scan_end_xy

        sample = self.surface.query(float(xy[0]), float(xy[1]))
        if not sample.valid:
            return None

        normal = sample.normal
        planar_tangent = np.array(
            [
                self.scan_end_xy[0] - self.scan_start_xy[0],
                self.scan_end_xy[1] - self.scan_start_xy[1],
                0.0,
            ]
        )
        tangent = planar_tangent - np.dot(planar_tangent, normal) * normal
        if np.linalg.norm(tangent) < 1.0e-9:
            tangent = np.array([0.0, 1.0, 0.0]) - normal[1] * normal
        tangent /= np.linalg.norm(tangent)
        lateral = np.cross(tangent, normal)
        lateral /= np.linalg.norm(lateral)
        tangent = np.cross(normal, lateral)
        base_rotation = np.column_stack((lateral, tangent, normal))
        rotation = base_rotation @ _rotation_from_rpy(*relative_rpy)
        quaternion = quaternion_from_rotation_matrix(rotation)

        position = sample.point.copy()
        if phase is Phase.APPROACH:
            position += self.approach_clearance * normal
        return TaskReference(
            phase=phase,
            position=position,
            quaternion=quaternion,
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            linear_acceleration=np.zeros(3),
            angular_acceleration=np.zeros(3),
            contact_point=sample.point.copy(),
            surface_normal=normal.copy(),
            probe_acoustic_axis=-rotation[:, 2],
            relative_rpy=relative_rpy,
            normal_force=normal_force,
            valid=True,
        )

    def _phase_values(
        self, time_seconds: float
    ) -> tuple[Phase, float, np.ndarray, float]:
        approach_end = self.approach_duration
        ramp_end = approach_end + self.contact_ramp_duration
        scan_end = ramp_end + self.scan_duration

        if time_seconds < approach_end:
            return Phase.APPROACH, 0.0, np.zeros(3), 0.0
        if time_seconds < ramp_end:
            u = 1.0 if self.contact_ramp_duration == 0.0 else (
                time_seconds - approach_end
            ) / self.contact_ramp_duration
            force_progress, _, _ = quintic_progress(u)
            return Phase.CONTACT_RAMP, u, np.zeros(3), self.target_force * force_progress
        if time_seconds <= scan_end:
            u = (time_seconds - ramp_end) / self.scan_duration
            return Phase.SURFACE_SCAN, u, np.zeros(3), self.target_force

        cycle_duration = (
            self.pitch_duration + 2.0 * self.neutral_duration + self.yaw_duration
        )
        if cycle_duration <= 0.0:
            return Phase.RETURN_NEUTRAL, 0.0, np.zeros(3), self.target_force
        cycle_time = (time_seconds - scan_end) % cycle_duration
        if cycle_time < self.pitch_duration:
            u = cycle_time / max(self.pitch_duration, 1.0e-12)
            pulse = math.sin(math.pi * u) ** 2
            return (
                Phase.PITCH_ONLY,
                u,
                np.array([0.0, self.reorientation_angle * pulse, 0.0]),
                self.target_force,
            )
        cycle_time -= self.pitch_duration
        if cycle_time < self.neutral_duration:
            return Phase.RETURN_NEUTRAL, 0.0, np.zeros(3), self.target_force
        cycle_time -= self.neutral_duration
        if cycle_time < self.yaw_duration:
            u = cycle_time / max(self.yaw_duration, 1.0e-12)
            pulse = math.sin(math.pi * u) ** 2
            return (
                Phase.YAW_ONLY,
                u,
                np.array([0.0, 0.0, self.reorientation_angle * pulse]),
                self.target_force,
            )
        return Phase.RETURN_NEUTRAL, 0.0, np.zeros(3), self.target_force

    def _derivatives(
        self, time_seconds: float, center: TaskReference
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Quintic scan endpoints are known exactly and should not acquire a
        # numerical derivative from a neighboring phase.
        ramp_end = self.approach_duration + self.contact_ramp_duration
        scan_end = ramp_end + self.scan_duration
        if center.phase is Phase.SURFACE_SCAN and (
            abs(time_seconds - ramp_end) < 1.0e-12
            or abs(time_seconds - scan_end) < 1.0e-12
        ):
            return (np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))

        dt = self.derivative_dt
        minus = self._raw_reference(max(0.0, time_seconds - dt))
        plus = self._raw_reference(time_seconds + dt)
        if minus is None or plus is None:
            return (np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
        linear_velocity = (plus.position - minus.position) / (2.0 * dt)
        linear_acceleration = (plus.position - 2.0 * center.position + minus.position) / (dt**2)
        rotation_minus = rotation_matrix_from_quaternion(minus.quaternion)
        rotation_center = rotation_matrix_from_quaternion(center.quaternion)
        rotation_plus = rotation_matrix_from_quaternion(plus.quaternion)
        angular_velocity = _angular_velocity(rotation_minus, rotation_plus, dt)
        omega_minus = _angular_velocity(rotation_minus, rotation_center, dt * 0.5)
        omega_plus = _angular_velocity(rotation_center, rotation_plus, dt * 0.5)
        angular_acceleration = (omega_plus - omega_minus) / dt
        return linear_velocity, linear_acceleration, angular_velocity, angular_acceleration
