"""Near-to-far ultrasound scan followed by fixed-point slice reorientation."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
import math

import numpy as np

from .surface_model import SurfaceMap
from .trajectory import (
    TaskReference,
    _angular_velocity,
    _rotation_from_rpy,
    quaternion_from_rotation_matrix,
    quintic_progress,
    rotation_matrix_from_quaternion,
)


class NearFarPhase(str, Enum):
    """Phases of the independent near-to-far comparison."""

    APPROACH_NEAR = "APPROACH_NEAR"
    CONTACT_RAMP = "CONTACT_RAMP"
    SCAN_NEAR_TO_FAR = "SCAN_NEAR_TO_FAR"
    SETTLE_FAR = "SETTLE_FAR"
    PITCH_ONLY = "PITCH_ONLY"
    RETURN_PITCH = "RETURN_PITCH"
    ROLL_ONLY = "ROLL_ONLY"
    RETURN_ROLL = "RETURN_ROLL"
    AXIAL_SLICE_90 = "AXIAL_SLICE_90"
    HOLD_FINAL_SLICE = "HOLD_FINAL_SLICE"


class NearFarTrajectory:
    """Scan toward the far torso end, then reorient at one fixed contact point.

    ``relative_rpy`` is expressed in the surface-aligned OSC task frame.  The
    pitch and axial-slice phases are intentionally disjoint so only one
    orientation coordinate changes at a time.
    """

    def __init__(
        self,
        surface: SurfaceMap,
        *,
        near_xy: tuple[float, float],
        far_xy: tuple[float, float],
        orientation_direction_xy: tuple[float, float] | None = None,
        approach_duration: float = 0.8,
        contact_ramp_duration: float = 0.8,
        scan_duration: float = 2.5,
        settle_duration: float = 0.4,
        pitch_duration: float = 1.5,
        return_pitch_duration: float = 0.8,
        axial_slice_duration: float = 2.0,
        pitch_angle: float = math.radians(-35.0),
        bend_axis: str = "pitch",
        axial_slice_angle: float = math.radians(90.0),
        approach_clearance: float = 0.005,
        contact_preload: float = 0.001,
        target_force: float = 15.0,
        derivative_dt: float = 1.0e-3,
    ) -> None:
        durations = (
            approach_duration,
            contact_ramp_duration,
            scan_duration,
            settle_duration,
            pitch_duration,
            return_pitch_duration,
            axial_slice_duration,
        )
        if any(value < 0.0 for value in durations) or scan_duration <= 0.0:
            raise ValueError(
                "durations must be nonnegative and scan_duration must be positive"
            )
        if not 0.0 < target_force <= 30.0:
            raise ValueError("target_force must be in (0, 30]")
        self.surface = surface
        self.near_xy = np.asarray(near_xy, dtype=np.float64)
        self.far_xy = np.asarray(far_xy, dtype=np.float64)
        if self.near_xy.shape != (2,) or self.far_xy.shape != (2,):
            raise ValueError("near_xy and far_xy must each contain x and y")
        if np.allclose(self.near_xy, self.far_xy):
            raise ValueError("near and far endpoints must be different")
        if orientation_direction_xy is None:
            direction = self.far_xy - self.near_xy
        else:
            direction = np.asarray(orientation_direction_xy, dtype=np.float64)
            if direction.shape != (2,):
                raise ValueError("orientation_direction_xy must contain x and y")
        if not np.isfinite(direction).all() or np.linalg.norm(direction) <= 1.0e-12:
            raise ValueError("orientation direction must be finite and nonzero")
        self.orientation_direction_xy = direction.copy()

        self.approach_duration = float(approach_duration)
        self.contact_ramp_duration = float(contact_ramp_duration)
        self.scan_duration = float(scan_duration)
        self.settle_duration = float(settle_duration)
        self.pitch_duration = float(pitch_duration)
        self.return_pitch_duration = float(return_pitch_duration)
        self.axial_slice_duration = float(axial_slice_duration)
        self.pitch_angle = float(pitch_angle)
        if bend_axis not in ("roll", "pitch"):
            raise ValueError("bend_axis must be 'roll' or 'pitch'")
        self.bend_axis = bend_axis
        self.axial_slice_angle = float(axial_slice_angle)
        self.approach_clearance = float(approach_clearance)
        self.contact_preload = float(contact_preload)
        self.target_force = float(target_force)
        self.derivative_dt = float(derivative_dt)
        self._last_safe: TaskReference | None = None

        for xy, name in ((self.near_xy, "near"), (self.far_xy, "far")):
            if not self.surface.query(float(xy[0]), float(xy[1])).valid:
                raise ValueError(f"{name} endpoint lies outside the valid surface")

    @property
    def total_duration(self) -> float:
        return (
            self.approach_duration
            + self.contact_ramp_duration
            + self.scan_duration
            + self.settle_duration
            + self.pitch_duration
            + self.return_pitch_duration
            + self.axial_slice_duration
        )

    @property
    def scan_start_time(self) -> float:
        return self.approach_duration + self.contact_ramp_duration

    @property
    def far_hold_start_time(self) -> float:
        return self.scan_start_time + self.scan_duration

    def reference(self, time_seconds: float) -> TaskReference:
        """Return the commanded surface reference, holding the last valid sample."""
        time_seconds = max(0.0, float(time_seconds))
        raw = self._raw_reference(time_seconds)
        if raw is None:
            if self._last_safe is None:
                raise ValueError("first trajectory reference is outside the surface")
            return replace(self._last_safe, valid=False)
        velocity, acceleration, angular_velocity, angular_acceleration = (
            self._derivatives(time_seconds, raw)
        )
        reference = replace(
            raw,
            linear_velocity=velocity,
            linear_acceleration=acceleration,
            angular_velocity=angular_velocity,
            angular_acceleration=angular_acceleration,
        )
        self._last_safe = reference
        return reference

    def _phase_values(
        self, time_seconds: float
    ) -> tuple[NearFarPhase, float, np.ndarray, float]:
        bend_index = 0 if self.bend_axis == "roll" else 1
        bend_phase = (
            NearFarPhase.ROLL_ONLY
            if self.bend_axis == "roll"
            else NearFarPhase.PITCH_ONLY
        )
        return_bend_phase = (
            NearFarPhase.RETURN_ROLL
            if self.bend_axis == "roll"
            else NearFarPhase.RETURN_PITCH
        )

        def bend_rpy(angle: float) -> np.ndarray:
            value = np.zeros(3, dtype=np.float64)
            value[bend_index] = angle
            return value

        approach_end = self.approach_duration
        ramp_end = approach_end + self.contact_ramp_duration
        scan_end = ramp_end + self.scan_duration
        settle_end = scan_end + self.settle_duration
        pitch_end = settle_end + self.pitch_duration
        return_end = pitch_end + self.return_pitch_duration
        axial_end = return_end + self.axial_slice_duration

        if time_seconds < approach_end:
            return NearFarPhase.APPROACH_NEAR, 0.0, np.zeros(3), 0.0
        if time_seconds < ramp_end:
            u = (time_seconds - approach_end) / max(
                self.contact_ramp_duration, 1.0e-12
            )
            force_progress, _, _ = quintic_progress(u)
            return (
                NearFarPhase.CONTACT_RAMP,
                u,
                np.zeros(3),
                self.target_force * force_progress,
            )
        if time_seconds <= scan_end:
            u = (time_seconds - ramp_end) / self.scan_duration
            return NearFarPhase.SCAN_NEAR_TO_FAR, u, np.zeros(3), self.target_force
        if self.settle_duration > 0.0 and time_seconds <= settle_end:
            u = (time_seconds - scan_end) / self.settle_duration
            return NearFarPhase.SETTLE_FAR, u, np.zeros(3), self.target_force
        if self.pitch_duration > 0.0 and time_seconds <= pitch_end:
            u = (time_seconds - settle_end) / self.pitch_duration
            progress, _, _ = quintic_progress(u)
            return (
                bend_phase,
                u,
                bend_rpy(self.pitch_angle * progress),
                self.target_force,
            )
        if self.return_pitch_duration > 0.0 and time_seconds < return_end:
            u = (time_seconds - pitch_end) / self.return_pitch_duration
            progress, _, _ = quintic_progress(u)
            return (
                return_bend_phase,
                u,
                bend_rpy(self.pitch_angle * (1.0 - progress)),
                self.target_force,
            )
        if self.axial_slice_duration > 0.0 and time_seconds <= axial_end:
            u = (time_seconds - return_end) / self.axial_slice_duration
            progress, _, _ = quintic_progress(u)
            return (
                NearFarPhase.AXIAL_SLICE_90,
                u,
                np.array([0.0, 0.0, self.axial_slice_angle * progress]),
                self.target_force,
            )
        return (
            NearFarPhase.HOLD_FINAL_SLICE,
            1.0,
            np.array([0.0, 0.0, self.axial_slice_angle]),
            self.target_force,
        )

    def _raw_reference(self, time_seconds: float) -> TaskReference | None:
        phase, progress, relative_rpy, normal_force = self._phase_values(time_seconds)
        if phase in (NearFarPhase.APPROACH_NEAR, NearFarPhase.CONTACT_RAMP):
            xy = self.near_xy
        elif phase is NearFarPhase.SCAN_NEAR_TO_FAR:
            scan_progress, _, _ = quintic_progress(progress)
            xy = self.near_xy + scan_progress * (self.far_xy - self.near_xy)
        else:
            xy = self.far_xy

        sample = self.surface.query(float(xy[0]), float(xy[1]))
        if not sample.valid:
            return None
        normal = sample.normal
        scan_direction = np.array(
            [
                self.orientation_direction_xy[0],
                self.orientation_direction_xy[1],
                0.0,
            ]
        )
        tangent = -scan_direction - np.dot(-scan_direction, normal) * normal
        if np.linalg.norm(tangent) < 1.0e-9:
            tangent = np.array([0.0, 1.0, 0.0]) - normal[1] * normal
        tangent /= np.linalg.norm(tangent)
        lateral = np.cross(tangent, normal)
        lateral /= np.linalg.norm(lateral)
        tangent = np.cross(normal, lateral)
        base_rotation = np.column_stack((lateral, tangent, normal))
        rotation = base_rotation @ _rotation_from_rpy(*relative_rpy)

        position = sample.point.copy()
        if phase is NearFarPhase.APPROACH_NEAR:
            position += self.approach_clearance * normal
        elif phase is NearFarPhase.CONTACT_RAMP:
            ramp_progress, _, _ = quintic_progress(progress)
            normal_offset = (
                (1.0 - ramp_progress) * self.approach_clearance
                - ramp_progress * self.contact_preload
            )
            position += normal_offset * normal

        return TaskReference(
            phase=phase,
            position=position,
            quaternion=quaternion_from_rotation_matrix(rotation),
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

    def _derivatives(
        self, time_seconds: float, center: TaskReference
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        scan_end = self.scan_start_time + self.scan_duration
        if center.phase is NearFarPhase.SCAN_NEAR_TO_FAR and (
            abs(time_seconds - self.scan_start_time) < 1.0e-12
            or abs(time_seconds - scan_end) < 1.0e-12
        ):
            zeros = np.zeros(3)
            return zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()

        dt = self.derivative_dt
        minus = self._raw_reference(max(0.0, time_seconds - dt))
        plus = self._raw_reference(time_seconds + dt)
        if minus is None or plus is None:
            zeros = np.zeros(3)
            return zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()
        linear_velocity = (plus.position - minus.position) / (2.0 * dt)
        linear_acceleration = (
            plus.position - 2.0 * center.position + minus.position
        ) / dt**2
        rotation_minus = rotation_matrix_from_quaternion(minus.quaternion)
        rotation_center = rotation_matrix_from_quaternion(center.quaternion)
        rotation_plus = rotation_matrix_from_quaternion(plus.quaternion)
        angular_velocity = _angular_velocity(rotation_minus, rotation_plus, dt)
        omega_minus = _angular_velocity(rotation_minus, rotation_center, 0.5 * dt)
        omega_plus = _angular_velocity(rotation_center, rotation_plus, 0.5 * dt)
        angular_acceleration = (omega_plus - omega_minus) / dt
        return (
            linear_velocity,
            linear_acceleration,
            angular_velocity,
            angular_acceleration,
        )
