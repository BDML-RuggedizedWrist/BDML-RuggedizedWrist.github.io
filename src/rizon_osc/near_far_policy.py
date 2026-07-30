"""Null-space posture targets for the independent near-to-far task."""

from __future__ import annotations

import numpy as np

from .near_far_trajectory import NearFarPhase


class NearFarRedundancyPolicy:
    """Hold the far-point arm posture while assigning orientation to J8/J9.

    This class only supplies ``nullspace_joint_pos_target`` to Isaac Lab's
    official OperationalSpaceController.  It never computes torques,
    pseudoinverses, Jacobian projectors, or joint commands.
    """

    def __init__(
        self,
        num_arm_joints: int = 7,
        *,
        wrist_bend_sign: float = -1.0,
    ) -> None:
        self.num_arm_joints = int(num_arm_joints)
        if self.num_arm_joints < 1:
            raise ValueError("num_arm_joints must be positive")
        if wrist_bend_sign not in (-1.0, 1.0):
            raise ValueError("wrist_bend_sign must be -1 or +1")
        self.wrist_bend_sign = float(wrist_bend_sign)
        self._initial_wrist: np.ndarray | None = None
        self._moving_arm_target: np.ndarray | None = None
        self._far_arm_target: np.ndarray | None = None

    @property
    def joint_count(self) -> int:
        return self.num_arm_joints + 2

    def _validate(self, joint_position: np.ndarray) -> np.ndarray:
        position = np.asarray(joint_position, dtype=np.float64)
        if position.shape != (self.joint_count,):
            raise ValueError(
                f"expected {self.joint_count} positions, got {position.shape}"
            )
        if not np.isfinite(position).all():
            raise ValueError("joint positions must be finite")
        return position

    def initialize(self, joint_position: np.ndarray) -> None:
        position = self._validate(joint_position)
        self._initial_wrist = position[-2:].copy()
        self._moving_arm_target = position[: self.num_arm_joints].copy()

    def begin_phase(
        self, phase: NearFarPhase | str, joint_position: np.ndarray
    ) -> None:
        if self._initial_wrist is None:
            raise RuntimeError("initialize must be called first")
        position = self._validate(joint_position)
        phase = NearFarPhase(phase)
        if phase in (
            NearFarPhase.APPROACH_NEAR,
            NearFarPhase.CONTACT_RAMP,
            NearFarPhase.SCAN_NEAR_TO_FAR,
        ):
            self._moving_arm_target = position[: self.num_arm_joints].copy()
        elif self._far_arm_target is None:
            self._far_arm_target = position[: self.num_arm_joints].copy()

    def target(
        self,
        current_joint_position: np.ndarray,
        *,
        phase: NearFarPhase | str,
        relative_pitch: float,
        relative_axial: float,
    ) -> np.ndarray:
        current_position = self._validate(current_joint_position)
        if self._initial_wrist is None or self._moving_arm_target is None:
            raise RuntimeError("initialize must be called first")
        phase = NearFarPhase(phase)
        fixed_point_phase = phase in (
            NearFarPhase.SETTLE_FAR,
            NearFarPhase.PITCH_ONLY,
            NearFarPhase.RETURN_PITCH,
            NearFarPhase.ROLL_ONLY,
            NearFarPhase.RETURN_ROLL,
            NearFarPhase.AXIAL_SLICE_90,
            NearFarPhase.HOLD_FINAL_SLICE,
        )
        arm_target = (
            self._far_arm_target
            if fixed_point_phase and self._far_arm_target is not None
            else current_position[: self.num_arm_joints]
        )
        pitch = float(relative_pitch)
        axial = float(relative_axial)
        if not np.isfinite([pitch, axial]).all():
            raise ValueError("relative orientation must be finite")
        target = np.empty(self.joint_count, dtype=np.float64)
        target[: self.num_arm_joints] = arm_target
        target[-2] = self._initial_wrist[0] + self.wrist_bend_sign * pitch
        target[-1] = self._initial_wrist[1] + axial
        return target
