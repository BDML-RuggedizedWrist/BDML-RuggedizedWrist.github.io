"""Null-space targets supplied to Isaac Lab OperationalSpaceController."""

from __future__ import annotations

import numpy as np


class RedundancyPolicy:
    """Provide Scheme A posture targets without computing joint efforts."""

    def __init__(
        self, num_arm_joints: int = 7, num_wrist_joints: int = 2
    ) -> None:
        if num_arm_joints < 1 or num_wrist_joints != 2:
            raise ValueError(
                "the Rizon comparison requires arm joints plus a 2-DoF wrist"
            )
        self.num_arm_joints = int(num_arm_joints)
        self.num_wrist_joints = int(num_wrist_joints)
        self._phase = ""
        self._run_initial_wrist: np.ndarray | None = None
        self._phase_start_arm: np.ndarray | None = None

    @property
    def _expected_joint_count(self) -> int:
        return self.num_arm_joints + self.num_wrist_joints

    def _position(self, joint_position: np.ndarray) -> np.ndarray:
        position = np.asarray(joint_position, dtype=np.float64)
        if position.shape != (self._expected_joint_count,):
            raise ValueError(
                f"expected {self._expected_joint_count} joint positions, "
                f"got {position.shape}"
            )
        if not np.isfinite(position).all():
            raise ValueError("joint positions must be finite")
        return position

    def initialize_run(self, joint_position: np.ndarray) -> None:
        if self._run_initial_wrist is not None:
            raise RuntimeError("run initial wrist baseline is already initialized")
        position = self._position(joint_position)
        self._run_initial_wrist = position[self.num_arm_joints :].copy()

    def begin_phase(self, phase: str, joint_position: np.ndarray) -> None:
        if self._run_initial_wrist is None:
            raise RuntimeError("initialize_run must be called before begin_phase")
        position = self._position(joint_position)
        self._phase = str(phase)
        self._phase_start_arm = position[: self.num_arm_joints].copy()

    def target(
        self,
        current_joint_position: np.ndarray,
        *,
        relative_pitch: float,
        relative_yaw: float,
    ) -> np.ndarray:
        current = self._position(current_joint_position)
        del current
        if self._run_initial_wrist is None:
            raise RuntimeError("initialize_run must be called before target")
        if self._phase_start_arm is None:
            raise RuntimeError("begin_phase must be called before target")
        pitch = float(relative_pitch)
        yaw = float(relative_yaw)
        if not np.isfinite([pitch, yaw]).all():
            raise ValueError("relative pitch and yaw must be finite")

        target = np.empty(self._expected_joint_count, dtype=np.float64)
        target[: self.num_arm_joints] = self._phase_start_arm
        with np.errstate(over="ignore"):
            target[self.num_arm_joints] = self._run_initial_wrist[0] - pitch
            target[self.num_arm_joints + 1] = self._run_initial_wrist[1] + yaw
        if not np.isfinite(target).all():
            raise ValueError("target joint positions must be finite")
        return target
