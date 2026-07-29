"""Null-space targets supplied to Isaac Lab OperationalSpaceController."""

from __future__ import annotations

import numpy as np

PITCH_PHASES = {"PITCH_ONLY", "CHALLENGE_PITCH_ONLY"}
YAW_PHASES = {"YAW_ONLY"}


class RedundancyPolicy:
    """Prefer the distal wrist without implementing a second controller.

    The returned vector is only a posture target for Isaac Lab OSC's built-in
    null-space controller. Primary pose and wrench torque always comes from
    ``OperationalSpaceController.compute``.
    """

    def __init__(self, num_arm_joints: int = 7, num_wrist_joints: int = 2) -> None:
        if num_arm_joints < 1 or num_wrist_joints != 2:
            raise ValueError("the Rizon comparison requires arm joints plus a 2-DoF wrist")
        self.num_arm_joints = int(num_arm_joints)
        self.num_wrist_joints = int(num_wrist_joints)
        self._phase = ""
        self._phase_start: np.ndarray | None = None

    def begin_phase(self, phase: str, joint_position: np.ndarray) -> None:
        position = np.asarray(joint_position, dtype=np.float64)
        expected = self.num_arm_joints + self.num_wrist_joints
        if position.shape != (expected,):
            raise ValueError(f"expected {expected} joint positions, got {position.shape}")
        self._phase = str(phase)
        self._phase_start = position.copy()

    def target(
        self,
        current_joint_position: np.ndarray,
        *,
        relative_pitch: float,
        relative_yaw: float,
    ) -> np.ndarray:
        current = np.asarray(current_joint_position, dtype=np.float64)
        if self._phase_start is None:
            self.begin_phase(self._phase or "SURFACE_SCAN", current)
        assert self._phase_start is not None
        target = self._phase_start.copy()
        if self._phase in PITCH_PHASES:
            target[-2] = -float(relative_pitch)
            target[-1] = 0.0
        elif self._phase in YAW_PHASES:
            target[-2] = 0.0
            target[-1] = float(relative_yaw)
        else:
            target[-2:] = 0.0
        return target
