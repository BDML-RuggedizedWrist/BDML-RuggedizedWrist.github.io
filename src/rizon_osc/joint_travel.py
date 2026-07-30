"""Phase-local joint travel accounting for seven- and nine-joint robots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointTravel:
    phase: str
    arm_7_rad: float
    arm_9_rad: float
    wrist_9_rad: float

    @property
    def reduction_percent(self) -> float | None:
        if self.arm_7_rad <= 1.0e-9:
            return None
        return 100.0 * (1.0 - self.arm_9_rad / self.arm_7_rad)


class JointTravelTracker:
    """Accumulate absolute joint increments since the current phase began."""

    @staticmethod
    def _positions(
        joint_7: np.ndarray, joint_9: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        position_7 = np.asarray(joint_7, dtype=float)
        position_9 = np.asarray(joint_9, dtype=float)
        if position_7.shape != (7,):
            raise ValueError(
                f"expected 7-joint position shape (7,), got {position_7.shape}"
            )
        if position_9.shape != (9,):
            raise ValueError(
                f"expected 9-joint position shape (9,), got {position_9.shape}"
            )
        return position_7, position_9

    def begin_phase(
        self, phase: str, joint_7: np.ndarray, joint_9: np.ndarray
    ) -> None:
        position_7, position_9 = self._positions(joint_7, joint_9)
        self._phase = str(phase)
        self._previous_7 = position_7.copy()
        self._previous_9 = position_9.copy()
        self._arm_7 = self._arm_9 = self._wrist_9 = 0.0

    def update(self, joint_7: np.ndarray, joint_9: np.ndarray) -> None:
        current_7, current_9 = self._positions(joint_7, joint_9)
        delta_7 = np.abs(current_7 - self._previous_7)
        delta_9 = np.abs(current_9 - self._previous_9)
        self._arm_7 += float(delta_7[:7].sum())
        self._arm_9 += float(delta_9[:7].sum())
        self._wrist_9 += float(delta_9[7:].sum())
        self._previous_7 = current_7.copy()
        self._previous_9 = current_9.copy()

    def snapshot(self) -> JointTravel:
        return JointTravel(
            self._phase, self._arm_7, self._arm_9, self._wrist_9
        )
