"""Pure state for a latched non-probe patient-collision stop."""

from dataclasses import dataclass
from enum import Enum


class CollisionLevel(str, Enum):
    CONTACT_OK = "CONTACT OK"
    NEAR_COLLISION = "NEAR COLLISION"
    COLLISION_STOP = "COLLISION STOP"


@dataclass(frozen=True)
class CollisionSnapshot:
    level: CollisionLevel
    current_force_n: float
    peak_force_n: float
    freeze_path: bool


class CollisionMonitor:
    def __init__(
        self, near_threshold_n: float = 0.5, stop_threshold_n: float = 2.0
    ) -> None:
        if not 0.0 <= near_threshold_n < stop_threshold_n:
            raise ValueError("collision thresholds must satisfy 0 <= near < stop")
        self.near_threshold_n = float(near_threshold_n)
        self.stop_threshold_n = float(stop_threshold_n)
        self._latched = False
        self._peak = 0.0

    def update(self, force_n: float) -> CollisionSnapshot:
        force = float(force_n)
        if force < 0.0:
            raise ValueError("collision force must be nonnegative")
        self._peak = max(self._peak, force)
        self._latched = self._latched or force >= self.stop_threshold_n
        if self._latched:
            level = CollisionLevel.COLLISION_STOP
        elif force >= self.near_threshold_n:
            level = CollisionLevel.NEAR_COLLISION
        else:
            level = CollisionLevel.CONTACT_OK
        return CollisionSnapshot(level, force, self._peak, self._latched)
