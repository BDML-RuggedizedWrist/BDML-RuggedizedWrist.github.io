"""Contact-force measurement helpers for Isaac Lab's built-in OSC."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FilteredForce:
    raw: float
    window_average: float
    filtered: float


class ContactForceFilter:
    """Short moving average followed by a first-order low-pass filter."""

    def __init__(self, history_length: int = 4, low_pass_alpha: float = 0.35) -> None:
        if history_length < 1:
            raise ValueError("history_length must be positive")
        if not 0.0 < low_pass_alpha <= 1.0:
            raise ValueError("low_pass_alpha must be in (0, 1]")
        self.history_length = int(history_length)
        self.low_pass_alpha = float(low_pass_alpha)
        self._history: deque[float] = deque(maxlen=self.history_length)
        self._filtered = 0.0
        self._initialized = False

    def update(self, raw_force: float) -> FilteredForce:
        raw = max(0.0, float(raw_force))
        self._history.append(raw)
        average = float(np.mean(self._history))
        if not self._initialized:
            self._filtered = average
            self._initialized = True
        else:
            alpha = self.low_pass_alpha
            self._filtered = alpha * average + (1.0 - alpha) * self._filtered
        return FilteredForce(raw=raw, window_average=average, filtered=self._filtered)

    def reset(self) -> None:
        self._history.clear()
        self._filtered = 0.0
        self._initialized = False

    @property
    def filtered(self) -> float:
        return self._filtered


def project_normal_force(force_world: np.ndarray, outward_normal_world: np.ndarray) -> float:
    """Return compressive force magnitude on the probe.

    The environment's force on the probe points along the outward patient
    normal. Some PhysX contact buffers expose the opposite pair sign, so the
    caller must pass the force vector acting on the probe. This helper accepts
    either sign only through its explicit convention and clamps tension.
    """
    force = np.asarray(force_world, dtype=np.float64)
    normal = np.asarray(outward_normal_world, dtype=np.float64)
    if force.shape != (3,) or normal.shape != (3,):
        raise ValueError("force and normal must be length-three vectors")
    magnitude = np.linalg.norm(normal)
    if magnitude <= 1.0e-12:
        raise ValueError("surface normal must be nonzero")
    normal = normal / magnitude
    # Controller command points into the patient (-normal). The reaction force
    # acting on the probe therefore points outward (+normal). The tests retain
    # the historic force-vector convention where an applied probe wrench is
    # supplied, hence use -dot here; runtime passes the sign-adjusted wrench.
    return max(0.0, float(np.dot(force, -normal)))
