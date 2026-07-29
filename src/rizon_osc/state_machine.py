"""Safety supervision independent from the nominal ultrasound task phase."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SafetyMode(str, Enum):
    TRACKING = "TRACKING"
    REACQUIRE = "REACQUIRE"
    INVALID_SURFACE = "INVALID_SURFACE"
    FORCE_HOLD = "FORCE_HOLD"


def phase_requires_contact(phase: str) -> bool:
    """Whether loss of contact should pause the nominal task clock."""
    return str(phase) not in ("APPROACH", "CONTACT_RAMP")


@dataclass(frozen=True)
class SupervisorState:
    mode: SafetyMode
    freeze_path: bool
    zero_force_command: bool
    reset_force_controller: bool
    contact_loss_duration: float
    max_contact_loss_duration: float


class ContactSupervisor:
    def __init__(
        self,
        *,
        contact_loss_limit: float = 0.1,
        reacquire_stable_time: float = 0.05,
        hard_force_limit: float = 35.0,
    ) -> None:
        self.contact_loss_limit = float(contact_loss_limit)
        self.reacquire_stable_time = float(reacquire_stable_time)
        self.hard_force_limit = float(hard_force_limit)
        self.reset()

    def reset(self) -> None:
        self.mode = SafetyMode.TRACKING
        self.contact_loss_duration = 0.0
        self.max_contact_loss_duration = 0.0
        self._stable_contact_duration = 0.0

    def update(
        self,
        *,
        dt: float,
        contact: bool,
        surface_valid: bool,
        measured_force: float,
        contact_phase: bool,
    ) -> SupervisorState:
        dt = max(0.0, float(dt))
        if not contact_phase:
            self.mode = SafetyMode.TRACKING
            self.contact_loss_duration = 0.0
            self._stable_contact_duration = 0.0
            return self._state(reset_force_controller=True)
        if not surface_valid:
            self.mode = SafetyMode.INVALID_SURFACE
            return self._state()
        if measured_force > self.hard_force_limit:
            self.mode = SafetyMode.FORCE_HOLD
            return self._state(zero_force_command=True)

        if contact:
            self.contact_loss_duration = 0.0
            if self.mode is SafetyMode.REACQUIRE:
                self._stable_contact_duration += dt
                if self._stable_contact_duration + 1.0e-12 >= self.reacquire_stable_time:
                    self.mode = SafetyMode.TRACKING
                    self._stable_contact_duration = 0.0
            else:
                self._stable_contact_duration = 0.0
                if self.mode in (SafetyMode.INVALID_SURFACE, SafetyMode.FORCE_HOLD):
                    self.mode = SafetyMode.TRACKING
        else:
            self._stable_contact_duration = 0.0
            self.contact_loss_duration += dt
            self.max_contact_loss_duration = max(
                self.max_contact_loss_duration, self.contact_loss_duration
            )
            if self.contact_loss_duration > self.contact_loss_limit:
                self.mode = SafetyMode.REACQUIRE
        return self._state()

    def _state(
        self,
        *,
        zero_force_command: bool = False,
        reset_force_controller: bool = False,
    ) -> SupervisorState:
        return SupervisorState(
            mode=self.mode,
            freeze_path=self.mode is not SafetyMode.TRACKING,
            zero_force_command=zero_force_command,
            reset_force_controller=reset_force_controller,
            contact_loss_duration=self.contact_loss_duration,
            max_contact_loss_duration=self.max_contact_loss_duration,
        )
