"""Pure text formatting for the persistent OSC comparison HUD."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HudSnapshot:
    phase: str
    force_7_n: float
    force_9_n: float
    arm_7_rad: float
    arm_9_rad: float
    wrist_9_rad: float
    reduction_percent: float | None
    collision_7: str
    collision_9: str


def format_hud(snapshot: HudSnapshot) -> str:
    """Format the comparison state without requiring an Isaac Sim UI."""
    reduction = (
        f"{snapshot.reduction_percent:.1f}%"
        if snapshot.reduction_percent is not None
        else "hidden until equal-accuracy gate"
    )
    return (
        f"Phase: {snapshot.phase}\n"
        "Measured normal force 7 / 9: "
        f"{snapshot.force_7_n:.1f} / {snapshot.force_9_n:.1f} N\n"
        "Phase arm travel 7 / 9: "
        f"{snapshot.arm_7_rad:.3f} / {snapshot.arm_9_rad:.3f} rad\n"
        f"9-DoF distal wrist travel: {snapshot.wrist_9_rad:.3f} rad\n"
        f"Validated main-arm reduction: {reduction}\n"
        f"Collision 7 / 9: {snapshot.collision_7} / {snapshot.collision_9}"
    )
