"""Lifecycle boundary for safely probing the authored supplemental wrist axes."""

from __future__ import annotations

from typing import Any, Protocol


class PrecheckArticulation(Protocol):
    """Minimal articulation writes required by one wrist precheck step."""

    def write_joint_position_to_sim_index(self, *, position: Any) -> None: ...

    def write_joint_velocity_to_sim_index(self, *, velocity: Any) -> None: ...

    def set_joint_effort_target_index(self, *, target: Any) -> None: ...

    def write_data_to_sim(self) -> None: ...


def stabilize_precheck_pair(
    *,
    red_robot: PrecheckArticulation,
    green_robot: PrecheckArticulation,
    red_position: Any,
    green_position: Any,
    red_zero_velocity: Any,
    green_zero_velocity: Any,
    red_zero_effort: Any,
    green_zero_effort: Any,
) -> None:
    """Commit deterministic position, velocity, and effort state for both sides."""
    robot_states = (
        (
            red_robot,
            red_position,
            red_zero_velocity,
            red_zero_effort,
        ),
        (
            green_robot,
            green_position,
            green_zero_velocity,
            green_zero_effort,
        ),
    )
    for robot, position, velocity, effort in robot_states:
        robot.write_joint_position_to_sim_index(position=position)
        robot.write_joint_velocity_to_sim_index(velocity=velocity)
        robot.set_joint_effort_target_index(target=effort)
    for robot, _, _, _ in robot_states:
        robot.write_data_to_sim()
