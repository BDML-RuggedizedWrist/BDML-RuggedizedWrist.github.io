import pytest


try:
    from rizon_osc.wrist_axis_precheck import stabilize_precheck_pair
except ImportError:
    stabilize_precheck_pair = None


class RecordingArticulation:
    """Small in-memory articulation implementing the precheck write boundary."""

    def __init__(self):
        self.position = None
        self.velocity = None
        self.effort = None
        self.committed_states = []

    def write_joint_position_to_sim_index(self, *, position):
        self.position = position

    def write_joint_velocity_to_sim_index(self, *, velocity):
        self.velocity = velocity

    def set_joint_effort_target_index(self, *, target):
        self.effort = target

    def write_data_to_sim(self):
        self.committed_states.append(
            (self.position, self.velocity, self.effort)
        )


def test_precheck_pair_commits_baseline_zero_velocity_and_zero_effort_for_both_sides():
    """Omitting RED from a wrist probe step must fail this regression test."""
    if stabilize_precheck_pair is None:
        pytest.fail("wrist-axis precheck pair lifecycle helper is missing")

    red = RecordingArticulation()
    green = RecordingArticulation()

    stabilize_precheck_pair(
        red_robot=red,
        green_robot=green,
        red_position=("red", "baseline"),
        green_position=("green", "pitch-trial"),
        red_zero_velocity=("red", "zero-velocity"),
        green_zero_velocity=("green", "zero-velocity"),
        red_zero_effort=("red", "zero-effort"),
        green_zero_effort=("green", "zero-effort"),
    )

    assert red.committed_states == [
        (
            ("red", "baseline"),
            ("red", "zero-velocity"),
            ("red", "zero-effort"),
        )
    ]
    assert green.committed_states == [
        (
            ("green", "pitch-trial"),
            ("green", "zero-velocity"),
            ("green", "zero-effort"),
        )
    ]
