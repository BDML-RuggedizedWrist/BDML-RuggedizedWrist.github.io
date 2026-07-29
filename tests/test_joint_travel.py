import numpy as np
import pytest

from rizon_osc.joint_travel import JointTravelTracker


def test_phase_travel_counts_only_absolute_incremental_motion():
    tracker = JointTravelTracker()
    tracker.begin_phase("PITCH_ONLY", np.zeros(7), np.zeros(9))
    tracker.update(
        np.array([0.1, 0, 0, 0, 0, 0, 0]),
        np.array([0.02, 0, 0, 0, 0, 0, 0, 0.2, 0]),
    )
    tracker.update(
        np.array([0.05, 0, 0, 0, 0, 0, 0]),
        np.array([0.01, 0, 0, 0, 0, 0, 0, 0.15, 0]),
    )

    result = tracker.snapshot()

    assert result.phase == "PITCH_ONLY"
    assert result.arm_7_rad == pytest.approx(0.15)
    assert result.arm_9_rad == pytest.approx(0.03)
    assert result.wrist_9_rad == pytest.approx(0.25)
    assert result.reduction_percent == pytest.approx(80.0)


@pytest.mark.parametrize(
    ("joint_7", "joint_9"),
    [
        (np.zeros(6), np.zeros(9)),
        (np.zeros(7), np.zeros(8)),
        (np.zeros((7, 1)), np.zeros(9)),
        (np.zeros(7), np.zeros((9, 1))),
    ],
)
def test_begin_phase_rejects_positions_with_wrong_shapes(joint_7, joint_9):
    tracker = JointTravelTracker()

    with pytest.raises(ValueError):
        tracker.begin_phase("PITCH_ONLY", joint_7, joint_9)


def test_update_rejects_positions_with_wrong_shapes():
    tracker = JointTravelTracker()
    tracker.begin_phase("PITCH_ONLY", np.zeros(7), np.zeros(9))

    with pytest.raises(ValueError):
        tracker.update(np.zeros(7), np.zeros(8))
