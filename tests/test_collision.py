import pytest

from rizon_osc.collision import CollisionLevel, CollisionMonitor


def test_collision_levels_and_latch():
    """A stop-level patient contact remains latched after force clears."""
    monitor = CollisionMonitor(near_threshold_n=0.5, stop_threshold_n=2.0)

    assert monitor.update(0.2).level is CollisionLevel.CONTACT_OK
    assert monitor.update(0.8).level is CollisionLevel.NEAR_COLLISION
    stopped = monitor.update(2.1)
    assert stopped.level is CollisionLevel.COLLISION_STOP
    assert stopped.freeze_path
    assert stopped.peak_force_n == pytest.approx(2.1)

    still_stopped = monitor.update(0.0)
    assert still_stopped.level is CollisionLevel.COLLISION_STOP
    assert still_stopped.freeze_path


def test_collision_monitor_rejects_negative_force():
    """Negative force magnitudes are invalid sensor inputs."""
    monitor = CollisionMonitor()

    with pytest.raises(ValueError, match="nonnegative"):
        monitor.update(-0.1)
