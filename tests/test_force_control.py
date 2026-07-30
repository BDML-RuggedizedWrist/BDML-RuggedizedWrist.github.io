import numpy as np
import pytest

from rizon_osc.force_control import ContactForceFilter, project_normal_force


def test_history_filter_rejects_one_sample_impulse():
    force_filter = ContactForceFilter(history_length=5, low_pass_alpha=1.0)
    for measured in (15.0, 15.0, 75.0, 15.0, 15.0):
        output = force_filter.update(measured)

    assert output.window_average == pytest.approx(27.0)
    assert output.filtered < 75.0


def test_projection_uses_current_normal_and_rejects_tension():
    normal = np.array([0.0, 0.0, 1.0])
    assert project_normal_force(np.array([2.0, -1.0, -12.0]), normal) == pytest.approx(12.0)
    assert project_normal_force(np.array([0.0, 0.0, 5.0]), normal) == pytest.approx(0.0)


def test_filter_reset_clears_history():
    force_filter = ContactForceFilter(history_length=3, low_pass_alpha=0.5)
    force_filter.update(12.0)
    force_filter.reset()

    output = force_filter.update(0.0)

    assert output.filtered == pytest.approx(0.0)
    assert output.window_average == pytest.approx(0.0)
