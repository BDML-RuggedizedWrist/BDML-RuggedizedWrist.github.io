"""Configuration values passed directly to IsaacLab's official OSC."""

from __future__ import annotations


def _common_kwargs() -> dict[str, object]:
    return {
        "impedance_mode": "variable_kp",
        "inertial_dynamics_decoupling": True,
        "partial_inertial_dynamics_decoupling": True,
        "gravity_compensation": True,
        "motion_damping_ratio_task": 1.0,
        "nullspace_control": "position",
        "nullspace_stiffness": 10.0,
        "nullspace_damping_ratio": 1.0,
    }


def pose_osc_kwargs() -> dict[str, object]:
    """Full-pose OSC used before contact and by the collision freeze path."""
    return {
        **_common_kwargs(),
        "target_types": ["pose_abs"],
        "motion_control_axes_task": [1, 1, 1, 1, 1, 1],
    }


def hybrid_osc_kwargs(force_gain: float) -> dict[str, object]:
    return {
        **_common_kwargs(),
        "target_types": ["pose_abs", "wrench_abs"],
        "motion_control_axes_task": [1, 1, 0, 1, 1, 1],
        "contact_wrench_control_axes_task": [0, 0, 1, 0, 0, 0],
        "contact_wrench_stiffness_task": [
            0.0,
            0.0,
            float(force_gain),
            0.0,
            0.0,
            0.0,
        ],
    }


def surface_scan_osc_kwargs(force_gain: float) -> dict[str, object]:
    """Full hybrid OSC for surface scanning."""
    return {
        **hybrid_osc_kwargs(force_gain),
    }


def variable_kp_command_parts(*, hybrid: bool) -> tuple[str, ...]:
    return ("pose", "wrench", "kp") if hybrid else ("pose", "kp")
