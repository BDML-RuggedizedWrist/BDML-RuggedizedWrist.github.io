from rizon_osc.osc_profile import (
    hybrid_osc_kwargs,
    pose_osc_kwargs,
    surface_scan_osc_kwargs,
    variable_kp_command_parts,
)


def test_hybrid_profile_matches_isaaclab_run_osc_structure():
    cfg = hybrid_osc_kwargs(force_gain=0.1)

    assert cfg["target_types"] == ["pose_abs", "wrench_abs"]
    assert cfg["impedance_mode"] == "variable_kp"
    assert cfg["inertial_dynamics_decoupling"] is True
    assert cfg["partial_inertial_dynamics_decoupling"] is True
    assert cfg["gravity_compensation"] is True
    assert cfg["motion_damping_ratio_task"] == 1.0
    assert cfg["motion_control_axes_task"] == [1, 1, 0, 1, 1, 1]
    assert cfg["contact_wrench_control_axes_task"] == [0, 0, 1, 0, 0, 0]
    assert cfg["contact_wrench_stiffness_task"] == [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
    assert cfg["nullspace_control"] == "position"
    assert cfg["nullspace_stiffness"] == 10.0
    assert cfg["nullspace_damping_ratio"] == 1.0


def test_pose_profile_is_official_six_axis_variable_kp_osc():
    cfg = pose_osc_kwargs()

    assert cfg["target_types"] == ["pose_abs"]
    assert cfg["impedance_mode"] == "variable_kp"
    assert cfg["inertial_dynamics_decoupling"] is True
    assert cfg["partial_inertial_dynamics_decoupling"] is True
    assert cfg["motion_control_axes_task"] == [1, 1, 1, 1, 1, 1]
    assert cfg["nullspace_stiffness"] == 10.0
    assert cfg["nullspace_damping_ratio"] == 1.0


def test_surface_scan_uses_full_official_hybrid_profile():
    cfg = surface_scan_osc_kwargs(force_gain=0.1)

    assert cfg["target_types"] == ["pose_abs", "wrench_abs"]
    assert cfg["motion_control_axes_task"] == [1, 1, 0, 1, 1, 1]
    assert cfg["contact_wrench_control_axes_task"] == [0, 0, 1, 0, 0, 0]
    assert cfg["contact_wrench_stiffness_task"] == [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
    assert cfg["nullspace_control"] == "position"
    assert cfg["nullspace_stiffness"] == 10.0


def test_variable_kp_command_layout_is_unambiguous():
    assert variable_kp_command_parts(hybrid=False) == ("pose", "kp")
    assert variable_kp_command_parts(hybrid=True) == ("pose", "wrench", "kp")
