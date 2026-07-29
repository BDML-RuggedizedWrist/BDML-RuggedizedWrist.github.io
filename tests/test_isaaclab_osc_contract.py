from pathlib import Path


RUNNER = Path(__file__).parents[1] / "scripts" / "run_osc_comparison.py"


def test_runner_uses_isaaclab_operational_space_controller_directly():
    source = RUNNER.read_text()

    assert "from isaaclab.controllers import (" in source
    assert "OperationalSpaceController" in source
    assert "OperationalSpaceControllerCfg" in source
    assert "OperationalSpaceController(" in source
    assert ".compute(" in source
    assert "WeightedOSC" not in source
    assert "force-only J^T term" not in source


def test_runner_uses_builtin_force_feedback_and_nullspace_interfaces():
    source = RUNNER.read_text()

    assert "hybrid_osc_kwargs" in source
    assert "current_ee_force_b=" in source
    assert "nullspace_joint_pos_target=" in source
    assert "AcceptanceMetrics(force_target=args_cli.normal_force)" in source


def test_runner_uses_tutorial_profile_without_local_osc_math():
    source = RUNNER.read_text()

    assert "hybrid_osc_kwargs" in source
    assert "pose_osc_kwargs" in source
    assert "torch.cat((pose_task, wrench_task, pose_kp_task)" in source
    assert "torch.pinverse" not in source
    assert "torch.linalg.pinv" not in source
    assert "jacobian_b.mT @" not in source
    assert "nullspace_stiffness = 55.0" not in source


def test_runner_shares_pre_collision_acquisition_command_and_task_frame():
    source = RUNNER.read_text()

    assert "shared_acquiring =" in source
    assert "task_frame = task_frame_b" in source
    assert "command = pose_command if use_pose_osc else hybrid_command" in source
    assert "red_use_pose_osc = use_pose_osc or collision_7.freeze_path" in source
    assert "osc_7 = pose_osc_7 if red_use_pose_osc else hybrid_osc_7" in source
    assert "osc_9 = pose_osc_9 if use_pose_osc else hybrid_osc_9" in source
    assert "current_task_frame_pose_b=red_task_frame" in source
    assert source.count("current_task_frame_pose_b=task_frame") == 1
    assert "acquiring_7 =" not in source
    assert "acquiring_9 =" not in source
    assert "task_frame_7 =" not in source
    assert "task_frame_9 =" not in source
    assert "command_7 =" not in source
    assert "command_9 =" not in source


def test_runner_uses_official_net_contact_force_history():
    source = RUNNER.read_text()

    assert "sensor.data.net_forces_w_history" in source
    sensor_helper = source.split("def sensor_reaction_force_w", maxsplit=1)[1]
    sensor_helper = sensor_helper.split("def maximum_patient_contact_force", maxsplit=1)[0]
    assert "force_matrix_w_history" not in sensor_helper


def test_runner_separates_probe_contact_from_patient_collision():
    source = RUNNER.read_text()

    assert "NON_PROBE_BODY_SUFFIXES" in source
    assert '"linear_probe"' not in source.split(
        "NON_PROBE_BODY_SUFFIXES =", maxsplit=1
    )[1].split(")", maxsplit=1)[0]
    assert "force_matrix_w_history" in source
    assert "maximum_patient_contact_force" in source


def test_relative_validation_report_is_anchored_to_project_root():
    source = RUNNER.read_text()

    assert "target = PROJECT_ROOT / target" in source


def test_simulation_shutdown_preserves_validation_exit_code():
    source = RUNNER.read_text()

    assert "simulation_app.close(exit_code=exit_code)" in source


def test_runtime_audits_actual_per_side_commands_and_safety_history():
    source = RUNNER.read_text()

    assert "commanded_force_7" in source
    assert "commanded_force_9" in source
    assert "command = pose_command if use_pose_osc else hybrid_command" in source
    assert "max_contact_loss_duration" in source
    assert "scenario_complete=" in source
