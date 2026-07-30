import ast
from dataclasses import replace
from pathlib import Path

from rizon_osc.validation_watchdog import (
    ValidationWatchdog,
    green_safety_reasons,
)
from rizon_osc.state_machine import SafetyMode


RUNNER = Path(__file__).parents[1] / "scripts" / "run_osc_comparison.py"
REDUNDANCY_POLICY = (
    Path(__file__).parents[1] / "src" / "rizon_osc" / "redundancy_policy.py"
)


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
    assert "surface_scan_osc_kwargs" in source
    assert "current_ee_force_b=" in source
    assert "nullspace_joint_pos_target=" in source
    assert "AcceptanceMetrics(force_target=args_cli.normal_force)" in source


def test_runner_uses_fast_scan_and_full_ninety_degree_cross_section():
    source = RUNNER.read_text()

    assert "scan_duration=2.0" in source
    assert "settle_duration=0.25" in source
    assert "pitch_duration=1.8" in source
    assert "neutral_duration=1.0" in source
    assert "yaw_duration=2.5" in source
    assert "pitch_angle=math.radians(-35.0)" in source
    assert "yaw_angle=math.radians(90.0)" in source
    assert "challenge_return_duration=2.0" in source


def test_runner_uses_tutorial_profile_without_local_osc_math():
    source = RUNNER.read_text()

    assert "hybrid_osc_kwargs" in source
    assert "pose_osc_kwargs" in source
    assert "(pose_task, wrench_task_7, pose_kp_task)" in source
    assert "(pose_task, wrench_task_9, pose_kp_task)" in source
    assert "torch.pinverse" not in source
    assert "torch.linalg.pinv" not in source
    assert "jacobian_b.mT @" not in source
    assert "nullspace_stiffness = 55.0" not in source


def test_redundancy_policy_is_scheme_a_target_without_runtime_kinematics():
    source = REDUNDANCY_POLICY.read_text()

    assert "def initialize_run(" in source
    assert "self._run_initial_wrist" in source
    assert (
        "target[: self.num_arm_joints] = self._phase_start_arm"
        in source
    )
    assert "self._run_initial_wrist[0] - pitch" in source
    assert "self._run_initial_wrist[1] + yaw" in source
    assert "pinv(" not in source
    assert "lstsq(" not in source
    assert "jacobian" not in source.lower()
    assert "projector" not in source.lower()


def test_runner_shares_initial_acquisition_then_isolates_per_robot_recovery():
    source = RUNNER.read_text()

    assert "initial_shared_acquiring =" in source
    assert "red_acquiring =" in source
    assert "green_acquiring =" in source
    assert "hybrid_command_7 =" in source
    assert "hybrid_command_9 =" in source
    assert "command_7 = red_pose_command if red_use_pose_osc else hybrid_command_7" in source
    assert "command_9 = green_pose_command if green_use_pose_osc else hybrid_command_9" in source
    assert "reference.phase is Phase.SURFACE_SCAN" in source
    assert "osc_7 = scan_osc_7" in source
    assert "osc_9 = scan_osc_9" in source
    assert "task_frame_7 =" in source
    assert "task_frame_9 =" in source
    assert "torch.allclose(command_7, command_9)" in source
    assert "torch.allclose(task_frame_7, task_frame_9)" in source
    assert (
        "if not green_safety_latched and not last_supervisor_9.freeze_path:"
        in source
    )


def test_runner_does_not_add_non_osc_wrist_damping():
    source = RUNNER.read_text()

    assert "active_wrist_ids = robot_9.find_joints" not in source
    assert "robot_9.write_joint_damping_to_sim_index(" not in source


def test_runner_uses_official_net_contact_force_history():
    source = RUNNER.read_text()

    assert "sensor.data.net_forces_w_history" in source
    sensor_helper = source.split("def sensor_reaction_force_w", maxsplit=1)[1]
    sensor_helper = sensor_helper.split("def maximum_patient_contact_force", maxsplit=1)[0]
    assert "force_matrix_w_history" not in sensor_helper


def test_robot_state_velocity_uses_the_same_link_frame_as_pose_and_jacobian():
    """A COM velocity must not be shifted again as though it were link velocity."""
    source = RUNNER.read_text()
    robot_state = source.split("def robot_state(", maxsplit=1)[1]
    robot_state = robot_state.split("def verify_wrist_axis_signs(", maxsplit=1)[0]

    assert "body_link_jacobian_w" in robot_state
    assert "body_link_vel_w" in robot_state
    assert "body_vel_w" not in robot_state


def test_runner_verifies_authored_wrist_axis_signs_before_control():
    """A swapped authored wrist axis must abort before any OSC torque exists."""
    source = RUNNER.read_text()

    assert "def verify_wrist_axis_signs(" in source
    assert "pitch_delta_task_rad" in source
    assert "yaw_delta_task_rad" in source
    assert "wrist_axis_check" in source
    assert 'if not report.get("wrist_axis_check", {}).get("passed", False):' in source
    assert source.index("if args_cli.validation_report is not None:") < source.index(
        'if not report.get("wrist_axis_check", {}).get("passed", False):'
    )


def test_wrist_axis_precheck_stabilizes_and_restores_both_robots():
    """Every wrist-sign physics step must keep RED from freely falling."""
    module = ast.parse(RUNNER.read_text())
    verify = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "verify_wrist_axis_signs"
    )
    argument_names = [argument.arg for argument in verify.args.args]
    assert "robot_7" in argument_names
    assert "robot_9" in argument_names

    nested_writer = next(
        node
        for node in verify.body
        if isinstance(node, ast.FunctionDef) and node.name == "write_and_measure"
    )
    nested_calls = [
        node for node in ast.walk(nested_writer) if isinstance(node, ast.Call)
    ]
    stabilize_call = next(
        node
        for node in nested_calls
        if ast.unparse(node.func) == "stabilize_precheck_pair"
    )
    stabilize_keywords = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in stabilize_call.keywords
    }
    assert stabilize_keywords == {
        "red_robot": "robot_7",
        "green_robot": "robot_9",
        "red_position": "baseline_joint_pos_7",
        "green_position": "position_9",
        "red_zero_velocity": "zero_joint_vel_7",
        "green_zero_velocity": "zero_joint_vel_9",
        "red_zero_effort": "zero_joint_effort_7",
        "green_zero_effort": "zero_joint_effort_9",
    }
    physics_step = next(
        node
        for node in nested_calls
        if ast.unparse(node.func) == "sim.step"
    )
    assert stabilize_call.lineno < physics_step.lineno

    verify_calls = [
        node for node in ast.walk(verify) if isinstance(node, ast.Call)
    ]
    measured_positions = [
        ast.unparse(node.args[0])
        for node in verify_calls
        if ast.unparse(node.func) == "write_and_measure"
    ]
    assert measured_positions == [
        "pitch_trial",
        "baseline_joint_pos_9",
        "yaw_trial",
        "baseline_joint_pos_9",
    ]

    restore_call = next(
        node
        for node in verify_calls
        if ast.unparse(node.func) == "stabilize_precheck_pair"
        and node is not stabilize_call
    )
    restore_keywords = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in restore_call.keywords
    }
    assert restore_call.lineno > max(
        node.lineno
        for node in verify_calls
        if ast.unparse(node.func) == "write_and_measure"
    )
    assert restore_keywords == {
        "red_robot": "robot_7",
        "green_robot": "robot_9",
        "red_position": "baseline_joint_pos_7",
        "green_position": "baseline_joint_pos_9",
        "red_zero_velocity": "zero_joint_vel_7",
        "green_zero_velocity": "zero_joint_vel_9",
        "red_zero_effort": "zero_joint_effort_7",
        "green_zero_effort": "zero_joint_effort_9",
    }

    run_simulator = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_simulator"
    )
    call_site = next(
        node
        for node in ast.walk(run_simulator)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "verify_wrist_axis_signs"
    )
    assert [ast.unparse(argument) for argument in call_site.args[:4]] == [
        "sim",
        "scene",
        "robot_7",
        "robot_9",
    ]


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
    assert "wrench_task_7 = wrench_task.clone()" in source
    assert "wrench_task_9 = wrench_task.clone()" in source
    assert "command_7 = red_pose_command if red_use_pose_osc else hybrid_command_7" in source
    assert "command_9 = green_pose_command if green_use_pose_osc else hybrid_command_9" in source
    assert "max_contact_loss_duration" in source
    assert "scenario_complete=" in source


def test_runner_integrates_phase_travel_metrics_and_persistent_hud():
    """The live runner must preserve the pure-module safety evidence."""
    source = RUNNER.read_text()

    assert "from rizon_osc.hud import HudSnapshot, format_hud" in source
    assert "from rizon_osc.joint_travel import JointTravelTracker" in source
    assert "travel_tracker.begin_phase(" in source
    assert "travel_tracker.update(" in source
    assert "travel = travel_tracker.snapshot()" in source
    assert "phase_arm_travel_7_rad=travel.arm_7_rad" in source
    assert "phase_arm_travel_9_rad=travel.arm_9_rad" in source
    assert "phase_wrist_travel_9_rad=travel.wrist_9_rad" in source
    assert 'ui.Window("OSC 7-DoF vs 9-DoF", width=520, height=220)' in source
    assert "hud_label.text = format_hud(" in source
    assert (
        "if not green_safety_latched and not last_supervisor_9.freeze_path:"
        in source
    )


def _resolved_robot_actuator_drives(*, wrist_active: bool):
    """Resolve the authored drive fields from ``make_robot_cfg`` for one side."""
    module = ast.parse(RUNNER.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_robot_cfg"
    )
    return_node = next(
        node for node in ast.walk(function) if isinstance(node, ast.Return)
    )
    articulation_call = return_node.value
    actuators_keyword = next(
        keyword for keyword in articulation_call.keywords if keyword.arg == "actuators"
    )

    def resolve(node):
        if isinstance(node, ast.IfExp):
            assert isinstance(node.test, ast.Name)
            assert node.test.id == "wrist_active"
            return resolve(node.body if wrist_active else node.orelse)
        return ast.literal_eval(node)

    return {
        resolve(name): {
            keyword.arg: resolve(keyword.value)
            for keyword in actuator.keywords
            if keyword.arg in {"joint_names_expr", "armature", "stiffness", "damping"}
        }
        for name, actuator in zip(
            actuators_keyword.value.keys, actuators_keyword.value.values, strict=True
        )
    }


def test_effort_controlled_osc_joints_disable_implicit_drives():
    """PhysX must not add passive drives after the official OSC effort."""
    red = _resolved_robot_actuator_drives(wrist_active=False)
    green = _resolved_robot_actuator_drives(wrist_active=True)
    arm_groups = {
        "shoulder_effort": ["joint[1-2]"],
        "elbow_effort": ["joint[3-4]"],
        "arm_wrist_effort": ["joint[5-7]"],
    }

    for name, joint_names in arm_groups.items():
        assert red[name] == {
            "joint_names_expr": joint_names,
            "stiffness": 0.0,
            "damping": 0.0,
        }
        assert green[name] == red[name]

    assert red["supplemental_wrist"] == {
        "joint_names_expr": ["wrist_.*_joint"],
        "armature": {
            "wrist_pitch_joint": 0.05,
            "wrist_roll_joint": 0.02,
        },
        "stiffness": 45.0,
        "damping": 7.0,
    }
    assert green["supplemental_wrist"] == {
        "joint_names_expr": ["wrist_.*_joint"],
        "armature": {
            "wrist_pitch_joint": 0.05,
            "wrist_roll_joint": 0.02,
        },
        "stiffness": 0.0,
        "damping": 0.0,
    }


def test_runner_initializes_green_policy_once_before_control_loop():
    module = ast.parse(RUNNER.read_text())
    run_simulator = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_simulator"
    )
    initialize_calls = [
        node
        for node in ast.walk(run_simulator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "initialize_run"
    ]
    control_loop = next(
        node for node in ast.walk(run_simulator) if isinstance(node, ast.While)
    )

    assert len(initialize_calls) == 1
    initialize_call = initialize_calls[0]
    assert initialize_call.lineno < control_loop.lineno
    assert initialize_call not in set(ast.walk(control_loop))
    assert ast.unparse(initialize_call.func.value) == "policy_9"
    assert ast.unparse(initialize_call.args[0]) == (
        "state_9[6][0].detach().cpu().numpy()"
    )


def test_phase_changes_do_not_recapture_the_run_initial_wrist():
    source = RUNNER.read_text()
    phase_block = source.split(
        "if reference.phase.value != previous_phase:", maxsplit=1
    )[1].split("green_null_target_np =", maxsplit=1)[0]

    assert "policy_9.begin_phase(" in phase_block
    assert "initialize_run(" not in phase_block


def test_runner_integrates_pure_validation_watchdog_after_physics_step():
    source = RUNNER.read_text()

    assert "from rizon_osc.validation_watchdog import (" in source
    assert "ValidationWatchdog" in source
    assert "WatchdogSample" in source
    assert "validation_watchdog.update(" in source
    assert source.rindex("post_state_9 = robot_state(") < source.index(
        "validation_watchdog.update("
    )
    assert 'report["validation_watchdog"] =' in source
    assert 'report["overall_pass"] = bool(' in source


def test_watchdog_does_not_change_official_osc_torque_boundary():
    module = ast.parse(RUNNER.read_text())
    run_simulator = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_simulator"
    )
    watchdog_calls = [
        node
        for node in ast.walk(run_simulator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
        and ast.unparse(node.func.value) == "validation_watchdog"
    ]

    assert len(watchdog_calls) == 1
    assert "torque_7 =" not in ast.unparse(watchdog_calls[0])
    assert "torque_9 =" not in ast.unparse(watchdog_calls[0])
    assert "set_joint_effort_target" not in ast.unparse(watchdog_calls[0])


def test_watchdog_uses_fresh_post_step_sensor_evidence():
    source = RUNNER.read_text()
    loop_source = source.split(
        "while simulation_app.is_running():", maxsplit=1
    )[1]

    physics_step = loop_source.index(
        "sim.step(render=not args_cli.headless)"
    )
    scene_update = loop_source.index("scene.update(dt)", physics_step)
    assert "post_reaction_7_w =" in loop_source
    assert "post_nonprobe_7 =" in loop_source
    post_reaction = loop_source.index("post_reaction_7_w =")
    post_nonprobe = loop_source.index("post_nonprobe_7 =")
    watchdog_update = loop_source.index("validation_watchdog.update(")
    max_steps = loop_source.index("if args_cli.max_steps > 0")
    assert (
        physics_step
        < scene_update
        < post_reaction
        < post_nonprobe
        < watchdog_update
        < max_steps
    )

    module = ast.parse(source)
    run_simulator = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_simulator"
    )
    calls = [node for node in ast.walk(run_simulator) if isinstance(node, ast.Call)]
    call_targets = [ast.unparse(node.func) for node in calls]
    assert call_targets.count("force_filter_7.update") == 1
    assert call_targets.count("force_filter_9.update") == 1
    assert call_targets.count("watchdog_force_filter_7.update") == 1
    assert call_targets.count("watchdog_force_filter_9.update") == 1
    assert call_targets.count("collision_monitor_7.update") == 1
    assert call_targets.count("collision_monitor_9.update") == 1

    watchdog_call = next(
        node
        for node in calls
        if ast.unparse(node.func) == "validation_watchdog.update"
    )
    sample_names = {
        node.id for node in ast.walk(watchdog_call) if isinstance(node, ast.Name)
    }
    assert {
        "watchdog_filtered_7",
        "watchdog_filtered_9",
        "post_nonprobe_7",
        "post_nonprobe_9",
    } <= sample_names
    assert "filtered_7" not in sample_names
    assert "filtered_9" not in sample_names


def _load_pure_runner_function(name: str):
    module = ast.parse(RUNNER.read_text())
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(functions) == 1
    function = functions[0]
    namespace = {
        "ValidationWatchdog": ValidationWatchdog,
        "green_safety_reasons": green_safety_reasons,
        "SafetyMode": SafetyMode,
    }
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[function], type_ignores=[])
            ),
            str(RUNNER),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def test_safety_latch_reasons_distinguish_recoverable_green_contact_loss():
    reason = _load_pure_runner_function("safety_latch_reason")

    assert (
        reason(
            collision_stop=True,
            supervisor_mode=SafetyMode.TRACKING,
            latch_contact_loss=False,
        )
        == "nonprobe_collision"
    )
    assert (
        reason(
            collision_stop=False,
            supervisor_mode=SafetyMode.FORCE_HOLD,
            latch_contact_loss=False,
        )
        == "normal_force_overload"
    )
    assert (
        reason(
            collision_stop=False,
            supervisor_mode=SafetyMode.INVALID_SURFACE,
            latch_contact_loss=False,
        )
        == "invalid_surface"
    )
    assert (
        reason(
            collision_stop=False,
            supervisor_mode=SafetyMode.REACQUIRE,
            latch_contact_loss=True,
        )
        == "contact_loss"
    )
    assert (
        reason(
            collision_stop=False,
            supervisor_mode=SafetyMode.REACQUIRE,
            latch_contact_loss=False,
        )
        is None
    )
    assert (
        reason(
            collision_stop=False,
            supervisor_mode=SafetyMode.TRACKING,
            latch_contact_loss=False,
            singularity_speed_guard=True,
        )
        == "singularity_speed_guard"
    )


def test_red_singularity_guard_only_applies_to_challenge_phases():
    guard = _load_pure_runner_function("red_singularity_speed_guard")

    assert guard(
        phase_name="PITCH_ONLY",
        max_main_arm_speed_rad_s=1.0,
    )
    assert guard(
        phase_name="CHALLENGE_PITCH_ONLY",
        max_main_arm_speed_rad_s=1.2,
    )
    assert not guard(
        phase_name="PITCH_ONLY",
        max_main_arm_speed_rad_s=0.99,
    )
    assert not guard(
        phase_name="APPROACH",
        max_main_arm_speed_rad_s=1.5,
    )
    assert not guard(
        phase_name="SURFACE_SCAN",
        max_main_arm_speed_rad_s=1.5,
    )


def test_both_collision_latches_hold_measured_pose_and_joint_posture():
    source = RUNNER.read_text()

    for side in ("red", "green"):
        state = "state_7" if side == "red" else "state_9"
        assert f"{side}_hold_task_frame = {state}[3].clone()" in source
        assert f"{side}_hold_null_target = {state}[6].clone()" in source
        assert f"if {side}_safety_latched:" in source
    assert "or green_safety_latched" in source
    assert "task_time = min(task_time + dt, trajectory.total_duration)" in source
    assert '"final_reference_hold" = completion_hold_reported' not in source
    assert 'report["final_reference_hold"] = completion_hold_reported' in source


def test_watchdog_report_schema_and_gate_cover_early_failure():
    augment_report = _load_pure_runner_function(
        "augment_validation_watchdog_report"
    )
    snapshot = ValidationWatchdog().snapshot()
    early_report = {
        "overall_pass": False,
        "reason": "authored wrist-axis sign verification failed",
    }

    result = augment_report(
        early_report,
        enabled=True,
        snapshot=snapshot,
    )

    assert result is early_report
    assert result["overall_pass"] is False
    assert result["validation_watchdog"] == {
        "enabled": True,
        "thresholds": {
            "wrist_limit_margin_rad": 0.02,
            "wrist_limit_duration_s": 0.10,
            "wrist_speed_rad_s": 2.05,
            "wrist_speed_duration_s": 0.10,
            "contact_loss_duration_s": 0.10,
            "normal_force_limit_n": 30.0,
            "nonprobe_collision_n": 2.0,
            "freeze_window_s": 0.25,
            "translation_command_m": 0.001,
            "translation_response_m": 0.0001,
            "rotation_command_deg": 0.5,
            "rotation_response_deg": 0.05,
        },
        **snapshot.as_dict(),
        "green_safety_reasons": [],
        "green_safety_passed": True,
    }
    failed_snapshot = replace(
        snapshot,
        passed=False,
        stop_requested=True,
        reasons=("nonfinite",),
        first_failure_step=1,
    )
    assert (
        augment_report(
            {"overall_pass": True},
            enabled=True,
            snapshot=failed_snapshot,
        )["overall_pass"]
        is False
    )
    assert (
        augment_report(
            {"overall_pass": True},
            enabled=False,
            snapshot=failed_snapshot,
        )["overall_pass"]
        is True
    )

    source = RUNNER.read_text()
    early_path = source.split(
        "if not wrist_axis_check.passed:", maxsplit=1
    )[1].split("state_7 = robot_state(", maxsplit=1)[0]
    assert "return augment_validation_watchdog_report(" in early_path
    assert source.count("return augment_validation_watchdog_report(") == 2


def test_validation_owns_separate_nonprobe_contact_sensors():
    module = ast.parse(RUNNER.read_text())
    main = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assignments = {
        target.id: node.value
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    for side in ("7", "9"):
        main_name = f"collision_sensors_{side}"
        watchdog_name = f"watchdog_collision_sensors_{side}"
        assert main_name in assignments
        assert watchdog_name in assignments
        main_value = assignments[main_name]
        watchdog_value = assignments[watchdog_name]

        assert isinstance(main_value, ast.Call)
        assert ast.unparse(main_value.func) == "make_patient_collision_sensors"
        assert ast.literal_eval(main_value.args[0]) == side
        assert isinstance(watchdog_value, ast.IfExp)
        assert ast.unparse(watchdog_value.test) == (
            "args_cli.validation_report is not None"
        )
        assert ast.unparse(watchdog_value.body.func) == (
            "make_patient_collision_sensors"
        )
        assert ast.literal_eval(watchdog_value.body.args[0]) == side
        assert isinstance(watchdog_value.orelse, ast.List)
        assert watchdog_value.orelse.elts == []

    sensor_factories = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "make_patient_collision_sensors"
    ]
    assert len(sensor_factories) == 4

    run_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "run_simulator"
    )
    run_keywords = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in run_call.keywords
    }
    assert run_keywords["collision_sensors_7"] == "collision_sensors_7"
    assert run_keywords["collision_sensors_9"] == "collision_sensors_9"
    assert run_keywords["watchdog_collision_sensors_7"] == (
        "watchdog_collision_sensors_7"
    )
    assert run_keywords["watchdog_collision_sensors_9"] == (
        "watchdog_collision_sensors_9"
    )


def test_watchdog_nonprobe_sensor_updates_are_isolated_and_same_step():
    module = ast.parse(RUNNER.read_text())
    run_simulator = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_simulator"
    )
    parameter_names = {
        argument.arg for argument in run_simulator.args.kwonlyargs
    }
    assert {
        "collision_sensors_7",
        "collision_sensors_9",
        "watchdog_collision_sensors_7",
        "watchdog_collision_sensors_9",
    } <= parameter_names

    nonprobe_calls = [
        node
        for node in ast.walk(run_simulator)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "maximum_patient_contact_force"
    ]
    sensor_arguments = [ast.unparse(node.args[0]) for node in nonprobe_calls]
    assert sensor_arguments.count("collision_sensors_7") == 1
    assert sensor_arguments.count("collision_sensors_9") == 1
    assert sensor_arguments.count("watchdog_collision_sensors_7") == 1
    assert sensor_arguments.count("watchdog_collision_sensors_9") == 1

    watchdog_sample = next(
        node
        for node in ast.walk(run_simulator)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "WatchdogSample"
    )
    red_collision_stop = next(
        keyword.value
        for keyword in watchdog_sample.keywords
        if keyword.arg == "red_collision_stop"
    )
    assert isinstance(red_collision_stop, ast.BoolOp)
    assert isinstance(red_collision_stop.op, ast.Or)
    assert ast.unparse(red_collision_stop.values[0]) == "red_safety_latched"
    same_step_collision = red_collision_stop.values[1]
    assert isinstance(same_step_collision, ast.BoolOp)
    assert isinstance(same_step_collision.op, ast.And)
    assert ast.unparse(same_step_collision.values[0]) == (
        "reference.phase in "
        "(Phase.CHALLENGE_TRANSIT, Phase.CHALLENGE_PITCH_ONLY, "
        "Phase.RETURN_NEUTRAL)"
    )
    assert ast.unparse(same_step_collision.values[1]) == (
        "post_nonprobe_7 >= ValidationWatchdog.NONPROBE_COLLISION_N"
    )
