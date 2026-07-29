import ast
from dataclasses import replace
from pathlib import Path

from rizon_osc.validation_watchdog import ValidationWatchdog


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


def test_runner_shares_pre_collision_acquisition_command_and_task_frame():
    source = RUNNER.read_text()

    assert "shared_acquiring =" in source
    assert "command_7 = pose_command if use_pose_osc else hybrid_command" in source
    assert "command_9 = pose_command if use_pose_osc else hybrid_command" in source
    assert "red_use_pose_osc = use_pose_osc or collision_7.freeze_path" in source
    assert "osc_7 = pose_osc_7 if red_use_pose_osc else hybrid_osc_7" in source
    assert "osc_9 = pose_osc_9 if use_pose_osc else hybrid_osc_9" in source
    assert "task_frame_7 =" in source
    assert "task_frame_9 =" in source
    assert "torch.allclose(command_7, command_9)" in source
    assert "torch.allclose(task_frame_7, task_frame_9)" in source
    assert "acquiring_7 =" not in source
    assert "acquiring_9 =" not in source


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
    assert "command_7 = pose_command if use_pose_osc else hybrid_command" in source
    assert "command_9 = pose_command if use_pose_osc else hybrid_command" in source
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
    assert "last_supervisor_7.freeze_path and not collision_7.freeze_path" in source


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
            if keyword.arg in {"joint_names_expr", "stiffness", "damping"}
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
        "stiffness": 45.0,
        "damping": 7.0,
    }
    assert green["supplemental_wrist"] == {
        "joint_names_expr": ["wrist_.*_joint"],
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
    namespace = {"ValidationWatchdog": ValidationWatchdog}
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
            "wrist_speed_rad_s": 1.99,
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
