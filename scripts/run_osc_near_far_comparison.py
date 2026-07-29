#!/usr/bin/env python3
"""Independent near-to-far 7-DoF/9-DoF ultrasound OSC comparison."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rizon_osc.collision import CollisionMonitor
from rizon_osc.force_control import ContactForceFilter
from rizon_osc.near_far_policy import NearFarRedundancyPolicy
from rizon_osc.near_far_trajectory import NearFarPhase, NearFarTrajectory
from rizon_osc.osc_profile import hybrid_osc_kwargs, pose_osc_kwargs
from rizon_osc.scene_assets import AssetPaths
from rizon_osc.state_machine import ContactSupervisor
from rizon_osc.surface_model import SurfaceMap
from rizon_osc.trajectory import (
    quaternion_from_rotation_matrix,
    quintic_progress,
    split_task_frame_rotation,
)
from rizon_osc.wrist_axis_precheck import stabilize_precheck_pair

from isaaclab.app import AppLauncher


DEFAULT_PATHS = AssetPaths.from_environment()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Zero keeps the GUI running until its window is closed manually.",
)
parser.add_argument("--normal_force", type=float, default=15.0)
parser.add_argument("--force_gain", type=float, default=0.8)
parser.add_argument("--robot_usd", type=Path, default=DEFAULT_PATHS.robot_wrapper)
parser.add_argument("--patient_usd", type=Path, default=DEFAULT_PATHS.patient_usd)
parser.add_argument("--surface_map", type=Path, default=DEFAULT_PATHS.surface_map)
parser.add_argument(
    "--contact_loss_limit",
    type=float,
    default=0.1,
    help="Contact loss duration before the affected robot independently reacquires.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.max_steps < 0:
    parser.error("--max_steps must be nonnegative")
if not 0.0 < args_cli.normal_force <= 30.0:
    parser.error("--normal_force must be in (0, 30]")
if args_cli.force_gain < 0.0:
    parser.error("--force_gain must be nonnegative")
if args_cli.contact_loss_limit <= 0.0:
    parser.error("--contact_loss_limit must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import (
    OperationalSpaceController,
    OperationalSpaceControllerCfg,
)
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_matrix,
    quat_inv,
    quat_slerp,
    subtract_frame_transforms,
)

if not args_cli.headless:
    import omni.ui as ui


ROBOT_USD = args_cli.robot_usd.expanduser().resolve()
PATIENT_USD = args_cli.patient_usd.expanduser().resolve()
SURFACE_MAP_PATH = args_cli.surface_map.expanduser().resolve()
GROUND_USD = DEFAULT_PATHS.ground_usd

ROBOT_Y_7 = -1.0
ROBOT_Y_9 = 1.0
ROBOT_ROOT_Z = 0.35
PATIENT_ROOT_X = 0.70057756
PATIENT_ROOT_Z = 0.28493234
PATIENT_ROOT_Y_FROM_ROBOT = -1.29225561
PATIENT_BED_LOCAL_CENTER_Y = 0.90
SURFACE_TRANSLATION_B = np.array(
    [PATIENT_ROOT_X, PATIENT_ROOT_Y_FROM_ROBOT, PATIENT_ROOT_Z - ROBOT_ROOT_Z]
)
PROBE_TIP_OFFSET = 0.13254
ULTRASOUND_GEL_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.02,
    dynamic_friction=0.01,
    restitution=0.0,
)

ROBOT_START = {
    "joint1": 0.0,
    "joint2": -0.698,
    "joint3": 0.0,
    "joint4": 1.571,
    "joint5": 0.0,
    "joint6": 0.698,
    "joint7": 0.0,
    "wrist_pitch_joint": 0.0,
    "wrist_roll_joint": 0.0,
}

NON_PROBE_BODY_SUFFIXES = (
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "flange",
    "flange/wrist_base",
    "flange/wrist_base/wrist_pitch_link",
    "flange/wrist_base/wrist_pitch_link/probe_roll_output",
)


@dataclass(frozen=True)
class WristAxisCheck:
    pitch_delta_task_rad: tuple[float, float, float]
    axial_delta_task_rad: tuple[float, float, float]
    passed: bool


def make_robot_cfg(root_y: float, *, wrist_active: bool) -> ArticulationCfg:
    """Fixed-base Rizon; official OSC supplies all active-joint efforts."""
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROBOT_USD),
            physics_material=ULTRASOUND_GEL_MATERIAL,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                rest_offset=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, root_y, ROBOT_ROOT_Z),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos=ROBOT_START,
        ),
        actuators={
            "shoulder_effort": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-2]"],
                effort_limit_sim=123.0,
                velocity_limit_sim=2.094,
                stiffness=0.0,
                damping=0.0,
            ),
            "elbow_effort": ImplicitActuatorCfg(
                joint_names_expr=["joint[3-4]"],
                effort_limit_sim=64.0,
                velocity_limit_sim=2.443,
                stiffness=0.0,
                damping=0.0,
            ),
            "arm_wrist_effort": ImplicitActuatorCfg(
                joint_names_expr=["joint[5-7]"],
                effort_limit_sim=39.0,
                velocity_limit_sim=4.887,
                stiffness=0.0,
                damping=0.0,
            ),
            "supplemental_wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_.*_joint"],
                effort_limit_sim=12.0,
                velocity_limit_sim=2.0,
                armature={
                    "wrist_pitch_joint": 0.05,
                    "wrist_roll_joint": 0.02,
                },
                stiffness=0.0 if wrist_active else 45.0,
                damping=0.0 if wrist_active else 7.0,
            ),
        },
    )


def patient_cfg(prim_path: str, robot_y: float) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PATIENT_USD),
            physics_material=ULTRASOUND_GEL_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                PATIENT_ROOT_X,
                robot_y + PATIENT_ROOT_Y_FROM_ROBOT,
                PATIENT_ROOT_Z,
            )
        ),
    )


def fixed_cuboid_cfg(
    prim_path: str,
    *,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                metallic=0.0,
                roughness=0.72,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
    )


@configclass
class NearFarSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(usd_path=str(GROUND_USD)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2600.0, color=(0.82, 0.86, 1.0)),
    )
    robot_7dof = make_robot_cfg(ROBOT_Y_7, wrist_active=False).replace(
        prim_path="{ENV_REGEX_NS}/Robot7DoF"
    )
    robot_9dof = make_robot_cfg(ROBOT_Y_9, wrist_active=True).replace(
        prim_path="{ENV_REGEX_NS}/Robot9DoF"
    )
    patient_7 = patient_cfg(
        "{ENV_REGEX_NS}/Patient7DoF/torso_collider", ROBOT_Y_7
    )
    patient_9 = patient_cfg(
        "{ENV_REGEX_NS}/Patient9DoF/torso_collider", ROBOT_Y_9
    )
    bed_7 = fixed_cuboid_cfg(
        "{ENV_REGEX_NS}/Patient7DoF/Bed",
        size=(0.85, 1.95, 0.075),
        position=(
            PATIENT_ROOT_X,
            ROBOT_Y_7 + PATIENT_ROOT_Y_FROM_ROBOT + PATIENT_BED_LOCAL_CENTER_Y,
            0.05,
        ),
        color=(0.55, 0.16, 0.16),
    )
    bed_9 = fixed_cuboid_cfg(
        "{ENV_REGEX_NS}/Patient9DoF/Bed",
        size=(0.85, 1.95, 0.075),
        position=(
            PATIENT_ROOT_X,
            ROBOT_Y_9 + PATIENT_ROOT_Y_FROM_ROBOT + PATIENT_BED_LOCAL_CENTER_Y,
            0.05,
        ),
        color=(0.14, 0.52, 0.22),
    )
    pedestal_7 = fixed_cuboid_cfg(
        "{ENV_REGEX_NS}/Pedestal7DoF",
        size=(0.32, 0.32, 0.35),
        position=(0.0, ROBOT_Y_7, 0.175),
        color=(0.22, 0.24, 0.28),
    )
    pedestal_9 = fixed_cuboid_cfg(
        "{ENV_REGEX_NS}/Pedestal9DoF",
        size=(0.32, 0.32, 0.35),
        position=(0.0, ROBOT_Y_9, 0.175),
        color=(0.22, 0.24, 0.28),
    )
    contact_7 = ContactSensorCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Robot7DoF/flange/wrist_base/wrist_pitch_link/"
            "probe_roll_output/linear_probe"
        ),
        update_period=0.0,
        history_length=4,
        track_air_time=True,
        force_threshold=0.5,
    )
    contact_9 = ContactSensorCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Robot9DoF/flange/wrist_base/wrist_pitch_link/"
            "probe_roll_output/linear_probe"
        ),
        update_period=0.0,
        history_length=4,
        track_air_time=True,
        force_threshold=0.5,
    )


def make_pose_osc(device: str) -> OperationalSpaceController:
    return OperationalSpaceController(
        OperationalSpaceControllerCfg(**pose_osc_kwargs()),
        num_envs=1,
        device=device,
    )


def make_hybrid_osc(device: str) -> OperationalSpaceController:
    return OperationalSpaceController(
        OperationalSpaceControllerCfg(
            **hybrid_osc_kwargs(force_gain=args_cli.force_gain)
        ),
        num_envs=1,
        device=device,
    )


def make_patient_collision_sensors(side: str) -> list[ContactSensor]:
    robot_name = f"Robot{side}DoF"
    patient_surface = (
        f"/World/envs/env_.*/Patient{side}DoF/torso_collider/"
        "upper_torso_contact_surface"
    )
    return [
        ContactSensor(
            ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/{robot_name}/{suffix}",
                update_period=0.0,
                history_length=2,
                filter_prim_paths_expr=[patient_surface],
            )
        )
        for suffix in NON_PROBE_BODY_SUFFIXES
    ]


def _tensor_data(proxy) -> torch.Tensor | None:
    if proxy is None:
        return None
    return proxy.torch if hasattr(proxy, "torch") else proxy


def sensor_reaction_force_w(sensor: ContactSensor, device: str) -> torch.Tensor:
    net_history = _tensor_data(sensor.data.net_forces_w_history)
    if net_history is not None:
        return torch.mean(net_history, dim=1).sum(dim=1)
    net_force = _tensor_data(sensor.data.net_forces_w)
    if net_force is not None:
        return net_force.sum(dim=1)
    return torch.zeros((1, 3), device=device)


def maximum_patient_contact_force(
    sensors: list[ContactSensor], dt: float
) -> float:
    maximum = 0.0
    for sensor in sensors:
        sensor.update(dt, force_recompute=True)
        history = _tensor_data(sensor.data.force_matrix_w_history)
        if history is None:
            raise RuntimeError("patient-filtered collision history unavailable")
        maximum = max(
            maximum,
            float(torch.linalg.vector_norm(history, dim=-1).max().item()),
        )
    return maximum


def robot_state(
    robot: Articulation,
    ee_body_idx: int,
    controlled_joint_ids: list[int],
) -> tuple[torch.Tensor, ...]:
    """Official OSC inputs evaluated at the linear probe acoustic face."""
    jacobian_body_idx = ee_body_idx - 1 if robot.is_fixed_base else ee_body_idx
    jacobian_joint_ids = [
        joint_id + robot.num_base_dofs for joint_id in controlled_joint_ids
    ]
    jacobian_w = robot.data.body_link_jacobian_w.torch[
        :, jacobian_body_idx, :, jacobian_joint_ids
    ]
    mass_matrix = robot.data.mass_matrix.torch[:, jacobian_joint_ids, :][
        :, :, jacobian_joint_ids
    ]
    gravity = robot.data.gravity_compensation_forces.torch[:, jacobian_joint_ids]

    root_pos_w = robot.data.root_pos_w.torch
    root_quat_w = robot.data.root_quat_w.torch
    ee_pos_w = robot.data.body_pos_w.torch[:, ee_body_idx]
    ee_quat_w = robot.data.body_quat_w.torch[:, ee_body_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )
    relative_velocity_w = (
        robot.data.body_link_vel_w.torch[:, ee_body_idx] - robot.data.root_vel_w.torch
    )
    linear_velocity_b = quat_apply_inverse(root_quat_w, relative_velocity_w[:, :3])
    angular_velocity_b = quat_apply_inverse(root_quat_w, relative_velocity_w[:, 3:])

    root_rotation_inverse = matrix_from_quat(quat_inv(root_quat_w))
    jacobian_b = jacobian_w.clone()
    jacobian_b[:, :3] = torch.bmm(root_rotation_inverse, jacobian_b[:, :3])
    jacobian_b[:, 3:] = torch.bmm(root_rotation_inverse, jacobian_b[:, 3:])

    tip_offset_local = torch.tensor(
        [[0.0, 0.0, -PROBE_TIP_OFFSET]],
        device=robot.device,
        dtype=ee_pos_b.dtype,
    )
    tip_offset_b = quat_apply(ee_quat_b, tip_offset_local)
    ee_pos_b = ee_pos_b + tip_offset_b
    angular_columns = jacobian_b[:, 3:, :].transpose(1, 2)
    shifted_linear = torch.cross(
        angular_columns,
        tip_offset_b.unsqueeze(1).expand_as(angular_columns),
        dim=-1,
    )
    jacobian_b[:, :3, :] += shifted_linear.transpose(1, 2)
    linear_velocity_b += torch.cross(angular_velocity_b, tip_offset_b, dim=-1)

    tip_offset_w = quat_apply(ee_quat_w, tip_offset_local)
    ee_pos_w = ee_pos_w + tip_offset_w
    return (
        jacobian_b,
        mass_matrix,
        gravity,
        torch.cat((ee_pos_b, ee_quat_b), dim=-1),
        torch.cat((linear_velocity_b, angular_velocity_b), dim=-1),
        torch.cat((ee_pos_w, ee_quat_w), dim=-1),
        robot.data.joint_pos.torch[:, controlled_joint_ids],
        robot.data.joint_vel.torch[:, controlled_joint_ids],
    )


def verify_wrist_axis_signs(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot_7: Articulation,
    robot_9: Articulation,
    ee_body_idx: int,
    controlled_joint_ids: list[int],
    task_frame_quat_b: torch.Tensor,
) -> WristAxisCheck:
    """Restore both robots while measuring the authored J8/J9 task-frame signs."""
    dt = sim.get_physics_dt()
    baseline_joint_pos_7 = robot_7.data.joint_pos.torch.clone()
    baseline_joint_pos_9 = robot_9.data.joint_pos.torch.clone()
    zero_joint_vel_7 = torch.zeros_like(robot_7.data.joint_vel.torch)
    zero_joint_vel_9 = torch.zeros_like(robot_9.data.joint_vel.torch)
    zero_joint_effort_7 = torch.zeros_like(baseline_joint_pos_7)
    zero_joint_effort_9 = torch.zeros_like(baseline_joint_pos_9)
    baseline_pose_b = robot_state(
        robot_9, ee_body_idx, controlled_joint_ids
    )[3].clone()

    def write_and_measure(position_9: torch.Tensor) -> torch.Tensor:
        stabilize_precheck_pair(
            red_robot=robot_7,
            green_robot=robot_9,
            red_position=baseline_joint_pos_7,
            green_position=position_9,
            red_zero_velocity=zero_joint_vel_7,
            green_zero_velocity=zero_joint_vel_9,
            red_zero_effort=zero_joint_effort_7,
            green_zero_effort=zero_joint_effort_9,
        )
        sim.step(render=False)
        scene.update(dt)
        return robot_state(robot_9, ee_body_idx, controlled_joint_ids)[3].clone()

    pitch_trial = baseline_joint_pos_9.clone()
    pitch_trial[:, controlled_joint_ids[-2]] += math.radians(1.0)
    pitch_pose_b = write_and_measure(pitch_trial)
    write_and_measure(baseline_joint_pos_9)
    axial_trial = baseline_joint_pos_9.clone()
    axial_trial[:, controlled_joint_ids[-1]] += math.radians(1.0)
    axial_pose_b = write_and_measure(axial_trial)
    write_and_measure(baseline_joint_pos_9)
    stabilize_precheck_pair(
        red_robot=robot_7,
        green_robot=robot_9,
        red_position=baseline_joint_pos_7,
        green_position=baseline_joint_pos_9,
        red_zero_velocity=zero_joint_vel_7,
        green_zero_velocity=zero_joint_vel_9,
        red_zero_effort=zero_joint_effort_7,
        green_zero_effort=zero_joint_effort_9,
    )

    _, pitch_delta_b = compute_pose_error(
        baseline_pose_b[:, :3],
        baseline_pose_b[:, 3:],
        pitch_pose_b[:, :3],
        pitch_pose_b[:, 3:],
    )
    _, axial_delta_b = compute_pose_error(
        baseline_pose_b[:, :3],
        baseline_pose_b[:, 3:],
        axial_pose_b[:, :3],
        axial_pose_b[:, 3:],
    )
    pitch_delta_task = quat_apply_inverse(task_frame_quat_b, pitch_delta_b)
    axial_delta_task = quat_apply_inverse(task_frame_quat_b, axial_delta_b)
    passed = bool(
        pitch_delta_task[0, 1] < -math.radians(0.5)
        and axial_delta_task[0, 2] > math.radians(0.5)
    )
    return WristAxisCheck(
        tuple(float(value) for value in pitch_delta_task[0].tolist()),
        tuple(float(value) for value in axial_delta_task[0].tolist()),
        passed,
    )


def task_tensors(
    reference,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_position_b = reference.position + SURFACE_TRANSLATION_B
    neutral_rotation, relative_rotation = split_task_frame_rotation(
        reference.quaternion, reference.relative_rpy
    )
    neutral_quaternion = quaternion_from_rotation_matrix(neutral_rotation)
    relative_quaternion = quaternion_from_rotation_matrix(relative_rotation)
    task_frame_b = torch.tensor(
        [[*target_position_b, *neutral_quaternion]],
        dtype=torch.float32,
        device=device,
    )
    pose_task = torch.tensor(
        [[0.0, 0.0, 0.0, *relative_quaternion]],
        dtype=torch.float32,
        device=device,
    )
    normal_b = torch.from_numpy(
        np.asarray(reference.surface_normal, dtype=np.float32)[None, :]
    ).to(device=device)
    wrench_task = torch.tensor(
        [[0.0, 0.0, -reference.normal_force, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    return task_frame_b, pose_task, normal_b, wrench_task


def compose_task_target(
    task_frame_b: torch.Tensor, pose_task: torch.Tensor
) -> torch.Tensor:
    position_b, quaternion_b = combine_frame_transforms(
        task_frame_b[:, :3],
        task_frame_b[:, 3:7],
        pose_task[:, :3],
        pose_task[:, 3:7],
    )
    return torch.cat((position_b, quaternion_b), dim=-1)


def world_pose(robot: Articulation, pose_b: torch.Tensor) -> torch.Tensor:
    position_w, quaternion_w = combine_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        pose_b[:, :3],
        pose_b[:, 3:7],
    )
    return torch.cat((position_w, quaternion_w), dim=-1)


def blend_startup_target(
    initial_pose_b: torch.Tensor,
    target_pose_b: torch.Tensor,
    normalized_time: float,
) -> torch.Tensor:
    progress, _, _ = quintic_progress(normalized_time)
    position = initial_pose_b[:, :3] + progress * (
        target_pose_b[:, :3] - initial_pose_b[:, :3]
    )
    quaternion = quat_slerp(
        initial_pose_b[0, 3:7].clone(),
        target_pose_b[0, 3:7].clone(),
        progress,
    ).unsqueeze(0)
    return torch.cat((position, quaternion), dim=-1)


def arrow_quaternion_from_x(direction: torch.Tensor) -> torch.Tensor:
    x_axis = torch.nn.functional.normalize(direction, dim=-1)
    z_hint = torch.tensor([[1.0, 0.0, 0.0]], device=direction.device)
    if torch.abs(torch.sum(x_axis * z_hint)).item() > 0.95:
        z_hint = torch.tensor([[0.0, 1.0, 0.0]], device=direction.device)
    y_axis = torch.nn.functional.normalize(
        torch.linalg.cross(z_hint, x_axis), dim=-1
    )
    z_axis = torch.linalg.cross(x_axis, y_axis)
    return quat_from_matrix(torch.stack((x_axis, y_axis, z_axis), dim=-1))


def create_markers() -> dict[str, VisualizationMarkers]:
    frame_cfg = FRAME_MARKER_CFG.copy()
    frame_cfg.markers["frame"].scale = (0.075, 0.075, 0.075)
    command_cfg = RED_ARROW_X_MARKER_CFG.copy()
    command_cfg.markers["arrow"].scale = (0.14, 0.025, 0.025)
    command_cfg.markers["arrow"].visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(1.0, 0.02, 0.78),
        emissive_color=(0.35, 0.0, 0.20),
    )
    measured_cfg = RED_ARROW_X_MARKER_CFG.copy()
    measured_cfg.markers["arrow"].scale = (0.14, 0.025, 0.025)
    measured_cfg.markers["arrow"].visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.0, 0.85, 1.0),
        emissive_color=(0.0, 0.20, 0.30),
    )
    markers: dict[str, VisualizationMarkers] = {}
    for side in ("7", "9"):
        markers[f"current_{side}"] = VisualizationMarkers(
            frame_cfg.replace(prim_path=f"/Visuals/NearFarCurrent{side}")
        )
        markers[f"target_{side}"] = VisualizationMarkers(
            frame_cfg.replace(prim_path=f"/Visuals/NearFarTarget{side}")
        )
        markers[f"command_{side}"] = VisualizationMarkers(
            command_cfg.replace(prim_path=f"/Visuals/NearFarCommandForce{side}")
        )
        markers[f"measured_{side}"] = VisualizationMarkers(
            measured_cfg.replace(prim_path=f"/Visuals/NearFarMeasuredForce{side}")
        )
    return markers


def choose_near_far(surface: SurfaceMap) -> tuple[tuple[float, float], tuple[float, float]]:
    """Use the calibrated Assembly3 scan-start to scan-end ordering.

    The robot's initial joint pose and patient transform are registered to
    ``scan_start_xy``.  A planar base-distance comparison is misleading here
    because reach depends on the full arm configuration, not only XY distance.
    """
    near = np.asarray(
        surface.metadata.get("scan_start_xy", [0.0, 1.18]), dtype=np.float64
    )
    far = np.asarray(
        surface.metadata.get("scan_end_xy", [0.0, 1.34]), dtype=np.float64
    )
    return tuple(near.tolist()), tuple(far.tolist())


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    surface: SurfaceMap,
    collision_sensors_7: list[ContactSensor],
    collision_sensors_9: list[ContactSensor],
) -> None:
    robot_7: Articulation = scene["robot_7dof"]
    robot_9: Articulation = scene["robot_9dof"]
    sensor_7: ContactSensor = scene["contact_7"]
    sensor_9: ContactSensor = scene["contact_9"]
    ee_7_idx = robot_7.find_bodies("linear_probe")[0][0]
    ee_9_idx = robot_9.find_bodies("linear_probe")[0][0]
    joints_7 = robot_7.find_joints("joint[1-7]")[0]
    joints_9 = robot_9.find_joints(["joint[1-7]", "wrist_.*_joint"])[0]
    locked_wrist_ids = robot_7.find_joints("wrist_.*_joint")[0]

    near_xy, far_xy = choose_near_far(surface)
    trajectory = NearFarTrajectory(
        surface,
        near_xy=near_xy,
        far_xy=far_xy,
        target_force=args_cli.normal_force,
        approach_duration=1.0,
        contact_ramp_duration=1.0,
        scan_duration=2.0,
        settle_duration=0.4,
        pitch_duration=1.5,
        return_pitch_duration=0.8,
        axial_slice_duration=2.0,
        pitch_angle=math.radians(-35.0),
        axial_slice_angle=math.radians(90.0),
    )
    print("[INFO] Independent task: near torso end -> far torso end")
    print(f"[INFO] near_xy={near_xy}, far_xy={far_xy}")
    print("[INFO] Far point: 15 N, pitch only, return, then probe-axis +90 deg")
    print("[INFO] Controller: isaaclab.controllers.OperationalSpaceController")
    print("[INFO] RED has J1-J7; GREEN far-point arm null target stays fixed")
    print("[INFO] MAGENTA=commanded force, CYAN=measured force")
    print("[INFO] Close the Isaac Sim window manually to stop the GUI.")

    pose_osc_7 = make_pose_osc(sim.device)
    pose_osc_9 = make_pose_osc(sim.device)
    hybrid_osc_7 = make_hybrid_osc(sim.device)
    hybrid_osc_9 = make_hybrid_osc(sim.device)
    dt = sim.get_physics_dt()

    default_pos_7 = robot_7.data.default_joint_pos.torch.clone()
    default_vel_7 = robot_7.data.default_joint_vel.torch.clone()
    default_pos_9 = robot_9.data.default_joint_pos.torch.clone()
    default_vel_9 = robot_9.data.default_joint_vel.torch.clone()
    zero_7 = torch.zeros_like(default_pos_7)
    zero_9 = torch.zeros_like(default_pos_9)
    for _ in range(8):
        robot_7.write_joint_position_to_sim_index(position=default_pos_7)
        robot_7.write_joint_velocity_to_sim_index(velocity=default_vel_7)
        robot_9.write_joint_position_to_sim_index(position=default_pos_9)
        robot_9.write_joint_velocity_to_sim_index(velocity=default_vel_9)
        robot_7.set_joint_effort_target_index(target=zero_7)
        robot_9.set_joint_effort_target_index(target=zero_9)
        robot_7.write_data_to_sim()
        robot_9.write_data_to_sim()
        sim.step(render=not args_cli.headless)
        scene.update(dt)

    state_7 = robot_state(robot_7, ee_7_idx, joints_7)
    state_9 = robot_state(robot_9, ee_9_idx, joints_9)
    initial_reference = trajectory.reference(0.0)
    initial_task_frame_b, initial_pose_task, _, _ = task_tensors(
        initial_reference, sim.device
    )
    wrist_axis_check = verify_wrist_axis_signs(
        sim,
        scene,
        robot_7,
        robot_9,
        ee_9_idx,
        joints_9,
        initial_task_frame_b[:, 3:7],
    )
    print(
        "[VALIDATION] wrist-axis signs pitch(task xyz)="
        f"{wrist_axis_check.pitch_delta_task_rad} "
        "axial(task xyz)="
        f"{wrist_axis_check.axial_delta_task_rad} "
        f"passed={wrist_axis_check.passed}"
    )
    if not wrist_axis_check.passed:
        raise RuntimeError("authored wrist-axis sign verification failed")
    state_7 = robot_state(robot_7, ee_7_idx, joints_7)
    state_9 = robot_state(robot_9, ee_9_idx, joints_9)
    startup_pose_b = state_7[3].clone()
    initial_target_b = compose_task_target(
        initial_task_frame_b, initial_pose_task
    )
    initial_position_error, initial_rotation_error = compute_pose_error(
        state_7[3][:, :3],
        state_7[3][:, 3:7],
        initial_target_b[:, :3],
        initial_target_b[:, 3:7],
    )
    print(
        "[DIAG] startup pose error="
        f"{1000.0 * torch.linalg.norm(initial_position_error).item():.3f} mm, "
        f"{math.degrees(torch.linalg.norm(initial_rotation_error).item()):.3f} deg"
    )
    policy_9 = NearFarRedundancyPolicy()
    policy_9.initialize(state_9[6][0].detach().cpu().numpy())
    null_target_7 = state_7[6].clone()

    supervisor_7 = ContactSupervisor(contact_loss_limit=args_cli.contact_loss_limit)
    supervisor_9 = ContactSupervisor(contact_loss_limit=args_cli.contact_loss_limit)
    force_filter_7 = ContactForceFilter(history_length=4, low_pass_alpha=0.5)
    force_filter_9 = ContactForceFilter(history_length=4, low_pass_alpha=0.5)
    collision_monitor_7 = CollisionMonitor()
    collision_monitor_9 = CollisionMonitor()
    markers = create_markers()

    hud_label = None
    if not args_cli.headless:
        hud_window = ui.Window(
            "OSC Near-to-Far 7-DoF vs 9-DoF", width=600, height=205
        )
        with hud_window.frame:
            with ui.VStack():
                hud_label = ui.Label("Waiting for OSC data…", word_wrap=True)

    effort_limits_7 = torch.tensor(
        [[123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0]],
        device=sim.device,
    )
    effort_limits_9 = torch.tensor(
        [[123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0, 12.0, 12.0]],
        device=sim.device,
    )
    rate_limits_7 = torch.tensor(
        [[300.0, 300.0, 240.0, 240.0, 180.0, 180.0, 180.0]],
        device=sim.device,
    )
    rate_limits_9 = torch.tensor(
        [[300.0, 300.0, 240.0, 240.0, 180.0, 180.0, 180.0, 120.0, 120.0]],
        device=sim.device,
    )
    applied_7 = torch.clamp(state_7[2].clone(), -effort_limits_7, effort_limits_7)
    applied_9 = torch.clamp(state_9[2].clone(), -effort_limits_9, effort_limits_9)
    locked_wrist_target = default_pos_7[:, locked_wrist_ids]
    pose_kp_task = torch.tensor(
        [[480.0, 480.0, 360.0, 240.0, 240.0, 240.0]],
        dtype=torch.float32,
        device=sim.device,
    )

    previous_phase: NearFarPhase | None = None
    previous_joint_7 = state_7[6].clone()
    previous_joint_9 = state_9[6].clone()
    phase_arm_travel_7 = 0.0
    phase_arm_travel_9 = 0.0
    phase_wrist_travel_9 = 0.0
    phase_start_q8_q9 = state_9[6][0, -2:].detach().cpu().numpy().copy()
    phase_force_7: list[float] = []
    phase_force_9: list[float] = []
    phase_summaries: dict[str, dict[str, object]] = {}
    task_time = 0.0
    step_count = 0
    collision_hold_frame_7 = state_7[3].clone()
    collision_hold_frame_9 = state_9[3].clone()
    collision_hold_pose_7 = torch.zeros_like(state_7[3])
    collision_hold_pose_9 = torch.zeros_like(state_9[3])
    collision_hold_pose_7[:, 6] = 1.0
    collision_hold_pose_9[:, 6] = 1.0
    red_collision_announced = False
    green_collision_announced = False

    while simulation_app.is_running():
        state_7 = robot_state(robot_7, ee_7_idx, joints_7)
        state_9 = robot_state(robot_9, ee_9_idx, joints_9)
        reference = trajectory.reference(task_time)
        task_frame_b, pose_task, normal_b, wrench_task = task_tensors(
            reference, sim.device
        )
        target_b = compose_task_target(task_frame_b, pose_task)
        if reference.phase is NearFarPhase.APPROACH_NEAR:
            target_7 = blend_startup_target(
                startup_pose_b,
                target_b,
                task_time / trajectory.approach_duration,
            )
            target_9 = blend_startup_target(
                startup_pose_b,
                target_b,
                task_time / trajectory.approach_duration,
            )
            task_frame_7 = target_7
            task_frame_9 = target_9
            pose_task_7 = torch.zeros_like(target_7)
            pose_task_9 = torch.zeros_like(target_9)
            pose_task_7[:, 6] = 1.0
            pose_task_9[:, 6] = 1.0
        else:
            task_frame_7 = task_frame_b
            task_frame_9 = task_frame_b
            pose_task_7 = pose_task
            pose_task_9 = pose_task

        if reference.phase is not previous_phase:
            if previous_phase is not None:
                phase_summaries[previous_phase.value] = {
                    "arm_travel_7_rad": phase_arm_travel_7,
                    "arm_travel_9_rad": phase_arm_travel_9,
                    "wrist_travel_9_rad": phase_wrist_travel_9,
                    "green_q8_q9_start_deg": np.degrees(
                        phase_start_q8_q9
                    ).tolist(),
                    "green_q8_q9_end_deg": np.degrees(
                        state_9[6][0, -2:].detach().cpu().numpy()
                    ).tolist(),
                    "mean_force_7_n": (
                        float(np.mean(phase_force_7)) if phase_force_7 else 0.0
                    ),
                    "mean_force_9_n": (
                        float(np.mean(phase_force_9)) if phase_force_9 else 0.0
                    ),
                    "min_force_9_n": (
                        float(np.min(phase_force_9)) if phase_force_9 else 0.0
                    ),
                    "max_force_9_n": (
                        float(np.max(phase_force_9)) if phase_force_9 else 0.0
                    ),
                }
            previous_phase = reference.phase
            null_target_7 = state_7[6].clone()
            policy_9.begin_phase(
                reference.phase, state_9[6][0].detach().cpu().numpy()
            )
            pose_osc_7.reset()
            pose_osc_9.reset()
            hybrid_osc_7.reset()
            hybrid_osc_9.reset()
            phase_arm_travel_7 = 0.0
            phase_arm_travel_9 = 0.0
            phase_wrist_travel_9 = 0.0
            phase_start_q8_q9 = (
                state_9[6][0, -2:].detach().cpu().numpy().copy()
            )
            phase_force_7 = []
            phase_force_9 = []
            print(
                f"[PHASE] {reference.phase.value} | "
                f"q8/q9={math.degrees(float(state_9[6][0, -2])):+.1f}/"
                f"{math.degrees(float(state_9[6][0, -1])):+.1f} deg"
            )

        green_null_target = torch.tensor(
            policy_9.target(
                state_9[6][0].detach().cpu().numpy(),
                phase=reference.phase,
                relative_pitch=float(reference.relative_rpy[1]),
                relative_axial=float(reference.relative_rpy[2]),
            )[None, :],
            dtype=state_9[6].dtype,
            device=sim.device,
        )

        collision_7 = collision_monitor_7.update(
            maximum_patient_contact_force(collision_sensors_7, dt)
        )
        collision_9 = collision_monitor_9.update(
            maximum_patient_contact_force(collision_sensors_9, dt)
        )
        if collision_7.freeze_path and not red_collision_announced:
            collision_hold_frame_7 = state_7[3].clone()
            collision_hold_pose_7 = torch.zeros_like(state_7[3])
            collision_hold_pose_7[:, 6] = 1.0
            pose_osc_7.reset()
            print(
                f"[SAFETY] RED non-probe collision stop: "
                f"{collision_7.current_force_n:.2f} N"
            )
            red_collision_announced = True
        if collision_9.freeze_path and not green_collision_announced:
            collision_hold_frame_9 = state_9[3].clone()
            collision_hold_pose_9 = torch.zeros_like(state_9[3])
            collision_hold_pose_9[:, 6] = 1.0
            pose_osc_9.reset()
            print(
                f"[SAFETY] GREEN non-probe collision stop: "
                f"{collision_9.current_force_n:.2f} N"
            )
            green_collision_announced = True

        reaction_7_b = quat_apply_inverse(
            robot_7.data.root_quat_w.torch,
            sensor_reaction_force_w(sensor_7, sim.device),
        )
        reaction_9_b = quat_apply_inverse(
            robot_9.data.root_quat_w.torch,
            sensor_reaction_force_w(sensor_9, sim.device),
        )
        measured_force_7 = float(
            torch.clamp(torch.sum(reaction_7_b * normal_b, dim=-1), min=0.0).item()
        )
        measured_force_9 = float(
            torch.clamp(torch.sum(reaction_9_b * normal_b, dim=-1), min=0.0).item()
        )
        filtered_7 = force_filter_7.update(measured_force_7).filtered
        filtered_9 = force_filter_9.update(measured_force_9).filtered
        if reference.phase not in (
            NearFarPhase.APPROACH_NEAR,
            NearFarPhase.CONTACT_RAMP,
        ):
            phase_force_7.append(filtered_7)
            phase_force_9.append(filtered_9)
        contact_required = reference.phase not in (
            NearFarPhase.APPROACH_NEAR,
            NearFarPhase.CONTACT_RAMP,
        )
        supervisor_state_7 = supervisor_7.update(
            dt=dt,
            contact=filtered_7 > 0.5,
            surface_valid=reference.valid,
            measured_force=filtered_7,
            contact_phase=contact_required,
        )
        supervisor_state_9 = supervisor_9.update(
            dt=dt,
            contact=filtered_9 > 0.5,
            surface_valid=reference.valid,
            measured_force=filtered_9,
            contact_phase=contact_required,
        )

        shared_contact_acquisition = (
            reference.phase is NearFarPhase.CONTACT_RAMP
            and (filtered_7 <= 0.5 or filtered_9 <= 0.5)
        )
        red_acquiring = shared_contact_acquisition or supervisor_state_7.freeze_path
        green_acquiring = (
            shared_contact_acquisition or supervisor_state_9.freeze_path
        )
        if red_acquiring and filtered_7 <= 0.5 and contact_required:
            task_frame_7 = task_frame_7.clone()
            task_frame_7[:, :3] -= 0.002 * normal_b
        if green_acquiring and filtered_9 <= 0.5 and contact_required:
            task_frame_9 = task_frame_9.clone()
            task_frame_9[:, :3] -= 0.002 * normal_b

        if collision_7.freeze_path:
            task_frame_7 = collision_hold_frame_7
            pose_task_7 = collision_hold_pose_7
        if collision_9.freeze_path:
            task_frame_9 = collision_hold_frame_9
            pose_task_9 = collision_hold_pose_9

        wrench_7 = wrench_task.clone()
        wrench_9 = wrench_task.clone()
        red_use_pose = (
            reference.phase is NearFarPhase.APPROACH_NEAR
            or red_acquiring
            or collision_7.freeze_path
        )
        green_use_pose = (
            reference.phase is NearFarPhase.APPROACH_NEAR
            or green_acquiring
            or collision_9.freeze_path
        )
        if supervisor_state_7.zero_force_command or red_use_pose:
            wrench_7.zero_()
        if supervisor_state_9.zero_force_command or green_use_pose:
            wrench_9.zero_()

        pose_command_7 = torch.cat((pose_task_7, pose_kp_task), dim=-1)
        pose_command_9 = torch.cat((pose_task_9, pose_kp_task), dim=-1)
        hybrid_command_7 = torch.cat(
            (pose_task_7, wrench_7, pose_kp_task), dim=-1
        )
        hybrid_command_9 = torch.cat(
            (pose_task_9, wrench_9, pose_kp_task), dim=-1
        )
        command_7 = pose_command_7 if red_use_pose else hybrid_command_7
        command_9 = pose_command_9 if green_use_pose else hybrid_command_9
        osc_7 = pose_osc_7 if red_use_pose else hybrid_osc_7
        osc_9 = pose_osc_9 if green_use_pose else hybrid_osc_9
        osc_7.set_command(
            command_7,
            current_ee_pose_b=state_7[3],
            current_task_frame_pose_b=task_frame_7,
        )
        osc_9.set_command(
            command_9,
            current_ee_pose_b=state_9[3],
            current_task_frame_pose_b=task_frame_9,
        )
        torque_7 = osc_7.compute(
            jacobian_b=state_7[0],
            current_ee_pose_b=state_7[3],
            current_ee_vel_b=state_7[4],
            current_ee_force_b=-reaction_7_b,
            mass_matrix=state_7[1],
            gravity=state_7[2],
            current_joint_pos=state_7[6],
            current_joint_vel=state_7[7],
            nullspace_joint_pos_target=null_target_7,
        )
        torque_9 = osc_9.compute(
            jacobian_b=state_9[0],
            current_ee_pose_b=state_9[3],
            current_ee_vel_b=state_9[4],
            current_ee_force_b=-reaction_9_b,
            mass_matrix=state_9[1],
            gravity=state_9[2],
            current_joint_pos=state_9[6],
            current_joint_vel=state_9[7],
            nullspace_joint_pos_target=green_null_target,
        )
        if step_count == 0:
            print(
                f"[DIAG] first raw OSC torque 7={torque_7[0].tolist()}\n"
                f"[DIAG] first raw OSC torque 9={torque_9[0].tolist()}"
            )
        torque_7 = torch.clamp(torque_7, -effort_limits_7, effort_limits_7)
        torque_9 = torch.clamp(torque_9, -effort_limits_9, effort_limits_9)
        applied_7 += torch.clamp(
            torque_7 - applied_7, -rate_limits_7 * dt, rate_limits_7 * dt
        )
        applied_9 += torch.clamp(
            torque_9 - applied_9, -rate_limits_9 * dt, rate_limits_9 * dt
        )
        robot_7.set_joint_effort_target_index(target=applied_7, joint_ids=joints_7)
        robot_7.set_joint_position_target_index(
            target=locked_wrist_target,
            joint_ids=locked_wrist_ids,
        )
        robot_9.set_joint_effort_target_index(target=applied_9, joint_ids=joints_9)
        robot_7.write_data_to_sim()
        robot_9.write_data_to_sim()

        actual_target_7_b = compose_task_target(task_frame_7, pose_task_7)
        actual_target_9_b = compose_task_target(task_frame_9, pose_task_9)
        target_7_w = world_pose(robot_7, actual_target_7_b)
        target_9_w = world_pose(robot_9, actual_target_9_b)
        markers["current_7"].visualize(state_7[5][:, :3], state_7[5][:, 3:7])
        markers["current_9"].visualize(state_9[5][:, :3], state_9[5][:, 3:7])
        markers["target_7"].visualize(target_7_w[:, :3], target_7_w[:, 3:7])
        markers["target_9"].visualize(target_9_w[:, :3], target_9_w[:, 3:7])
        force_direction_b = -normal_b
        command_quaternion_b = arrow_quaternion_from_x(force_direction_b)
        measured_quaternion_b = arrow_quaternion_from_x(normal_b)
        for side, robot, actual_target_b, commanded, measured in (
            (
                "7",
                robot_7,
                actual_target_7_b,
                0.0 if red_use_pose else abs(float(wrench_7[0, 2])),
                filtered_7,
            ),
            (
                "9",
                robot_9,
                actual_target_9_b,
                0.0 if green_use_pose else abs(float(wrench_9[0, 2])),
                filtered_9,
            ),
        ):
            command_arrow_b = torch.cat(
                (
                    actual_target_b[:, :3] - 0.08 * force_direction_b,
                    command_quaternion_b,
                ),
                dim=-1,
            )
            measured_arrow_b = command_arrow_b.clone()
            measured_arrow_b[:, 0] -= 0.075
            measured_arrow_b[:, 3:7] = measured_quaternion_b
            command_arrow_w = world_pose(robot, command_arrow_b)
            measured_arrow_w = world_pose(robot, measured_arrow_b)
            command_scale = torch.tensor(
                [[max(0.001, commanded / args_cli.normal_force), 1.0, 1.0]],
                device=sim.device,
            )
            measured_scale = torch.tensor(
                [[max(0.001, measured / args_cli.normal_force), 1.0, 1.0]],
                device=sim.device,
            )
            markers[f"command_{side}"].visualize(
                command_arrow_w[:, :3],
                command_arrow_w[:, 3:7],
                command_scale,
            )
            markers[f"measured_{side}"].visualize(
                measured_arrow_w[:, :3],
                measured_arrow_w[:, 3:7],
                measured_scale,
            )

        sim.step(render=not args_cli.headless)
        scene.update(dt)
        step_count += 1
        if not supervisor_state_9.freeze_path and not collision_9.freeze_path:
            task_time = min(task_time + dt, trajectory.total_duration + dt)

        post_state_7 = robot_state(robot_7, ee_7_idx, joints_7)
        post_state_9 = robot_state(robot_9, ee_9_idx, joints_9)
        delta_7 = torch.abs(post_state_7[6] - previous_joint_7)
        delta_9 = torch.abs(post_state_9[6] - previous_joint_9)
        phase_arm_travel_7 += float(torch.sum(delta_7[:, :7]).item())
        phase_arm_travel_9 += float(torch.sum(delta_9[:, :7]).item())
        phase_wrist_travel_9 += float(torch.sum(delta_9[:, 7:]).item())
        previous_joint_7 = post_state_7[6].clone()
        previous_joint_9 = post_state_9[6].clone()

        metric_interval = max(1, round(0.1 / dt))
        if step_count % metric_interval == 0:
            reduction = (
                100.0 * (1.0 - phase_arm_travel_9 / phase_arm_travel_7)
                if phase_arm_travel_7 > 1.0e-8
                else 0.0
            )
            text = (
                f"{reference.phase.value} | force 7/9: "
                f"{filtered_7:.1f}/{filtered_9:.1f} N (target 15 N)\n"
                f"phase main-arm travel 7/9: "
                f"{phase_arm_travel_7:.3f}/{phase_arm_travel_9:.3f} rad | "
                f"9DoF reduction: {reduction:.1f}%\n"
                f"green wrist travel: {phase_wrist_travel_9:.3f} rad | "
                f"q8/q9: {math.degrees(float(post_state_9[6][0, -2])):+.1f}/"
                f"{math.degrees(float(post_state_9[6][0, -1])):+.1f} deg | "
                f"collision 7/9: {collision_7.level.value}/"
                f"{collision_9.level.value}"
            )
            if hud_label is not None:
                hud_label.text = text
            print(f"[METRIC t={task_time:.2f}] {text.replace(chr(10), ' | ')}")

        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            break

    if previous_phase is not None:
        phase_summaries[previous_phase.value] = {
            "arm_travel_7_rad": phase_arm_travel_7,
            "arm_travel_9_rad": phase_arm_travel_9,
            "wrist_travel_9_rad": phase_wrist_travel_9,
            "green_q8_q9_start_deg": np.degrees(phase_start_q8_q9).tolist(),
            "green_q8_q9_end_deg": np.degrees(
                post_state_9[6][0, -2:].detach().cpu().numpy()
            ).tolist(),
            "mean_force_7_n": (
                float(np.mean(phase_force_7)) if phase_force_7 else 0.0
            ),
            "mean_force_9_n": (
                float(np.mean(phase_force_9)) if phase_force_9 else 0.0
            ),
            "min_force_9_n": (
                float(np.min(phase_force_9)) if phase_force_9 else 0.0
            ),
            "max_force_9_n": (
                float(np.max(phase_force_9)) if phase_force_9 else 0.0
            ),
        }
    final_position_error_9, final_rotation_error_9 = compute_pose_error(
        post_state_9[3][:, :3],
        post_state_9[3][:, 3:7],
        actual_target_9_b[:, :3],
        actual_target_9_b[:, 3:7],
    )
    summary = {
        "controller": "isaaclab.controllers.OperationalSpaceController",
        "physics_steps": step_count,
        "task_time_s": task_time,
        "trajectory_duration_s": trajectory.total_duration,
        "final_phase": reference.phase.value,
        "normal_force_target_n": args_cli.normal_force,
        "green_max_contact_loss_s": (
            supervisor_state_9.max_contact_loss_duration
        ),
        "green_nonprobe_collision_peak_n": collision_9.peak_force_n,
        "green_collision_latched": collision_9.freeze_path,
        "green_final_force_n": filtered_9,
        "green_final_position_error_mm": (
            1000.0 * torch.linalg.norm(final_position_error_9).item()
        ),
        "green_final_orientation_error_deg": math.degrees(
            torch.linalg.norm(final_rotation_error_9).item()
        ),
        "phase_summaries": phase_summaries,
    }
    print("[FINAL SUMMARY] " + json.dumps(summary, sort_keys=True))


def main() -> int:
    for path, label in (
        (ROBOT_USD, "robot wrapper"),
        (PATIENT_USD, "patient USD"),
        (SURFACE_MAP_PATH, "surface map"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    surface = SurfaceMap.load(SURFACE_MAP_PATH)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=0.004,
            render_interval=4,
            device=args_cli.device,
            use_fabric=True,
        )
    )
    sim.set_camera_view(eye=(2.9, 3.6, 2.15), target=(0.55, -0.20, 0.60))
    scene = InteractiveScene(NearFarSceneCfg(num_envs=1, env_spacing=3.0))
    collision_sensors_7 = make_patient_collision_sensors("7")
    collision_sensors_9 = make_patient_collision_sensors("9")
    sim.reset()
    scene.update(sim.get_physics_dt())
    print("[INFO] Fixed Assembly3 near-to-far scene ready.")
    run_simulator(
        sim,
        scene,
        surface,
        collision_sensors_7,
        collision_sensors_9,
    )
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        simulation_app.close(exit_code=exit_code)
    raise SystemExit(exit_code)
