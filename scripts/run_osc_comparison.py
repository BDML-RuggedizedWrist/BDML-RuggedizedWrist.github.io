#!/usr/bin/env python3
"""Isaac Lab OSC comparison for 7-DoF and 9-DoF Rizon ultrasound contact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rizon_osc.force_control import ContactForceFilter
from rizon_osc.metrics import AcceptanceMetrics, MetricSample
from rizon_osc.osc_profile import hybrid_osc_kwargs, pose_osc_kwargs
from rizon_osc.redundancy_policy import RedundancyPolicy
from rizon_osc.scene_assets import AssetPaths
from rizon_osc.state_machine import ContactSupervisor, phase_requires_contact
from rizon_osc.surface_model import SurfaceMap
from rizon_osc.trajectory import (
    Phase,
    SurfaceTrajectory,
    quaternion_from_rotation_matrix,
    rotation_matrix_from_quaternion,
)

from isaaclab.app import AppLauncher


DEFAULT_PATHS = AssetPaths.from_environment()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Zero keeps the GUI open until the user closes it.",
)
parser.add_argument("--normal_force", type=float, default=15.0)
parser.add_argument(
    "--force_gain",
    type=float,
    default=0.8,
    help="Isaac Lab OSC contact_wrench_stiffness_task gain on task Z.",
)
parser.add_argument(
    "--robot_usd", type=Path, default=DEFAULT_PATHS.robot_wrapper
)
parser.add_argument(
    "--patient_usd", type=Path, default=DEFAULT_PATHS.patient_usd
)
parser.add_argument(
    "--surface_map", type=Path, default=DEFAULT_PATHS.surface_map
)
parser.add_argument(
    "--validation_report",
    type=Path,
    default=None,
    help="Write a JSON acceptance report. A failed full run exits nonzero.",
)
parser.add_argument(
    "--contact_loss_limit",
    type=float,
    default=0.1,
    help="Seconds without contact before path freeze; use a larger value only for diagnostics.",
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
    subtract_frame_transforms,
)
from pxr import Usd, UsdGeom


ROBOT_USD = args_cli.robot_usd.expanduser().resolve()
PATIENT_USD = args_cli.patient_usd.expanduser().resolve()
SURFACE_MAP_PATH = args_cli.surface_map.expanduser().resolve()
GROUND_USD = DEFAULT_PATHS.ground_usd

ROBOT_Y_7 = -1.0
ROBOT_Y_9 = 1.0
ROBOT_ROOT_Z = 0.35
# Registered from the exact Rizon/wrist/probe default pose: Assembly3's scan
# start is 5 mm below the acoustic face, so approach begins without an
# artificial 47 mm lateral repositioning transient.
PATIENT_ROOT_X = 0.70057756
PATIENT_ROOT_Z = 0.28493234
# Assembly3 local y=1.18 maps to the initial probe base y=-0.11225561.
PATIENT_ROOT_Y_FROM_ROBOT = -1.29225561
PATIENT_BED_LOCAL_CENTER_Y = 0.90
SURFACE_TRANSLATION_B = np.array(
    [PATIENT_ROOT_X, PATIENT_ROOT_Y_FROM_ROBOT, PATIENT_ROOT_Z - ROBOT_ROOT_Z]
)
PROBE_TIP_OFFSET = 0.13254
ULTRASOUND_GEL_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.08,
    dynamic_friction=0.05,
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


def make_robot_cfg(root_y: float, wrist_active: bool) -> ArticulationCfg:
    """Create an explicitly fixed Rizon with effort-controlled arm joints."""
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
                damping=5.0,
            ),
            "elbow_effort": ImplicitActuatorCfg(
                joint_names_expr=["joint[3-4]"],
                effort_limit_sim=64.0,
                velocity_limit_sim=2.443,
                stiffness=0.0,
                damping=4.0,
            ),
            "arm_wrist_effort": ImplicitActuatorCfg(
                joint_names_expr=["joint[5-7]"],
                effort_limit_sim=39.0,
                velocity_limit_sim=4.887,
                stiffness=0.0,
                damping=2.0,
            ),
            "supplemental_wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_.*_joint"],
                effort_limit_sim=12.0,
                velocity_limit_sim=2.0,
                stiffness=0.0 if wrist_active else 45.0,
                damping=4.0 if wrist_active else 7.0,
            ),
        },
    )


def patient_cfg(prim_path: str, robot_y: float) -> AssetBaseCfg:
    """Spawn the project-local static patient asset."""
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
    """A collision-enabled static cuboid (no rigid-body API)."""
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, metallic=0.0, roughness=0.72
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
    )


@configclass
class ComparisonSceneCfg(InteractiveSceneCfg):
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
    cfg = OperationalSpaceControllerCfg(**pose_osc_kwargs())
    return OperationalSpaceController(cfg, num_envs=1, device=device)


def make_hybrid_osc(device: str) -> OperationalSpaceController:
    """Create Isaac Lab's closed-loop hybrid motion/force OSC."""
    cfg = OperationalSpaceControllerCfg(
        **hybrid_osc_kwargs(force_gain=args_cli.force_gain)
    )
    return OperationalSpaceController(cfg, num_envs=1, device=device)


def robot_state(
    robot: Articulation,
    ee_body_idx: int,
    controlled_joint_ids: list[int],
) -> tuple[torch.Tensor, ...]:
    """Return official OSC inputs evaluated at the probe acoustic face."""
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
        robot.data.body_vel_w.torch[:, ee_body_idx] - robot.data.root_vel_w.torch
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
    linear_velocity_b += torch.cross(
        angular_velocity_b, tip_offset_b, dim=-1
    )

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


def _tensor_data(proxy) -> torch.Tensor | None:
    if proxy is None:
        return None
    return proxy.torch if hasattr(proxy, "torch") else proxy


def sensor_reaction_force_w(sensor: ContactSensor, device: str) -> torch.Tensor:
    """Average net probe contact force exactly as Isaac Lab's OSC tutorial."""
    net_history = _tensor_data(sensor.data.net_forces_w_history)
    if net_history is not None:
        return torch.mean(net_history, dim=1).sum(dim=1)
    net_force = _tensor_data(sensor.data.net_forces_w)
    if net_force is not None:
        return net_force.sum(dim=1)
    return torch.zeros((1, 3), device=device)


def task_tensors(
    trajectory_reference,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform one Assembly3-local reference into the common robot-base task."""
    target_position_b = trajectory_reference.position + SURFACE_TRANSLATION_B
    target_rotation = rotation_matrix_from_quaternion(
        trajectory_reference.quaternion
    )
    relative_rotation = _rotation_from_rpy(trajectory_reference.relative_rpy)
    neutral_rotation = target_rotation @ relative_rotation.T
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
    target_b = torch.tensor(
        [[*target_position_b, *trajectory_reference.quaternion]],
        dtype=torch.float32,
        device=device,
    )
    normal_b = torch.from_numpy(
        np.asarray(trajectory_reference.surface_normal, dtype=np.float32)[None, :]
    ).to(device=device)
    wrench_task = torch.tensor(
        [[0.0, 0.0, -trajectory_reference.normal_force, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    return task_frame_b, pose_task, target_b, normal_b, wrench_task


def _rotation_from_rpy(relative_rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = relative_rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def world_pose(robot: Articulation, pose_b: torch.Tensor) -> torch.Tensor:
    position_w, quaternion_w = combine_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        pose_b[:, :3],
        pose_b[:, 3:7],
    )
    return torch.cat((position_w, quaternion_w), dim=-1)


def compose_task_target(
    task_frame_b: torch.Tensor, pose_task: torch.Tensor
) -> torch.Tensor:
    """Compose the actual pose command supplied to Isaac Lab OSC."""
    position_b, quaternion_b = combine_frame_transforms(
        task_frame_b[:, :3],
        task_frame_b[:, 3:7],
        pose_task[:, :3],
        pose_task[:, 3:7],
    )
    return torch.cat((position_b, quaternion_b), dim=-1)


def arrow_quaternion_from_x(direction: torch.Tensor) -> torch.Tensor:
    x_axis = torch.nn.functional.normalize(direction, dim=-1)
    z_hint = torch.tensor(
        [[1.0, 0.0, 0.0]], device=direction.device
    ).expand_as(x_axis)
    if torch.abs(torch.sum(x_axis * z_hint)).item() > 0.95:
        z_hint = torch.tensor(
            [[0.0, 1.0, 0.0]], device=direction.device
        ).expand_as(x_axis)
    y_axis = torch.nn.functional.normalize(
        torch.linalg.cross(z_hint, x_axis), dim=-1
    )
    z_axis = torch.linalg.cross(x_axis, y_axis)
    return quat_from_matrix(torch.stack((x_axis, y_axis, z_axis), dim=-1))


def static_transform_snapshot(stage: Usd.Stage) -> dict[str, np.ndarray]:
    paths = (
        "/World/envs/env_0/Robot7DoF",
        "/World/envs/env_0/Robot9DoF",
        "/World/envs/env_0/Patient7DoF/torso_collider",
        "/World/envs/env_0/Patient9DoF/torso_collider",
        "/World/envs/env_0/Patient7DoF/Bed",
        "/World/envs/env_0/Patient9DoF/Bed",
        "/World/envs/env_0/Pedestal7DoF",
        "/World/envs/env_0/Pedestal9DoF",
    )
    snapshot: dict[str, np.ndarray] = {}
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise RuntimeError(f"Required fixed prim is missing: {path}")
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        snapshot[path] = np.asarray(matrix, dtype=np.float64)
    return snapshot


def maximum_static_drift(
    stage: Usd.Stage, initial: dict[str, np.ndarray]
) -> float:
    maximum = 0.0
    for path, reference in initial.items():
        matrix = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        current = np.asarray(matrix, dtype=np.float64)
        maximum = max(maximum, float(np.max(np.abs(current - reference))))
    return maximum


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
    return {
        "current_7": VisualizationMarkers(
            frame_cfg.replace(prim_path="/Visuals/Current7")
        ),
        "current_9": VisualizationMarkers(
            frame_cfg.replace(prim_path="/Visuals/Current9")
        ),
        "target_7": VisualizationMarkers(
            frame_cfg.replace(prim_path="/Visuals/Target7")
        ),
        "target_9": VisualizationMarkers(
            frame_cfg.replace(prim_path="/Visuals/Target9")
        ),
        "command_7": VisualizationMarkers(
            command_cfg.replace(prim_path="/Visuals/CommandForce7")
        ),
        "command_9": VisualizationMarkers(
            command_cfg.replace(prim_path="/Visuals/CommandForce9")
        ),
        "measured_7": VisualizationMarkers(
            measured_cfg.replace(prim_path="/Visuals/MeasuredForce7")
        ),
        "measured_9": VisualizationMarkers(
            measured_cfg.replace(prim_path="/Visuals/MeasuredForce9")
        ),
    }


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    surface: SurfaceMap,
) -> dict:
    robot_7: Articulation = scene["robot_7dof"]
    robot_9: Articulation = scene["robot_9dof"]
    sensor_7: ContactSensor = scene["contact_7"]
    sensor_9: ContactSensor = scene["contact_9"]
    ee_7_idx = robot_7.find_bodies("linear_probe")[0][0]
    ee_9_idx = robot_9.find_bodies("linear_probe")[0][0]
    joints_7 = robot_7.find_joints("joint[1-7]")[0]
    joints_9 = robot_9.find_joints(["joint[1-7]", "wrist_.*_joint"])[0]
    locked_wrist_ids = robot_7.find_joints("wrist_.*_joint")[0]

    print("[INFO] Controller: isaaclab.controllers.OperationalSpaceController")
    print("[INFO] Contact control: built-in current_ee_force_b feedback")
    print("[INFO] RED: Rizon joints 1-7; supplemental wrist locked")
    print("[INFO] GREEN: all 9 joints; built-in nullspace position target favors wrist")
    print("[INFO] MAGENTA = commanded force; CYAN = measured force")
    print("[INFO] Close the Isaac Sim window manually to stop the GUI.")

    pose_osc_7 = make_pose_osc(sim.device)
    pose_osc_9 = make_pose_osc(sim.device)
    hybrid_osc_7 = make_hybrid_osc(sim.device)
    hybrid_osc_9 = make_hybrid_osc(sim.device)
    dt = sim.get_physics_dt()

    scan_start = tuple(surface.metadata.get("scan_start_xy", [0.0, 1.18]))
    scan_end = tuple(surface.metadata.get("scan_end_xy", [0.0, 1.34]))
    trajectory = SurfaceTrajectory(
        surface,
        scan_start_xy=scan_start,
        scan_end_xy=scan_end,
        approach_duration=1.0,
        contact_ramp_duration=0.5,
        scan_duration=4.0,
        pitch_duration=1.2,
        neutral_duration=0.5,
        yaw_duration=1.2,
        approach_clearance=0.012,
        contact_preload=0.004,
        target_force=args_cli.normal_force,
        reorientation_angle=math.radians(20.0),
    )

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
    initial_delta, initial_rotation = compute_pose_error(
        state_7[3][:, :3],
        state_7[3][:, 3:7],
        state_9[3][:, :3],
        state_9[3][:, 3:7],
    )
    print(
        "[INFO] Initial 7/9 probe mismatch: "
        f"{1000 * torch.linalg.norm(initial_delta).item():.3f} mm, "
        f"{math.degrees(torch.linalg.norm(initial_rotation).item()):.3f} deg"
    )
    initial_reference = trajectory.reference(0.0)
    _, _, initial_target_b, initial_normal_b, _ = task_tensors(
        initial_reference, sim.device
    )
    initial_position_error, initial_orientation_error = compute_pose_error(
        state_7[3][:, :3],
        state_7[3][:, 3:7],
        initial_target_b[:, :3],
        initial_target_b[:, 3:7],
    )
    initial_probe_rotation = matrix_from_quat(state_7[3][:, 3:7])
    initial_probe_axis = -initial_probe_rotation[:, :, 2]
    print(
        "[DIAG] initial current pose_b="
        f"{state_7[3][0].tolist()}\n"
        f"[DIAG] initial target pose_b={initial_target_b[0].tolist()}\n"
        f"[DIAG] initial surface normal_b={initial_normal_b[0].tolist()} "
        f"probe_axis_b={initial_probe_axis[0].tolist()}\n"
        f"[DIAG] initial pose error="
        f"{1000 * torch.linalg.norm(initial_position_error).item():.3f} mm, "
        f"{math.degrees(torch.linalg.norm(initial_orientation_error).item()):.3f} deg"
    )

    stage = sim.stage
    fixed_initial = static_transform_snapshot(stage)
    markers = create_markers()
    policy_9 = RedundancyPolicy()
    supervisor_7 = ContactSupervisor(contact_loss_limit=args_cli.contact_loss_limit)
    supervisor_9 = ContactSupervisor(contact_loss_limit=args_cli.contact_loss_limit)
    force_filter_7 = ContactForceFilter(history_length=4, low_pass_alpha=0.5)
    force_filter_9 = ContactForceFilter(history_length=4, low_pass_alpha=0.5)
    metrics = AcceptanceMetrics(force_target=args_cli.normal_force)

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
    null_target_7 = state_7[6].clone()
    previous_phase = ""
    previous_joint_7 = state_7[6].clone()
    previous_joint_9 = state_9[6].clone()
    arm_travel_7 = 0.0
    arm_travel_9 = 0.0
    wrist_travel_9 = 0.0
    phase_arm_travel_7 = 0.0
    phase_arm_travel_9 = 0.0
    phase_wrist_travel_9 = 0.0
    task_time = 0.0
    step_count = 0
    force_diag_printed = False
    last_supervisor_7 = supervisor_7.update(
        dt=0.0,
        contact=False,
        surface_valid=True,
        measured_force=0.0,
        contact_phase=False,
    )
    last_supervisor_9 = last_supervisor_7
    pose_kp_task = torch.tensor(
        [[360.0, 360.0, 360.0, 120.0, 120.0, 120.0]],
        dtype=torch.float32,
        device=sim.device,
    )

    while simulation_app.is_running():
        state_7 = robot_state(robot_7, ee_7_idx, joints_7)
        state_9 = robot_state(robot_9, ee_9_idx, joints_9)
        reference = trajectory.reference(task_time)
        task_frame_b, pose_task, target_b, normal_b, wrench_task = task_tensors(
            reference, sim.device
        )

        if reference.phase.value != previous_phase:
            previous_phase = reference.phase.value
            phase_arm_travel_7 = 0.0
            phase_arm_travel_9 = 0.0
            phase_wrist_travel_9 = 0.0
            null_target_7 = state_7[6].clone()
            policy_9.begin_phase(
                previous_phase, state_9[6][0].detach().cpu().numpy()
            )
            pose_osc_7.reset()
            pose_osc_9.reset()
            hybrid_osc_7.reset()
            hybrid_osc_9.reset()
            print(f"[PHASE] {previous_phase}")
        green_null_target_np = policy_9.target(
            state_9[6][0].detach().cpu().numpy(),
            relative_pitch=float(reference.relative_rpy[1]),
            relative_yaw=float(reference.relative_rpy[2]),
        )
        green_null_target = torch.tensor(
            green_null_target_np[None, :],
            dtype=state_9[6].dtype,
            device=sim.device,
        )

        reaction_7_w = sensor_reaction_force_w(sensor_7, sim.device)
        reaction_9_w = sensor_reaction_force_w(sensor_9, sim.device)
        reaction_7_b = quat_apply_inverse(
            robot_7.data.root_quat_w.torch, reaction_7_w
        )
        reaction_9_b = quat_apply_inverse(
            robot_9.data.root_quat_w.torch, reaction_9_w
        )
        # Isaac Lab OSC expects the wrench exerted by the end effector. The
        # contact sensor reports the equal-and-opposite patient reaction.
        applied_force_7_b = -reaction_7_b
        applied_force_9_b = -reaction_9_b
        measured_force_7 = float(
            torch.clamp(torch.sum(reaction_7_b * normal_b, dim=-1), min=0.0).item()
        )
        measured_force_9 = float(
            torch.clamp(torch.sum(reaction_9_b * normal_b, dim=-1), min=0.0).item()
        )
        filtered_7 = force_filter_7.update(measured_force_7).filtered
        filtered_9 = force_filter_9.update(measured_force_9).filtered
        contact_phase = phase_requires_contact(reference.phase.value)
        last_supervisor_7 = supervisor_7.update(
            dt=dt,
            contact=filtered_7 > 0.5,
            surface_valid=reference.valid,
            measured_force=filtered_7,
            contact_phase=contact_phase,
        )
        last_supervisor_9 = supervisor_9.update(
            dt=dt,
            contact=filtered_9 > 0.5,
            surface_valid=reference.valid,
            measured_force=filtered_9,
            contact_phase=contact_phase,
        )
        if last_supervisor_7.zero_force_command or last_supervisor_9.zero_force_command:
            wrench_task.zero_()
        pose_command = torch.cat((pose_task, pose_kp_task), dim=-1)
        hybrid_command = torch.cat((pose_task, wrench_task, pose_kp_task), dim=-1)

        shared_acquiring = (
            reference.phase is Phase.CONTACT_RAMP
            and (filtered_7 <= 0.5 or filtered_9 <= 0.5)
        ) or (
            (last_supervisor_7.freeze_path and filtered_7 <= 0.5)
            or (last_supervisor_9.freeze_path and filtered_9 <= 0.5)
        )
        task_frame = task_frame_b
        if shared_acquiring and reference.phase is not Phase.CONTACT_RAMP:
            task_frame = task_frame_b.clone()
            task_frame[:, :3] -= 0.002 * normal_b
        use_pose_osc = reference.phase is Phase.APPROACH or shared_acquiring
        command = pose_command if use_pose_osc else hybrid_command
        osc_7 = pose_osc_7 if use_pose_osc else hybrid_osc_7
        osc_9 = pose_osc_9 if use_pose_osc else hybrid_osc_9
        commanded_force = abs(float(wrench_task[0, 2].item())) if not use_pose_osc else 0.0
        commanded_force_7 = commanded_force
        commanded_force_9 = commanded_force
        actual_target_b = compose_task_target(task_frame, pose_task)
        references_identical = True
        osc_7.set_command(
            command,
            current_ee_pose_b=state_7[3],
            current_task_frame_pose_b=task_frame,
        )
        osc_9.set_command(
            command,
            current_ee_pose_b=state_9[3],
            current_task_frame_pose_b=task_frame,
        )
        torque_7 = osc_7.compute(
            jacobian_b=state_7[0],
            current_ee_pose_b=state_7[3],
            current_ee_vel_b=state_7[4],
            current_ee_force_b=applied_force_7_b,
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
            current_ee_force_b=applied_force_9_b,
            mass_matrix=state_9[1],
            gravity=state_9[2],
            current_joint_pos=state_9[6],
            current_joint_vel=state_9[7],
            nullspace_joint_pos_target=green_null_target,
        )
        if reference.normal_force >= 0.99 * args_cli.normal_force and not force_diag_printed:
            desired_force_b = -args_cli.normal_force * normal_b
            print(
                "[DIAG] 15N desired force_b="
                f"{desired_force_b[0].tolist()}\n"
                f"[DIAG] at 15N current tip_b={state_7[3][0, :3].tolist()} "
                f"surface target_b={target_b[0, :3].tolist()}"
            )
            force_diag_printed = True
        if step_count == 0:
            print(
                "[DIAG] first raw OSC torque 7="
                f"{torque_7[0].tolist()}\n"
                "[DIAG] first raw OSC torque 9="
                f"{torque_9[0].tolist()}"
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
            target=locked_wrist_target, joint_ids=locked_wrist_ids
        )
        robot_9.set_joint_effort_target_index(target=applied_9, joint_ids=joints_9)
        robot_7.write_data_to_sim()
        robot_9.write_data_to_sim()

        target_7_w = world_pose(robot_7, actual_target_b)
        target_9_w = world_pose(robot_9, actual_target_b)
        markers["current_7"].visualize(state_7[5][:, :3], state_7[5][:, 3:7])
        markers["current_9"].visualize(state_9[5][:, :3], state_9[5][:, 3:7])
        markers["target_7"].visualize(target_7_w[:, :3], target_7_w[:, 3:7])
        markers["target_9"].visualize(target_9_w[:, :3], target_9_w[:, 3:7])
        force_direction_b = -normal_b
        force_quaternion_b = arrow_quaternion_from_x(force_direction_b)
        reaction_quaternion_b = arrow_quaternion_from_x(normal_b)
        marker_values = {
            "7": (robot_7, actual_target_b, commanded_force_7, filtered_7),
            "9": (robot_9, actual_target_b, commanded_force_9, filtered_9),
        }
        for side, (robot, actual_target_b, commanded_force, measured_force) in marker_values.items():
            command_arrow_b = torch.cat(
                (
                    actual_target_b[:, :3] - 0.08 * force_direction_b,
                    force_quaternion_b,
                ),
                dim=-1,
            )
            measured_arrow_b = command_arrow_b.clone()
            measured_arrow_b[:, 0] -= 0.075
            measured_arrow_b[:, 3:7] = reaction_quaternion_b
            command_arrow_w = world_pose(robot, command_arrow_b)
            measured_arrow_w = world_pose(robot, measured_arrow_b)
            command_scale = torch.tensor(
                [
                    [
                        max(0.001, commanded_force / args_cli.normal_force),
                        1.0,
                        1.0,
                    ]
                ],
                device=sim.device,
            )
            measured_scale = torch.tensor(
                [
                    [
                        max(0.001, measured_force / args_cli.normal_force),
                        1.0,
                        1.0,
                    ]
                ],
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
        if not (last_supervisor_7.freeze_path or last_supervisor_9.freeze_path):
            task_time += dt

        delta_7 = torch.abs(state_7[6] - previous_joint_7)
        delta_9 = torch.abs(state_9[6] - previous_joint_9)
        arm_travel_7 += float(torch.sum(delta_7[:, :7]).item())
        arm_travel_9 += float(torch.sum(delta_9[:, :7]).item())
        wrist_travel_9 += float(torch.sum(delta_9[:, 7:]).item())
        phase_arm_travel_7 += float(torch.sum(delta_7[:, :7]).item())
        phase_arm_travel_9 += float(torch.sum(delta_9[:, :7]).item())
        phase_wrist_travel_9 += float(torch.sum(delta_9[:, 7:]).item())
        previous_joint_7 = state_7[6].clone()
        previous_joint_9 = state_9[6].clone()

        metric_interval = max(1, round(0.1 / dt))
        if step_count % metric_interval == 0:
            position_error_7, rotation_error_7 = compute_pose_error(
                state_7[3][:, :3],
                state_7[3][:, 3:7],
                target_b[:, :3],
                target_b[:, 3:7],
            )
            position_error_9, rotation_error_9 = compute_pose_error(
                state_9[3][:, :3],
                state_9[3][:, 3:7],
                target_b[:, :3],
                target_b[:, 3:7],
            )
            tangent_error_7 = torch.linalg.norm(
                position_error_7
                - torch.sum(position_error_7 * normal_b, dim=-1, keepdim=True)
                * normal_b
            ).item()
            tangent_error_9 = torch.linalg.norm(
                position_error_9
                - torch.sum(position_error_9 * normal_b, dim=-1, keepdim=True)
                * normal_b
            ).item()
            normal_position_error_7 = float(
                torch.sum(position_error_7 * normal_b).item()
            )
            normal_position_error_9 = float(
                torch.sum(position_error_9 * normal_b).item()
            )
            normal_velocity_7 = float(
                torch.sum(state_7[4][:, :3] * normal_b).item()
            )
            normal_velocity_9 = float(
                torch.sum(state_9[4][:, :3] * normal_b).item()
            )
            rotation_7 = matrix_from_quat(state_7[3][:, 3:7])
            rotation_9 = matrix_from_quat(state_9[3][:, 3:7])
            orientation_error_7_deg = math.degrees(
                float(torch.linalg.norm(rotation_error_7).item())
            )
            orientation_error_9_deg = math.degrees(
                float(torch.linalg.norm(rotation_error_9).item())
            )
            acoustic_7 = -rotation_7[:, :, 2]
            acoustic_9 = -rotation_9[:, :, 2]
            desired_acoustic = -normal_b
            angle_7 = math.degrees(
                math.acos(
                    float(
                        torch.clamp(
                            torch.sum(acoustic_7 * desired_acoustic), -1.0, 1.0
                        ).item()
                    )
                )
            )
            angle_9 = math.degrees(
                math.acos(
                    float(
                        torch.clamp(
                            torch.sum(acoustic_9 * desired_acoustic), -1.0, 1.0
                        ).item()
                    )
                )
            )
            drift = maximum_static_drift(stage, fixed_initial)
            if reference.phase not in (Phase.APPROACH, Phase.CONTACT_RAMP):
                metrics.add(
                    MetricSample(
                        phase=reference.phase.value,
                        commanded_force_7=commanded_force_7,
                        commanded_force_9=commanded_force_9,
                        measured_force_7=filtered_7,
                        measured_force_9=filtered_9,
                        normal_angle_7_deg=angle_7,
                        normal_angle_9_deg=angle_9,
                        orientation_error_7_deg=orientation_error_7_deg,
                        orientation_error_9_deg=orientation_error_9_deg,
                        tangent_error_7_m=tangent_error_7,
                        tangent_error_9_m=tangent_error_9,
                        contact_loss_7_s=last_supervisor_7.max_contact_loss_duration,
                        contact_loss_9_s=last_supervisor_9.max_contact_loss_duration,
                        arm_travel_7_rad=arm_travel_7,
                        arm_travel_9_rad=arm_travel_9,
                        wrist_travel_9_rad=wrist_travel_9,
                        static_drift_m=drift,
                        references_identical=references_identical,
                    )
                )
            raw_reduction_text = "n/a"
            if arm_travel_7 > 1.0e-8 and tangent_error_7 < 0.01 and tangent_error_9 < 0.01:
                raw_reduction_text = (
                    f"{100 * (1 - arm_travel_9 / arm_travel_7):.1f}%"
                )
            print(
                f"[METRIC t={task_time:5.2f}s {reference.phase.value}] "
                f"IsaacLab OSC | force cmd 7/9="
                f"{commanded_force_7:4.1f}/{commanded_force_9:4.1f} N "
                f"meas 7/9={filtered_7:4.1f}/{filtered_9:4.1f} N | "
                f"normal angle 7/9={angle_7:4.1f}/{angle_9:4.1f} deg | "
                f"orientation err={orientation_error_7_deg:4.1f}/"
                f"{orientation_error_9_deg:4.1f} deg | "
                f"tangent err={1000*tangent_error_7:4.1f}/"
                f"{1000*tangent_error_9:4.1f} mm | "
                f"normal pos err={1000*normal_position_error_7:+5.1f}/"
                f"{1000*normal_position_error_9:+5.1f} mm | "
                f"normal vel={1000*normal_velocity_7:+5.1f}/"
                f"{1000*normal_velocity_9:+5.1f} mm/s | "
                f"arm travel={arm_travel_7:5.2f}/{arm_travel_9:5.2f} rad "
                f"raw reduction={raw_reduction_text} | "
                f"wrist9={wrist_travel_9:5.2f} rad | "
                f"phase arm={phase_arm_travel_7:4.2f}/"
                f"{phase_arm_travel_9:4.2f} wrist9={phase_wrist_travel_9:4.2f} | "
                f"static drift={drift:.2e}"
            )

        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            break

    scenario_complete = task_time + 0.5 * dt >= trajectory.total_duration
    report = metrics.report(scenario_complete=scenario_complete)
    report["controller"] = (
        "isaaclab.controllers.OperationalSpaceController"
    )
    report["task_time_s"] = task_time
    report["physics_steps"] = step_count
    report["normal_force_target_n"] = args_cli.normal_force
    report["scenario_total_duration_s"] = trajectory.total_duration
    return report


def main() -> int:
    for path, label in (
        (ROBOT_USD, "robot wrapper"),
        (PATIENT_USD, "patient USD"),
        (SURFACE_MAP_PATH, "surface map"),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {label}: {path}\n"
                "Run tools/prepare_local_assets.py with env_isaacsim first."
            )
    surface = SurfaceMap.load(SURFACE_MAP_PATH)
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.004,
        render_interval=4,
        device=args_cli.device,
        use_fabric=True,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.9, 3.6, 2.15), target=(0.55, -0.20, 0.60))
    scene = InteractiveScene(ComparisonSceneCfg(num_envs=1, env_spacing=3.0))
    sim.reset()
    scene.update(sim.get_physics_dt())
    print("[INFO] Fixed Assembly3 scene ready.")
    report = run_simulator(sim, scene, surface)
    if args_cli.validation_report is not None:
        target = args_cli.validation_report.expanduser()
        if not target.is_absolute():
            target = PROJECT_ROOT / target
        target = target.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"[INFO] Validation report: {target}")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("overall_pass", False) else 2
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
