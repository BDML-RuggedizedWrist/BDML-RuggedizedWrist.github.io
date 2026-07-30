#!/usr/bin/env python3
"""Isaac Lab OSC comparison for 7-DoF and 9-DoF Rizon ultrasound contact."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
from rizon_osc.hud import HudSnapshot, format_hud
from rizon_osc.joint_travel import JointTravelTracker
from rizon_osc.metrics import AcceptanceMetrics, MetricSample
from rizon_osc.osc_profile import (
    completion_hold_osc_kwargs,
    hybrid_osc_kwargs,
    pose_osc_kwargs,
    safety_hold_osc_kwargs,
    surface_scan_osc_kwargs,
)
from rizon_osc.redundancy_policy import RedundancyPolicy
from rizon_osc.scene_assets import AssetPaths
from rizon_osc.state_machine import (
    ContactSupervisor,
    SafetyMode,
    phase_requires_contact,
)
from rizon_osc.surface_model import SurfaceMap
from rizon_osc.trajectory import (
    Phase,
    SurfaceTrajectory,
    quintic_progress,
    quaternion_from_rotation_matrix,
    split_task_frame_rotation,
)
from rizon_osc.validation_watchdog import (
    ValidationWatchdog,
    WatchdogSample,
    green_safety_reasons,
)
from rizon_osc.wrist_axis_precheck import stabilize_precheck_pair

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
parser.add_argument(
    "--record_side",
    choices=("7dof", "9dof", "both"),
    default=None,
    help="Isolate one setup and record a clean, matched side-view demo.",
)
parser.add_argument(
    "--record_output",
    type=Path,
    default=None,
    help="Viewport-only MP4 output. Requires --record_side and --max_steps.",
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
if (args_cli.record_side is None) != (args_cli.record_output is None):
    parser.error("--record_side and --record_output must be supplied together")
if args_cli.record_output is not None and args_cli.max_steps <= 0:
    parser.error("--record_output requires a positive --max_steps")
if (
    args_cli.record_side == "both"
    and "{side}" not in str(args_cli.record_output)
):
    parser.error("--record_side both requires {side} in --record_output")
if args_cli.record_output is not None:
    args_cli.enable_cameras = True

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
from isaaclab.sensors import CameraCfg, ContactSensor, ContactSensorCfg
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
from pxr import Usd, UsdGeom
from rizon_osc.demo_recording import (
    CameraVideoRecorder,
    isolate_comparison_side,
    side_camera,
)

if not args_cli.headless:
    import omni.ui as ui


ROBOT_USD = args_cli.robot_usd.expanduser().resolve()
PATIENT_USD = args_cli.patient_usd.expanduser().resolve()
SURFACE_MAP_PATH = args_cli.surface_map.expanduser().resolve()
GROUND_USD = DEFAULT_PATHS.ground_usd

# Keep the public comparison cameras identical while preventing the opposite
# experiment from entering either frame.  The ordinary interactive layout stays
# compact; recording mode only translates the two otherwise identical setups.
_RECORDING_LATERAL_OFFSET = 6.0 if args_cli.record_side is not None else 1.0
ROBOT_Y_7 = -_RECORDING_LATERAL_OFFSET
ROBOT_Y_9 = _RECORDING_LATERAL_OFFSET
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
    if args_cli.record_output is not None:
        recording_camera_7 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/RecordingCamera7",
            update_period=0.0,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=25.0,
                focus_distance=4.0,
                horizontal_aperture=22.0,
                clipping_range=(0.05, 100.0),
            ),
        )
        recording_camera_9 = CameraCfg(
            prim_path="{ENV_REGEX_NS}/RecordingCamera9",
            update_period=0.0,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=25.0,
                focus_distance=4.0,
                horizontal_aperture=22.0,
                clipping_range=(0.05, 100.0),
            ),
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


def make_safety_hold_osc(device: str) -> OperationalSpaceController:
    """Create an official six-axis pose hold plus joint-posture OSC."""
    cfg = OperationalSpaceControllerCfg(**safety_hold_osc_kwargs())
    return OperationalSpaceController(cfg, num_envs=1, device=device)


def make_completion_hold_osc(device: str) -> OperationalSpaceController:
    """Create the official hybrid OSC used after the trajectory ends."""
    cfg = OperationalSpaceControllerCfg(
        **completion_hold_osc_kwargs(force_gain=args_cli.force_gain)
    )
    return OperationalSpaceController(cfg, num_envs=1, device=device)


def make_surface_scan_osc(device: str) -> OperationalSpaceController:
    """Create the official hybrid OSC with redundant probe-axis spin."""
    cfg = OperationalSpaceControllerCfg(
        **surface_scan_osc_kwargs(force_gain=args_cli.force_gain)
    )
    return OperationalSpaceController(cfg, num_envs=1, device=device)


def make_patient_collision_sensors(side: str) -> list[ContactSensor]:
    """Create patient-filtered non-probe contact sensors for one robot."""
    robot_name = f"Robot{side}DoF"
    patient_name = f"Patient{side}DoF"
    patient_surface = (
        f"/World/envs/env_.*/{patient_name}/torso_collider/"
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


@dataclass(frozen=True)
class WristAxisCheck:
    """Measured orientation signs for the authored green wrist axes."""

    pitch_delta_task_rad: tuple[float, float, float]
    yaw_delta_task_rad: tuple[float, float, float]
    passed: bool


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


def verify_wrist_axis_signs(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot_7: Articulation,
    robot_9: Articulation,
    ee_body_idx: int,
    controlled_joint_ids: list[int],
    task_frame_quat_b: torch.Tensor,
) -> WristAxisCheck:
    """Probe each green wrist joint in physics before OSC torque is enabled."""
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
        return robot_state(
            robot_9, ee_body_idx, controlled_joint_ids
        )[3].clone()

    pitch_trial = baseline_joint_pos_9.clone()
    pitch_trial[:, controlled_joint_ids[-2]] += math.radians(1.0)
    pitch_pose_b = write_and_measure(pitch_trial)
    write_and_measure(baseline_joint_pos_9)

    yaw_trial = baseline_joint_pos_9.clone()
    yaw_trial[:, controlled_joint_ids[-1]] += math.radians(1.0)
    yaw_pose_b = write_and_measure(yaw_trial)
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
    _, yaw_delta_b = compute_pose_error(
        baseline_pose_b[:, :3],
        baseline_pose_b[:, 3:],
        yaw_pose_b[:, :3],
        yaw_pose_b[:, 3:],
    )
    pitch_delta_task = quat_apply_inverse(task_frame_quat_b, pitch_delta_b)
    yaw_delta_task = quat_apply_inverse(task_frame_quat_b, yaw_delta_b)
    passed = bool(
        pitch_delta_task[0, 1] < -math.radians(0.5)
        and yaw_delta_task[0, 2] > math.radians(0.5)
    )
    return WristAxisCheck(
        tuple(float(value) for value in pitch_delta_task[0].tolist()),
        tuple(float(value) for value in yaw_delta_task[0].tolist()),
        passed,
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


def maximum_patient_contact_force(
    sensors: list[ContactSensor], dt: float
) -> float:
    """Return the maximum patient-filtered non-probe contact magnitude."""
    maximum = 0.0
    for sensor in sensors:
        sensor.update(dt, force_recompute=True)
        history = _tensor_data(sensor.data.force_matrix_w_history)
        if history is None:
            raise RuntimeError("patient-filtered force history is unavailable")
        maximum = max(
            maximum,
            float(torch.linalg.vector_norm(history, dim=-1).max().item()),
        )
    return maximum


def task_tensors(
    trajectory_reference,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform one Assembly3-local reference into the common robot-base task."""
    target_position_b = trajectory_reference.position + SURFACE_TRANSLATION_B
    neutral_rotation, relative_rotation = split_task_frame_rotation(
        trajectory_reference.quaternion, trajectory_reference.relative_rpy
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


def identity_pose_task(reference: torch.Tensor) -> torch.Tensor:
    """Return an identity pose command using Isaac Lab's xyzw quaternion order."""
    identity = torch.zeros_like(reference)
    identity[:, 6] = 1.0
    return identity


def safety_latch_reason(
    *,
    collision_stop: bool,
    supervisor_mode: SafetyMode,
    latch_contact_loss: bool,
    singularity_speed_guard: bool = False,
) -> str | None:
    """Classify conditions that must never auto-resume into nominal motion."""
    if collision_stop:
        return "nonprobe_collision"
    if supervisor_mode is SafetyMode.FORCE_HOLD:
        return "normal_force_overload"
    if supervisor_mode is SafetyMode.INVALID_SURFACE:
        return "invalid_surface"
    if singularity_speed_guard:
        return "singularity_speed_guard"
    if latch_contact_loss and supervisor_mode is SafetyMode.REACQUIRE:
        return "contact_loss"
    return None


def red_singularity_speed_guard(
    *,
    phase_name: str,
    max_main_arm_speed_rad_s: float,
    threshold_rad_s: float = 1.0,
) -> bool:
    """Preempt the red arm's high-speed singular transient in challenge phases."""
    challenge_phases = (
        "PITCH_ONLY",
        "RETURN_PITCH",
        "YAW_ONLY",
        "RETURN_YAW",
        "CHALLENGE_TRANSIT",
        "CHALLENGE_PITCH_ONLY",
        "RETURN_NEUTRAL",
    )
    return (
        phase_name in challenge_phases
        and max_main_arm_speed_rad_s >= threshold_rad_s
    )


def blend_startup_target(
    initial_pose_b: torch.Tensor,
    target_pose_b: torch.Tensor,
    normalized_time: float,
) -> torch.Tensor:
    """Quintic pose blend that removes the discontinuity at OSC startup."""
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


def augment_validation_watchdog_report(
    report: dict,
    *,
    enabled: bool,
    snapshot,
) -> dict:
    """Attach the pure watchdog schema and apply its validation gate."""
    report["validation_watchdog"] = {
        "enabled": enabled,
        "thresholds": {
            "wrist_limit_margin_rad": ValidationWatchdog.LIMIT_MARGIN_RAD,
            "wrist_limit_duration_s": ValidationWatchdog.LIMIT_DURATION_S,
            "wrist_speed_rad_s": ValidationWatchdog.SPEED_THRESHOLD_RAD_S,
            "wrist_speed_duration_s": ValidationWatchdog.SPEED_DURATION_S,
            "contact_loss_duration_s": (
                ValidationWatchdog.CONTACT_LOSS_DURATION_S
            ),
            "normal_force_limit_n": ValidationWatchdog.FORCE_LIMIT_N,
            "nonprobe_collision_n": (
                ValidationWatchdog.NONPROBE_COLLISION_N
            ),
            "freeze_window_s": ValidationWatchdog.FREEZE_WINDOW_S,
            "translation_command_m": (
                ValidationWatchdog.TRANSLATION_COMMAND_M
            ),
            "translation_response_m": (
                ValidationWatchdog.TRANSLATION_RESPONSE_M
            ),
            "rotation_command_deg": 0.5,
            "rotation_response_deg": 0.05,
        },
        **snapshot.as_dict(),
    }
    report["validation_watchdog"]["green_safety_reasons"] = list(
        green_safety_reasons(snapshot.reasons)
    )
    report["validation_watchdog"]["green_safety_passed"] = not bool(
        green_safety_reasons(snapshot.reasons)
    )
    report["overall_pass"] = bool(
        report.get("overall_pass", False)
        and (
            not enabled
            or not green_safety_reasons(snapshot.reasons)
        )
    )
    return report


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    surface: SurfaceMap,
    *,
    collision_sensors_7: list[ContactSensor],
    collision_sensors_9: list[ContactSensor],
    watchdog_collision_sensors_7: list[ContactSensor],
    watchdog_collision_sensors_9: list[ContactSensor],
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
    wrist_limits_9 = (
        robot_9.data.joint_pos_limits.torch[0, joints_9[-2:], :]
        .detach()
        .cpu()
        .numpy()
    )

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
    safety_hold_osc_7 = make_safety_hold_osc(sim.device)
    safety_hold_osc_9 = make_safety_hold_osc(sim.device)
    completion_hold_osc_9 = make_completion_hold_osc(sim.device)
    scan_osc_7 = make_surface_scan_osc(sim.device)
    scan_osc_9 = make_surface_scan_osc(sim.device)
    dt = sim.get_physics_dt()
    watchdog_enabled = args_cli.validation_report is not None
    validation_watchdog = ValidationWatchdog()
    watchdog_snapshot = validation_watchdog.snapshot()

    scan_start = tuple(surface.metadata.get("scan_start_xy", [0.0, 1.18]))
    scan_end = tuple(surface.metadata.get("scan_end_xy", [0.0, 1.34]))
    trajectory = SurfaceTrajectory(
        surface,
        scan_start_xy=scan_start,
        scan_end_xy=scan_end,
        approach_duration=1.0,
        contact_ramp_duration=1.0,
        scan_duration=2.0,
        settle_duration=0.25,
        pitch_duration=1.8,
        neutral_duration=1.0,
        yaw_duration=2.5,
        pitch_angle=math.radians(-35.0),
        yaw_angle=math.radians(90.0),
        approach_clearance=0.005,
        contact_preload=0.001,
        target_force=args_cli.normal_force,
        reorientation_angle=math.radians(20.0),
        challenge_return_duration=2.0,
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
    initial_task_frame_b, initial_pose_task, initial_target_b, initial_normal_b, _ = task_tensors(
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
        "yaw(task xyz)="
        f"{wrist_axis_check.yaw_delta_task_rad} "
        f"passed={wrist_axis_check.passed}"
    )
    if not wrist_axis_check.passed:
        report = {
            "overall_pass": False,
            "reason": "authored wrist-axis sign verification failed",
            "wrist_axis_check": asdict(wrist_axis_check),
        }
        return augment_validation_watchdog_report(
            report,
            enabled=watchdog_enabled,
            snapshot=watchdog_snapshot,
        )
    state_7 = robot_state(robot_7, ee_7_idx, joints_7)
    state_9 = robot_state(robot_9, ee_9_idx, joints_9)
    startup_pose_b = state_7[3].clone()
    policy_9 = RedundancyPolicy()
    policy_9.initialize_run(state_9[6][0].detach().cpu().numpy())
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
    supervisor_7 = ContactSupervisor(
        contact_loss_limit=args_cli.contact_loss_limit,
        hard_force_limit=30.0,
    )
    supervisor_9 = ContactSupervisor(
        contact_loss_limit=args_cli.contact_loss_limit,
        hard_force_limit=30.0,
    )
    force_filter_7 = ContactForceFilter(history_length=4, low_pass_alpha=0.5)
    force_filter_9 = ContactForceFilter(history_length=4, low_pass_alpha=0.5)
    watchdog_force_filter_7 = ContactForceFilter(
        history_length=4, low_pass_alpha=0.5
    )
    watchdog_force_filter_9 = ContactForceFilter(
        history_length=4, low_pass_alpha=0.5
    )
    collision_monitor_7 = CollisionMonitor()
    collision_monitor_9 = CollisionMonitor()
    metrics = AcceptanceMetrics(force_target=args_cli.normal_force)
    travel_tracker = JointTravelTracker()
    latest_metric_values = np.empty(0, dtype=np.float64)
    hud_window = None
    hud_label = None
    if not args_cli.headless and args_cli.record_output is None:
        hud_window = ui.Window("OSC 7-DoF vs 9-DoF", width=520, height=220)
        with hud_window.frame:
            with ui.VStack():
                hud_label = ui.Label("Waiting for first OSC metric…", word_wrap=True)

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
    red_safety_phase: str | None = None
    task_time = 0.0
    step_count = 0
    force_diag_printed = False
    completion_hold_reported = False
    last_supervisor_7 = supervisor_7.update(
        dt=0.0,
        contact=False,
        surface_valid=True,
        measured_force=0.0,
        contact_phase=False,
    )
    last_supervisor_9 = last_supervisor_7
    pose_kp_task = torch.tensor(
        [[480.0, 480.0, 360.0, 240.0, 240.0, 240.0]],
        dtype=torch.float32,
        device=sim.device,
    )
    red_safety_latched = False
    green_safety_latched = False
    red_hold_task_frame = state_7[3].clone()
    green_hold_task_frame = state_9[3].clone()
    red_hold_null_target = state_7[6].clone()
    green_hold_null_target = state_9[6].clone()
    red_safety_event: dict[str, object] | None = None
    green_safety_event: dict[str, object] | None = None

    recordings: list[CameraVideoRecorder] = []
    if args_cli.record_side is not None:
        if args_cli.record_side != "both":
            isolate_comparison_side(
                stage,
                args_cli.record_side,
                (
                    "/Visuals/Current7",
                    "/Visuals/Target7",
                    "/Visuals/CommandForce7",
                    "/Visuals/MeasuredForce7",
                ),
                (
                    "/Visuals/Current9",
                    "/Visuals/Target9",
                    "/Visuals/CommandForce9",
                    "/Visuals/MeasuredForce9",
                ),
            )
        sides = (
            ("7dof", ROBOT_Y_7, "recording_camera_7"),
            ("9dof", ROBOT_Y_9, "recording_camera_9"),
        )
        for side, robot_y, camera_name in sides:
            if args_cli.record_side not in (side, "both"):
                continue
            camera = scene[camera_name]
            camera_eye, camera_target = side_camera(robot_y)
            camera.set_world_poses_from_view(
                torch.tensor([camera_eye], device=sim.device),
                torch.tensor([camera_target], device=sim.device),
            )
            output_path = Path(
                str(args_cli.record_output).format(side=side)
            )
            recordings.append(CameraVideoRecorder(camera, output_path))

    while simulation_app.is_running():
        state_7 = robot_state(robot_7, ee_7_idx, joints_7)
        state_9 = robot_state(robot_9, ee_9_idx, joints_9)
        reference = trajectory.reference(task_time)
        task_frame_b, pose_task, target_b, normal_b, wrench_task = task_tensors(
            reference, sim.device
        )
        if reference.phase is Phase.APPROACH:
            target_b = blend_startup_target(
                startup_pose_b,
                target_b,
                task_time / trajectory.approach_duration,
            )
            task_frame_b = target_b.clone()
            pose_task = torch.zeros_like(target_b)
            pose_task[:, 6] = 1.0
        collision_7 = collision_monitor_7.update(
            maximum_patient_contact_force(collision_sensors_7, dt)
        )
        collision_9 = collision_monitor_9.update(
            maximum_patient_contact_force(collision_sensors_9, dt)
        )
        if reference.phase.value != previous_phase:
            previous_phase = reference.phase.value
            travel_tracker.begin_phase(
                reference.phase.value,
                state_7[6][0].detach().cpu().numpy(),
                state_9[6][0].detach().cpu().numpy(),
            )
            null_target_7 = state_7[6].clone()
            policy_9.begin_phase(
                reference.phase.value, state_9[6][0].detach().cpu().numpy()
            )
            pose_osc_7.reset()
            pose_osc_9.reset()
            hybrid_osc_7.reset()
            hybrid_osc_9.reset()
            safety_hold_osc_7.reset()
            safety_hold_osc_9.reset()
            completion_hold_osc_9.reset()
            scan_osc_7.reset()
            scan_osc_9.reset()
            q8_q9 = state_9[6][0, -2:].detach().cpu().numpy()
            print(
                f"[PHASE] {previous_phase} "
                f"green_q8={math.degrees(float(q8_q9[0])):.1f}deg "
                f"green_q9={math.degrees(float(q8_q9[1])):.1f}deg"
            )
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
        max_main_arm_speed_7 = float(
            torch.max(torch.abs(state_7[7][:, :7])).item()
        )
        red_speed_guard = red_singularity_speed_guard(
            phase_name=reference.phase.value,
            max_main_arm_speed_rad_s=max_main_arm_speed_7,
        )
        red_reason = safety_latch_reason(
            collision_stop=collision_7.freeze_path,
            supervisor_mode=last_supervisor_7.mode,
            latch_contact_loss=True,
            singularity_speed_guard=red_speed_guard,
        )
        green_reason = safety_latch_reason(
            collision_stop=collision_9.freeze_path,
            supervisor_mode=last_supervisor_9.mode,
            latch_contact_loss=False,
        )
        if red_reason is not None and not red_safety_latched:
            red_safety_latched = True
            red_safety_phase = reference.phase.value
            red_hold_task_frame = state_7[3].clone()
            red_hold_null_target = state_7[6].clone()
            safety_hold_osc_7.reset()
            red_safety_event = {
                "reason": red_reason,
                "phase": reference.phase.value,
                "task_time_s": task_time,
                "measured_force_n": filtered_7,
                "nonprobe_force_n": collision_7.current_force_n,
                "arm_travel_at_latch_rad": arm_travel_7,
                "max_joint_speed_rad_s": max_main_arm_speed_7,
                "max_tip_position_error_m": 0.0,
                "max_tip_orientation_error_deg": 0.0,
            }
            print(
                "[SAFETY RED] irreversible OSC hold: "
                f"reason={red_reason} phase={reference.phase.value} "
                f"t={task_time:.3f}s force={filtered_7:.2f}N "
                f"nonprobe={collision_7.current_force_n:.2f}N"
            )
        if green_reason is not None and not green_safety_latched:
            green_safety_latched = True
            green_hold_task_frame = state_9[3].clone()
            green_hold_null_target = state_9[6].clone()
            safety_hold_osc_9.reset()
            green_safety_event = {
                "reason": green_reason,
                "phase": reference.phase.value,
                "task_time_s": task_time,
                "measured_force_n": filtered_9,
                "nonprobe_force_n": collision_9.current_force_n,
                "arm_travel_at_latch_rad": arm_travel_9,
                "max_joint_speed_rad_s": float(
                    torch.max(torch.abs(state_9[7])).item()
                ),
                "max_tip_position_error_m": 0.0,
                "max_tip_orientation_error_deg": 0.0,
            }
            print(
                "[SAFETY GREEN] irreversible OSC hold: "
                f"reason={green_reason} phase={reference.phase.value} "
                f"t={task_time:.3f}s force={filtered_9:.2f}N "
                f"nonprobe={collision_9.current_force_n:.2f}N"
            )
        wrench_task_7 = wrench_task.clone()
        wrench_task_9 = wrench_task.clone()
        if last_supervisor_7.zero_force_command:
            wrench_task_7.zero_()
        if last_supervisor_9.zero_force_command:
            wrench_task_9.zero_()
        hybrid_command_7 = torch.cat(
            (pose_task, wrench_task_7, pose_kp_task), dim=-1
        )
        hybrid_command_9 = torch.cat(
            (pose_task, wrench_task_9, pose_kp_task), dim=-1
        )

        initial_shared_acquiring = (
            reference.phase is Phase.CONTACT_RAMP
            and not red_safety_latched
            and not green_safety_latched
            and (filtered_7 <= 0.5 or filtered_9 <= 0.5)
        )
        red_acquiring = (
            not red_safety_latched
            and (initial_shared_acquiring or last_supervisor_7.freeze_path)
        )
        green_acquiring = (
            not green_safety_latched
            and (initial_shared_acquiring or last_supervisor_9.freeze_path)
        )
        task_frame_7 = task_frame_b
        task_frame_9 = task_frame_b
        if (
            red_acquiring
            and filtered_7 <= 0.5
            and reference.phase is not Phase.CONTACT_RAMP
        ):
            task_frame_7 = task_frame_b.clone()
            task_frame_7[:, :3] -= 0.002 * normal_b
        if (
            green_acquiring
            and filtered_9 <= 0.5
            and reference.phase is not Phase.CONTACT_RAMP
        ):
            task_frame_9 = task_frame_b.clone()
            task_frame_9[:, :3] -= 0.002 * normal_b
        red_pose_task = pose_task
        green_pose_task = pose_task
        if red_safety_latched:
            task_frame_7 = red_hold_task_frame
            red_pose_task = identity_pose_task(pose_task)
            wrench_task_7.zero_()
        if green_safety_latched:
            task_frame_9 = green_hold_task_frame
            green_pose_task = identity_pose_task(pose_task)
            wrench_task_9.zero_()
        red_use_pose_osc = (
            reference.phase is Phase.APPROACH
            or red_acquiring
            or red_safety_latched
        )
        green_use_pose_osc = (
            reference.phase is Phase.APPROACH
            or green_acquiring
            or green_safety_latched
        )
        red_pose_command = torch.cat((red_pose_task, pose_kp_task), dim=-1)
        green_pose_command = torch.cat((green_pose_task, pose_kp_task), dim=-1)
        command_7 = red_pose_command if red_use_pose_osc else hybrid_command_7
        command_9 = green_pose_command if green_use_pose_osc else hybrid_command_9
        if red_safety_latched:
            osc_7 = safety_hold_osc_7
        elif red_use_pose_osc:
            osc_7 = pose_osc_7
        elif reference.phase is Phase.SURFACE_SCAN:
            osc_7 = scan_osc_7
        else:
            osc_7 = hybrid_osc_7
        if green_safety_latched:
            osc_9 = safety_hold_osc_9
        elif green_use_pose_osc:
            osc_9 = pose_osc_9
        elif completion_hold_reported:
            osc_9 = completion_hold_osc_9
        elif reference.phase is Phase.SURFACE_SCAN:
            osc_9 = scan_osc_9
        else:
            osc_9 = hybrid_osc_9
        commanded_force_7_raw = abs(float(wrench_task_7[0, 2].item()))
        commanded_force_9_raw = abs(float(wrench_task_9[0, 2].item()))
        commanded_force_7 = (
            0.0
            if red_use_pose_osc
            else commanded_force_7_raw
        )
        commanded_force_9 = (
            0.0
            if green_use_pose_osc
            else commanded_force_9_raw
        )
        actual_target_7_b = compose_task_target(task_frame_7, red_pose_task)
        actual_target_9_b = compose_task_target(task_frame_9, green_pose_task)
        if red_safety_latched or green_safety_latched:
            references_identical = False
        else:
            references_identical = (
                command_7.shape == command_9.shape
                and torch.allclose(command_7, command_9)
                and torch.allclose(task_frame_7, task_frame_9)
                and not red_safety_latched
                and not green_safety_latched
            )
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
            current_ee_force_b=applied_force_7_b,
            mass_matrix=state_7[1],
            gravity=state_7[2],
            current_joint_pos=state_7[6],
            current_joint_vel=state_7[7],
            nullspace_joint_pos_target=(
                red_hold_null_target if red_safety_latched else null_target_7
            ),
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
            nullspace_joint_pos_target=(
                green_hold_null_target
                if green_safety_latched
                else green_null_target
            ),
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

        target_7_w = world_pose(robot_7, actual_target_7_b)
        target_9_w = world_pose(robot_9, actual_target_9_b)
        markers["current_7"].visualize(state_7[5][:, :3], state_7[5][:, 3:7])
        markers["current_9"].visualize(state_9[5][:, :3], state_9[5][:, 3:7])
        markers["target_7"].visualize(target_7_w[:, :3], target_7_w[:, 3:7])
        markers["target_9"].visualize(target_9_w[:, :3], target_9_w[:, 3:7])
        force_direction_b = -normal_b
        force_quaternion_b = arrow_quaternion_from_x(force_direction_b)
        reaction_quaternion_b = arrow_quaternion_from_x(normal_b)
        marker_values = {
            "7": (robot_7, actual_target_7_b, commanded_force_7, filtered_7),
            "9": (robot_9, actual_target_9_b, commanded_force_9, filtered_9),
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
        for recording in recordings:
            recording.capture(step_count)
        if not green_safety_latched and not last_supervisor_9.freeze_path:
            task_time = min(task_time + dt, trajectory.total_duration)
        if (
            task_time >= trajectory.total_duration
            and not completion_hold_reported
        ):
            completion_hold_reported = True
            completion_hold_osc_9.reset()
            print(
                "[TASK] trajectory complete; holding final 15 N reference "
                "with Isaac Lab OSC until the window is closed."
            )

        post_state_7 = robot_state(robot_7, ee_7_idx, joints_7)
        post_state_9 = robot_state(robot_9, ee_9_idx, joints_9)
        travel_tracker.update(
            post_state_7[6][0].detach().cpu().numpy(),
            post_state_9[6][0].detach().cpu().numpy(),
        )
        delta_7 = torch.abs(post_state_7[6] - previous_joint_7)
        delta_9 = torch.abs(post_state_9[6] - previous_joint_9)
        arm_travel_7 += float(torch.sum(delta_7[:, :7]).item())
        arm_travel_9 += float(torch.sum(delta_9[:, :7]).item())
        wrist_travel_9 += float(torch.sum(delta_9[:, 7:]).item())
        previous_joint_7 = post_state_7[6].clone()
        previous_joint_9 = post_state_9[6].clone()
        if red_safety_event is not None:
            hold_position_error_7, hold_rotation_error_7 = compute_pose_error(
                post_state_7[3][:, :3],
                post_state_7[3][:, 3:7],
                red_hold_task_frame[:, :3],
                red_hold_task_frame[:, 3:7],
            )
            red_safety_event["max_tip_position_error_m"] = max(
                float(red_safety_event["max_tip_position_error_m"]),
                float(torch.linalg.vector_norm(hold_position_error_7).item()),
            )
            red_safety_event["max_tip_orientation_error_deg"] = max(
                float(red_safety_event["max_tip_orientation_error_deg"]),
                math.degrees(
                    float(torch.linalg.vector_norm(hold_rotation_error_7).item())
                ),
            )
        if green_safety_event is not None:
            hold_position_error_9, hold_rotation_error_9 = compute_pose_error(
                post_state_9[3][:, :3],
                post_state_9[3][:, 3:7],
                green_hold_task_frame[:, :3],
                green_hold_task_frame[:, 3:7],
            )
            green_safety_event["max_tip_position_error_m"] = max(
                float(green_safety_event["max_tip_position_error_m"]),
                float(torch.linalg.vector_norm(hold_position_error_9).item()),
            )
            green_safety_event["max_tip_orientation_error_deg"] = max(
                float(green_safety_event["max_tip_orientation_error_deg"]),
                math.degrees(
                    float(torch.linalg.vector_norm(hold_rotation_error_9).item())
                ),
            )

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
            travel = travel_tracker.snapshot()
            next_phase = trajectory.reference(task_time).phase
            phase_completed_9 = (
                next_phase is not reference.phase
                or task_time >= trajectory.total_duration
            )
            phase_completed_7 = (
                phase_completed_9
                and red_safety_phase != reference.phase.value
            )
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
                        phase_arm_travel_7_rad=travel.arm_7_rad,
                        phase_arm_travel_9_rad=travel.arm_9_rad,
                        phase_wrist_travel_9_rad=travel.wrist_9_rad,
                        nonprobe_force_7_n=collision_7.current_force_n,
                        nonprobe_force_9_n=collision_9.current_force_n,
                        collision_stop_7=red_safety_latched,
                        collision_stop_9=green_safety_latched,
                        completed_7=phase_completed_7,
                        completed_9=phase_completed_9,
                    )
                )
            if hud_label is not None:
                hud_label.text = format_hud(
                    HudSnapshot(
                        phase=reference.phase.value,
                        force_7_n=filtered_7,
                        force_9_n=filtered_9,
                        arm_7_rad=travel.arm_7_rad,
                        arm_9_rad=travel.arm_9_rad,
                        wrist_9_rad=travel.wrist_9_rad,
                        reduction_percent=travel.reduction_percent,
                        collision_7=collision_7.level.value,
                        collision_9=collision_9.level.value,
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
                f"q8/q9={math.degrees(float(state_9[6][0, -2].item())):+5.1f}/"
                f"{math.degrees(float(state_9[6][0, -1].item())):+6.1f} deg | "
                f"phase arm={travel.arm_7_rad:4.2f}/"
                f"{travel.arm_9_rad:4.2f} wrist9={travel.wrist_9_rad:4.2f} | "
                f"static drift={drift:.2e}"
            )
            latest_metric_values = np.asarray(
                [
                    tangent_error_7,
                    tangent_error_9,
                    normal_position_error_7,
                    normal_position_error_9,
                    normal_velocity_7,
                    normal_velocity_9,
                    orientation_error_7_deg,
                    orientation_error_9_deg,
                    angle_7,
                    angle_9,
                    drift,
                    travel.arm_7_rad,
                    travel.arm_9_rad,
                    travel.wrist_9_rad,
                ],
                dtype=np.float64,
            )

        if watchdog_enabled:
            post_reaction_7_w = sensor_reaction_force_w(sensor_7, sim.device)
            post_reaction_9_w = sensor_reaction_force_w(sensor_9, sim.device)
            post_reaction_7_b = quat_apply_inverse(
                robot_7.data.root_quat_w.torch, post_reaction_7_w
            )
            post_reaction_9_b = quat_apply_inverse(
                robot_9.data.root_quat_w.torch, post_reaction_9_w
            )
            post_measured_force_7 = float(
                torch.clamp(
                    torch.sum(post_reaction_7_b * normal_b, dim=-1),
                    min=0.0,
                ).item()
            )
            post_measured_force_9 = float(
                torch.clamp(
                    torch.sum(post_reaction_9_b * normal_b, dim=-1),
                    min=0.0,
                ).item()
            )
            watchdog_filtered_7 = watchdog_force_filter_7.update(
                post_measured_force_7
            ).filtered
            watchdog_filtered_9 = watchdog_force_filter_9.update(
                post_measured_force_9
            ).filtered
            post_nonprobe_7 = maximum_patient_contact_force(
                watchdog_collision_sensors_7, dt
            )
            post_nonprobe_9 = maximum_patient_contact_force(
                watchdog_collision_sensors_9, dt
            )
            finite_payloads = tuple(
                tensor.detach().cpu().numpy()
                for tensor in (
                    *state_7,
                    *state_9,
                    *post_state_7,
                    *post_state_9,
                    command_7,
                    command_9,
                    torque_7,
                    torque_9,
                    applied_7,
                    applied_9,
                )
            )
            watchdog_snapshot = validation_watchdog.update(
                WatchdogSample(
                    step=step_count,
                    dt_s=dt,
                    phase=reference.phase.value,
                    wrist_position_rad=post_state_9[6][0, 7:]
                    .detach()
                    .cpu()
                    .numpy(),
                    wrist_velocity_rad_s=post_state_9[7][0, 7:]
                    .detach()
                    .cpu()
                    .numpy(),
                    wrist_limits_rad=wrist_limits_9,
                    contact_required=phase_requires_contact(
                        reference.phase.value
                    ),
                    contact_present=np.array(
                        [
                            watchdog_filtered_7 > 0.5,
                            watchdog_filtered_9 > 0.5,
                        ],
                        dtype=bool,
                    ),
                    measured_normal_force_n=np.array(
                        [watchdog_filtered_7, watchdog_filtered_9],
                        dtype=np.float64,
                    ),
                    nonprobe_force_n=np.array(
                        [post_nonprobe_7, post_nonprobe_9],
                        dtype=np.float64,
                    ),
                    red_collision_stop=(
                        red_safety_latched
                        or (
                            reference.phase
                            in (
                                Phase.CHALLENGE_TRANSIT,
                                Phase.CHALLENGE_PITCH_ONLY,
                                Phase.RETURN_NEUTRAL,
                            )
                            and post_nonprobe_7
                            >= ValidationWatchdog.NONPROBE_COLLISION_N
                        )
                    ),
                    target_position_m=np.stack(
                        (
                            actual_target_7_b[0, :3].detach().cpu().numpy(),
                            actual_target_9_b[0, :3].detach().cpu().numpy(),
                        )
                    ),
                    measured_position_m=np.stack(
                        (
                            post_state_7[3][0, :3].detach().cpu().numpy(),
                            post_state_9[3][0, :3].detach().cpu().numpy(),
                        )
                    ),
                    target_quaternion_wxyz=np.stack(
                        (
                            actual_target_7_b[0, 3:7]
                            .detach()
                            .cpu()
                            .numpy(),
                            actual_target_9_b[0, 3:7]
                            .detach()
                            .cpu()
                            .numpy(),
                        )
                    ),
                    measured_quaternion_wxyz=np.stack(
                        (
                            post_state_7[3][0, 3:7]
                            .detach()
                            .cpu()
                            .numpy(),
                            post_state_9[3][0, 3:7]
                            .detach()
                            .cpu()
                            .numpy(),
                        )
                    ),
                    finite_payloads=finite_payloads
                    + (
                        np.asarray(
                            [
                                watchdog_filtered_7,
                                watchdog_filtered_9,
                                post_nonprobe_7,
                                post_nonprobe_9,
                            ]
                        ),
                        latest_metric_values,
                    ),
                )
            )
            stop_reasons = green_safety_reasons(watchdog_snapshot.reasons)
            if stop_reasons:
                print(
                    "[WATCHDOG] validation stopped: "
                    + ", ".join(stop_reasons)
                )
                break

        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            break

    for recording in recordings:
        recording.finish()
    scenario_complete = task_time + 0.5 * dt >= trajectory.total_duration
    report = metrics.report(scenario_complete=scenario_complete)
    report["controller"] = (
        "isaaclab.controllers.OperationalSpaceController"
    )
    report["task_time_s"] = task_time
    report["physics_steps"] = step_count
    report["normal_force_target_n"] = args_cli.normal_force
    report["scenario_total_duration_s"] = trajectory.total_duration
    report["wrist_axis_check"] = asdict(wrist_axis_check)
    if red_safety_event is not None:
        final_position_error_7, final_rotation_error_7 = compute_pose_error(
            post_state_7[3][:, :3],
            post_state_7[3][:, 3:7],
            red_hold_task_frame[:, :3],
            red_hold_task_frame[:, 3:7],
        )
        red_safety_event["arm_travel_after_latch_rad"] = (
            arm_travel_7
            - float(red_safety_event["arm_travel_at_latch_rad"])
        )
        red_safety_event["final_joint_displacement_l1_rad"] = float(
            torch.sum(torch.abs(previous_joint_7 - red_hold_null_target)).item()
        )
        red_safety_event["final_tip_position_error_m"] = float(
            torch.linalg.vector_norm(final_position_error_7).item()
        )
        red_safety_event["final_tip_orientation_error_deg"] = math.degrees(
            float(torch.linalg.vector_norm(final_rotation_error_7).item())
        )
    if green_safety_event is not None:
        final_position_error_9, final_rotation_error_9 = compute_pose_error(
            post_state_9[3][:, :3],
            post_state_9[3][:, 3:7],
            green_hold_task_frame[:, :3],
            green_hold_task_frame[:, 3:7],
        )
        green_safety_event["arm_travel_after_latch_rad"] = (
            arm_travel_9
            - float(green_safety_event["arm_travel_at_latch_rad"])
        )
        green_safety_event["final_joint_displacement_l1_rad"] = float(
            torch.sum(torch.abs(previous_joint_9 - green_hold_null_target)).item()
        )
        green_safety_event["final_tip_position_error_m"] = float(
            torch.linalg.vector_norm(final_position_error_9).item()
        )
        green_safety_event["final_tip_orientation_error_deg"] = math.degrees(
            float(torch.linalg.vector_norm(final_rotation_error_9).item())
        )
    report["safety_latches"] = {
        "red": red_safety_event,
        "green": green_safety_event,
    }
    report["final_reference_hold"] = completion_hold_reported
    return augment_validation_watchdog_report(
        report,
        enabled=watchdog_enabled,
        snapshot=watchdog_snapshot,
    )


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
    if args_cli.record_side is None:
        sim.set_camera_view(
            eye=(2.9, 3.6, 2.15), target=(0.55, -0.20, 0.60)
        )
    else:
        robot_y = ROBOT_Y_7 if args_cli.record_side == "7dof" else ROBOT_Y_9
        eye, target = side_camera(robot_y)
        sim.set_camera_view(eye=eye, target=target)
    scene = InteractiveScene(ComparisonSceneCfg(num_envs=1, env_spacing=3.0))
    if args_cli.record_side not in (None, "both"):
        isolate_comparison_side(
            sim.stage, args_cli.record_side, (), ()
        )
    collision_sensors_7 = make_patient_collision_sensors("7")
    collision_sensors_9 = make_patient_collision_sensors("9")
    watchdog_collision_sensors_7 = (
        make_patient_collision_sensors("7")
        if args_cli.validation_report is not None
        else []
    )
    watchdog_collision_sensors_9 = (
        make_patient_collision_sensors("9")
        if args_cli.validation_report is not None
        else []
    )
    sim.reset()
    scene.update(sim.get_physics_dt())
    print("[INFO] Fixed Assembly3 scene ready.")
    report = run_simulator(
        sim,
        scene,
        surface,
        collision_sensors_7=collision_sensors_7,
        collision_sensors_9=collision_sensors_9,
        watchdog_collision_sensors_7=watchdog_collision_sensors_7,
        watchdog_collision_sensors_9=watchdog_collision_sensors_9,
    )
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
    if not report.get("wrist_axis_check", {}).get("passed", False):
        print("[ERROR] Wrist-axis sign verification failed; task control was aborted.")
        return 2
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
