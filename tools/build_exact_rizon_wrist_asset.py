#!/usr/bin/env python3
"""Build the exact Rizon 4s -> 2-DoF wrist -> linear probe USD chain.

The robot is referenced verbatim from the first working Isaac project.  Only
the wrist/probe geometry (converted from the user's Onshape URDF and probe
STL) is reused.  The exported URDF omitted the terminal roll DoF, so this file
keeps the motor/housing fixed to the pitch link and authors a separate,
centered terminal output revolute link for the probe.
"""

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RIZON_USD = Path(
    "/home/bdml-sim/isaac_projects/flexiv_rizon4s/assets/rizon4s.usd"
)
DEFAULT_WRIST_GEOMETRY_USD = Path(
    "/home/bdml-sim/isaac_projects/rizon4s_wrist_osc/assets/"
    "rizon4s_wrist_linear.usd/rizon4s_wrist_linear/payloads/base.usda"
)
DEFAULT_OUTPUT_USD = PROJECT_ROOT / "generated/rizon4s_exact_wrist_probe.usda"

SOURCE_WRIST = (
    "/rizon4s_wrist_linear/Geometry/base_link/link1/link2/link3/link4/"
    "link5/link6/link7/flange/wrist_base"
)

ROOT = "/Rizon4s"
FLANGE = f"{ROOT}/flange"
WRIST_BASE = f"{FLANGE}/wrist_base"
PITCH_LINK = f"{WRIST_BASE}/wrist_pitch_link"
ROLL_LINK = f"{PITCH_LINK}/wrist_roll_link"
SOURCE_PROBE = f"{SOURCE_WRIST}/wrist_pitch_link/wrist_roll_link/linear_probe"
TOOL_ROLL_LINK = f"{PITCH_LINK}/probe_roll_output"
PROBE = f"{TOOL_ROLL_LINK}/linear_probe"
PROBE_TIP = f"{PROBE}/probe_tip"

# The omitted terminal shaft is concentric with the circular roll-output CAD,
# not with the origin of the fixed Onshape subassembly.  Its centerline in the
# old roll-link frame is x=-19.2142 mm, y=-13.9599 mm.  Transforming the point
# at the original probe mounting plane (z=35 mm) into wrist_pitch_link gives
# this centered terminal frame.  The quaternion preserves the real shaft axis.
TOOL_ROLL_POS_IN_PITCH = (-0.0011588743, -0.0316500, 0.0121950452)
TOOL_ROLL_ROT_IN_PITCH = (0.9988782, 0.0, -0.04735377, 0.0)
# Rotate both roll-joint local frames 180 degrees about X.  This preserves the
# q=0 pose while reversing the authored local +Z axis, so a positive ninth
# joint command is positive task-frame yaw.
ROLL_JOINT_ROT0 = (0.0, 0.9988782, 0.0, 0.04735377)
ROLL_JOINT_ROT1 = (0.0, 1.0, 0.0, 0.0)
# Keep the probe rigid-body/control frame unchanged and rotate only its CAD,
# collision proxy, and visible contact pad +90 degrees about local Z.  This
# gives the requested axial mounting without redefining the OSC end-effector
# frame or consuming the ninth joint's travel.
PROBE_BODY_ROT = (0.0, 1.0, 0.0, 0.0)
AXIAL_GEOMETRY_ROT = (0.70710678, 0.0, 0.0, 0.70710678)


def add_body(stage: Usd.Stage, path: str, mass: float) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"Missing referenced body prim: {path}")
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    # A small diagonal inertia is intentionally conservative.  The detailed
    # CAD is visual-only; collision and inertia use stable primitive proxies.
    inertia = max(1.0e-5, mass * 7.0e-4)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia, inertia, inertia))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))


def add_collision_box(
    stage: Usd.Stage,
    parent: str,
    name: str,
    size: tuple[float, float, float],
    center: tuple[float, float, float],
) -> None:
    cube = UsdGeom.Cube.Define(stage, f"{parent}/{name}")
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube)
    xform.SetTranslate(Gf.Vec3d(*center))
    xform.SetScale(Gf.Vec3f(*size))
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)


def add_output_flange(stage: Usd.Stage, parent: str) -> None:
    """Add a small visible/collidable shaft that alone follows joint nine."""
    shaft = UsdGeom.Cylinder.Define(stage, f"{parent}/terminal_output_flange")
    shaft.CreateAxisAttr(UsdGeom.Tokens.z)
    shaft.CreateRadiusAttr(0.023)
    shaft.CreateHeightAttr(0.010)
    UsdGeom.XformCommonAPI(shaft).SetTranslate(Gf.Vec3d(0.0, 0.0, -0.005))
    shaft.CreateDisplayColorAttr([Gf.Vec3f(0.16, 0.48, 0.58)])
    UsdPhysics.CollisionAPI.Apply(shaft.GetPrim()).CreateCollisionEnabledAttr(True)


def rotate_probe_geometry_about_z(stage: Usd.Stage) -> None:
    """Rotate the transducer CAD/collision 90 degrees, preserving link axes."""
    transducer = stage.GetPrimAtPath(f"{PROBE}/linear_transducer")
    probe_box = stage.GetPrimAtPath(f"{PROBE}/box")
    probe_tip = stage.GetPrimAtPath(f"{PROBE}/probe_tip")
    if not transducer or not probe_box or not probe_tip:
        raise RuntimeError("Incomplete linear-probe geometry subtree")

    # +90 degrees about Z maps (x, y) -> (-y, x).
    old_translation = transducer.GetAttribute("xformOp:translate").Get()
    transducer.GetAttribute("xformOp:translate").Set(
        Gf.Vec3d(-old_translation[1], old_translation[0], old_translation[2])
    )
    transducer.GetAttribute("xformOp:orient").Set(
        Gf.Quatd(AXIAL_GEOMETRY_ROT[0], Gf.Vec3d(*AXIAL_GEOMETRY_ROT[1:]))
    )
    probe_box.GetAttribute("xformOp:orient").Set(
        Gf.Quatf(AXIAL_GEOMETRY_ROT[0], Gf.Vec3f(*AXIAL_GEOMETRY_ROT[1:]))
    )
    probe_tip.GetAttribute("xformOp:orient").Set(
        Gf.Quatf(AXIAL_GEOMETRY_ROT[0], Gf.Vec3f(*AXIAL_GEOMETRY_ROT[1:]))
    )


def set_joint_frames(
    joint,
    pos0,
    rot0=(1.0, 0.0, 0.0, 0.0),
    rot1=(1.0, 0.0, 0.0, 0.0),
) -> None:
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    joint.CreateLocalRot0Attr(Gf.Quatf(rot0[0], Gf.Vec3f(*rot0[1:])))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(rot1[0], Gf.Vec3f(*rot1[1:])))


def add_fixed_joint(stage: Usd.Stage, name: str, body0: str, body1: str, pos0, rot0=(1, 0, 0, 0)):
    joint = UsdPhysics.FixedJoint.Define(stage, f"{ROOT}/joints/{name}")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    set_joint_frames(joint, pos0, rot0)


def add_revolute_joint(
    stage: Usd.Stage,
    name: str,
    body0: str,
    body1: str,
    pos0,
    rot0,
    axis: str,
    limits: tuple[float, float],
    rot1=(1.0, 0.0, 0.0, 0.0),
):
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{ROOT}/joints/{name}")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateAxisAttr(axis)
    joint.CreateLowerLimitAttr(limits[0])
    joint.CreateUpperLimitAttr(limits[1])
    set_joint_frames(joint, pos0, rot0, rot1)

    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateMaxForceAttr(12.0)
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.0)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rizon-usd", type=Path, default=DEFAULT_RIZON_USD)
    parser.add_argument(
        "--wrist-geometry-usd", type=Path, default=DEFAULT_WRIST_GEOMETRY_USD
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_USD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rizon_usd = args.rizon_usd.expanduser().resolve()
    wrist_geometry_usd = args.wrist_geometry_usd.expanduser().resolve()
    output_usd = args.output.expanduser().resolve()
    for path in (rizon_usd, wrist_geometry_usd):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, ROOT)
    root.GetPrim().GetReferences().AddReference(str(rizon_usd), ROOT)
    stage.SetDefaultPrim(root.GetPrim())

    # The referenced subtree contains the wrist CAD from Onshape.  Its final
    # fixed subassembly is motor/housing geometry, not the omitted terminal
    # output DoF, so it deliberately remains visual geometry on PITCH_LINK.
    wrist = stage.DefinePrim(WRIST_BASE, "Xform")
    wrist.GetReferences().AddReference(str(wrist_geometry_usd), SOURCE_WRIST)

    # Resolve references before applying physics schemas to their link prims.
    stage.Load()
    old_probe = stage.GetPrimAtPath(f"{ROLL_LINK}/linear_probe")
    if not old_probe:
        raise RuntimeError("Original linear-probe geometry is missing from the wrist import")
    old_probe.SetActive(False)

    # Re-reference only the downloaded linear transducer beneath a new body.
    # This makes the ninth joint a real PhysX revolute joint while avoiding the
    # previous failure where an entire fixed motor subassembly swept through
    # the bracket.  At q=0 the new frame coincides with the real output shaft.
    tool_roll = UsdGeom.Xform.Define(stage, TOOL_ROLL_LINK)
    tool_xform = UsdGeom.XformCommonAPI(tool_roll)
    tool_xform.SetTranslate(Gf.Vec3d(*TOOL_ROLL_POS_IN_PITCH))
    tool_xform.SetRotate((0.0, -5.429, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    probe_prim = stage.DefinePrim(PROBE, "Xform")
    probe_prim.GetReferences().AddReference(str(wrist_geometry_usd), SOURCE_PROBE)
    stage.Load()
    rotate_probe_geometry_about_z(stage)

    add_body(stage, WRIST_BASE, 0.90)
    add_body(stage, PITCH_LINK, 0.55)
    add_body(stage, TOOL_ROLL_LINK, 0.10)
    add_body(stage, PROBE, 0.22)
    # Generic UsdFileCfg contact activation stops at the nested wrist rigid
    # body.  The probe-force and non-probe collision sensors therefore need
    # explicit reporting schemas on every supplemental wrist body they read.
    for body_path in (WRIST_BASE, PITCH_LINK, TOOL_ROLL_LINK, PROBE):
        body = stage.GetPrimAtPath(body_path)
        body.AddAppliedSchema("PhysxContactReportAPI")
        body.CreateAttribute(
            "physxContactReport:threshold", Sdf.ValueTypeNames.Float
        ).Set(0.0)

    # Conservative proxies: detailed CAD remains visible, while these simple
    # convex shapes make contact robust and fast.  Adjacent self-collision is
    # disabled at articulation level, but all external contacts are active.
    add_collision_box(stage, WRIST_BASE, "wrist_base_collision", (0.095, 0.082, 0.100), (0.0, 0.010, 0.030))
    add_collision_box(stage, PITCH_LINK, "pitch_link_collision", (0.080, 0.070, 0.075), (0.0, -0.010, -0.015))
    add_output_flange(stage, TOOL_ROLL_LINK)

    # The imported probe box spans 64 x 24 x 132 mm and is aligned with its
    # STL.  Reuse it as the collision shape so the acoustic face is accurate.
    probe_box = stage.GetPrimAtPath(f"{PROBE}/box")
    if not probe_box:
        raise RuntimeError("Linear probe collision box is missing from the imported wrist subtree")
    UsdPhysics.CollisionAPI.Apply(probe_box).CreateCollisionEnabledAttr(True)
    UsdGeom.Imageable(probe_box).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    # Connection chain: exact Rizon flange -> pitch (Y) -> newly authored
    # centered terminal roll (Z) -> downloaded linear ultrasound probe.
    add_fixed_joint(stage, "flange_to_wrist", FLANGE, WRIST_BASE, (0.0, 0.0, 0.04545))
    add_revolute_joint(
        stage,
        "wrist_pitch_joint",
        WRIST_BASE,
        PITCH_LINK,
        (-2.29851e-16, 0.0324, 0.0605),
        (1.0, 0.0, 0.0, 0.0),
        "Y",
        (-90.0, 90.0),
    )
    add_revolute_joint(
        stage,
        "wrist_roll_joint",
        PITCH_LINK,
        TOOL_ROLL_LINK,
        TOOL_ROLL_POS_IN_PITCH,
        ROLL_JOINT_ROT0,
        "Z",
        (-180.0, 180.0),
        ROLL_JOINT_ROT1,
    )
    # Flip the probe at its mount so its acoustic face points along the
    # Rizon tool direction (toward the phantom), not back through the wrist.
    # The referenced STL spans local -Z from its mounting body.
    probe_prim = stage.GetPrimAtPath(PROBE)
    probe_translate = probe_prim.GetAttribute("xformOp:translate")
    probe_translate.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    probe_orient = probe_prim.GetAttribute("xformOp:orient")
    probe_orient.Set(Gf.Quatf(PROBE_BODY_ROT[0], Gf.Vec3f(*PROBE_BODY_ROT[1:])))
    add_fixed_joint(
        stage,
        "roll_to_linear_probe",
        TOOL_ROLL_LINK,
        PROBE,
        (0.0, 0.0, 0.0),
        PROBE_BODY_ROT,
    )

    # Bright contact pad makes the ultrasound probe unmistakable in the GUI.
    pad = UsdGeom.Cube.Define(stage, f"{PROBE}/acoustic_contact_face")
    pad.CreateSizeAttr(1.0)
    pad_xform = UsdGeom.XformCommonAPI(pad)
    pad_xform.SetTranslate(Gf.Vec3d(0.0, 0.0, -0.1327))
    pad_xform.SetScale(Gf.Vec3f(0.055, 0.018, 0.0025))
    pad_xform.SetRotate((0.0, 0.0, 90.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    pad.CreateDisplayColorAttr([Gf.Vec3f(0.0, 0.95, 1.0)])

    # A thin identification collar remains visible from the wide comparison
    # camera while leaving the downloaded probe CAD clearly recognizable.
    collar = UsdGeom.Cube.Define(stage, f"{PROBE}/probe_identification_collar")
    collar.CreateSizeAttr(1.0)
    collar_xform = UsdGeom.XformCommonAPI(collar)
    collar_xform.SetTranslate(Gf.Vec3d(0.0, 0.0, -0.018))
    collar_xform.SetScale(Gf.Vec3f(0.069, 0.029, 0.010))
    collar_xform.SetRotate((0.0, 0.0, 90.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    collar.CreateDisplayColorAttr([Gf.Vec3f(0.0, 0.72, 0.95)])

    stage.GetRootLayer().Save()
    print(f"[OK] Exact robot + 2-DoF wrist + probe asset: {output_usd}")


if __name__ == "__main__":
    main()
