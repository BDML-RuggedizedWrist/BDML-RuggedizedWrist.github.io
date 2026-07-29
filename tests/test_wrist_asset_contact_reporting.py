"""Regression coverage for wrist contact sensors in the generated USD."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from pxr import Usd, UsdGeom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER = PROJECT_ROOT / "tools/build_exact_rizon_wrist_asset.py"
CONTACT_REPORT_BODIES = (
    "/Rizon4s/flange/wrist_base",
    "/Rizon4s/flange/wrist_base/wrist_pitch_link",
    "/Rizon4s/flange/wrist_base/wrist_pitch_link/probe_roll_output",
    "/Rizon4s/flange/wrist_base/wrist_pitch_link/probe_roll_output/linear_probe",
)
ROLL_OUTPUT = CONTACT_REPORT_BODIES[2]
PITCH_JOINT = "/Rizon4s/joints/wrist_pitch_joint"
ROLL_JOINT = "/Rizon4s/joints/wrist_roll_joint"


def _quat_components(value) -> tuple[float, float, float, float]:
    imaginary = value.GetImaginary()
    return (value.GetReal(), imaginary[0], imaginary[1], imaginary[2])


def _vec_components(value) -> tuple[float, float, float]:
    return (value[0], value[1], value[2])


def _matrix_components(value) -> tuple[float, ...]:
    return tuple(value[row][column] for row in range(4) for column in range(4))


def _has_contact_reporter(prim) -> bool:
    api_schemas = prim.GetMetadata("apiSchemas")
    return api_schemas is not None and "PhysxContactReportAPI" in api_schemas.explicitItems


def test_generated_wrist_bodies_publish_contact_reporters(tmp_path: Path):
    """Every body used by the wrist collision sensors must report contact."""
    output = tmp_path / "rizon_wrist.usda"
    subprocess.run(
        [sys.executable, str(BUILDER), "--output", str(output)],
        check=True,
        cwd=PROJECT_ROOT,
    )

    stage = Usd.Stage.Open(str(output))
    missing = []
    for path in CONTACT_REPORT_BODIES:
        body = stage.GetPrimAtPath(path)
        if not _has_contact_reporter(body):
            missing.append(path)
            continue
        assert body.GetAttribute("physxContactReport:threshold").Get() == 0.0

    assert missing == []
    reported_bodies = {
        str(prim.GetPath())
        for prim in Usd.PrimRange(stage.GetPrimAtPath(CONTACT_REPORT_BODIES[0]))
        if _has_contact_reporter(prim)
    }
    assert reported_bodies == set(CONTACT_REPORT_BODIES)


def test_roll_joint_reverses_local_z_without_changing_its_zero_pose(tmp_path: Path):
    """Positive joint nine must be the task frame's positive yaw direction."""
    output = tmp_path / "rizon_wrist.usda"
    subprocess.run(
        [sys.executable, str(BUILDER), "--output", str(output)],
        check=True,
        cwd=PROJECT_ROOT,
    )

    stage = Usd.Stage.Open(str(output))
    joint = stage.GetPrimAtPath(ROLL_JOINT)

    assert joint.GetAttribute("physics:lowerLimit").Get() == pytest.approx(-180.0)
    assert joint.GetAttribute("physics:upperLimit").Get() == pytest.approx(180.0)
    assert _vec_components(joint.GetAttribute("physics:localPos0").Get()) == pytest.approx(
        (-0.0011588743, -0.0316500, 0.0121950452)
    )
    assert _vec_components(joint.GetAttribute("physics:localPos1").Get()) == pytest.approx(
        (0.0, 0.0, 0.0)
    )
    assert _quat_components(joint.GetAttribute("physics:localRot0").Get()) == pytest.approx((
        0.0,
        0.9988782,
        0.0,
        0.04735377,
    ))
    assert _quat_components(joint.GetAttribute("physics:localRot1").Get()) == pytest.approx((
        0.0,
        1.0,
        0.0,
        0.0,
    ))
    assert _matrix_components(
        UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(ROLL_OUTPUT))
    ) == pytest.approx((
        0.09461219871070535, -1.1184439479743238e-7, -0.9955142047480354, 0.0,
        2.6226830692094465e-7, 0.9999999999999618, -8.742277429334211e-8, 0.0,
        0.9955142047480071, -2.528205641017351e-7, 0.09461219871073102, 0.0,
        0.25714501602650425, -0.11225006888928585, 1.2561588694660617, 1.0,
    ))


def test_pitch_joint_has_ninety_degree_scan_margin(tmp_path: Path):
    output = tmp_path / "rizon_wrist.usda"
    subprocess.run(
        [sys.executable, str(BUILDER), "--output", str(output)],
        check=True,
        cwd=PROJECT_ROOT,
    )

    stage = Usd.Stage.Open(str(output))
    joint = stage.GetPrimAtPath(PITCH_JOINT)

    assert joint.GetAttribute("physics:lowerLimit").Get() == pytest.approx(-90.0)
    assert joint.GetAttribute("physics:upperLimit").Get() == pytest.approx(90.0)
