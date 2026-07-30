#!/usr/bin/env python3
"""Build project-local robot, patient, collider, and surface assets."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rizon_osc.scene_assets import AssetPaths  # noqa: E402
from rizon_osc.surface_model import SurfaceMap  # noqa: E402


def define_mesh(
    stage: Usd.Stage,
    path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int32)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(triangles.shape[0], 3, dtype=np.int32))
    )
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return mesh


def surface_collision_mesh(surface: SurfaceMap) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(surface.x_grid, surface.y_grid)
    vertices = np.column_stack((xx.ravel(), yy.ravel(), surface.height.ravel()))
    width = surface.x_grid.size
    faces: list[tuple[int, int, int]] = []
    for iy in range(surface.y_grid.size - 1):
        for ix in range(surface.x_grid.size - 1):
            if not np.all(surface.valid_mask[iy : iy + 2, ix : ix + 2]):
                continue
            lower_left = iy * width + ix
            lower_right = lower_left + 1
            upper_left = lower_left + width
            upper_right = upper_left + 1
            faces.append((lower_left, lower_right, upper_right))
            faces.append((lower_left, upper_right, upper_left))
    if not faces:
        raise RuntimeError("surface map produced no collision triangles")
    return vertices, np.asarray(faces, dtype=np.int32)


def build_patient_usd(
    patient_stl: Path,
    surface_path: Path,
    output_path: Path,
) -> None:
    source_mesh = trimesh.load_mesh(patient_stl, process=False)
    if not isinstance(source_mesh, trimesh.Trimesh):
        source_mesh = trimesh.util.concatenate(tuple(source_mesh.geometry.values()))
    surface = SurfaceMap.load(surface_path)
    collision_vertices, collision_faces = surface_collision_mesh(surface)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/torso_collider")
    stage.SetDefaultPrim(root.GetPrim())

    visual = define_mesh(
        stage,
        "/torso_collider/full_patient_visual",
        np.asarray(source_mesh.vertices),
        np.asarray(source_mesh.faces),
    )
    visual.CreatePurposeAttr(UsdGeom.Tokens.render)
    visual.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.50, 0.38)])

    collider = define_mesh(
        stage,
        "/torso_collider/upper_torso_contact_surface",
        collision_vertices,
        collision_faces,
    )
    collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
    collider.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr("none")

    stage.GetRootLayer().Save()
    print(f"[OK] Static Assembly3 patient + torso collider: {output_path}")


def parse_args() -> argparse.Namespace:
    defaults = AssetPaths.from_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rizon-usd", type=Path, default=defaults.base_rizon_usd)
    parser.add_argument(
        "--wrist-geometry-usd", type=Path, default=defaults.wrist_geometry_usd
    )
    parser.add_argument("--patient-stl", type=Path, default=defaults.patient_stl)
    parser.add_argument("--patient-urdf", type=Path, default=defaults.patient_urdf)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = AssetPaths.from_environment()
    paths = AssetPaths(
        generated_dir=defaults.generated_dir,
        base_rizon_usd=args.rizon_usd.expanduser().resolve(),
        wrist_geometry_usd=args.wrist_geometry_usd.expanduser().resolve(),
        patient_stl=args.patient_stl.expanduser().resolve(),
        patient_urdf=args.patient_urdf.expanduser().resolve(),
        ground_usd=defaults.ground_usd,
    )
    paths.require_sources()
    paths.generated_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools/preprocess_assembly3_surface.py"),
            "--stl",
            str(paths.patient_stl),
            "--output",
            str(paths.surface_map),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools/build_exact_rizon_wrist_asset.py"),
            "--rizon-usd",
            str(paths.base_rizon_usd),
            "--wrist-geometry-usd",
            str(paths.wrist_geometry_usd),
            "--output",
            str(paths.robot_wrapper),
        ],
        check=True,
    )
    build_patient_usd(paths.patient_stl, paths.surface_map, paths.patient_usd)
    print("[OK] Local Isaac assets are ready:")
    print(f"  robot:  {paths.robot_wrapper}")
    print(f"  patient:{paths.patient_usd}")
    print(f"  surface:{paths.surface_map}")


if __name__ == "__main__":
    main()
