#!/usr/bin/env python3
"""Convert the downloaded Assembly3 chest into a compact height/normal map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rizon_osc.surface_model import SurfaceMap  # noqa: E402


DEFAULT_STL = Path(
    "/home/bdml-sim/Downloads/Assembly 3/assembly_3/meshes/Part_2.stl"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_surface_map(
    vertices: np.ndarray,
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    resolution: float,
    neighbor_radius: float,
    smooth_sigma_cells: float,
) -> SurfaceMap:
    """Sample the highest local STL surface over a regular XY grid."""
    x_grid = np.arange(x_limits[0], x_limits[1] + 0.5 * resolution, resolution)
    y_grid = np.arange(y_limits[0], y_limits[1] + 0.5 * resolution, resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    query_xy = np.column_stack((xx.ravel(), yy.ravel()))

    tree = cKDTree(vertices[:, :2])
    distances, indices = tree.query(query_xy, k=64, workers=-1)
    nearby = distances <= neighbor_radius
    candidate_z = vertices[indices, 2]
    candidate_z = np.where(nearby, candidate_z, -np.inf)
    height = np.max(candidate_z, axis=1).reshape(xx.shape)
    valid = np.isfinite(height)
    if np.count_nonzero(valid) < 0.8 * valid.size:
        coverage = 100.0 * np.count_nonzero(valid) / valid.size
        raise RuntimeError(
            f"Only {coverage:.1f}% of the requested ROI has mesh support; "
            "adjust the ROI or --neighbor-radius."
        )

    # Normalized Gaussian filtering prevents invalid cells from biasing the
    # chest height toward zero.
    numerator = gaussian_filter(np.where(valid, height, 0.0), smooth_sigma_cells)
    denominator = gaussian_filter(valid.astype(np.float64), smooth_sigma_cells)
    smoothed = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1.0e-8,
    )
    dz_dy, dz_dx = np.gradient(smoothed, y_grid, x_grid)
    normals = np.stack((-dz_dx, -dz_dy, np.ones_like(smoothed)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    valid &= np.isfinite(smoothed) & np.isfinite(normals).all(axis=-1)
    return SurfaceMap(x_grid, y_grid, smoothed, normals, valid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", type=Path, default=DEFAULT_STL)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "generated/assembly3_torso_surface.npz",
    )
    parser.add_argument("--x-min", type=float, default=-0.12)
    parser.add_argument("--x-max", type=float, default=0.12)
    parser.add_argument("--y-min", type=float, default=1.12)
    parser.add_argument("--y-max", type=float, default=1.40)
    parser.add_argument("--resolution", type=float, default=0.004)
    parser.add_argument("--neighbor-radius", type=float, default=0.008)
    parser.add_argument(
        "--smooth-sigma-cells",
        type=float,
        default=8.0,
        help=(
            "Gaussian smoothing for the clinical contact surface. Eight 4-mm "
            "cells suppress STL facet jumps while retaining torso curvature."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.stl.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Assembly3 STL not found: {source}")
    if args.x_min >= args.x_max or args.y_min >= args.y_max:
        raise ValueError("ROI minimums must be below maximums")
    if (
        args.resolution <= 0.0
        or args.neighbor_radius <= 0.0
        or args.smooth_sigma_cells <= 0.0
    ):
        raise ValueError(
            "resolution, neighbor radius, and smooth sigma must be positive"
        )

    mesh = trimesh.load_mesh(source, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    print(f"[surface] source: {source}")
    print(f"[surface] mesh bounds: {mesh.bounds.tolist()}")
    surface = build_surface_map(
        np.asarray(mesh.vertices),
        x_limits=(args.x_min, args.x_max),
        y_limits=(args.y_min, args.y_max),
        resolution=args.resolution,
        neighbor_radius=args.neighbor_radius,
        smooth_sigma_cells=args.smooth_sigma_cells,
    )
    metadata = {
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "coordinate_convention": "Assembly3 STL local x/y grid, highest z, outward +z",
        "roi": {
            "x": [args.x_min, args.x_max],
            "y": [args.y_min, args.y_max],
        },
        "resolution": args.resolution,
        "neighbor_radius": args.neighbor_radius,
        "smooth_sigma_cells": args.smooth_sigma_cells,
        "scan_start_xy": [0.0, 1.18],
        "scan_end_xy": [0.0, 1.34],
    }
    surface.save(args.output, metadata=metadata)
    summary = {
        "output": str(args.output.resolve()),
        "shape": list(surface.height.shape),
        "valid_fraction": float(np.mean(surface.valid_mask)),
        "height_range": [
            float(np.nanmin(surface.height)),
            float(np.nanmax(surface.height)),
        ],
        "metadata": metadata,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
