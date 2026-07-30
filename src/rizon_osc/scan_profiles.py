"""Calibrated path profiles shared by independent ultrasound comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .surface_model import SurfaceMap


@dataclass(frozen=True)
class ScanProfile:
    """Geometry and timing for one independent comparison task."""

    name: str
    description: str
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    orientation_direction_xy: tuple[float, float] | None
    scan_duration: float
    bend_axis: str
    bend_angle: float
    wrist_bend_sign: float
    axial_slice_angle: float


def _valid_inset(axis: np.ndarray, requested_inset: float) -> tuple[float, float]:
    """Return two interior coordinates without leaving a narrow surface grid."""
    span = float(axis[-1] - axis[0])
    inset = min(float(requested_inset), 0.2 * span)
    return float(axis[0] + inset), float(axis[-1] - inset)


def near_to_far_profile(surface: SurfaceMap) -> ScanProfile:
    """Return the existing longitudinal torso scan profile."""
    start = np.asarray(
        surface.metadata.get("scan_start_xy", [0.0, 1.18]), dtype=np.float64
    )
    end = np.asarray(
        surface.metadata.get("scan_end_xy", [0.0, 1.34]), dtype=np.float64
    )
    return ScanProfile(
        name="near_to_far",
        description="near torso end -> far torso end",
        start_xy=tuple(start.tolist()),
        end_xy=tuple(end.tolist()),
        orientation_direction_xy=None,
        scan_duration=2.0,
        bend_axis="pitch",
        bend_angle=math.radians(-35.0),
        wrist_bend_sign=-1.0,
        axial_slice_angle=math.radians(90.0),
    )


def cross_waist_profile(
    surface: SurfaceMap,
    *,
    surface_translation_xy: tuple[float, float],
) -> ScanProfile:
    """Scan laterally from the base-near waist side to the opposite side.

    The endpoint ordering is derived in the robot-base plane, so it stays
    correct if the registered patient translation changes.
    """
    left_x, right_x = _valid_inset(surface.x_grid, requested_inset=0.05)
    metadata_start = np.asarray(
        surface.metadata.get("scan_start_xy", [0.0, 1.18]), dtype=np.float64
    )
    waist_y = float(
        surface.metadata.get(
            "waist_y",
            np.clip(metadata_start[1], surface.y_grid[0], surface.y_grid[-1]),
        )
    )
    waist_y = float(np.clip(waist_y, surface.y_grid[0], surface.y_grid[-1]))
    candidates = (
        np.array([left_x, waist_y], dtype=np.float64),
        np.array([right_x, waist_y], dtype=np.float64),
    )
    translation = np.asarray(surface_translation_xy, dtype=np.float64)
    if translation.shape != (2,) or not np.isfinite(translation).all():
        raise ValueError("surface_translation_xy must be a finite XY pair")
    ordered = sorted(candidates, key=lambda xy: float(np.linalg.norm(xy + translation)))
    start, end = ordered
    for point, label in ((start, "base-near"), (end, "opposite")):
        if not surface.query(float(point[0]), float(point[1])).valid:
            raise ValueError(f"{label} waist endpoint lies outside the valid surface")
    return ScanProfile(
        name="cross_waist",
        description="base-near waist side -> opposite waist side",
        start_xy=tuple(start.tolist()),
        end_xy=tuple(end.tolist()),
        # Keep the same probe heading used by the calibrated longitudinal
        # task while translating laterally across the waist.
        orientation_direction_xy=(0.0, 1.0),
        scan_duration=2.4,
        bend_axis="pitch",
        bend_angle=math.radians(-40.0),
        wrist_bend_sign=-1.0,
        axial_slice_angle=math.radians(90.0),
    )


def scan_profile(
    surface: SurfaceMap,
    *,
    task_variant: str,
    surface_translation_xy: tuple[float, float],
) -> ScanProfile:
    """Resolve a named task without changing the common official-OSC runtime."""
    if task_variant == "near_to_far":
        return near_to_far_profile(surface)
    if task_variant == "cross_waist":
        return cross_waist_profile(
            surface,
            surface_translation_xy=surface_translation_xy,
        )
    raise ValueError(f"unknown task variant: {task_variant}")
