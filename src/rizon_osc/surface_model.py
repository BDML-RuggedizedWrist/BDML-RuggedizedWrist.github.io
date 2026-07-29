"""Regular-grid representation of the Assembly3 contact surface."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SurfaceSample:
    """A single point and outward normal sampled from a surface map."""

    point: np.ndarray
    height: float
    normal: np.ndarray
    valid: bool


class SurfaceMap:
    """Height field with bilinear height and normal interpolation.

    Array convention is ``height[y_index, x_index]``. A query is valid only
    when all four corners of its interpolation cell are valid; the runtime
    never extrapolates across missing torso geometry.
    """

    def __init__(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        height: np.ndarray,
        normals: np.ndarray,
        valid_mask: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.x_grid = np.asarray(x_grid, dtype=np.float64)
        self.y_grid = np.asarray(y_grid, dtype=np.float64)
        self.height = np.asarray(height, dtype=np.float64)
        self.normals = np.asarray(normals, dtype=np.float64)
        self.valid_mask = np.asarray(valid_mask, dtype=bool)
        self.metadata = dict(metadata or {})

        expected_shape = (self.y_grid.size, self.x_grid.size)
        if self.x_grid.ndim != 1 or self.y_grid.ndim != 1:
            raise ValueError("x_grid and y_grid must be one-dimensional")
        if self.x_grid.size < 2 or self.y_grid.size < 2:
            raise ValueError("surface grid must contain at least two points per axis")
        if self.height.shape != expected_shape:
            raise ValueError(f"height shape must be {expected_shape}, got {self.height.shape}")
        if self.normals.shape != (*expected_shape, 3):
            raise ValueError(
                f"normals shape must be {(*expected_shape, 3)}, got {self.normals.shape}"
            )
        if self.valid_mask.shape != expected_shape:
            raise ValueError(
                f"valid_mask shape must be {expected_shape}, got {self.valid_mask.shape}"
            )
        if np.any(np.diff(self.x_grid) <= 0.0) or np.any(np.diff(self.y_grid) <= 0.0):
            raise ValueError("surface grid axes must be strictly increasing")
        if not np.isfinite(self.height[self.valid_mask]).all():
            raise ValueError("valid surface heights must be finite")

        magnitudes = np.linalg.norm(self.normals, axis=-1)
        if np.any(magnitudes[self.valid_mask] <= 1.0e-12):
            raise ValueError("valid surface normals must be nonzero")
        self.normals = self.normals.copy()
        self.normals[self.valid_mask] /= magnitudes[self.valid_mask, None]

    @property
    def metadata_json(self) -> str:
        """Canonical serialized metadata stored alongside the arrays."""
        return json.dumps(self.metadata, sort_keys=True, separators=(",", ":"))

    def query(self, x: float, y: float) -> SurfaceSample:
        """Sample the map without extrapolating."""
        x = float(x)
        y = float(y)
        if (
            x < self.x_grid[0]
            or x > self.x_grid[-1]
            or y < self.y_grid[0]
            or y > self.y_grid[-1]
        ):
            return self._invalid_sample(x, y)

        ix = min(int(np.searchsorted(self.x_grid, x, side="right") - 1), self.x_grid.size - 2)
        iy = min(int(np.searchsorted(self.y_grid, y, side="right") - 1), self.y_grid.size - 2)
        ix = max(ix, 0)
        iy = max(iy, 0)

        corners_valid = self.valid_mask[iy : iy + 2, ix : ix + 2]
        if not bool(np.all(corners_valid)):
            return self._invalid_sample(x, y)

        x0, x1 = self.x_grid[ix : ix + 2]
        y0, y1 = self.y_grid[iy : iy + 2]
        tx = (x - x0) / (x1 - x0)
        ty = (y - y0) / (y1 - y0)
        weights = np.array(
            [[(1.0 - tx) * (1.0 - ty), tx * (1.0 - ty)], [(1.0 - tx) * ty, tx * ty]]
        )
        height = float(np.sum(weights * self.height[iy : iy + 2, ix : ix + 2]))
        normal = np.sum(weights[..., None] * self.normals[iy : iy + 2, ix : ix + 2], axis=(0, 1))
        magnitude = float(np.linalg.norm(normal))
        if magnitude <= 1.0e-12 or not np.isfinite(magnitude):
            return self._invalid_sample(x, y)
        normal /= magnitude
        return SurfaceSample(
            point=np.array([x, y, height], dtype=np.float64),
            height=height,
            normal=normal,
            valid=True,
        )

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        """Save the map as a compressed, pickle-free NumPy archive."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        effective_metadata = self.metadata if metadata is None else dict(metadata)
        np.savez_compressed(
            target,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            height=self.height,
            normals=self.normals,
            valid_mask=self.valid_mask,
            metadata_json=np.asarray(
                json.dumps(effective_metadata, sort_keys=True, separators=(",", ":"))
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SurfaceMap":
        """Load a surface map without enabling object deserialization."""
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            return cls(
                archive["x_grid"],
                archive["y_grid"],
                archive["height"],
                archive["normals"],
                archive["valid_mask"],
                metadata=metadata,
            )

    @staticmethod
    def _invalid_sample(x: float, y: float) -> SurfaceSample:
        return SurfaceSample(
            point=np.array([x, y, np.nan], dtype=np.float64),
            height=float("nan"),
            normal=np.full(3, np.nan, dtype=np.float64),
            valid=False,
        )


def transform_sample(
    sample: SurfaceSample, rotation: np.ndarray, translation: np.ndarray
) -> SurfaceSample:
    """Apply a rigid transform to a valid sample."""
    if not sample.valid:
        return sample
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("rotation must be 3x3 and translation must be length 3")
    point = rotation @ sample.point + translation
    normal = rotation @ sample.normal
    normal /= np.linalg.norm(normal)
    return SurfaceSample(point=point, height=float(point[2]), normal=normal, valid=True)
