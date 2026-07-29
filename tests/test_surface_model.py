import json

import numpy as np
import pytest

from rizon_osc.surface_model import SurfaceMap, SurfaceSample, transform_sample


@pytest.fixture
def plane_map() -> SurfaceMap:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    xx, yy = np.meshgrid(x, y)
    height = 1.0 + 0.2 * xx - 0.1 * yy
    normal = np.array([-0.2, 0.1, 1.0])
    normal /= np.linalg.norm(normal)
    normals = np.broadcast_to(normal, (*height.shape, 3)).copy()
    return SurfaceMap(x, y, height, normals, np.ones_like(height, dtype=bool))


def test_query_bilinearly_interpolates_height_and_normal(plane_map):
    sample = plane_map.query(0.25, 1.5)

    assert sample.valid
    assert sample.height == pytest.approx(1.0 + 0.2 * 0.25 - 0.1 * 1.5)
    assert np.linalg.norm(sample.normal) == pytest.approx(1.0)
    assert sample.normal == pytest.approx(np.array([-0.2, 0.1, 1.0]) / np.sqrt(1.05))


def test_query_rejects_outside_bounds_and_invalid_cell(plane_map):
    assert not plane_map.query(-0.01, 0.5).valid

    invalid = plane_map.valid_mask.copy()
    invalid[1, 1] = False
    surface = SurfaceMap(
        plane_map.x_grid,
        plane_map.y_grid,
        plane_map.height,
        plane_map.normals,
        invalid,
    )
    assert not surface.query(0.5, 0.5).valid


def test_transform_sample_rotates_normal_and_translates_point():
    sample = SurfaceSample(
        point=np.array([1.0, 2.0, 3.0]),
        height=3.0,
        normal=np.array([1.0, 0.0, 0.0]),
        valid=True,
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = transform_sample(sample, rotation, np.array([0.5, -1.0, 2.0]))

    assert transformed.point == pytest.approx(np.array([-1.5, 0.0, 5.0]))
    assert transformed.normal == pytest.approx(np.array([0.0, 1.0, 0.0]))
    assert transformed.valid


def test_round_trip_preserves_source_metadata(tmp_path, plane_map):
    target = tmp_path / "surface.npz"
    metadata = {
        "source_path": "/tmp/patient.stl",
        "source_sha256": "a" * 64,
        "coordinate_convention": "x/y grid, outward +z",
    }
    plane_map.save(target, metadata=metadata)

    loaded = SurfaceMap.load(target)

    assert loaded.metadata == metadata
    assert json.loads(loaded.metadata_json)["source_sha256"] == "a" * 64
    assert loaded.query(0.5, 0.5).height == pytest.approx(1.05)


def test_constructor_rejects_non_unit_or_wrong_shape_data():
    with pytest.raises(ValueError, match="shape"):
        SurfaceMap(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.zeros((2, 2)),
            np.zeros((2, 2, 2)),
            np.ones((2, 2), dtype=bool),
        )
