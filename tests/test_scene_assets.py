from pathlib import Path

from rizon_osc.scene_assets import (
    DEFAULT_PATHS,
    PATIENT_COLLIDER_BODY,
    PROBE_BODY_NAME,
    STATIC_ASSETS,
    ROBOT_ROOT_FIXED,
)


def test_patient_beds_pedestals_and_robot_roots_are_fixed():
    required = {
        "patient_7",
        "patient_9",
        "bed_7",
        "bed_9",
        "pedestal_7",
        "pedestal_9",
    }
    by_name = {asset.name: asset for asset in STATIC_ASSETS}

    assert required <= by_name.keys()
    assert all(by_name[name].fixed for name in required)
    assert all(not by_name[name].dynamic for name in required)
    assert ROBOT_ROOT_FIXED


def test_contact_body_names_match_sensor_filters():
    assert PROBE_BODY_NAME == "linear_probe"
    assert PATIENT_COLLIDER_BODY == "torso_collider"


def test_generated_assets_are_project_local_not_tmp():
    assert DEFAULT_PATHS.generated_dir.name == "generated"
    assert not str(DEFAULT_PATHS.generated_dir).startswith("/tmp")
    assert DEFAULT_PATHS.robot_wrapper.parent == DEFAULT_PATHS.generated_dir
    assert DEFAULT_PATHS.patient_usd.parent == DEFAULT_PATHS.generated_dir


def test_downloaded_sources_are_plain_configurable_paths():
    assert isinstance(DEFAULT_PATHS.patient_stl, Path)
    assert isinstance(DEFAULT_PATHS.patient_urdf, Path)
    assert DEFAULT_PATHS.patient_stl.suffix.lower() == ".stl"
    assert DEFAULT_PATHS.patient_urdf.suffix.lower() == ".urdf"
