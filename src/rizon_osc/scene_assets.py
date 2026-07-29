"""Declarative local asset paths and physical-fixation requirements."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_BODY_NAME = "linear_probe"
PATIENT_COLLIDER_BODY = "torso_collider"
ROBOT_ROOT_FIXED = True


@dataclass(frozen=True)
class AssetPaths:
    generated_dir: Path
    base_rizon_usd: Path
    wrist_geometry_usd: Path
    patient_stl: Path
    patient_urdf: Path
    ground_usd: Path

    @property
    def robot_wrapper(self) -> Path:
        return self.generated_dir / "rizon4s_exact_wrist_probe.usda"

    @property
    def patient_usd(self) -> Path:
        return self.generated_dir / "assembly3_patient.usd"

    @property
    def surface_map(self) -> Path:
        return self.generated_dir / "assembly3_torso_surface.npz"

    @classmethod
    def from_environment(cls) -> "AssetPaths":
        """Resolve overridable sources while keeping outputs project-local."""

        def path_env(name: str, default: str | Path) -> Path:
            return Path(os.environ.get(name, str(default))).expanduser().resolve()

        return cls(
            generated_dir=PROJECT_ROOT / "generated",
            base_rizon_usd=path_env(
                "RIZON_BASE_USD",
                "/home/bdml-sim/isaac_projects/flexiv_rizon4s/assets/rizon4s.usd",
            ),
            wrist_geometry_usd=path_env(
                "RIZON_WRIST_GEOMETRY_USD",
                "/home/bdml-sim/isaac_projects/rizon4s_wrist_osc/assets/"
                "rizon4s_wrist_linear.usd/rizon4s_wrist_linear/payloads/base.usda",
            ),
            patient_stl=path_env(
                "ASSEMBLY3_STL",
                "/home/bdml-sim/Downloads/Assembly 3/assembly_3/meshes/Part_2.stl",
            ),
            patient_urdf=path_env(
                "ASSEMBLY3_URDF",
                "/home/bdml-sim/Downloads/Assembly 3/assembly_3/urdf/assembly_3.urdf",
            ),
            ground_usd=path_env(
                "RIZON_GROUND_USD",
                "/home/bdml-sim/isaac_projects/flexiv_rizon4s/assets/"
                "Grid/default_environment.usd",
            ),
        )

    def require_sources(self) -> None:
        missing = [
            path
            for path in (
                self.base_rizon_usd,
                self.wrist_geometry_usd,
                self.patient_stl,
                self.patient_urdf,
            )
            if not path.is_file()
        ]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"Required local source assets are missing:\n{formatted}")


@dataclass(frozen=True)
class StaticAssetSpec:
    name: str
    fixed: bool
    dynamic: bool
    collision_enabled: bool


STATIC_ASSETS = (
    StaticAssetSpec("patient_7", fixed=True, dynamic=False, collision_enabled=True),
    StaticAssetSpec("patient_9", fixed=True, dynamic=False, collision_enabled=True),
    StaticAssetSpec("bed_7", fixed=True, dynamic=False, collision_enabled=True),
    StaticAssetSpec("bed_9", fixed=True, dynamic=False, collision_enabled=True),
    StaticAssetSpec("pedestal_7", fixed=True, dynamic=False, collision_enabled=True),
    StaticAssetSpec("pedestal_9", fixed=True, dynamic=False, collision_enabled=True),
)

DEFAULT_PATHS = AssetPaths.from_environment()
