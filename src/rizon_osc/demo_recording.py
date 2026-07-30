"""Viewport-only recording helpers for the public comparison demos."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Iterable

import imageio_ffmpeg
import numpy as np
from pxr import Usd, UsdGeom


def side_camera(
    robot_y: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return a repeatable oblique side view centered on one setup."""
    eye = (2.55, robot_y + 1.90, 1.48)
    target = (0.43, robot_y - 0.24, 0.66)
    return eye, target


def _set_visible(stage: Usd.Stage, path: str, visible: bool) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        return
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def isolate_comparison_side(
    stage: Usd.Stage,
    side: str,
    marker_paths_7: Iterable[str],
    marker_paths_9: Iterable[str],
) -> None:
    """Show one setup and hide the opposite setup and its markers."""
    if side not in ("7dof", "9dof"):
        raise ValueError(f"Unsupported recording side: {side}")
    selected_digit = "7" if side == "7dof" else "9"
    hidden_digit = "9" if selected_digit == "7" else "7"

    for suffix in ("Robot", "Patient", "Pedestal"):
        _set_visible(
            stage,
            f"/World/envs/env_0/{suffix}{selected_digit}DoF",
            True,
        )
        _set_visible(
            stage,
            f"/World/envs/env_0/{suffix}{hidden_digit}DoF",
            False,
        )

    for path in marker_paths_7:
        _set_visible(stage, path, selected_digit == "7")
    for path in marker_paths_9:
        _set_visible(stage, path, selected_digit == "9")



class CameraVideoRecorder:
    """Stream selected Isaac camera frames directly into an H.264 movie."""

    def __init__(
        self,
        camera,
        output_path: Path,
        *,
        sample_stride: int = 4,
        fps: int = 30,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.camera = camera
        self.output_path = output_path.expanduser().resolve()
        self.sample_stride = int(sample_stride)
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self.frame_count = 0
        if self.output_path.suffix.lower() != ".mp4":
            raise ValueError("Recording output must end in .mp4")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[RECORD] camera-only capture started: {self.output_path}")

    def capture(self, physics_step: int) -> None:
        if physics_step % self.sample_stride != 0:
            return
        rgb = self.camera.data.output["rgb"]
        if hasattr(rgb, "torch"):
            rgb = rgb.torch
        if hasattr(rgb, "detach"):
            frame = rgb[0, ..., :3].detach().cpu().numpy()
        else:
            frame = np.asarray(rgb)[0, ..., :3]
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape != (self.height, self.width, 3):
            raise RuntimeError(
                f"Unexpected recording frame shape: {frame.shape}"
            )
        if self._process.stdin is None:
            raise RuntimeError("Recording process has no input stream")
        self._process.stdin.write(frame.tobytes())
        self.frame_count += 1

    def finish(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        return_code = self._process.wait(timeout=180)
        if return_code != 0:
            raise RuntimeError(
                f"Video encoder failed with exit code {return_code}"
            )
        print(
            f"[RECORD] camera capture complete: {self.output_path} "
            f"({self.frame_count} frames)"
        )
