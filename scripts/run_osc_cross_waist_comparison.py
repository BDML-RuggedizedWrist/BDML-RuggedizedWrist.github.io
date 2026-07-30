#!/usr/bin/env python3
"""Independent cross-waist 7-DoF/9-DoF ultrasound OSC comparison."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


RUNNER = Path(__file__).with_name("run_osc_near_far_comparison.py")

if "--task_variant" not in sys.argv:
    sys.argv[1:1] = ["--task_variant", "cross_waist"]

runpy.run_path(str(RUNNER), run_name="__main__")
