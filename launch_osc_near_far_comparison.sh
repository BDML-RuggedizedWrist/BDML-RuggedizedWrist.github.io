#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/bdml-sim/IsaacLab}"
ISAACSIM_PYTHON="${ISAACSIM_PYTHON:-/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python}"
cd "$ISAACLAB_ROOT"

if [[ " $* " == *" --viz none "* ]] || [[ " $* " == *" --headless "* ]]; then
  exec "$ISAACSIM_PYTHON" \
    "$PROJECT_ROOT/scripts/run_osc_near_far_comparison.py" "$@"
else
  exec "$ISAACSIM_PYTHON" \
    "$PROJECT_ROOT/scripts/run_osc_near_far_comparison.py" \
    --viz kit "$@"
fi
