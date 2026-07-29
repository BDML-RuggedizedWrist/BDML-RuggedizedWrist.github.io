#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/bdml-sim/IsaacLab}"
ISAACSIM_PYTHON="${ISAACSIM_PYTHON:-/home/bdml-sim/anaconda3/envs/env_isaacsim/bin/python}"
cd "$ISAACLAB_ROOT"

# The desktop launch uses Kit, while automated checks may explicitly request
# ``--viz none``.  Do not inject two conflicting visualizer selections.
if [[ " $* " == *" --viz none "* ]] || [[ " $* " == *" --headless "* ]]; then
  exec "$ISAACSIM_PYTHON" \
    "$PROJECT_ROOT/scripts/run_osc_comparison.py" "$@"
else
  exec "$ISAACSIM_PYTHON" \
    "$PROJECT_ROOT/scripts/run_osc_comparison.py" \
    --viz kit "$@"
fi
