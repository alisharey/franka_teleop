#!/usr/bin/env bash
# Exports recorded episodes for BAKU training (see export_episodes.py).
#   tools/baku/export.sh --out ~/baku_export --task "put the cube in the box" ~/franka_teleop_data/2026*_episode_*
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/humble/setup.bash
source "$ROOT/install/setup.bash"
exec python3 "$ROOT/tools/baku/export_episodes.py" "$@"
