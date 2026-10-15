#!/usr/bin/env bash
# Runs a trained BAKU snapshot on the robot (see deploy.py --help).
# Start ./record.sh --policy in another terminal first (cameras, recorder, teleop).
#   vla/baku/deploy.sh vla/baku/runs/<run>/snapshot/60000.pt --record http://localhost:8765
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
if [[ ! -x "$HERE/.venv/bin/python" || ! -f "$ROOT/vla/BAKU/baku/franka/policy.py" ]]; then
  echo "Run vla/baku/setup_local.sh first."; exit 1
fi
source /opt/ros/humble/setup.bash
source "$ROOT/install/setup.bash"
exec "$HERE/.venv/bin/python" "$HERE/deploy.py" "$@"
