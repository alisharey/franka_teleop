#!/usr/bin/env bash
# Runs the policy server alone (normally ./record.sh --deploy starts it), e.g.
#   vla/baku/deploy.sh --no-motion      # preview only: never sends actions or home requests
# See deploy.py --help.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
if [[ ! -x "$HERE/.venv/bin/python" || ! -f "$ROOT/vla/BAKU/baku/franka/policy.py" ]]; then
  echo "Run vla/baku/setup_local.sh first."; exit 1
fi
source /opt/ros/humble/setup.bash
source "$ROOT/install/setup.bash"
exec "$HERE/.venv/bin/python" "$HERE/deploy.py" "$@"
