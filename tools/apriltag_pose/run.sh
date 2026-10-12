#!/usr/bin/env bash
# Live AprilTag (tagStandard41h12) pose estimation with a RealSense camera.
# Self-contained: its own Python environment in .venv, nothing shared with the
# rest of the repo. Options are passed through, see ./run.sh --help.
#   ./run.sh                          # tag id 4, printed 100 mm wide (black square 55.6 mm)
#   ./run.sh --tag-size 0.100         # if the black square itself is 100 mm
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
if [[ ! -x .venv/bin/python ]]; then
  echo "First run: creating the Python environment..."
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
exec .venv/bin/python apriltag_pose.py "$@"
