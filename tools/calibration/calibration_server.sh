#!/usr/bin/env bash
# Camera calibration page (http://localhost:8770): hand-guide the arm in
# Programming mode, take snapshots, calibrate. Read-only: never moves the robot.
# Close ./record.sh and realsense-viewer first (they hold the cameras).
#   tools/calibration/calibration_server.sh [--session ~/franka_calibration/<folder>]
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$HERE/.venv/bin/python" ]]; then  # OpenCV 4.x: the 5.0.0 Python package lacks calibrateHandEye
  /usr/bin/python3 -m venv "$HERE/.venv"
  "$HERE/.venv/bin/pip" install -q "opencv-python>=4.8,<5" "numpy<2.3" pupil-apriltags pyrealsense2
fi
exec "$HERE/.venv/bin/python" "$HERE/calibration_server.py" "$@"
