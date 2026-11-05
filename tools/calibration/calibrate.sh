#!/usr/bin/env bash
# Camera calibration from one recorded episode and the AprilTag on the table.
#
# Record an episode (./record.sh, Quest or gamepad) in which the arm stops at
# 15-25 poses where the WRIST camera sees the tag, tilting the gripper 20-40 deg
# about different axes between stops (pause ~1 s at each); the third-person
# camera should see the tag too. Then:
#
#   tools/calibration/calibrate.sh ~/franka_teleop_data/<episode> [options]
#
# Writes <episode>/calibration/calibration.json (T_base_cam_2, T_ee_cam_1,
# T_base_tag, intrinsics, residuals) and check images, and copies the JSON to
# config/camera_calibration.json. Options: see calibrate_cameras.py --help.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
EPISODE="$(cd "$1" && pwd)"; shift
if [[ ! -x "$HERE/.venv/bin/python" ]]; then  # OpenCV 4.x: the 5.0.0 Python package lacks calibrateHandEye
  /usr/bin/python3 -m venv "$HERE/.venv"
  "$HERE/.venv/bin/pip" install -q "opencv-python>=4.8,<5" "numpy<2.3" pupil-apriltags pyrealsense2
fi
POSES="$EPISODE/calibration/robot_poses.npz"
mkdir -p "$EPISODE/calibration"
(
  set +u
  source /opt/ros/humble/setup.bash
  source "$ROOT/install/setup.bash"
  python3 "$HERE/extract_robot_poses.py" "$EPISODE" "$POSES"
)
"$HERE/.venv/bin/python" "$HERE/calibrate_cameras.py" "$EPISODE" --poses "$POSES" "$@"
cp "$EPISODE/calibration/calibration.json" "$ROOT/config/camera_calibration.json"
echo "copied to config/camera_calibration.json"
