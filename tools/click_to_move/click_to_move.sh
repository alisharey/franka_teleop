#!/usr/bin/env bash
# Click a point in a camera view, the gripper goes there (tools/click_to_move/click_to_move.py):
# starts the cameras (no recorder), the click page (http://localhost:8780, EdgeTAM obstacle tracking) and the
# teleop listening for its targets. Ctrl-C ends everything. Close ./record.sh first (it holds the cameras and the robot).
#   tools/click_to_move/click_to_move.sh
set -eo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
source /opt/ros/humble/setup.bash
source install/setup.bash
mkdir -p log
STAMP=$(date +%Y%m%d_%H%M%S)
# own sessions (setsid): Ctrl-C is for the teleop; the cleanup below stops these
setsid ros2 launch franka_teleop recording.launch.py config:=config/recording.yaml recorder:=false > "log/click_cameras_$STAMP.log" 2>&1 &
CAMERAS=$!
# EdgeTAM weights in data/models/hf (downloaded once; offline afterwards)
export HF_HOME="$PWD/data/models/hf"
[[ -d "$HF_HOME/hub/models--yonigozlan--EdgeTAM-hf" ]] && export HF_HUB_OFFLINE=1
setsid tools/click_to_move/.venv/bin/python tools/click_to_move/click_to_move.py "$@" > "log/click_page_$STAMP.log" 2>&1 &
PAGE=$!
cleanup() {
  kill -TERM "$PAGE" 2> /dev/null  # a background python ignores SIGINT
  kill -INT -- "-$CAMERAS" 2> /dev/null
  for _ in $(seq 1 50); do kill -0 "$CAMERAS" 2> /dev/null || break; sleep 0.2; done
  kill -TERM -- "-$CAMERAS" 2> /dev/null || true
}
trap cleanup EXIT
trap ':' INT
echo "Click page: http://localhost:8780 (logs: log/click_*_$STAMP.log). Keep the robot's stop button in reach."
install/franka_teleop/lib/franka_teleop/gamepad_teleop --config config/teleop.conf --input policy --enable-motion
