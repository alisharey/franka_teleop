#!/usr/bin/env bash
# One terminal for a recording session:
#   1. starts both cameras + the recorder/web UI in the background (log in log/),
#   2. opens the web UI,
#   3. runs gamepad_teleop in the foreground; when it exits (Ctrl-C or an error)
#      you can restart it or quit.
# Quitting stops the cameras and recorder; a running episode is saved first.
#
#   ./record.sh                 # default configs
#   ./record.sh --no-browser    # don't open the web UI
#   ./record.sh --no-teleop     # cameras + recorder only
#   ./record.sh --quest         # Meta Quest right controller instead of the gamepad
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")"
TELEOP_CONFIG=config/teleop.conf
RECORDING_CONFIG=config/recording.yaml
OPEN_BROWSER=1
RUN_TELEOP=1
INPUT=gamepad
for argument in "$@"; do
  case "$argument" in
    --no-browser) OPEN_BROWSER=0 ;;
    --no-teleop) RUN_TELEOP=0 ;;
    --quest) INPUT=quest ;;
    -h|--help) sed -n '2,12p' "$0" | cut -c3-; exit 0 ;;
    *) echo "unknown option: $argument (see --help)"; exit 2 ;;
  esac
done

if [[ ! -f install/setup.bash ]]; then
  echo "Not built yet. Run: source /opt/ros/humble/setup.bash && colcon build --symlink-install"
  exit 1
fi
set +u  # ROS setup scripts use unset variables
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source install/setup.bash
set -u
if ! ros2 pkg prefix realsense2_camera > /dev/null 2>&1; then
  echo "The RealSense ROS driver is missing. Install it with:"
  echo "  sudo apt install -y ros-humble-realsense2-camera ros-humble-rosbag2-storage-mcap"
  exit 1
fi

PORT=$(sed -n 's/^http_port:[[:space:]]*\([0-9]\+\).*/\1/p' "$RECORDING_CONFIG")
PORT=${PORT:-8765}
URL="http://localhost:$PORT"
mkdir -p log
LOG="log/recording_$(date +%Y%m%d_%H%M%S).log"

# Own session (setsid): a Ctrl-C meant for the teleop must not reach the
# recorder while it is recording.
setsid ros2 launch franka_teleop recording.launch.py "config:=$RECORDING_CONFIG" > "$LOG" 2>&1 &
LAUNCH_PID=$!

stop_recording_stack() {
  trap - EXIT INT TERM
  if kill -0 "$LAUNCH_PID" 2> /dev/null; then
    echo "Stopping cameras and recorder (a running episode is saved first)..."
    kill -INT -- "-$LAUNCH_PID" 2> /dev/null
    wait "$LAUNCH_PID" 2> /dev/null
  fi
  echo "Done. Log: $LOG"
}
trap stop_recording_stack EXIT
trap 'exit 130' TERM

echo "Starting cameras and recorder (log: $LOG)..."
for _ in $(seq 1 60); do
  if curl -fs "$URL/api/status" > /dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$LAUNCH_PID" 2> /dev/null; then
    echo "The recording launch exited. Last lines of $LOG:"
    tail -n 20 "$LOG"
    exit 1
  fi
  sleep 0.5
done
if ! curl -fs "$URL/api/status" > /dev/null 2>&1; then
  echo "The web UI did not come up at $URL. Last lines of $LOG:"
  tail -n 20 "$LOG"
  exit 1
fi
echo "Web UI: $URL"
if [[ $OPEN_BROWSER == 1 ]] && command -v xdg-open > /dev/null; then
  xdg-open "$URL" > /dev/null 2>&1 &
fi

# Ctrl-C should end the teleop (which handles it itself), not this script.
# A no-op trap (unlike ignoring) is reset to default in child processes.
trap ':' INT

if [[ $RUN_TELEOP == 0 ]]; then
  echo "Cameras and recorder are running. Press Enter to stop."
  read -r || true
  exit 0
fi

TELEOP=install/franka_teleop/lib/franka_teleop/gamepad_teleop
while true; do
  echo
  if [[ $INPUT == quest ]]; then
    echo "Starting the teleop with the Meta Quest: hold the right GRIP to move, TRIGGER toggles the gripper,"
    echo "A starts/stops a recording; Ctrl-C ends the teleop."
  else
    echo "Starting gamepad_teleop: hold RB to move. Y starts/stops a recording; Ctrl-C ends the teleop."
  fi
  "$TELEOP" --config "$TELEOP_CONFIG" --input "$INPUT" --enable-motion
  if ! kill -0 "$LAUNCH_PID" 2> /dev/null; then
    echo "The cameras/recorder stopped unexpectedly. Last lines of $LOG:"
    tail -n 20 "$LOG"
    exit 1
  fi
  echo
  if ! read -r -p "Teleop ended. Enter = restart teleop, q = quit (saves any running episode): " answer; then
    exit 0
  fi
  [[ $answer == q* ]] && exit 0
done
