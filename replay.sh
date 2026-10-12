#!/usr/bin/env bash
# Replays a recorded episode on the robot:
#   1. turns the episode into delta end-effector pose actions (scripts/episode_actions),
#   2. moves the arm to the episode's start joints, then executes the actions
#      (src/episode_replay.cpp). The arm moves only while RB is held.
#
#   ./replay.sh EPISODE [--mode accumulate|from-current] [--rate HZ] [--speed F] [--no-gripper]
#
# EPISODE: an episode folder, its folder name, or its name (e.g. episode_0002).
# --mode accumulate    (default) each target = previous target + action
# --mode from-current  each action is added to where the arm actually is (like a policy)
# --rate               action rate in Hz (default 15)
# --speed              0 < F <= 1, slows the replay down (default 1 = recorded speed)
# Actions and a per-step log land in <data_dir>/replays/<episode>/.
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")"
MODE=accumulate
RATE=15
SPEED=1
EXTRA=()
EPISODE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --rate) RATE="$2"; shift 2 ;;
    --speed) SPEED="$2"; shift 2 ;;
    --no-gripper) EXTRA+=(--no-gripper); shift ;;
    -h|--help) sed -n '2,14p' "$0" | cut -c3-; exit 0 ;;
    -*) echo "unknown option: $1 (see --help)"; exit 2 ;;
    *) EPISODE="$1"; shift ;;
  esac
done
if [[ -z $EPISODE ]]; then
  echo "Which episode? See --help."
  exit 2
fi
if [[ ! -x install/franka_teleop/lib/franka_teleop/episode_replay ]]; then
  echo "Not built yet. Run: source /opt/ros/humble/setup.bash && colcon build --symlink-install"
  exit 1
fi
if pgrep -x gamepad_teleop > /dev/null; then
  echo "gamepad_teleop is running; only one program can control the robot. End it first."
  exit 1
fi
set +u  # ROS setup scripts use unset variables
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source install/setup.bash
set -u

DATA_DIR=$(sed -n 's/^data_dir:[[:space:]]*\([^#]*\).*/\1/p' config/recording.yaml | sed 's/[[:space:]]*$//')
DATA_DIR=${DATA_DIR/#\~/$HOME}
if [[ -d $EPISODE ]]; then
  FOLDER=$(readlink -f "$EPISODE")
elif [[ -d $DATA_DIR/$EPISODE ]]; then
  FOLDER=$DATA_DIR/$EPISODE
else
  FOLDER=$(grep -l "\"name\": \"$EPISODE\"" "$DATA_DIR"/*/meta.json 2> /dev/null | head -1 | xargs -r dirname)
fi
if [[ -z ${FOLDER:-} || ! -f $FOLDER/meta.json ]]; then
  echo "No episode '$EPISODE' in $DATA_DIR."
  exit 1
fi

OUT="$DATA_DIR/replays/$(basename "$FOLDER")"
mkdir -p "$OUT"
ACTIONS="$OUT/actions_${RATE}hz.txt"
PYTHONNOUSERSITE=1 install/franka_teleop/lib/franka_teleop/episode_actions "$FOLDER" --rate "$RATE" \
  --teleop-config config/teleop.conf --out "$ACTIONS" || exit 1
LOG="$OUT/$(date +%Y%m%d_%H%M%S)_${MODE}_${RATE}hz_x${SPEED}.csv"
exec install/franka_teleop/lib/franka_teleop/episode_replay --actions "$ACTIONS" --mode "$MODE" \
  --speed "$SPEED" --log "$LOG" "${EXTRA[@]}"
