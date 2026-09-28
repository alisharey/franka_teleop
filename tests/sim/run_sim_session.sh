#!/usr/bin/env bash
# Runs the real gamepad_teleop binary (built against tests/sim/fake_franka.cpp)
# through a scripted gamepad session and checks the trace.
# Usage: run_sim_session.sh <sim_teleop binary> <work dir>
set -euo pipefail
BIN="$1"
WORK="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$WORK"
FIFO="$WORK/joystick.fifo"
rm -f "$FIFO"
mkfifo "$FIFO"
sed -e "s#^joystick_path = .*#joystick_path = $FIFO#" -e "s#^gripper_enabled = .*#gripper_enabled = false#" \
  "$HERE/../../config/teleop.conf" > "$WORK/sim.conf"

python3 "$HERE/gamepad_script.py" "$FIFO" > "$WORK/script.log" 2>&1 &
SCRIPT_PID=$!
sleep 0.2
SIM_TRACE="$WORK/trace.csv" "$BIN" --config "$WORK/sim.conf" --enable-motion > "$WORK/teleop.log" 2>&1 &
TELEOP_PID=$!
sleep 30.0
kill -INT "$TELEOP_PID" 2>/dev/null || true
set +e
wait "$TELEOP_PID"
STATUS=$?
set -e
kill "$SCRIPT_PID" 2>/dev/null || true
echo "---- teleop output (exit $STATUS) ----"
cat "$WORK/teleop.log"
python3 "$HERE/check_trace.py" "$WORK/trace.csv" "$WORK/script.log"
exit $STATUS
