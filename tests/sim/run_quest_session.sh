#!/usr/bin/env bash
# Runs the real gamepad_teleop in input=quest mode against the simulated FR3
# (fake_franka.cpp) with a scripted headset (fake_adb.py), then checks the trace.
# Usage: run_quest_session.sh <sim_teleop binary> <work dir>
set -euo pipefail
BIN="$1"
WORK="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$WORK"
# Home = the simulated arm's start joints (fake_franka.cpp), so the flick at the end returns there.
sed -e "s#^gripper_enabled = .*#gripper_enabled = false#" -e "/^home_joints/d" -e "/^quest_connection/d" -e "/^quest_rotation_scale/d" \
  "$HERE/../../config/teleop.conf" > "$WORK/sim_quest.conf"
echo "home_joints = 0.036286, -1.367388, -0.180505, -2.794799, -0.043301, 2.076045, 0.407056" >> "$WORK/sim_quest.conf"
QUEST_ADB="$HERE/fake_adb.py" SIM_TRACE="$WORK/trace.csv" "$BIN" --config "$WORK/sim_quest.conf" --input quest \
  --enable-motion > "$WORK/teleop.log" 2>&1 &
TELEOP_PID=$!
sleep 26
kill -INT "$TELEOP_PID" 2>/dev/null || true
set +e
wait "$TELEOP_PID"
STATUS=$?
set -e
echo "---- teleop output (exit $STATUS) ----"
cat "$WORK/teleop.log"
python3 "$HERE/check_quest_trace.py" "$WORK/trace.csv"
exit $STATUS
