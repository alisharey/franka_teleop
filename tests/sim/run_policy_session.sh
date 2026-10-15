#!/usr/bin/env bash
# Runs the real gamepad_teleop in input=policy mode against the simulated FR3
# (fake_franka.cpp) with a scripted policy (fake_policy.py), then checks the trace.
# Usage: run_policy_session.sh <sim_teleop binary> <work dir>
set -euo pipefail
BIN="$1"
WORK="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$WORK"
# Home = the simulated arm's start joints (fake_franka.cpp), so the home request returns there.
sed -e "s#^gripper_enabled = .*#gripper_enabled = false#" -e "/^home_joints/d" -e "/^policy_/d" \
  "$HERE/../../config/teleop.conf" > "$WORK/sim_policy.conf"
echo "home_joints = 0.036286, -1.367388, -0.180505, -2.794799, -0.043301, 2.076045, 0.407056" >> "$WORK/sim_policy.conf"
echo "policy_port = 47821" >> "$WORK/sim_policy.conf"
SIM_TRACE="$WORK/trace.csv" "$BIN" --config "$WORK/sim_policy.conf" --input policy \
  --enable-motion > "$WORK/teleop.log" 2>&1 &
TELEOP_PID=$!
python3 "$HERE/fake_policy.py" 47821 &
POLICY_PID=$!
sleep 24
kill -INT "$TELEOP_PID" 2>/dev/null || true
kill "$POLICY_PID" 2>/dev/null || true
set +e
wait "$TELEOP_PID"
STATUS=$?
set -e
echo "---- teleop output (exit $STATUS) ----"
cat "$WORK/teleop.log"
python3 "$HERE/check_policy_trace.py" "$WORK/trace.csv"
exit $STATUS
