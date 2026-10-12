#!/usr/bin/env bash
# Homes the Franka Hand: the fingers close and open fully once to calibrate.
# Needed after the robot is powered on; until then the gripper rejects
# open/close commands (it reports a max width of ~0). Keep the fingers clear.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
IP=$(sed -n 's/^robot_ip[[:space:]]*=[[:space:]]*\([0-9.]*\).*/\1/p' config/teleop.conf)
exec install/franka_teleop/lib/franka_teleop/home_gripper "$IP"
