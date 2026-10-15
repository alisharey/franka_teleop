#!/usr/bin/env bash
# Saves the arm's current joint positions (no motion) as `home_joints` in
# config/teleop.conf: the pose a downward flick of the Quest stick returns to.
# Move the arm there first (teleop or hand-guiding), close the teleop, then run.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
IP=$(sed -n 's/^robot_ip[[:space:]]*=[[:space:]]*\([0-9.]*\).*/\1/p' config/teleop.conf)
exec install/franka_teleop/lib/franka_teleop/save_home_pose "$IP" config/teleop.conf
