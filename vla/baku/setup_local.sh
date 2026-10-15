#!/usr/bin/env bash
# One-time setup of policy deployment on the robot PC:
#   1. checks out the BAKU submodule (vla/BAKU: alisharey/BAKU, branch franka-teleop),
#   2. creates vla/baku/.venv (Python 3.10 with the system packages, so ROS 2
#      Humble's rclpy works) with the versions of the cluster's `baku` env.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
git -C "$ROOT" submodule update --init vla/BAKU   # not recursive: BAKU's Metaworld/Open-Teach are not needed
if [[ ! -x "$HERE/.venv/bin/python" ]]; then
  /usr/bin/python3 -m venv --system-site-packages "$HERE/.venv"
fi
PIP="$HERE/.venv/bin/pip"
"$PIP" install --upgrade pip
"$PIP" install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
"$PIP" install "numpy<2" hydra-core==1.1.0 omegaconf==2.1.2 einops==0.7.0 diffusers==0.11.1 \
  huggingface-hub==0.33.0 tqdm scipy==1.13.1 matplotlib
"$PIP" install --no-deps robomimic==0.2.0   # BAKU only uses robomimic.utils.tensor_utils
set +u
source /opt/ros/humble/setup.bash
"$HERE/.venv/bin/python" -c "import torch, cv2, rclpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
