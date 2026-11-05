#!/usr/bin/env bash
# Runs an Isaac Lab script of this folder (SCRIPT=cup_demos.py by default, or replica.py) with the
# cluster's native Isaac Lab (~/isaaclab-cluster setup).
# Launch as a direct srun step on an allocated node (the open-file limit matters for RTX):
#   srun --overlap --jobid=<job> /usr/bin/env OMNI_KIT_ACCEPT_EULA=YES [SCRIPT=replica.py] <this script> --data DIR --out DIR
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then echo "Run on an allocated compute node (srun)." >&2; exit 2; fi
source "${ISAACLAB_CONDA_SH:-/apps/local/anaconda2023/etc/profile.d/conda.sh}"
LUSTRE_ROOT="${ISAACLAB_LUSTRE_ROOT:-/l/users/${USER}}"
conda activate "${LUSTRE_ROOT}/conda-envs/isaaclab-2.3.2"
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="${LUSTRE_ROOT}/.cache/isaaclab-2.3.2" XDG_CONFIG_HOME="${LUSTRE_ROOT}/.config/isaaclab-2.3.2"
export XDG_DATA_HOME="${LUSTRE_ROOT}/.local/share/isaaclab-2.3.2"
export CUDA_CACHE_PATH="${XDG_CACHE_HOME}/nvidia/ComputeCache"
unset DISPLAY
echo "node=$(hostname) nofile=$(ulimit -Sn)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${LUSTRE_ROOT}/isaaclab/native-2.3.2/IsaacLab"
timeout --signal=TERM --kill-after=60s "${TIMEOUT:-45m}" ./isaaclab.sh -p "$HERE/${SCRIPT:-cup_demos.py}" "$@"
