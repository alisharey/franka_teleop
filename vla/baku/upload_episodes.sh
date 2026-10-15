#!/usr/bin/env bash
# Exports recorded episodes (robot state + TeleMoMa actions + color videos, no
# depth) and uploads them to the cluster, ready for franka/build_dataset.py.
#   vla/baku/upload_episodes.sh [--task "text"] ~/franka_teleop_data/2026*_episode_*
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
source "$HERE/cluster.env"
EXPORT="${BAKU_EXPORT_DIR:-$HOME/franka_baku_export}"
"$ROOT/tools/baku/export.sh" --out "$EXPORT" "$@"
ssh "$CLUSTER_HOST" mkdir -p "$CLUSTER_EXPORT"
rsync -av "$EXPORT/" "$CLUSTER_HOST:$CLUSTER_EXPORT/"
