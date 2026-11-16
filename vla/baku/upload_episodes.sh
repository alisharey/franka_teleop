#!/usr/bin/env bash
# Exports recorded episodes (robot state + TeleMoMa actions + color videos, no
# depth) and uploads them to the cluster, ready for franka/build_dataset.py.
# The export only exists in a temporary folder until it is uploaded.
#   vla/baku/upload_episodes.sh [--task "text"] data/teleop/2026*_episode_*
#   CLUSTER_EXPORT=<cluster folder> vla/baku/upload_episodes.sh ...   # another destination
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DEST="${CLUSTER_EXPORT:-}"
source "$HERE/cluster.env"
CLUSTER_EXPORT="${DEST:-$CLUSTER_EXPORT}"
EXPORT="$(mktemp -d -t franka_baku_export.XXXXXX)"
trap 'rm -rf "$EXPORT"' EXIT
"$ROOT/tools/baku/export.sh" --out "$EXPORT" "$@"
ssh "$CLUSTER_HOST" mkdir -p "$CLUSTER_EXPORT"
rsync -a "$EXPORT/" "$CLUSTER_HOST:$CLUSTER_EXPORT/"
echo "uploaded $(ls "$EXPORT" | wc -l) episodes to $CLUSTER_HOST:$CLUSTER_EXPORT"
