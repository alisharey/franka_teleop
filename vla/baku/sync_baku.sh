#!/usr/bin/env bash
# The BAKU code lives on the branch franka-teleop of alisharey/BAKU (submodule vla/BAKU,
# cluster checkout $CLUSTER_BAKU). This PC has no GitHub login, so commits are made
# on the cluster:
#   vla/baku/sync_baku.sh push   # copy local edits of vla/BAKU/baku to the cluster checkout
#                                # (then commit and push there)
#   vla/baku/sync_baku.sh pull   # update vla/BAKU to the branch's latest commit
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
source "$HERE/cluster.env"
case "${1:-}" in
  push) rsync -av --exclude=__pycache__ --exclude=exp_local "$ROOT/vla/BAKU/baku/" "$CLUSTER_HOST:$CLUSTER_BAKU/baku/"
        ssh "$CLUSTER_HOST" git -C "$CLUSTER_BAKU" status --short ;;
  pull) git -C "$ROOT" submodule update --remote vla/BAKU
        git -C "$ROOT/vla/BAKU" log --oneline -1 ;;
  *) sed -n '2,8p' "$0"; exit 2 ;;
esac
