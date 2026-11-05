#!/usr/bin/env bash
# Copies one trained snapshot (and its run's .hydra config) from the cluster for deploy.sh.
#   vla/baku/fetch_run.sh exp_local/franka_teleop_resnet_train/<run>/snapshot/60000.pt [local name]
# (path relative to BAKU/baku on the cluster, or absolute). The local folder is
# runs/<local name> (default: the run's folder name); the Policy panel lists it
# by that name. Also copies observation.json (how the training images were made, e.g. the
# sim2real background filter; deploy.py applies it) from the run folder, else from the
# training dataset folder (dataloader.bc_dataset.path), when one exists. Prints the local path.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster.env"
SNAPSHOT="$1"
[[ "$SNAPSHOT" = /* ]] || SNAPSHOT="$CLUSTER_BAKU/baku/$SNAPSHOT"
RUN="$(dirname "$(dirname "$SNAPSHOT")")"
LOCAL="$HERE/runs/${2:-$(basename "$RUN")}"
mkdir -p "$LOCAL/snapshot"
rsync -a "$CLUSTER_HOST:$RUN/.hydra" "$LOCAL/"
DATA="$(sed -n '/^  bc_dataset:/,/^  [a-z]/s/^    path: *//p' "$LOCAL/.hydra/config.yaml" | head -n 1)"
for SOURCE in "$RUN/observation.json" ${DATA:+"$DATA/observation.json"}; do
  if rsync -a "$CLUSTER_HOST:$SOURCE" "$LOCAL/" 2> /dev/null; then break; fi
done
if [[ -f "$LOCAL/observation.json" ]]; then
  echo "observation: $(tr -d '\n' < "$LOCAL/observation.json")" >&2
fi
rsync -a --info=progress2 "$CLUSTER_HOST:$SNAPSHOT" "$LOCAL/snapshot/"
echo "$LOCAL/snapshot/$(basename "$SNAPSHOT")"
