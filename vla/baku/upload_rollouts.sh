#!/usr/bin/env bash
# Converts the labelled policy rollouts (data/rollouts/<run>/) into the Failure_Detection
# format (failure_detection_export.py) in a temporary folder and uploads them to the HPC store. Runs
# relabelled since an earlier upload replace their old files there; nothing is kept locally.
#   vla/baku/upload_rollouts.sh
#   ROLLOUT_STORE=<cluster folder> vla/baku/upload_rollouts.sh    # another collection
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
source "$HERE/cluster.env"
STORE="${ROLLOUT_STORE:-$CLUSTER_ROLLOUTS}"
CALIBRATION="$(python3 -c "import json,os;print(os.path.basename(json.load(open('$ROOT/config/camera_calibration.json'))['session']))" 2> /dev/null || echo unversioned)"
TMP="$(mktemp -d -t franka_rollout_upload.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

ssh "$CLUSTER_HOST" mkdir -p "$STORE"
# the store's task numbering; only a store without one starts a new one (a failed copy must not renumber)
if ssh "$CLUSTER_HOST" test -f "$STORE/tasks.json"; then
  scp -q "$CLUSTER_HOST:$STORE/tasks.json" "$TMP/tasks.json"
fi
"$HERE/.venv/bin/python" "$HERE/failure_detection_export.py" --out "$TMP/export" --tasks "$TMP/tasks.json" \
  --suite "$(basename "$STORE")" --camera-calibration "$CALIBRATION"

# a relabelled run has the other succ flag in its file names: remove that outdated copy
( cd "$TMP/export" && ls *.csv 2> /dev/null | sed -E 's/--succ([01])\.csv$/ \1/' |
    awk '{ print $1 "--succ" (1 - $2) }' ) > "$TMP/stale.txt"
if [[ -s "$TMP/stale.txt" ]]; then
  ssh "$CLUSTER_HOST" "cd '$STORE' && while read -r stem; do rm -f \"\$stem\".csv \"\$stem\".pkl \"\$stem\".meta.json \"\$stem\".npz; done" < "$TMP/stale.txt"
fi
rsync -a "$TMP/export/" "$CLUSTER_HOST:$STORE/"
echo "store: $CLUSTER_HOST:$STORE ($(ssh "$CLUSTER_HOST" "ls '$STORE' | grep -c '\.csv$'") rollouts)"
