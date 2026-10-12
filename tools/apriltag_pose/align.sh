#!/usr/bin/env bash
# Camera re-placement against the fixed AprilTag: web page at http://localhost:8770.
# Same self-contained environment as run.sh. Options: ./align.sh --help
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
if [[ ! -x .venv/bin/python ]]; then
  echo "First run: creating the Python environment..."
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
exec .venv/bin/python camera_align.py "$@"
