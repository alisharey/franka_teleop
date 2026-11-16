"""Policy runs saved by vla/baku/deploy.py (record.sh --deploy), for the web UI's Rollouts tab.

One folder per run, <YYYYmmdd_HHMMSS>_<checkpoint>_task<N>:
    rollout.json    task, checkpoint, steps, end reason, outcome (the label), provenance
    rollout.npz     per policy step: actions, entropies, features, the policy's input images, timing
    recording/      the recorder episode of the run (videos, robot bag), when it was recorded
"""

import json
import os
from pathlib import Path
import re
import shutil
import time

from .episodes import folder_size

OUTCOMES = ('success', 'failure', 'intervention', 'discard')
_ID = re.compile(r'^\d{8}_\d{6}_[A-Za-z0-9_.-]+$')


class RolloutStore:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, rollout_id):
        if not _ID.match(rollout_id):
            raise KeyError(rollout_id)
        path = self.root / rollout_id
        if not (path / 'rollout.json').is_file():
            raise KeyError(rollout_id)
        return path

    def version(self):
        """Changes when runs are added or their rollout.json is rewritten (labels, a recording moved in)."""
        try:
            newest = max((p.stat().st_mtime_ns for p in self.root.glob('*/rollout.json')), default=0)
            return '%d-%d' % (self.root.stat().st_mtime_ns, newest)
        except OSError:
            return '0'

    def list(self):
        rollouts = []
        if not self.root.is_dir():
            return rollouts
        for path in sorted(self.root.iterdir(), reverse=True):
            if not _ID.match(path.name):
                continue
            try:
                info = json.loads((path / 'rollout.json').read_text())
            except (OSError, ValueError):
                continue
            recording = path / 'recording'
            duration = None
            try:
                duration = json.loads((recording / 'meta.json').read_text()).get('duration_s')
            except (OSError, ValueError):
                pass
            checkpoint = info.get('checkpoint') or {}
            rollouts.append({
                'id': path.name, 'task': info.get('task'), 'checkpoint': checkpoint.get('label'),
                'steps': info.get('steps'), 'rate_hz': info.get('rate_hz'), 'ended': info.get('ended'),
                'outcome': info.get('outcome'), 'duration_s': duration,
                'has_replay': (recording / 'replay.mp4').is_file(),
                'size_mb': round(folder_size(path) / 1e6, 1)})
        return rollouts

    def label(self, rollout_id, outcome):
        """Sets (None: clears) the outcome in rollout.json, keeping its other fields."""
        if outcome is not None and outcome not in OUTCOMES:
            raise ValueError('outcome must be one of %s or null' % ', '.join(OUTCOMES))
        path = self.path(rollout_id) / 'rollout.json'
        info = json.loads(path.read_text())
        info['outcome'] = outcome
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(info, indent=1))
        os.replace(temporary, path)

    def delete(self, rollout_id):
        """Removes the run folder for good (rollout data, label, recording)."""
        path = self.path(rollout_id)
        if time.time() - max(f.stat().st_mtime for f in [path, *path.iterdir()]) < 30:
            raise RuntimeError('this run is still being saved; try again in a moment')
        shutil.rmtree(path)
