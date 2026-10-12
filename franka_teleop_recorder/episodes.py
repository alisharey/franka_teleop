"""Episode folders on disk: create, list, rename, update metadata, move to trash.

Layout of one episode (folder name: <YYYYmmdd_HHMMSS>_<name>):
    meta.json                 name, task, success, notes, timing, per-stream stats,
                              camera intrinsics, teleop config + git commit
    bag/                      ROS 2 bag with every non-image topic (1 kHz /teleop/state, ...)
    <cam>_color.mp4           H.264, 30 fps
    <cam>_color_frames.csv    frame index -> camera timestamp, receive time (ns)
    <cam>_depth.mkv           16-bit depth aligned to color, lossless FFV1
    <cam>_depth_frames.csv
    replay.mp4                side-by-side replay of all color views
"""

import datetime
import json
import os
from pathlib import Path
import re
import shutil
import threading

FORMAT = 'franka_teleop_episode/1'
BUSY = ('recording', 'processing')
_SLUG = re.compile(r'[^A-Za-z0-9_-]+')
_ID = re.compile(r'^\d{8}_\d{6}(_[A-Za-z0-9_-]+)?$')


def slugify(name):
    slug = _SLUG.sub('_', name.strip()).strip('_')
    return slug[:80]


def is_episode_id(value):
    return bool(_ID.match(value))


class EpisodeStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.trash = self.root / '.trash'
        self._lock = threading.Lock()
        self.version = 0  # bumped on every change, so the web UI knows to reload

    def _changed(self):  # also called by the recorder after it writes meta.json
        self.version += 1

    def path(self, episode_id):
        if not is_episode_id(episode_id):
            raise KeyError(episode_id)
        path = self.root / episode_id
        if not (path / 'meta.json').is_file():
            raise KeyError(episode_id)
        return path

    def create(self, name, task, now=None):
        now = now or datetime.datetime.now()
        stamp = now.strftime('%Y%m%d_%H%M%S')
        with self._lock:
            name = name.strip() or self._next_default_name()
            slug = slugify(name)
            episode_id = '%s_%s' % (stamp, slug) if slug else stamp
            suffix = 1
            while (self.root / episode_id).exists():
                suffix += 1
                episode_id = '%s_%s_%d' % (stamp, slug, suffix) if slug else '%s_%d' % (stamp, suffix)
            path = self.root / episode_id
            path.mkdir()
            meta = {
                'format': FORMAT,
                'id': episode_id,
                'name': name,
                'task': task,
                'success': None,
                'notes': '',
                'created': now.isoformat(timespec='seconds'),
                'status': 'recording',
            }
            write_json(path / 'meta.json', meta)
            self._changed()
        return episode_id, path, meta

    def recover_interrupted(self):
        """Episodes still marked recording/processing were cut off by a crash or
        kill; mark them so they can be replayed, edited or deleted."""
        recovered = []
        for path in self.root.iterdir():
            meta_path = path / 'meta.json'
            if not (path.is_dir() and is_episode_id(path.name) and meta_path.is_file()):
                continue
            try:
                meta = read_json(meta_path)
            except (OSError, ValueError):
                continue
            if meta.get('status') in BUSY:
                meta['status'] = 'interrupted'
                meta.setdefault('warnings', []).append(
                    'recorder stopped before this episode was saved; files may be incomplete')
                write_json(meta_path, meta)
                recovered.append(path.name)
        if recovered:
            self._changed()
        return recovered

    def _next_default_name(self):
        count = sum(1 for p in self.root.iterdir() if p.is_dir() and is_episode_id(p.name))
        return 'episode_%04d' % (count + 1)

    def read(self, episode_id):
        return read_json(self.path(episode_id) / 'meta.json')

    def write(self, episode_id, meta):
        with self._lock:
            write_json(self.path(episode_id) / 'meta.json', meta)
            self._changed()

    def list(self):
        episodes = []
        for path in sorted(self.root.iterdir(), reverse=True):
            if not (path.is_dir() and is_episode_id(path.name)):
                continue
            try:
                meta = read_json(path / 'meta.json')
            except (OSError, ValueError):
                continue
            meta['id'] = path.name
            meta['has_replay'] = (path / 'replay.mp4').is_file()
            meta['size_mb'] = round(folder_size(path) / 1e6, 1)
            episodes.append(meta)
        return episodes

    def update(self, episode_id, name=None, task=None, success=None, notes=None, clear_success=False):
        """Edits metadata; a new name also renames the folder. Returns the (new) id."""
        with self._lock:
            path = self.path(episode_id)
            meta = read_json(path / 'meta.json')
            if meta.get('status') in BUSY:
                raise RuntimeError('this episode is still being saved; try again in a moment')
            if task is not None:
                meta['task'] = task
            if notes is not None:
                meta['notes'] = notes
            if success is not None or clear_success:
                meta['success'] = None if clear_success else bool(success)
            new_id = episode_id
            if name is not None and name.strip() and name.strip() != meta.get('name'):
                meta['name'] = name.strip()
                slug = slugify(meta['name'])
                stamp = episode_id[:15]
                new_id = '%s_%s' % (stamp, slug) if slug else stamp
                if new_id != episode_id:
                    if (self.root / new_id).exists():
                        raise FileExistsError('an episode named %r already exists' % new_id)
                    os.rename(path, self.root / new_id)
                    path = self.root / new_id
            meta['id'] = new_id
            write_json(path / 'meta.json', meta)
            self._changed()
        return new_id

    def move_to_trash(self, episode_id):
        with self._lock:
            path = self.path(episode_id)
            if read_json(path / 'meta.json').get('status') in BUSY:
                raise RuntimeError('this episode is still being saved; try again in a moment')
            self.trash.mkdir(exist_ok=True)
            target = self.trash / episode_id
            suffix = 1
            while target.exists():
                suffix += 1
                target = self.trash / ('%s_%d' % (episode_id, suffix))
            shutil.move(str(path), str(target))
            self._changed()


def read_json(path):
    with open(path) as stream:
        return json.load(stream)


def write_json(path, data):
    temporary = Path(str(path) + '.tmp')
    with open(temporary, 'w') as stream:
        json.dump(data, stream, indent=2, default=str)
        stream.write('\n')
    os.replace(temporary, path)


def folder_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total
