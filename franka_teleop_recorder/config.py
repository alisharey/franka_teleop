"""Loads config/recording.yaml."""

from dataclasses import dataclass, field
from pathlib import Path
import re

import yaml

CAMERA_NAME = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')


@dataclass
class CameraConfig:
    name: str
    serial: str


@dataclass
class RecordingConfig:
    data_dir: Path
    http_host: str = '127.0.0.1'
    http_port: int = 8765
    record_button: int = 4
    quest_record_button: int = 0
    bag_storage: str = 'mcap'
    cameras: list = field(default_factory=list)
    width: int = 640
    height: int = 480
    fps: int = 30
    depth: bool = True
    imu: bool = False
    color_crf: int = 18
    color_preset: str = 'veryfast'


def load_config(path) -> RecordingConfig:
    with open(path) as stream:
        raw = yaml.safe_load(stream) or {}
    camera = raw.get('camera', {})
    video = raw.get('video', {})
    cameras = [CameraConfig(name=str(c['name']), serial=str(c['serial'])) for c in raw.get('cameras', [])]
    if not cameras:
        raise ValueError('recording config needs at least one camera')
    names = [c.name for c in cameras]
    if len(set(names)) != len(names) or not all(CAMERA_NAME.match(n) for n in names):
        raise ValueError('camera names must be unique identifiers (letters, digits, _): %s' % names)
    if raw.get('bag_storage', 'mcap') not in ('mcap', 'sqlite3'):
        raise ValueError("bag_storage must be 'mcap' or 'sqlite3'")
    return RecordingConfig(
        data_dir=Path(str(raw.get('data_dir', '~/franka_teleop_data'))).expanduser(),
        http_host=str(raw.get('http_host', '127.0.0.1')),
        http_port=int(raw.get('http_port', 8765)),
        record_button=int(raw.get('record_button', 4)),
        quest_record_button=int(raw.get('quest_record_button', 0)),
        bag_storage=str(raw.get('bag_storage', 'mcap')),
        cameras=cameras,
        width=int(camera.get('width', 640)),
        height=int(camera.get('height', 480)),
        fps=int(camera.get('fps', 30)),
        depth=bool(camera.get('depth', True)),
        imu=bool(camera.get('imu', False)),
        color_crf=int(video.get('color_crf', 18)),
        color_preset=str(video.get('color_preset', 'veryfast')),
    )
