#!/usr/bin/env python3
"""One episode's robot motion for a sim replay, aligned to a camera's frames.

Writes <out>.npz with, for every frame of the camera:
    t (s), q (7), gripper_width (m), o_t_ee (4x4, base frame)
Needs ROS (rosbag2_py): run with the system Python and ROS sourced.

    python3 sim2real/scene/export_replay.py ~/franka_teleop_data/<episode> out.npz [--camera cam_2]
"""
import site
import sys

_user_site = site.getusersitepackages()  # ~/.local numpy 2.x breaks ROS Humble (see scripts/recorder)
sys.path[:] = [entry for entry in sys.path if not entry.startswith(_user_site)]

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('episode')
    parser.add_argument('out')
    parser.add_argument('--camera', default='cam_2')
    args = parser.parse_args()
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from franka_teleop.msg import TeleopState

    episode = Path(args.episode)
    meta = json.loads((episode / 'meta.json').read_text())
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(episode / 'bag'), storage_id=meta['bag'].get('storage', 'mcap')),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/teleop/state']))
    stamps, q, width, pose = [], [], [], []
    count = 0
    while reader.has_next():
        _, data, _ = reader.read_next()
        count += 1
        if count % 5:
            continue
        s = deserialize_message(data, TeleopState)
        stamps.append(s.header.stamp.sec * 1_000_000_000 + s.header.stamp.nanosec)
        q.append(np.array(s.q))
        width.append(s.gripper_width)
        pose.append(np.array(s.o_t_ee).reshape(4, 4, order='F'))
    stamps = np.array(stamps)
    with open(episode / meta['cameras'][args.camera]['streams']['color']['frames_csv']) as f:
        frames = np.array([int(r['stamp_ns']) for r in csv.DictReader(f)], dtype=np.int64)
    index = np.clip(np.searchsorted(stamps, frames), 0, len(stamps) - 1)
    np.savez(args.out, t=(frames - frames[0]) * 1e-9, q=np.array(q)[index], gripper_width=np.array(width)[index],
             o_t_ee=np.array(pose)[index], frame_stamp_ns=frames)
    print('%s: %d %s frames, %.1f s' % (meta['id'], len(frames), args.camera, (frames[-1] - frames[0]) * 1e-9))


if __name__ == '__main__':
    main()
