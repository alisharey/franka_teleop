#!/usr/bin/env python3
"""Step 1 of calibrate.sh: the robot's end-effector pose over an episode.

Reads /teleop/state from the episode's bag and writes <out>.npz with
    stamp_ns (N,)       system time of each sample (100 Hz, every 10th tick)
    o_t_ee   (N, 4, 4)  end effector in the base frame (libfranka O_T_EE)
    dq_max   (N,)       largest joint speed [rad/s] (to pick moments the arm is still)
Needs ROS (rosbag2_py), so it runs with the system Python.
"""
import site
import sys

_user_site = site.getusersitepackages()  # ~/.local numpy 2.x breaks ROS Humble (see scripts/recorder)
sys.path[:] = [entry for entry in sys.path if not entry.startswith(_user_site)]

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402


def main():
    episode, out = Path(sys.argv[1]), Path(sys.argv[2])
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from franka_teleop.msg import TeleopState

    meta = json.loads((episode / 'meta.json').read_text())
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(episode / 'bag'),
                                          storage_id=meta.get('bag', {}).get('storage', 'mcap')),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/teleop/state']))
    stamps, poses, speeds = [], [], []
    count = 0
    while reader.has_next():
        _, data, _ = reader.read_next()
        count += 1
        if count % 10:
            continue
        state = deserialize_message(data, TeleopState)
        stamps.append(state.header.stamp.sec * 1_000_000_000 + state.header.stamp.nanosec)
        poses.append(np.array(state.o_t_ee).reshape(4, 4, order='F'))  # libfranka: column-major
        speeds.append(float(np.max(np.abs(state.dq))))
    if not stamps:
        sys.exit('%s has no /teleop/state: record the calibration episode with the teleop running' % episode)
    np.savez(out, stamp_ns=np.array(stamps, dtype=np.int64), o_t_ee=np.array(poses), dq_max=np.array(speeds))
    print('robot poses: %d samples over %.1f s' % (len(stamps), (stamps[-1] - stamps[0]) * 1e-9))


if __name__ == '__main__':
    main()
