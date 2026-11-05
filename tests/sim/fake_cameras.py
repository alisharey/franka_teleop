#!/usr/bin/env python3
"""Publishes a recorded episode's color videos as the camera topics
(/cameras/<cam>/color/image_raw, rgb8, 30 Hz, looping), for dry runs of the
policy server without the RealSense cameras.

    python3 tests/sim/fake_cameras.py ~/franka_teleop_data/<episode> [--depth] [--state]

--depth  also the aligned depth (/cameras/<cam>/aligned_depth_to_color/image_raw,
         16UC1 mm, same stamp as the color frame), from the recorded depth videos
--state  also /teleop/state at the frame rate: the episode bag's robot state at the
         wrist camera's recorded frame time (O_T_EE, q, gripper), stamped like the frames
         (needs install/setup.bash sourced). For the sim2real background filter in preview.
"""
import site
import sys

_user_site = site.getusersitepackages()  # ~/.local numpy 2.x breaks ROS Humble (see scripts/recorder)
sys.path[:] = [entry for entry in sys.path if not entry.startswith(_user_site)]

import argparse  # noqa: E402
import array  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402

WRIST = 'cam_1'


class Stream:
    """One camera's color video (and depth video) read in lockstep, restartable."""

    def __init__(self, episode, streams, depth):
        self.color_path = episode / streams['color']['file']
        self.depth_path = episode / streams['depth']['file'] if depth else None
        self.width, self.height = streams['color']['width'], streams['color']['height']
        with open(episode / streams['color']['frames_csv']) as f:
            self.stamps = np.array([int(r['stamp_ns']) for r in csv.DictReader(f)], dtype=np.int64)
        self.capture = self.depth = None
        self.restart()

    def restart(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(str(self.color_path))
        if self.depth is not None:
            self.depth.kill()
            self.depth.wait()
        if self.depth_path is not None:
            self.depth = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-i', str(self.depth_path),
                                           '-pix_fmt', 'gray16le', '-f', 'rawvideo', '-'], stdout=subprocess.PIPE)
        self.index = 0

    def read(self):
        """(BGR, depth uint16 or None) of the next frame, or None at the end."""
        ok, bgr = self.capture.read()
        if not ok:
            return None
        depth = None
        if self.depth is not None:
            size = self.width * self.height * 2
            chunk = self.depth.stdout.read(size)
            if len(chunk) < size:
                return None
            depth = np.frombuffer(chunk, np.uint16).reshape(self.height, self.width)
        self.index += 1
        return bgr, depth


def bag_states(episode, meta):
    """(stamps_ns, [TeleopState]) of the episode bag's /teleop/state, every 10th message."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from franka_teleop.msg import TeleopState
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(episode / 'bag'), storage_id=meta['bag'].get('storage', 'mcap')),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/teleop/state']))
    stamps, states, count = [], [], 0
    while reader.has_next():
        _, data, _ = reader.read_next()
        count += 1
        if count % 10:
            continue
        state = deserialize_message(data, TeleopState)
        stamps.append(state.header.stamp.sec * 1_000_000_000 + state.header.stamp.nanosec)
        states.append(state)
    return np.array(stamps, dtype=np.int64), states


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('episode')
    parser.add_argument('--depth', action='store_true', help='also publish the aligned depth')
    parser.add_argument('--state', action='store_true', help='also publish /teleop/state from the episode bag')
    args = parser.parse_args()
    episode = Path(args.episode)
    meta = json.loads((episode / 'meta.json').read_text())
    rclpy.init()
    node = Node('fake_cameras')
    streams = {}
    for camera, info in meta['cameras'].items():
        prefix = '/cameras/%s' % camera
        streams[camera] = (Stream(episode, info['streams'], args.depth),
                           node.create_publisher(Image, prefix + '/color/image_raw', 10),
                           node.create_publisher(Image, prefix + '/aligned_depth_to_color/image_raw', 10)
                           if args.depth else None)
    state_publisher = None
    if args.state:
        from franka_teleop.msg import TeleopState
        state_stamps, states = bag_states(episode, meta)
        state_publisher = node.create_publisher(TeleopState, '/teleop/state', 100)
    print('publishing %s%s%s from %s' % (sorted(streams), ' + depth' if args.depth else '',
                                         ' + /teleop/state' if args.state else '', episode), flush=True)
    period = 1 / 30.0
    next_tick = time.monotonic()
    while rclpy.ok():
        frames = {camera: stream.read() for camera, (stream, _, _) in streams.items()}
        if any(frame is None for frame in frames.values()):  # loop all cameras together
            for stream, _, _ in streams.values():
                stream.restart()
            frames = {camera: stream.read() for camera, (stream, _, _) in streams.items()}
        stamp = node.get_clock().now().to_msg()
        for camera, (stream, color_publisher, depth_publisher) in streams.items():
            bgr, depth = frames[camera]
            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = '%s_color_optical_frame' % camera
            msg.height, msg.width = bgr.shape[:2]
            msg.encoding = 'rgb8'
            msg.step = msg.width * 3
            msg.data = array.array('B', cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes())  # fast path (bytes: per-byte check)
            color_publisher.publish(msg)
            if depth_publisher is not None:
                msg = Image()
                msg.header.stamp = stamp
                msg.header.frame_id = '%s_color_optical_frame' % camera
                msg.height, msg.width = depth.shape
                msg.encoding = '16UC1'
                msg.step = msg.width * 2
                msg.data = array.array('B', depth.tobytes())
                depth_publisher.publish(msg)
        if state_publisher is not None:
            wrist = streams[WRIST][0] if WRIST in streams else next(iter(streams.values()))[0]
            recorded = wrist.stamps[min(wrist.index - 1, len(wrist.stamps) - 1)]
            state = states[min(int(np.searchsorted(state_stamps, recorded)), len(states) - 1)]
            state.header.stamp = stamp
            state_publisher.publish(state)
        next_tick += period
        time.sleep(max(0.0, next_tick - time.monotonic()))


if __name__ == '__main__':
    main()
