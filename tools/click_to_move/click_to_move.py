#!/usr/bin/env python3
"""Click a point in either camera view and the gripper goes there, around an obstacle.

Target: the clicked pixel's depth (aligned to color) gives a 3D point in that camera, turned into the robot base
frame with config/camera_calibration.json: the static camera directly (T_base_cam_cam_2), the wrist camera
through the current end-effector pose (O_T_EE * T_ee_cam_cam_1). The end-effector point (libfranka O_T_EE, the
Franka Hand's TCP between the fingertips) goes to that point plus a hover height along the table's normal (from
the calibration board lying on the table), pointing into the table, keeping its rotation about that axis.

Obstacle: "Select obstacle", then a click on it starts EdgeTAM (tracker.py) on that camera; every tracked mask
plus depth becomes the obstacle:
  --object points (default): the mask's depth points (1 cm voxels), each inflated by --inflate. Any shape (a tall
    ruler); only what the camera sees, so the far side of a thick object is missing.
  --object cup: one of the digital twin's cups (--cup, auto by colour), a vertical cylinder of its own size behind
    the visible surface; only points within the cup's size count (a mask leaking onto a hand lifting it is
    ignored); a lifted cup floats above the table.
Keep-out region for the TCP, inflated by the gripper's reach (11 cm: from their meshes, every part of the Franka
Hand, the wrist camera and its mount is within 10.5 cm of the TCP's axis, in any rotation) plus a clearance, and
above the obstacle by the clearance plus the 1 cm the fingertips reach below the TCP:
  points: around each point, from the reach below it to that above it (only the gripper is kept clear: the arm is
    not checked, e.g. under an overhang);
  cup: the cylinder, reaching down to the table (the hand and arm are above the TCP: it cannot pass under).
Both are drawn in both camera views.

Path: straight when it stays out of the keep-out region; else around it (the shortest way along tangents and an
arc that stays inside the teleop's workspace fence); else over it. Targets inside the keep-out region are
refused. The plan is drawn before Go; during the move the obstacle estimate is frozen (the arm passing in front
of the camera would disturb the tracking) and the arm stops if its TCP gets within the uninflated reach.

Motion goes through gamepad_teleop --input policy (src/policy_input.h, UDP), so its speed limits, workspace
fence and contact limit apply; without packets for policy_timeout_ms the arm stops. Started by
click_to_move.sh, which also starts the cameras and the teleop.

    http://localhost:8780
"""
import argparse
import json
import math
import os
import random
import re
import signal
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Image

from franka_teleop.msg import TeleopState
from tracker import Tracker

ROOT = Path(__file__).resolve().parents[2]
PACKET = struct.Struct('<4sIQQ7d')  # src/policy_input.h
RATE_HZ = 30.0
# The teleop moves at gain (5) x the distance to the target, capped at 0.25 m/s and 0.4 rad/s
# (config/teleop.conf): limiting how far ahead each target is sets the speed, --speed x those caps.
TELEOP_GAIN, MAX_LINEAR_M_S, MAX_ANGULAR_RAD_S = 5.0, 0.25, 0.40
REACHED_M, REACHED_DEG, REACHED_TICKS = 0.003, 1.0, 10
LOOKAHEAD_M = 0.03       # how far along the path the TCP's target runs ahead
PATH_EXTRA_M = 0.01      # the path keeps this much more than the keep-out radius (tracking lag on the arc)
# From the meshes (Franka Hand from the Isaac Panda USD, the v7 wrist-camera mount and Intel's D435 model): every
# part of the gripper is within 10.5 cm of its axis (hand +-10.4 cm along the fingers, the camera 10.5 cm), and
# the fingertips reach 0.9 cm below the TCP.
FINGERTIP_BELOW_TCP_M = 0.01
# The digital twin's cups (sim2real/scenes/cups_20261024/scene.json on the digital-twin branch):
# height, larger diameter [m]. --cup auto picks one by the tracked pixels' colour.
VOXEL_M = 0.01           # --object points: the mask's depth points are averaged in voxels this size
CUPS = {'black': (0.0635, 0.0815), 'blue': (0.099, 0.0874), 'red': (0.0907, 0.0856)}


def euler_xyz(r):
    """scipy 'xyz' (extrinsic) angles of r = Rz(c) Ry(b) Rx(a), as TeleMoMa's actions use."""
    return np.array([math.atan2(r[2, 1], r[2, 2]), -math.asin(max(-1.0, min(1.0, r[2, 0]))), math.atan2(r[1, 0], r[0, 0])])


def angle_deg(r):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2))))


def workspace_fence(path):
    """{'x': (lo, hi), 'y': ..., 'z': ...} of the teleop's workspace fence (config/teleop.conf)."""
    fence = {}
    for axis in 'xyz':
        m = re.search(r'^workspace_%s_m\s*=\s*([-\d.]+)\s*,\s*([-\d.]+)' % axis, Path(path).read_text(), re.M)
        fence[axis] = (float(m.group(1)), float(m.group(2))) if m else (-1e9, 1e9)
    return fence


class Ros:
    """Camera images, aligned depth, intrinsics and the robot state from the running stack."""

    def __init__(self, cameras):
        rclpy.init()
        self.node = rclpy.create_node('click_to_move')
        self.lock = threading.Lock()
        self.color, self.depth, self.K, self.stamp, self.state, self.state_time = {}, {}, {}, {}, None, 0.0
        for cam in cameras:
            prefix = '/cameras/%s' % cam
            self.node.create_subscription(Image, prefix + '/color/image_raw',
                                          lambda m, c=cam: self._image(c, m), qos_profile_sensor_data)
            self.node.create_subscription(Image, prefix + '/aligned_depth_to_color/image_raw',
                                          lambda m, c=cam: self._depth(c, m), qos_profile_sensor_data)
            self.node.create_subscription(CameraInfo, prefix + '/color/camera_info',
                                          lambda m, c=cam: self._info(c, m), qos_profile_sensor_data)
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.node.create_subscription(TeleopState, '/teleop/state', self._state, qos, raw=True)
        self._last_decode = 0.0
        threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()

    def _image(self, cam, m):
        if m.encoding not in ('rgb8', 'bgr8'):
            return
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * 3].reshape(m.height, m.width, 3)
        with self.lock:
            self.color[cam] = img if m.encoding == 'bgr8' else img[:, :, ::-1]
            self.stamp[cam] = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec

    def _depth(self, cam, m):
        if m.encoding not in ('16UC1', 'mono16'):
            return
        with self.lock:
            self.depth[cam] = np.frombuffer(m.data, np.uint16).reshape(m.height, m.step // 2)[:, :m.width].copy()

    def _info(self, cam, m):
        with self.lock:
            self.K[cam] = np.array(m.k, float).reshape(3, 3)

    def _state(self, data):
        now = time.monotonic()
        if now - self._last_decode < 0.005:  # 1 kHz topic: 200 per second is plenty
            return
        self._last_decode = now
        s = deserialize_message(data, TeleopState)
        sample = {'ee': np.array(s.o_t_ee, float).reshape(4, 4, order='F'), 'gripper_width': float(s.gripper_width),
                  'motion_enabled': bool(s.motion_enabled), 'dq_max': float(np.max(np.abs(s.dq))),
                  'q': np.array(s.q, float), 'f_t_ee': np.array(s.f_t_ee, float).reshape(4, 4, order='F')}
        with self.lock:
            self.state, self.state_time = sample, now

    def robot(self):
        with self.lock:
            return None if self.state is None or time.monotonic() - self.state_time > 0.3 else dict(self.state)

    def frame(self, cam):
        with self.lock:
            return self.color.get(cam), self.depth.get(cam), self.K.get(cam)

    def rgb_frame(self, cam):
        """For the tracker: (RGB, depth, K, stamp)."""
        with self.lock:
            color = self.color.get(cam)
            return (None if color is None else color[:, :, ::-1].copy(), self.depth.get(cam), self.K.get(cam),
                    self.stamp.get(cam))


class Table:
    """The table frame: origin on the calibration board, 'up' = its normal; 2D coordinates in the table plane."""

    def __init__(self, calib):
        board = np.array(calib['T_base_tag']) if 'T_base_tag' in calib else np.eye(4)
        up = -board[:3, 2] if 'T_base_tag' in calib else np.array([0.0, 0.0, 1.0])
        self.up = up / np.linalg.norm(up) * (1 if up[2] > 0 else -1)
        self.origin = board[:3, 3]
        e1 = np.array([1.0, 0, 0]) - self.up[0] * self.up
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = np.cross(self.up, self.e1)

    def xy(self, p):
        d = np.asarray(p) - self.origin
        return np.stack([d @ self.e1, d @ self.e2], -1)

    def height(self, p):
        return (np.asarray(p) - self.origin) @ self.up

    def point(self, xy, h):
        xy, h = np.atleast_2d(xy), np.atleast_1d(h)
        return self.origin + xy[:, :1] * self.e1 + xy[:, 1:2] * self.e2 + h[:, None] * self.up


class Mover:
    """Sends end-effector targets to gamepad_teleop --input policy as TeleMoMa actions, along a path."""

    def __init__(self, ros, port, speed):
        self.ros = ros
        self.step_m = speed * MAX_LINEAR_M_S / TELEOP_GAIN
        self.step_rad = speed * MAX_ANGULAR_RAD_S / TELEOP_GAIN
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = ('127.0.0.1', port)
        self.session = random.getrandbits(32) | 1
        self.sequence = 0
        self.moving = False
        self.message = 'Click a point in a camera view.'
        self.stop_requested = False
        self.lock = threading.Lock()

    def _send(self, action, enabled):
        self.sequence += 1
        self.socket.sendto(PACKET.pack(b'FPA1', 1 if enabled else 0, self.sequence, self.session << 32,
                                       *[float(v) for v in action]), self.address)

    def hold(self):
        for _ in range(3):
            self._send(np.zeros(7), False)
            time.sleep(0.01)

    def follow(self, path, rotation, guard):
        """path: (N, 3) TCP positions (base frame, every ~1 cm), rotation: the TCP's final orientation,
        guard(position) -> None or the reason to stop at once."""
        with self.lock:
            if self.moving:
                raise RuntimeError('already moving: Stop first')
            self.moving, self.stop_requested = True, False
        threading.Thread(target=self._run, args=(np.asarray(path), rotation, guard), daemon=True).start()

    def stop(self):
        self.stop_requested = True

    def _run(self, path, rotation, guard):
        started, reached, reason, index = time.monotonic(), 0, 'stopped', 0
        lookahead = max(1, int(round(LOOKAHEAD_M / 0.01)))
        try:
            while not self.stop_requested:
                robot = self.ros.robot()
                if robot is None:
                    reason = 'no robot state (is the teleop running?)'
                    break
                ee = robot['ee']
                position = ee[:3, 3]
                problem = guard(position)
                if problem:
                    reason = 'STOPPED: ' + problem
                    break
                # progress along the path: the closest point in a window ahead, never backwards
                window = path[index:index + 3 * lookahead + 1]
                index += int(np.argmin(np.linalg.norm(window - position, axis=1)))
                target = path[min(index + lookahead, len(path) - 1)]
                goal = path[-1]
                rot = rotation @ ee[:3, :3].T
                deg = angle_deg(rot)
                dist_goal = float(np.linalg.norm(goal - position))
                if dist_goal < REACHED_M and deg < REACHED_DEG:
                    reached += 1
                    if reached >= REACHED_TICKS:
                        reason = 'arrived (%.1f mm, %.1f deg off)' % (dist_goal * 1000, deg)
                        break
                else:
                    reached = 0
                if time.monotonic() - started > 60:
                    reason = 'not reached in 60 s (%.0f mm off): workspace fence or contact?' % (dist_goal * 1000)
                    break
                delta = target - position
                action = np.zeros(7)
                action[:3] = delta * min(1.0, self.step_m / max(np.linalg.norm(delta), 1e-9))
                euler = euler_xyz(rot)
                action[3:6] = euler * min(1.0, self.step_rad / max(np.linalg.norm(euler), 1e-9))
                action[6] = 1.0 if robot['gripper_width'] > 0.04 else 0.0  # keep the gripper as it is
                self._send(action, True)
                self.message = 'moving: %.0f mm to go (path point %d/%d)' % (dist_goal * 1000, index + 1, len(path))
                time.sleep(1.0 / RATE_HZ)
        finally:
            self.hold()
            self.moving = False
            self.message = reason


class Demo:
    def __init__(self, args):
        calib = json.loads(Path(args.calibration).read_text())
        self.T_base_cam2 = np.array(calib['T_base_cam_' + args.static])
        self.T_ee_cam1 = np.array(calib['T_ee_cam_' + args.wrist])
        self.calibration_session = Path(calib.get('session', '?')).name
        self.table = Table(calib)
        self.up = self.table.up
        print('up (table normal) in the base frame: %s, %.1f deg from the base z axis' % (
            self.up.round(4), math.degrees(math.acos(min(1.0, self.up[2])))), flush=True)
        self.fence = workspace_fence(args.teleop_config)
        self.hand_radius = args.hand_radius
        self.clearance = args.clearance
        self.wrist, self.static = args.wrist, args.static
        self.ros = Ros([args.wrist, args.static])
        self.mover = Mover(self.ros, args.port_udp, args.speed)
        self.tracker = Tracker(self.ros.rgb_frame)
        self.point = None        # clicked surface point in the base frame
        self.hover_m = args.hover
        self.object, self.cup_arg, self.inflate = args.object, args.cup, args.inflate
        self.cup = None          # name of the tracked cup (CUPS), picked on the first fit after a select
        # {'xy': axis in the table plane, 'radius', 'bottom', 'top', 'centre', 'time'}; --object points also
        # 'cloud' (N, 3), and xy / radius are then a circle around all its inflated points
        self.obstacle = None
        self.frozen = None       # obstacle used by the running move
        self.plan = None         # {'path': (N, 3), 'kind', 'rotation'} or {'error'}
        self.lock = threading.Lock()
        threading.Thread(target=self._obstacle_loop, daemon=True).start()

    # ---------------------------------------------------------------- geometry
    def camera_pose(self, cam, robot):
        """T_base_cam of a camera now (the wrist camera moves with the end effector)."""
        if cam == self.static:
            return self.T_base_cam2
        return None if robot is None else robot['ee'] @ self.T_ee_cam1

    def keepout(self, obstacle):
        """Keep-out region for the TCP: (radius around the axis, height above the table). The fingertips reach
        FINGERTIP_BELOW_TCP_M below the TCP, so the region's top is raised by that too."""
        return (obstacle['radius'] + self.hand_radius + self.clearance,
                obstacle['top'] + self.clearance + FINGERTIP_BELOW_TCP_M)

    def _columns(self, points, obstacle, reach, above, below):
        """Which points are within reach (in the table plane) of an obstacle point, at most above higher and below
        lower than it."""
        points = np.atleast_2d(points)
        cloud = obstacle['cloud']
        d = np.linalg.norm(self.table.xy(points)[:, None] - self.table.xy(cloud)[None], axis=2)
        dh = self.table.height(points)[:, None] - self.table.height(cloud)[None]
        return ((d < reach) & (dh < above) & (dh > -below)).any(axis=1)

    def _pick_cup(self, result):
        if self.cup_arg != 'auto':
            return self.cup_arg
        r, g, b = result['color'][result['mask']].reshape(-1, 3).mean(0)
        return 'black' if max(r, g, b) < 70 else 'red' if r > b else 'blue'

    def _fit(self, result, seed):
        """The obstacle from the tracked mask and its depth (--object), or None."""
        cam, mask, depth, K = result['cam'], result['mask'], result['depth'], result['K']
        if mask.shape != depth.shape:
            return None
        robot = self.ros.robot()
        pose = self.camera_pose(cam, robot)
        if pose is None or (cam == self.wrist and robot['dq_max'] > 0.02):
            return None
        inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # edge pixels have flying depth
        if inner.sum() < 30:
            inner = mask
        v, u = np.nonzero(inner)
        z = depth[v, u].astype(float) * 0.001
        ok = (z > 0.1) & (z < 3.0)
        if ok.sum() < 20:
            return None
        u, v, z = u[ok], v[ok], z[ok]
        p = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1) @ pose[:3, :3].T + pose[:3, 3]
        p = p[self.table.height(p) > 0.004]  # above the table
        if len(p) < 20:
            return None
        if self.object == 'points':
            return self._fit_points(p)
        if self.cup is None:
            self.cup = self._pick_cup(result)
        return self._fit_cup(p, pose, seed, CUPS[self.cup])

    def _fit_points(self, p):
        """The depth points averaged in VOXEL_M voxels (voxels with under 3 points are dropped: flying depth)."""
        keys = np.floor(p / VOXEL_M).astype(np.int64)
        _, index, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        index = index.ravel()
        sums = np.zeros((len(counts), 3))
        np.add.at(sums, index, p)
        cloud = (sums / counts[:, None])[counts >= 3]
        if len(cloud) < 3:
            return None
        xy, h = self.table.xy(cloud), self.table.height(cloud)
        c = (xy.min(axis=0) + xy.max(axis=0)) / 2
        bottom = float(h.min()) - self.inflate
        bottom = 0.0 if bottom < 0.015 else bottom
        top = float(h.max()) + self.inflate
        return {'cloud': cloud, 'xy': c, 'radius': float(np.linalg.norm(xy - c, axis=1).max()) + self.inflate,
                'bottom': bottom, 'top': top, 'points': len(cloud), 'centre': self.table.point(c, (bottom + top) / 2)[0]}

    def _fit_cup(self, p, pose, seed, size):
        """A cup of the known size (height, diameter): only points within its size of the last centre (seed)
        count, so a mask leaking onto the hand or arm holding it is ignored; the top is the highest point seen,
        the bottom one cup height below (a lifted cup floats above the table)."""
        reach = math.hypot(size[1] / 2, size[0] / 2) + 0.015
        centre = seed if seed is not None else np.median(p, axis=0)
        for _ in range(3):
            near = np.linalg.norm(p - centre, axis=1) < reach
            if near.sum() < 20:  # lost the seed (moved fast): start from all points
                near = np.ones(len(p), bool)
            centre = np.median(p[near], axis=0)
        p = p[near]
        h = self.table.height(p)
        p = p[h > np.percentile(h, 97) - size[0] - 0.01]  # nothing more than a cup height below its top
        xy, h = self.table.xy(p), self.table.height(p)
        visible = np.median(xy, axis=0)
        view = visible - self.table.xy(pose[:3, 3])
        view /= np.linalg.norm(view)
        top = float(np.percentile(h, 97))
        radius = size[1] / 2
        axis = visible + view * radius * 0.7  # the visible surface is curved: its median lies in front of the axis
        bottom = top - size[0]
        bottom = 0.0 if bottom < 0.015 else bottom  # standing on the table
        return {'xy': axis, 'radius': radius, 'bottom': bottom, 'top': top, 'points': len(p),
                'centre': self.table.point(axis, (bottom + top) / 2)[0]}

    def _obstacle_loop(self):
        last = None
        while True:
            result = self.tracker.latest()
            if result is None or result['time'] == last:
                time.sleep(0.02)
                continue
            last = result['time']
            with self.lock:
                seed = None if self.obstacle is None else self.obstacle['centre']
            fit = self._fit(result, seed)
            with self.lock:
                if fit is None or self.tracker.latest() is None:  # cleared meanwhile
                    continue
                old = self.obstacle
                if 'cloud' in fit or old is None or np.linalg.norm(old['centre'] - fit['centre']) > 0.05:
                    self.obstacle = dict(fit, time=time.monotonic())
                else:  # smooth the noise of single frames
                    a = 0.3
                    self.obstacle = {k: (1 - a) * old[k] + a * fit[k] for k in ('xy', 'radius', 'bottom', 'top', 'centre')}
                    self.obstacle.update(points=fit['points'], time=time.monotonic())

    # ---------------------------------------------------------------- planning
    def goal(self, robot):
        """End-effector goal: the hover height above the clicked point along the table's normal, the gripper's
        z axis pointing into the table (= -up), its current rotation about that axis kept."""
        z = -self.up
        x = robot['ee'][:3, 0] - np.dot(robot['ee'][:3, 0], z) * z  # current x, made perpendicular to z
        if np.linalg.norm(x) < 1e-6:
            x = np.cross(z, [0.0, 1.0, 0.0])
        x /= np.linalg.norm(x)
        target = np.eye(4)
        target[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
        target[:3, 3] = self.point + self.hover_m * self.up
        return target

    def _inside(self, points, obstacle, margin=0.0):
        """Which TCP positions are in the keep-out region (shrunk by margin)."""
        if 'cloud' in obstacle:
            reach = self.inflate + self.hand_radius + self.clearance - margin
            return self._columns(points, obstacle, reach, self.inflate + self.clearance + FINGERTIP_BELOW_TCP_M - margin, reach)
        radius, top = self.keepout(obstacle)
        d = np.linalg.norm(self.table.xy(points) - obstacle['xy'], axis=1)
        return (d < radius - margin) & (self.table.height(points) < top - margin)

    def _in_fence(self, points):
        p = np.atleast_2d(points)
        return all(((p[:, i] >= lo - 1e-3) & (p[:, i] <= hi + 1e-3)).all() for i, (lo, hi) in enumerate(self.fence.values()))

    @staticmethod
    def _resample(points, step=0.01):
        points = np.asarray(points, float)
        out = [points[0]]
        for a, b in zip(points[:-1], points[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
            out += [a + (b - a) * (k / n) for k in range(1, n + 1)]
        return np.array(out)

    def _around(self, start, goal, obstacle):
        """Shortest way around the keep-out cylinder in the table plane (tangents and an arc), heights changing
        evenly along it; the candidates inside the workspace fence, shortest first."""
        radius = self.keepout(obstacle)[0] + PATH_EXTRA_M
        c = obstacle['xy']
        s2, g2 = self.table.xy(start), self.table.xy(goal)
        hs, hg = self.table.height(start), self.table.height(goal)

        def tangents(p):
            d = np.linalg.norm(p - c)
            base, a = math.atan2(*(p - c)[::-1]), math.acos(min(1.0, radius / d))
            return [base + a, base - a]
        candidates = []
        for ts in tangents(s2):
            for tg in tangents(g2):
                for direction in (1, -1):
                    sweep = (tg - ts) * direction % (2 * math.pi)
                    if sweep > math.pi * 1.5:
                        continue
                    angles = ts + direction * np.linspace(0, sweep, max(2, int(math.degrees(sweep) / 4) + 1))
                    arc = c + radius * np.stack([np.cos(angles), np.sin(angles)], 1)
                    xy = np.vstack([s2, arc, g2])
                    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
                    frac = np.concatenate([[0], np.cumsum(seg)]) / max(seg.sum(), 1e-9)
                    path = self._resample(self.table.point(xy, hs + (hg - hs) * frac))
                    if not self._inside(path, obstacle, margin=0.002).any():
                        candidates.append((seg.sum(), path))
        return [p for _, p in sorted(candidates, key=lambda t: t[0]) if self._in_fence(p)]

    def make_plan(self, robot, obstacle):
        goal = self.goal(robot)
        start = robot['ee'][:3, 3]
        rotation = goal[:3, :3]
        if not self._in_fence(goal[:3, 3]):
            return {'error': 'the target is outside the workspace fence %s' % self.fence}
        if obstacle is None:
            return {'kind': 'straight (no obstacle)', 'path': self._resample([start, goal[:3, 3]]), 'rotation': rotation}
        radius, top = self.keepout(obstacle)
        if self._inside(goal[:3, 3][None], obstacle)[0]:
            d = np.linalg.norm(self.table.xy(goal[:3, 3]) - obstacle['xy'])
            if 'cloud' in obstacle:
                return {'error': 'the target is too close to the obstacle: the gripper needs %.0f cm around it '
                                 '(or raise the hover above %.0f cm)' % ((self.inflate + self.hand_radius + self.clearance) * 100, top * 100)}
            return {'error': 'the target is too close to the obstacle: %.0f cm from its axis, the gripper needs %.0f cm '
                             '(or raise the hover above %.0f cm)' % (d * 100, radius * 100, top * 100)}
        straight = self._resample([start, goal[:3, 3]])
        if not self._inside(straight, obstacle).any():
            return {'kind': 'straight (clear of the obstacle)', 'path': straight, 'rotation': rotation}
        if not self._inside(start[None], obstacle)[0]:
            around = self._around(start, goal[:3, 3], obstacle)
            if around:
                return {'kind': 'around the obstacle', 'path': around[0], 'rotation': rotation}
        # over it: up to the keep-out top (plus the path margin), across, down
        h_over = max(top + PATH_EXTRA_M, self.table.height(start), self.table.height(goal[:3, 3]))
        s2, g2 = self.table.xy(start), self.table.xy(goal[:3, 3])
        over = self._resample(np.vstack([start, self.table.point(s2, h_over), self.table.point(g2, h_over), goal[:3, 3][None]]))
        if not self._in_fence(over):
            return {'error': 'no path: around leaves the workspace fence and over is too high'}
        return {'kind': 'over the obstacle', 'path': over, 'rotation': rotation}

    def guard(self, obstacle):
        """Stops the move if the TCP comes within the uninflated reach (the hand would touch the obstacle)."""
        def check(position):
            if 'cloud' in obstacle:
                if self._columns(position, obstacle, self.inflate + self.hand_radius, self.inflate,
                                 self.inflate + self.hand_radius)[0]:
                    return 'the gripper came within its reach of the obstacle'
                return None
            d = np.linalg.norm(self.table.xy(position) - obstacle['xy'])
            if d < obstacle['radius'] + self.hand_radius and self.table.height(position) < obstacle['top']:
                return 'the gripper came within %.0f cm of the obstacle axis' % (d * 100)
            return None
        return check

    # ---------------------------------------------------------------- page actions
    def pixel_point(self, cam, fx, fy):
        color, depth, K = self.ros.frame(cam)
        if color is None or depth is None or K is None:
            raise RuntimeError('no image, depth or intrinsics from %s yet' % cam)
        robot = self.ros.robot()
        if cam == self.wrist and robot is None:
            raise RuntimeError('no robot state: the wrist camera needs the end-effector pose')
        if cam == self.wrist and robot['dq_max'] > 0.02:
            raise RuntimeError('the arm is moving: click the wrist view while it is still')
        h, w = depth.shape
        u, v = fx * w, fy * h
        win = depth[max(int(v) - 4, 0):int(v) + 5, max(int(u) - 4, 0):int(u) + 5]
        valid = win[win > 0]
        if valid.size < 5:
            raise RuntimeError('no depth at that pixel (too close, shiny or black surface?)')
        z = float(np.median(valid)) * 0.001
        p_cam = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, 1.0])
        return (self.camera_pose(cam, robot) @ p_cam)[:3], u, v, z

    def click(self, cam, fx, fy):
        point, _, _, z = self.pixel_point(cam, fx, fy)
        self.point = point
        robot = self.ros.robot()
        if robot is not None and not self.mover.moving:
            with self.lock:
                self.plan = self.make_plan(robot, self.obstacle)
        return {'point': point.round(4).tolist(), 'camera': cam, 'depth_m': round(z, 3), 'plan': self.plan_summary()}

    def select_obstacle(self, cam, fx, fy):
        _, u, v, _ = self.pixel_point(cam, fx, fy)
        if self.mover.moving:
            raise RuntimeError('wait until the move has finished')
        with self.lock:
            self.obstacle, self.plan, self.cup = None, None, None
        self.tracker.start(cam, u, v)
        return {'tracking': cam}

    def clear_obstacle(self):
        self.tracker.clear()
        with self.lock:
            self.obstacle, self.plan, self.cup = None, None, None

    def go(self, hover_m, clearance_m):
        if self.point is None:
            raise RuntimeError('click a target point first')
        robot = self.ros.robot()
        if robot is None:
            raise RuntimeError('no robot state: is the teleop running?')
        self.hover_m = max(0.02, float(hover_m))
        self.clearance = max(0.0, float(clearance_m))
        with self.lock:
            obstacle = None if self.obstacle is None else dict(self.obstacle)
            if obstacle is not None and time.monotonic() - obstacle['time'] > 2.0:
                raise RuntimeError('the obstacle is not being tracked right now: select it again')
            plan = self.make_plan(robot, obstacle)
            self.plan, self.frozen = plan, obstacle
        if 'error' in plan:
            raise RuntimeError(plan['error'])
        guard = self.guard(obstacle) if obstacle else (lambda position: None)
        self.mover.follow(plan['path'], plan['rotation'], guard)
        return {'plan': self.plan_summary()}

    def plan_summary(self):
        plan = self.plan
        if plan is None:
            return None
        if 'error' in plan:
            return {'error': plan['error']}
        return {'kind': plan['kind'], 'length_m': round(float(np.sum(np.linalg.norm(np.diff(plan['path'], axis=0), axis=1))), 3)}

    def status(self):
        robot = self.ros.robot()
        with self.lock:
            obstacle = self.frozen if self.mover.moving and self.frozen else self.obstacle
        out = {'point': None if self.point is None else self.point.round(4).tolist(), 'hover_m': self.hover_m,
               'clearance_m': self.clearance, 'moving': self.mover.moving, 'message': self.mover.message,
               'calibration': self.calibration_session, 'plan': self.plan_summary(),
               'tracker': {'ready': self.tracker.ready, 'error': self.tracker.error, 'camera': self.tracker.cam,
                           'ms': None if self.tracker.ms is None else round(self.tracker.ms),
                           'fps': None if self.tracker.fps is None else round(self.tracker.fps, 1)},
               'robot': None if robot is None else {'ee_xyz': robot['ee'][:3, 3].round(4).tolist(),
                                                    'motion_enabled': robot['motion_enabled']}}
        if obstacle:
            radius, top = self.keepout(obstacle)
            centre = self.table.point(obstacle['xy'], 0.0)[0]
            out['obstacle'] = {'centre': centre.round(3).tolist(), 'diameter_cm': round(obstacle['radius'] * 200, 1),
                               'label': ('%d points, +%g cm' % (obstacle['points'], round(self.inflate * 100, 1))
                                         if 'cloud' in obstacle else '%s cup' % self.cup),
                               'bottom_cm': round(obstacle['bottom'] * 100, 1),
                               'height_cm': round(obstacle['top'] * 100, 1), 'keepout_radius_cm': round(radius * 100, 1),
                               'keepout_height_cm': round(top * 100, 1),
                               'age_s': round(time.monotonic() - obstacle['time'], 1), 'frozen': self.mover.moving}
        return out

    # ---------------------------------------------------------------- drawing
    def _draw_cylinder(self, img, px, xy, radius, top, color, fill, low=0.0):
        angles = np.linspace(0, 2 * math.pi, 33)
        ring = xy + radius * np.stack([np.cos(angles), np.sin(angles)], 1)
        bottom = [px(p) for p in self.table.point(ring, np.full(len(ring), low))]
        upper = [px(p) for p in self.table.point(ring, np.full(len(ring), top))]
        pts = [p for p in bottom + upper if p is not None]
        if len(pts) < 3:
            return
        if fill:
            overlay = img.copy()
            cv2.fillConvexPoly(overlay, cv2.convexHull(np.array(pts, np.int32)), color)
            cv2.addWeighted(overlay, fill, img, 1 - fill, 0, img)
        for ring_px in (bottom, upper):
            if None not in ring_px:
                cv2.polylines(img, [np.array(ring_px, np.int32)], True, color, 1, cv2.LINE_AA)
        for k in range(0, 32, 4):
            if bottom[k] and upper[k]:
                cv2.line(img, bottom[k], upper[k], color, 1, cv2.LINE_AA)

    def view(self, cam):
        color, _, K = self.ros.frame(cam)
        if color is None or K is None:
            img = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(img, 'waiting for %s' % cam, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return img
        img = color.copy()
        result = self.tracker.latest()
        if result is not None and result['cam'] == cam and result['mask'].shape == img.shape[:2]:
            contours, _ = cv2.findContours(result['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (255, 255, 0), 2)
        robot = self.ros.robot()
        pose = self.camera_pose(cam, robot)
        if pose is None:
            return img
        cam_from_base = np.linalg.inv(pose)

        def px(p):
            q = cam_from_base[:3, :3] @ p + cam_from_base[:3, 3]
            if q[2] < 0.05:
                return None
            u, v = K[0, 0] * q[0] / q[2] + K[0, 2], K[1, 1] * q[1] / q[2] + K[1, 2]
            return (int(u), int(v)) if -2000 < u < 4000 and -2000 < v < 4000 else None
        with self.lock:
            obstacle = self.frozen if self.mover.moving and self.frozen else self.obstacle
            plan = self.plan
        if obstacle:
            radius, top = self.keepout(obstacle)
            low = max(0.0, obstacle['bottom'] - self.hand_radius - self.clearance) if 'cloud' in obstacle else 0.0
            self._draw_cylinder(img, px, obstacle['xy'], radius, top, (0, 140, 255), 0.18, low)  # keep-out for the TCP
            if 'cloud' in obstacle:  # each point as a disc of the inflation radius
                overlay = img.copy()
                for p in obstacle['cloud']:
                    c, q = px(p), cam_from_base[:3, :3] @ p + cam_from_base[:3, 3]
                    if c:
                        cv2.circle(overlay, c, max(2, int(K[0, 0] * self.inflate / q[2])), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            else:
                self._draw_cylinder(img, px, obstacle['xy'], obstacle['radius'], obstacle['top'], (0, 0, 255), 0.45,
                                    obstacle['bottom'])  # object
        if plan and 'path' in plan:
            pts = [px(p) for p in plan['path'][::2]]
            pts = [p for p in pts if p is not None]
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32)], False, (255, 0, 255), 2, cv2.LINE_AA)
        if self.point is not None:
            for p, color_, label in ((self.point, (0, 0, 255), 'point'), (self.point + self.hover_m * self.up, (0, 200, 255), 'goal')):
                c = px(p)
                if c:
                    cv2.drawMarker(img, c, color_, cv2.MARKER_CROSS, 22, 2)
                    cv2.putText(img, label, (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_, 1, cv2.LINE_AA)
        if robot is not None and cam == self.static:
            c = px(robot['ee'][:3, 3])
            if c:
                cv2.circle(img, c, 7, (0, 255, 0), 2)
                cv2.putText(img, 'gripper', (c[0] + 8, c[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return img


PAGE = '''<!doctype html><html><head><meta charset="utf-8"><title>Click to move</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --bg:#f6f6f4; --panel:#fff; --text:#1d1d1b; --muted:#6b6b66; --line:#deded8; --accent:#2f6fde; --bad:#c8372d; --warn:#c77700; }
@media (prefers-color-scheme: dark) { :root { --bg:#151514; --panel:#1f1f1d; --text:#ecece8; --muted:#9a9a93; --line:#34342f; } }
body { margin:0; background:var(--bg); color:var(--text); font:14px system-ui, sans-serif; }
main { max-width:1400px; margin:0 auto; padding:16px; display:grid; gap:14px; }
.views { display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:14px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; }
img { width:100%; display:block; border-radius:6px; cursor:crosshair; background:#000; }
.bar { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
button { padding:8px 16px; border-radius:8px; border:1px solid var(--line); background:var(--panel); color:var(--text); font:inherit; cursor:pointer; }
button.go { background:var(--accent); border-color:var(--accent); color:#fff; } button.stop { background:var(--bad); border-color:var(--bad); color:#fff; font-weight:650; }
button.armed { background:var(--warn); border-color:var(--warn); color:#fff; }
input { width:60px; padding:6px; border-radius:6px; border:1px solid var(--line); background:var(--bg); color:var(--text); font:inherit; }
.muted { color:var(--muted); } h2 { margin:0 0 8px; font-size:14px; }
</style></head><body><main>
<div class="panel bar"><b>Click to move</b>
  <button id="obst">Select obstacle</button><button id="clear">Clear obstacle</button>
  <span>hover <input id="hover" type="number" step="1" min="2" value="10"> cm</span>
  <span>clearance <input id="clear-cm" type="number" step="1" min="0" value="CLEARANCE"> cm</span>
  <button class="go" id="go">Go</button><button class="stop" id="stop">Stop</button></div>
<div class="panel bar"><span id="msg">Click a target point, or Select obstacle and click on it.</span></div>
<div class="panel bar muted" id="info">&nbsp;</div>
<div class="views">
  <div class="panel"><h2>wrist camera (click while the arm is still)</h2><img data-cam="WRIST" src="/stream/WRIST"></div>
  <div class="panel"><h2>static camera</h2><img data-cam="STATIC" src="/stream/STATIC"></div>
</div>
<div class="panel muted">Yellow outline: EdgeTAM's mask. Red: the obstacle from the mask and depth (inflated points, or the cup). Orange cylinder:
the keep-out region for the gripper's centre (the obstacle + the hand's reach + the clearance; with points, the
planner goes around this circle, and checks straight paths against every point). Magenta: the planned path.
Red / orange crosses: the clicked point and where the gripper's centre goes. Green circle: the gripper's centre from its joint readings.
Keep the robot's stop button in reach.</div>
</main><script>
const $ = (id) => document.getElementById(id);
let selecting = false;
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const d = await r.json(); if (!r.ok || d.ok === false) throw new Error(d.error || r.statusText); return d;
}
function planText(p) { return !p ? '' : p.error ? 'Plan: ' + p.error : `Plan: ${p.kind}, ${p.length_m} m.`; }
$('obst').addEventListener('click', () => { selecting = !selecting; $('obst').classList.toggle('armed', selecting);
  $('msg').textContent = selecting ? 'Click on the obstacle in a camera view.' : ''; });
$('clear').addEventListener('click', () => api('/api/obstacle/clear', {}).then(() => { $('msg').textContent = 'Obstacle cleared.'; }));
for (const img of document.querySelectorAll('img')) img.addEventListener('click', async (e) => {
  const b = img.getBoundingClientRect(), at = {cam: img.dataset.cam, x: (e.clientX - b.left) / b.width, y: (e.clientY - b.top) / b.height};
  try {
    if (selecting) { await api('/api/obstacle', at); selecting = false; $('obst').classList.remove('armed');
                     $('msg').textContent = 'Tracking the obstacle; now click a target point.'; }
    else { const r = await api('/api/click', at);
           $('msg').textContent = `Target x ${r.point[0].toFixed(3)} y ${r.point[1].toFixed(3)} z ${r.point[2].toFixed(3)} m. ${planText(r.plan)} Check the drawing, then Go.`; }
  } catch (err) { $('msg').textContent = err.message; }
});
$('go').addEventListener('click', async () => { try { const r = await api('/api/go', {hover_m: Number($('hover').value) / 100, clearance_m: Number($('clear-cm').value) / 100});
  $('msg').textContent = planText(r.plan); } catch (e) { $('msg').textContent = e.message; } });
$('stop').addEventListener('click', () => api('/api/stop', {}).catch(() => {}));
let lastMove = null;
async function poll() {
  try { const s = await api('/api/status'), t = s.tracker, o = s.obstacle;
    let info = !s.robot ? 'No robot state: is the teleop running? ' : '';
    info += !t.ready ? (t.error || 'EdgeTAM loading...') : t.camera ? `EdgeTAM tracking in ${t.camera}: ${t.fps ?? '-'} fps (${t.ms ?? '-'} ms per frame). ` : 'EdgeTAM ready. ';
    if (o) info += `Obstacle (${o.label}): ${o.diameter_cm} cm wide, from ${o.bottom_cm} to ${o.height_cm} cm at x ${o.centre[0]} y ${o.centre[1]}; keep-out ${o.keepout_radius_cm} cm around its axis up to ${o.keepout_height_cm} cm${o.frozen ? ' (frozen during the move)' : o.age_s > 2 ? ' (not tracked for ' + o.age_s + ' s)' : ''}.`;
    $('info').textContent = info;
    if (lastMove !== null && (s.moving || s.message !== lastMove)) $('msg').textContent = s.message;  // only new ones
    lastMove = s.message;
  } catch (e) {}
  setTimeout(poll, 300);
}
poll();
</script></body></html>'''


def serve(demo, port):
    page = PAGE.replace('WRIST', demo.wrist).replace('STATIC', demo.static).replace(
        'CLEARANCE', '%g' % round(demo.clearance * 100)).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, data, code=200):
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(page)))
                self.end_headers()
                self.wfile.write(page)
            elif self.path == '/api/status':
                self._json(dict(demo.status(), ok=True))
            elif self.path.startswith('/stream/') and self.path[8:] in (demo.wrist, demo.static):
                cam = self.path[8:]
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                try:
                    while True:
                        jpg = cv2.imencode('.jpg', demo.view(cam), [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(jpg) + jpg + b'\r\n')
                        time.sleep(0.07)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._json({'ok': False, 'error': 'not found'}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
            try:
                if self.path == '/api/click':
                    return self._json(dict(demo.click(body['cam'], float(body['x']), float(body['y'])), ok=True))
                if self.path == '/api/obstacle':
                    return self._json(dict(demo.select_obstacle(body['cam'], float(body['x']), float(body['y'])), ok=True))
                if self.path == '/api/obstacle/clear':
                    demo.clear_obstacle()
                    return self._json({'ok': True})
                if self.path == '/api/go':
                    return self._json(dict(demo.go(body.get('hover_m', demo.hover_m), body.get('clearance_m', demo.clearance)), ok=True))
                if self.path == '/api/stop':
                    demo.mover.stop()
                    return self._json({'ok': True})
                self._json({'ok': False, 'error': 'not found'}, 404)
            except (RuntimeError, KeyError, ValueError) as error:
                self._json({'ok': False, 'error': str(error)}, 409)

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    print('Click to move: http://localhost:%d (calibration %s)' % (port, demo.calibration_session), flush=True)
    server.serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', type=int, default=8780)
    p.add_argument('--port-udp', type=int, default=47820, help='policy_port of config/teleop.conf')
    p.add_argument('--calibration', default=str(ROOT / 'config' / 'camera_calibration.json'))
    p.add_argument('--teleop-config', default=str(ROOT / 'config' / 'teleop.conf'), help='for its workspace fence')
    p.add_argument('--wrist', default='cam_1')
    p.add_argument('--static', default='cam_2')
    p.add_argument('--hover', type=float, default=0.10, help='default height above the clicked point [m]')
    p.add_argument('--speed', type=float, default=0.6, help='fraction of the teleop speed caps (0.25 m/s, 0.4 rad/s)')
    p.add_argument('--hand-radius', type=float, default=0.11,
                   help='reach of the gripper around the TCP [m]: 10.5 cm from the meshes (hand, wrist camera and mount)')
    p.add_argument('--object', default='points', choices=['points', 'cup'],
                   help='points: the inflated depth points of the mask (any shape); cup: a digital-twin cup')
    p.add_argument('--inflate', type=float, default=0.01, help='--object points: each point inflated by [m]')
    p.add_argument('--cup', default='auto', choices=['auto', *CUPS], help='--object cup: which (auto: by colour)')
    p.add_argument('--clearance', type=float, default=0.03, help='extra distance kept from the obstacle [m]')
    args = p.parse_args()
    demo = Demo(args)

    def quit_(*_):  # set after rclpy.init, which otherwise takes SIGINT/SIGTERM and keeps the server running
        demo.mover.stop()
        demo.mover.hold()
        os._exit(0)
    signal.signal(signal.SIGTERM, quit_)
    signal.signal(signal.SIGINT, quit_)
    serve(demo, args.port)


if __name__ == '__main__':
    main()
