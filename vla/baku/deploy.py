#!/usr/bin/env python3
"""Policy server: runs a Franka BAKU snapshot on the robot, controlled from the
recorder web page (Policy panel) or its HTTP API.

    camera topics -> BAKU -> UDP -> gamepad_teleop --input policy (src/policy_input.h)

Started by ./record.sh --deploy (cameras, recorder, teleop and this server).
The teleop executes each action as it executed the demonstrations: target =
pose + action, tracked with the Quest's gains, speed caps, IK and workspace fence.

Modes (state machine, one at a time):
    load      load a checkpoint from vla/baku/runs/*/snapshot/*.pt
    home      ask the teleop for the saved home pose; "arrived" from /teleop/state
    preview   run the policy on the live cameras and show its actions; NOTHING is
              sent to the robot
    run       an episode: actions are sent (enabled) at the dataset rate until
              Stop or max steps, then it waits for a label
    label     success / failure / intervention / discard for the last episode (optional,
              can be changed; the episode is saved when it ends)

--no-motion refuses home and run (preview only).

HTTP API (127.0.0.1:<port>, JSON):
    GET  /api/policy/status
    POST /api/policy/load    {"checkpoint": path}
    POST /api/policy/home
    POST /api/policy/preview {"on": true|false}
    POST /api/policy/start   {"home_first": bool, "record": bool}
    POST /api/policy/stop
    POST /api/policy/label   {"outcome": "success"|"failure"|"intervention"|"discard"}
    POST /api/policy/task    {"index": n}

Per rollout: <out>/<id>/rollout.npz + rollout.json (actions sent and raw, the
policy's input images, VQ-BeT entropies/probabilities, full code distributions,
picked codes, predicted action chunk, action-token feature, timing), with
"record" a recorder episode (cameras + robot bag), and with --safe-export the
failure-detection files task<T>--ep<E>--succ<0|1>.{csv,pkl,meta.json}.

Observation: a run trained on processed camera images says so in
<run>/observation.json (fetch_run.sh copies it), e.g. the sim2real background filter
    {"type": "background_filter", "scene": "sim2real/scenes/cups_20261024/scene.json"}
(optional "calibration", "box"; paths relative to the repository). The server then also
subscribes to the aligned depth topics and applies sim2real/observation.py to every
frame before the policy's preprocessing (wrist camera: with O_T_EE at the frame's
stamp). Without the file the policy sees the raw color frames.
"""

import argparse
import json
import math
import pickle
import random
import signal
import socket
import struct
import sys
import threading
import time
import traceback
import urllib.request
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# BAKU with the Franka add-ons: the vla/BAKU submodule (branch franka-teleop of alisharey/BAKU)
sys.path.insert(0, str(HERE.parent / 'BAKU' / 'baku'))

PACKET = struct.Struct('<4sIQQ7d')  # src/policy_input.h
ACTION_NAMES = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'dgripper']
HOME_TOLERANCE_RAD = 0.01
HOME_TIMEOUT_S = 40.0
OBSERVATION_FILE = 'observation.json'
STATE_SAMPLE_S = 0.01      # /teleop/state (1 kHz) decoded at most every 10 ms (status, wrist-filter pose)
PAIR_TOLERANCE_NS = 5_000_000  # color and aligned depth of one camera share a stamp (enable_sync)
POSE_TOLERANCE_S = 0.1     # wrist filter: the nearest O_T_EE sample must be this close to the frame


def read_home_joints(config_path):
    for line in Path(config_path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line.startswith('home_joints') and '=' in line:
            return np.array([float(v) for v in line.split('=', 1)[1].split(',')])
    return None


class RosIO:
    """Latest color frames per camera (+ aligned depth while enabled) and samples of /teleop/state."""

    def __init__(self, cameras):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from franka_teleop.msg import TeleopState

        self._deserialize = __import__('rclpy.serialization', fromlist=['deserialize_message']).deserialize_message
        self._state_type = TeleopState
        self._image_type, self._image_qos = Image, qos_profile_sensor_data
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)  # Ctrl-C / SIGTERM handled in main()
        self.node = Node('baku_policy_server')
        self.lock = threading.Lock()
        self.cameras = list(cameras)
        self.frames = {}  # camera -> deque of (receive time, RGB, stamp_ns), newest last
        self.depths = {}  # camera -> deque of (receive time, uint16 mm, stamp_ns), while depth is on
        self._depth_subscriptions = []
        self.poses = deque(maxlen=400)  # (stamp_ns, O_T_EE 4x4), ~4 s
        self.state = None
        self.state_time = None
        self._last_decode = 0.0
        for camera in cameras:
            self.node.create_subscription(Image, '/cameras/%s/color/image_raw' % camera,
                                          lambda msg, c=camera: self._on_image(c, msg), qos_profile_sensor_data)
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100, reliability=ReliabilityPolicy.RELIABLE)
        self.node.create_subscription(TeleopState, '/teleop/state', self._on_state, qos, raw=True)
        threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()

    def set_depth(self, on):
        """Subscribes to (or drops) /cameras/<cam>/aligned_depth_to_color/image_raw (16UC1, mm)."""
        with self.lock:
            if bool(on) == bool(self._depth_subscriptions):
                return
            if on:
                self._depth_subscriptions = [self.node.create_subscription(
                    self._image_type, '/cameras/%s/aligned_depth_to_color/image_raw' % camera,
                    lambda msg, c=camera: self._on_depth(c, msg), self._image_qos) for camera in self.cameras]
            else:
                for subscription in self._depth_subscriptions:
                    self.node.destroy_subscription(subscription)
                self._depth_subscriptions = []
                self.depths = {}

    @staticmethod
    def _stamp_ns(msg):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def _on_image(self, camera, msg):
        if msg.encoding not in ('rgb8', 'bgr8'):
            return
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
        image = image.reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            image = image[:, :, ::-1]
        with self.lock:
            self.frames.setdefault(camera, deque(maxlen=6)).append(
                (time.monotonic(), np.ascontiguousarray(image), self._stamp_ns(msg)))

    def _on_depth(self, camera, msg):
        if msg.encoding not in ('16UC1', 'mono16'):
            return
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
        with self.lock:
            if self._depth_subscriptions:
                self.depths.setdefault(camera, deque(maxlen=6)).append(
                    (time.monotonic(), depth.copy(), self._stamp_ns(msg)))

    def _on_state(self, data):
        now = time.monotonic()
        if now - self._last_decode < STATE_SAMPLE_S:  # 1 kHz topic: decode ~100 per second
            return
        self._last_decode = now
        state = self._deserialize(data, self._state_type)
        sample = {'q': np.array(state.q), 'ee_xyz': [float(v) for v in state.o_t_ee[12:15]],
                  'motion_enabled': bool(state.motion_enabled), 'deadman': bool(state.deadman),
                  'gripper_width': float(state.gripper_width), 'robot_mode': int(state.robot_mode),
                  'dq_max': float(np.max(np.abs(state.dq)))}
        pose = np.array(state.o_t_ee).reshape(4, 4, order='F')  # libfranka: column-major
        with self.lock:
            self.state, self.state_time = sample, now
            self.poses.append((self._stamp_ns(state), pose))

    def latest_frames(self, depth=False):
        """camera -> (receive time, RGB, stamp_ns, depth or None). With depth: the newest color
        frame that has its aligned depth frame (same stamp); a camera without one is left out."""
        with self.lock:
            frames = {c: list(q) for c, q in self.frames.items() if q}
            depths = {c: list(q) for c, q in self.depths.items() if q}
        out = {}
        for camera, recent in frames.items():
            if not depth:
                out[camera] = recent[-1] + (None,)
                continue
            for received, color, stamp in reversed(recent):
                match = [d for d in depths.get(camera, []) if abs(d[2] - stamp) <= PAIR_TOLERANCE_NS]
                if match:
                    out[camera] = (received, color, stamp, match[-1][1])
                    break
        return out

    def depth_ages(self):
        now = time.monotonic()
        with self.lock:
            return {c: round(now - q[-1][0], 2) if q else None
                    for c, q in ((c, self.depths.get(c)) for c in self.cameras)}

    def ee_pose_at(self, stamp_ns):
        """O_T_EE of the /teleop/state sample nearest to stamp_ns (None if none within POSE_TOLERANCE_S)."""
        with self.lock:
            poses = list(self.poses)
        if not poses:
            return None
        stamp, pose = min(poses, key=lambda p: abs(p[0] - stamp_ns))
        return pose if abs(stamp - stamp_ns) <= POSE_TOLERANCE_S * 1e9 else None

    def latest_pose(self):
        """Newest O_T_EE (None without a recent /teleop/state)."""
        with self.lock:
            if not self.poses or self.state_time is None or time.monotonic() - self.state_time > 1.0:
                return None
            return self.poses[-1][1]

    def robot(self):
        with self.lock:
            if self.state is None or time.monotonic() - self.state_time > 1.0:
                return None
            return dict(self.state)


def read_observation(checkpoint):
    """<run>/observation.json next to snapshot/ (None: raw camera frames)."""
    path = Path(checkpoint).parents[1] / OBSERVATION_FILE
    if not path.exists():
        return None
    spec = json.loads(path.read_text())
    if spec.get('type') != 'background_filter':
        raise RuntimeError('%s: unknown observation type %r' % (path, spec.get('type')))
    return spec


class BackgroundObservation:
    """observation.json type background_filter: sim2real/observation.py (the training data's
    filter) on each live color frame with its aligned depth; real camera depth scale."""

    label = 'background filter'
    needs_depth = True

    def __init__(self, spec):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from sim2real.observation import DEFAULT_BOX, DEFAULT_CALIBRATION, DEFAULT_SCENE, WRIST, BackgroundFilter
        self.spec = spec
        scene = ROOT / spec.get('scene', DEFAULT_SCENE)  # relative to the repository (or absolute)
        calibration = ROOT / spec.get('calibration', DEFAULT_CALIBRATION)
        box = tuple(spec.get('box', DEFAULT_BOX))
        self.filter = BackgroundFilter(scene, calibration, box, sim=False)
        self.wrist = WRIST

    def needs_pose(self, camera):
        return camera == self.wrist

    def apply(self, camera, color, depth_mm, ee_pose=None):
        return self.filter.apply(camera, color, depth_mm, ee_pose)


class Link:
    """UDP to gamepad_teleop --input policy."""

    def __init__(self, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = ('127.0.0.1', port)
        self.sequence = 0
        # The teleop takes a new session as a restart (sync, no home move) and each count
        # increment within a session as one home request (src/gamepad_teleop.cpp).
        self.session = random.getrandbits(32) | 1
        self.home_requests = 0

    def send(self, action, enabled):
        self.sequence += 1
        self.socket.sendto(PACKET.pack(b'FPA1', 1 if enabled else 0, self.sequence,
                                       (self.session << 32) | (self.home_requests & 0xFFFFFFFF),
                                       *[float(v) for v in action]), self.address)

    def hold(self):
        self.send(np.zeros(7), False)


class Recorder:
    """The recording web UI's API (franka_teleop_recorder/web.py)."""

    def __init__(self, url):
        self.url = url.rstrip('/') if url else None

    def _post(self, path, body):
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read())

    def start(self, task, name):
        self._post('/api/settings', {'task': task, 'next_name': name})
        reply = self._post('/api/record', {'action': 'start'})
        if not reply.get('ok'):
            raise RuntimeError('recorder: %s' % reply.get('error'))
        return reply['id']

    def stop(self):
        return self._post('/api/record', {'action': 'stop'}).get('id')

    def label(self, episode_id, success, notes):
        return self._post('/api/episodes/%s' % episode_id, {'success': success, 'notes': notes})


def checkpoint_label(path):
    path = Path(path)
    step = path.stem
    label = '%sk' % (int(step) // 1000) if step.isdigit() and int(step) % 1000 == 0 else step
    return '%s / %s' % (path.parents[1].name, label), label


class Server:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.state = 'idle'  # idle loading homing preview running label
        self.message = 'Load a checkpoint.'
        self.error = None
        self.policy = None
        self.loaded = None
        self.observation = None  # BackgroundObservation when the run has observation.json
        self.step = 0
        self.last = {}
        self.log = None
        self.record_episode = None
        self.last_run = None  # the last episode: saved at once; its label can be set or changed later
        self.stop_requested = False
        self.want_record = False
        self.home_joints = read_home_joints(args.teleop_config)
        self.link = Link(args.port)
        self.recorder = Recorder(args.record)
        self.ros = RosIO(['cam_1', 'cam_2'])
        self.worker = None

    # ------------------------------------------------------------ helpers
    def checkpoints(self):
        found = sorted(Path(self.args.runs).glob('*/snapshot/*.pt'),
                       key=lambda p: (p.parents[1].name, int(p.stem) if p.stem.isdigit() else 0))
        return [{'path': str(p), 'label': checkpoint_label(p)[0]} for p in found]

    def robot_status(self):
        robot = self.ros.robot()
        if robot is None:
            return {'connected': False}
        out = {'connected': True, 'ee_xyz': [round(v, 4) for v in robot['ee_xyz']],
               'motion_enabled': robot['motion_enabled'], 'gripper_width': round(robot['gripper_width'], 4),
               'moving': robot['dq_max'] > 0.01}
        if self.home_joints is not None:
            error = float(np.max(np.abs(robot['q'] - self.home_joints)))
            out.update(home_error_rad=round(error, 4), at_home=error < HOME_TOLERANCE_RAD)
        return out

    def camera_ages(self):
        now = time.monotonic()
        frames = self.ros.latest_frames()
        return {c: round(now - frames[c][0], 2) if c in frames else None for c in ('cam_1', 'cam_2')}

    def fresh_frames(self):
        """{policy key: RGB frame} with the run's observation applied (background filter: with
        the aligned depth and, wrist camera, O_T_EE at the frame's stamp), or (None, what is
        missing) when a camera, its depth or the pose is older than --camera-timeout-s."""
        observation = self.observation
        frames = self.ros.latest_frames(depth=observation is not None)
        now = time.monotonic()
        out = {}
        for key, camera in self.policy.cameras.items():
            if camera not in frames or now - frames[camera][0] > self.args.camera_timeout_s:
                return None, ('%s color+depth' if observation else 'camera %s') % camera
            received, color, stamp, depth = frames[camera]
            if observation is None:
                out[key] = color
                continue
            pose = None
            if observation.needs_pose(camera):
                pose = self.ros.ee_pose_at(stamp)
                if pose is None:
                    return None, 'robot pose (/teleop/state) for the %s filter' % camera
            out[key] = observation.apply(camera, color, depth, pose)
        if getattr(self.policy, 'needs_robot', False):  # e.g. the sim2real SAC actor: + EE position, gripper width
            robot, pose = self.ros.robot(), self.ros.latest_pose()
            if robot is None or pose is None:
                return None, 'robot state (/teleop/state)'
            self.policy.set_robot(pose, robot['gripper_width'])
        return out, None

    def status(self):
        with self.lock:
            last = {}
            if self.last:
                action = self.last.get('action')
                last = {'action': None if action is None else [round(float(v), 4) for v in action],
                        'entropy': self.last.get('entropy_avg'), 'codes': self.last.get('codes'),
                        'chunk_xyz_mm': self.last.get('chunk_xyz_mm'), 'compute_ms': self.last.get('compute_ms'),
                        'filter_ms': self.last.get('filter_ms')}
            observation = self.observation
            return {
                'ok': True, 'state': self.state, 'message': self.message, 'error': self.error,
                'no_motion': self.args.no_motion, 'checkpoints': self.checkpoints(),
                'loaded': self.loaded, 'tasks': self.policy.info['task_texts'] if self.policy else [],
                'task_index': self.policy.task_index if self.policy else None,
                'step': self.step, 'max_steps': self.args.max_steps,
                'rate_hz': round(self.policy.rate_hz, 2) if self.policy else None,
                'last': last, 'cameras': self.camera_ages(), 'robot': self.robot_status(),
                'observation': observation.label if observation else None,
                'depth': self.ros.depth_ages() if observation else None,
                'home_saved': self.home_joints is not None, 'gpu_free_mb': gpu_free_mb(),
                'last_run': None if self.last_run is None else {
                    k: self.last_run[k] for k in ('id', 'steps', 'outcome', 'reason', 'record_episode')},
                'recorder': bool(self.recorder.url),
            }

    def _set(self, state, message, error=None):
        with self.lock:
            self.state, self.message, self.error = state, message, error
        print('[%s] %s%s' % (state, message, ('  ERROR: ' + error) if error else ''), flush=True)

    def _require(self, *states):
        if self.state not in states:
            raise RuntimeError('busy: %s (%s)' % (self.state, self.message))

    def _start_worker(self, target, *args):
        def run():
            try:
                target(*args)
            except Exception as error:  # report on the page, keep the server up
                traceback.print_exc()
                if self.state in ('running', 'homing'):  # preview and loading never send anything
                    self.link.hold()
                if self.state == 'running' and self.record_episode:
                    try:
                        self.recorder.stop()
                    except OSError:
                        pass
                self._set('idle', 'Stopped after an error.', str(error))
        self.stop_requested = False
        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    # ------------------------------------------------------------ commands
    def load(self, checkpoint):
        with self.lock:
            self._require('idle')
            if checkpoint not in [c['path'] for c in self.checkpoints()]:
                raise RuntimeError('unknown checkpoint %s' % checkpoint)
            self._set('loading', 'Loading %s ...' % checkpoint_label(checkpoint)[0])
        self._start_worker(self._load, checkpoint)

    def _load(self, checkpoint):
        import torch
        from franka.policy import FrankaPolicy
        self.policy = None
        self.loaded = None
        self.observation = None
        self.ros.set_depth(False)
        torch.cuda.empty_cache()
        spec = read_observation(checkpoint)
        observation = BackgroundObservation(spec) if spec else None
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from sim2real.sac_policy import SACPolicy, is_sac_run
        policy_class = SACPolicy if is_sac_run(checkpoint) else FrankaPolicy  # <run>/policy.json type "sac"
        policy = policy_class(checkpoint, device=self.args.device, max_steps=self.args.max_steps)
        if getattr(policy, 'needs_robot', False):
            policy.set_robot(np.eye(4), 0.08)  # for the warm-up below
        blank = {k: np.zeros((480, 640, 3), np.uint8) for k in policy.pixel_keys}
        if observation:  # also builds the filter's per-pixel rays
            blank = {k: observation.apply(policy.cameras[k], v, np.zeros(v.shape[:2], np.uint16), np.eye(4))
                     for k, v in blank.items()}
            self.ros.set_depth(True)
        policy.act(blank)  # warm up CUDA kernels
        policy.reset()
        self.policy = policy
        self.observation = observation
        name, short = checkpoint_label(checkpoint)
        self.loaded = {'path': checkpoint, 'label': name, 'short': short, 'task': policy.task_text,
                       'observation': spec, 'policy': 'sac' if policy_class is SACPolicy else 'baku'}
        self._set('idle', 'Loaded %s%s.' % (name, ' (observation: %s)' % observation.label if observation else ''))

    def set_task(self, index):
        with self.lock:
            self._require('idle')
            self.policy.set_task(int(index))
            self.loaded['task'] = self.policy.task_text

    def home(self):
        with self.lock:
            if self.args.no_motion:
                raise RuntimeError('--no-motion: home is disabled')
            self._require('idle')
            if self.home_joints is None:
                raise RuntimeError('no home_joints in %s (tools/save_home_pose.sh)' % self.args.teleop_config)
            self._set('homing', 'Moving to the home pose ...')
        self._start_worker(self._home_then, None)

    def _home(self):
        for _ in range(3):  # announce this sender's session first, so the request below counts
            self.link.hold()
            time.sleep(0.02)
        self.link.home_requests += 1
        started = time.monotonic()
        while True:
            self.link.hold()  # carries the home request; keeps the arm otherwise still
            robot = self.ros.robot()
            elapsed = time.monotonic() - started
            if robot is None:
                if elapsed > 3.0:
                    raise RuntimeError('no /teleop/state: is the teleop running (record.sh --deploy)?')
            elif elapsed > 1.0 and robot['dq_max'] < 0.01 and self.robot_status().get('at_home'):
                return
            if self.stop_requested:
                self.link.home_requests += 1  # a second request cancels the teleop's home move
                for _ in range(3):
                    self.link.hold()
                raise RuntimeError('home move stopped')
            if elapsed > HOME_TIMEOUT_S:
                raise RuntimeError('home pose not reached in %.0f s' % HOME_TIMEOUT_S)
            time.sleep(0.05)

    def _home_then(self, then):
        self._home()
        if then is None:
            self._set('idle', 'At the home pose.')
        else:
            then()

    def preview(self, on):
        with self.lock:
            if on:
                self._require('idle')
                if self.policy is None:
                    raise RuntimeError('load a checkpoint first')
                self._set('preview', 'Preview: the policy runs on the live cameras; nothing is sent to the robot.')
                self._start_worker(self._loop, False)
            elif self.state == 'preview':
                self.stop_requested = True

    def start(self, home_first, record):
        with self.lock:
            if self.args.no_motion:
                raise RuntimeError('--no-motion: running episodes is disabled (use preview)')
            self._require('idle')
            if self.policy is None:
                raise RuntimeError('load a checkpoint first')
            frames, missing = self.fresh_frames()
            if frames is None:
                raise RuntimeError('no recent %s' % missing)
            if self.ros.robot() is None:
                raise RuntimeError('no /teleop/state: is the teleop running (record.sh --deploy)?')
            self.want_record = bool(record and self.recorder.url)
            if home_first:
                self._set('homing', 'Moving to the home pose, then starting ...')
                self._start_worker(self._home_then, lambda: self._loop(True))
            else:
                self._set('running', 'Starting ...')
                self._start_worker(self._loop, True)

    def stop(self):
        with self.lock:
            self.stop_requested = True

    def _loop(self, execute):
        policy = self.policy
        policy.reset()
        self.step = 0
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log = {'id': '%s_%s_task%d' % (stamp, self.loaded['short'], policy.task_index), 'checkpoint': self.loaded,
               't': [], 'action': [], 'raw_action': [], 'entropy_avg': [], 'token_entropy': [], 'token_prob': [],
               'code_probs': [], 'codes': [], 'action_chunk': [], 'feature': [], 'images': [], 'compute_ms': [],
               'camera_age_s': [], 'filter_ms': [], 'executed': execute}
        self.record_episode = None
        if execute:
            if self.want_record:
                self.record_episode = self.recorder.start(policy.task_text, 'policy_%s_%s' % (
                    self.loaded['short'], stamp))
            self._set('running', 'Running %s (Stop to end) ...' % self.loaded['label'])
        period = 1.0 / policy.rate_hz
        start = time.monotonic()
        next_tick = start
        late = 0
        reason = 'stopped'
        stale_since = None
        message = self.message
        while not self.stop_requested:
            if execute and self.ros.robot() is None:
                reason = 'no robot state (teleop stopped?)'
                break
            began = time.monotonic()
            frames, missing = self.fresh_frames()  # with the run's observation (background filter) applied
            if frames is None:  # hold still; give up if the camera stays stale
                if execute:
                    self.link.hold()
                stale_since = stale_since or time.monotonic()
                if time.monotonic() - stale_since > 1.0:
                    if execute:
                        reason = '%s stopped' % missing
                        break
                    with self.lock:
                        self.message = 'Preview waiting: no recent %s.' % missing
                time.sleep(0.02)
                next_tick = time.monotonic()
                continue
            if stale_since is not None:
                stale_since = None
                with self.lock:
                    self.message = message
            filtered = time.monotonic()
            policy_input = policy.preprocess(frames)  # the policy's input images (BAKU 128x128, SAC 84x84), also logged
            raw = policy.act(policy_input, preprocessed=True)
            compute_ms = (time.monotonic() - filtered) * 1000
            filter_ms = (filtered - began) * 1000
            action = raw.copy()
            norm = np.linalg.norm(action[:3])
            if norm > self.args.max_position_m:
                action[:3] *= self.args.max_position_m / norm
            action[3:6] = np.clip(action[3:6], -self.args.max_rotation_rad, self.args.max_rotation_rad)
            if execute:
                self.link.send(action, True)
            entropy, prob = policy.token_columns()
            last = policy.last
            chunk = last.get('action_chunk')
            with self.lock:
                self.step = policy.step_index
                self.last = {'action': action, 'entropy_avg': last.get('entropy_avg'),
                             'codes': None if last.get('codes') is None else [int(c) for c in last['codes']],
                             'chunk_xyz_mm': None if chunk is None else
                             [round(float(v), 1) for v in 1000 * chunk[:, :3].sum(axis=0)],
                             'compute_ms': round(compute_ms, 1),
                             'filter_ms': round(filter_ms, 1) if self.observation else None}
            ages = self.camera_ages()
            log['t'].append(began - start)
            log['action'].append(action)
            log['raw_action'].append(raw)
            log['entropy_avg'].append(last.get('entropy_avg', math.nan))
            log['token_entropy'].append(entropy)
            log['token_prob'].append(prob)
            log['code_probs'].append(last.get('code_probs', np.full((2, 16), np.nan)))
            log['codes'].append(last.get('codes', np.full(2, -1)))
            log['action_chunk'].append(chunk if chunk is not None else np.full((policy.info['num_queries'], 7), np.nan))
            log['feature'].append(last.get('feature', np.full(256, np.nan, np.float32)))
            log['images'].append(np.stack([policy_input[k] for k in policy.pixel_keys]))
            log['compute_ms'].append(compute_ms)
            if self.observation:
                log['filter_ms'].append(filter_ms)
            log['camera_age_s'].append([ages[c] if ages[c] is not None else math.nan for c in ('cam_1', 'cam_2')])
            if policy.step_index >= self.args.max_steps:
                if not execute:
                    policy.reset()  # preview keeps going
                    continue
                reason = 'max steps (%d)' % self.args.max_steps
                break
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                late += 1
                next_tick = time.monotonic()
        log['late_ticks'] = late
        if not execute:
            self._set('idle', 'Preview stopped (%d steps, nothing sent).' % len(log['action']))
            return
        self.link.hold()
        if self.record_episode:
            try:
                self.recorder.stop()
            except OSError as error:
                print('recorder stop failed: %s' % error, flush=True)
        run = self.save(log, reason)
        with self.lock:
            self.last_run = run
        self._set('idle', 'Episode ended (%s) after %d steps; saved %s. Label it below (optional).' % (
            reason, run['steps'], run['id']))

    def save(self, log, reason):
        """Writes the rollout at once (unlabeled); label() adds or changes the outcome later."""
        steps = len(log['action'])
        folder = Path(self.args.out) / log['id']
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.array(v) for k, v in log.items()
                  if k not in ('id', 'checkpoint', 'executed', 'late_ticks') and len(v)}
        if arrays:
            np.savez_compressed(folder / 'rollout.npz', **arrays)
        info = {'checkpoint': log['checkpoint'], 'task_index': self.policy.task_index, 'task': self.policy.task_text,
                'rate_hz': self.policy.rate_hz, 'steps': steps, 'late_ticks': log['late_ticks'], 'ended': reason,
                'outcome': None, 'recorder_episode': self.record_episode,
                'clip': {'position_m': self.args.max_position_m, 'rotation_rad': self.args.max_rotation_rad},
                'arrays': {k: list(np.shape(v)) for k, v in arrays.items()}}
        (folder / 'rollout.json').write_text(json.dumps(info, indent=1))
        return {'id': log['id'], 'folder': folder, 'info': info, 'log': log, 'steps': steps, 'reason': reason,
                'outcome': None, 'record_episode': self.record_episode, 'export': None,
                'task_index': self.policy.task_index}

    def label(self, outcome):
        """Sets (or changes) the outcome of the last episode: rollout.json, the recorder
        episode's success flag and, with --safe-export, the failure-detection files."""
        outcome = {'s': 'success', 'f': 'failure', 'i': 'intervention', 'x': 'discard'}.get(outcome, outcome)
        if outcome not in ('success', 'failure', 'intervention', 'discard'):
            raise RuntimeError('outcome must be success, failure, intervention or discard')
        with self.lock:
            run = self.last_run
            if run is None:
                raise RuntimeError('no episode to label yet')
            run['outcome'] = outcome
            run['info']['outcome'] = outcome
            (run['folder'] / 'rollout.json').write_text(json.dumps(run['info'], indent=1))
            success = outcome == 'success'
            notes = 'BAKU policy rollout: %s, task %d, %s' % (run['log']['checkpoint']['path'], run['task_index'],
                                                             outcome)
            if run['export']:  # relabelled: replace the earlier export
                for path in run['export']:
                    Path(path).unlink(missing_ok=True)
                run['export'] = None
            if self.args.safe_export and outcome != 'discard' and run['steps']:
                stem = write_safe_export(Path(self.args.safe_export), run['log'], self.policy, self.args, success,
                                         outcome == 'intervention', notes, run['task_index'])
                run['export'] = [str(stem) + ext for ext in ('.csv', '.pkl', '.meta.json')]
            self.message = 'Last episode %s: %s.' % (run['id'], outcome)
        if run['record_episode']:
            episode, value = run['record_episode'], None if outcome == 'discard' else success

            def label_when_saved():  # the recorder refuses edits while it still writes the episode
                for _ in range(60):
                    try:
                        self.recorder.label(episode, value, notes)
                        return
                    except OSError:
                        time.sleep(1.0)
                print('recorder: could not label episode %s' % episode, flush=True)
            threading.Thread(target=label_when_saved, daemon=True).start()


def write_safe_export(folder, log, policy, args, success, intervention, notes, task_id):
    """interpretability/monitor.py's per-rollout files for Failure_Detection."""
    import torch
    folder.mkdir(parents=True, exist_ok=True)
    used = [int(p.name.split('--ep')[1].split('--')[0]) for p in folder.glob('task%d--ep*--succ*.csv' % task_id)]
    episode = max(used) + 1 if used else 0
    stem = folder / ('task%d--ep%d--succ%d' % (task_id, episode, 1 if success else 0))
    header = ['action/%s' % n for n in ACTION_NAMES] + ['action/token_%d_entropy' % i for i in range(7)] + \
             ['action/token_%d_prob' % i for i in range(7)]
    with open(str(stem) + '.csv', 'w') as f:
        f.write(','.join(header) + '\n')
        for action, entropy, prob in zip(log['action'], log['token_entropy'], log['token_prob']):
            f.write(','.join('%.9g' % v for v in [*action, *entropy, *prob]) + '\n')
    with open(str(stem) + '.pkl', 'wb') as f:
        pickle.dump({'hidden_states': torch.tensor(np.array(log['feature']), dtype=torch.float32).unsqueeze(1),
                     'task_suite_name': args.suite_name, 'task_id': task_id, 'task_description': policy.task_text,
                     'episode_idx': episode, 'episode_success': 1 if success else 0}, f)
    meta = {'policy': 'baku', 'suite': args.suite_name, 'kind': 'natural', 'task_id': task_id,
            'episode_idx': episode, 'success': 1 if success else 0, 'checkpoint': policy.snapshot,
            'demo_dataset_version': {'path': policy.info['dataset']['path'],
                                     'episodes': policy.info['dataset']['episode_ids']},
            'robot_id': args.robot_id, 'control_hz': policy.rate_hz,
            'camera_calibration_version': args.camera_calibration,
            'action_filter': 'deploy clip |pos| <= %g m, |rot| <= %g rad; gamepad_teleop input=policy: target '
                             'tracking (quest gains), speed caps, workspace fence, IK' % (args.max_position_m,
                                                                                          args.max_rotation_rad),
            'executed_action': 'the CSV holds the actions sent (after the clip); the robot tracked them through '
                               'the teleop controller (recorded in the bag when the episode was recorded)',
            'intervention': bool(intervention), 'notes': notes, 'task_description': policy.task_text,
            'observation': (log['checkpoint'].get('observation') or {}).get('type', 'camera')}
    Path(str(stem) + '.meta.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
    return stem


def gpu_free_mb():
    try:
        import torch
        free, _ = torch.cuda.mem_get_info()
        return int(free / 2**20)
    except Exception:
        return None


def make_http(server, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, payload):
            body = json.dumps(payload, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')  # the recorder page (another port) calls this
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()

        def do_GET(self):
            if self.path == '/api/policy/status':
                return self._send(200, server.status())
            self._send(404, {'ok': False, 'error': 'not found'})

        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length') or 0)
                body = json.loads(self.rfile.read(length) or b'{}')
                route = self.path
                if route == '/api/policy/load':
                    server.load(body['checkpoint'])
                elif route == '/api/policy/home':
                    server.home()
                elif route == '/api/policy/preview':
                    server.preview(bool(body.get('on', True)))
                elif route == '/api/policy/start':
                    server.start(bool(body.get('home_first')), bool(body.get('record', True)))
                elif route == '/api/policy/stop':
                    server.stop()
                elif route == '/api/policy/label':
                    server.label(body.get('outcome', ''))
                elif route == '/api/policy/task':
                    server.set_task(body['index'])
                else:
                    return self._send(404, {'ok': False, 'error': 'not found'})
                return self._send(200, {'ok': True})
            except Exception as error:
                return self._send(409, {'ok': False, 'error': str(error)})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--http-port', type=int, default=8766, help='API port (the recorder page calls it)')
    parser.add_argument('--runs', default=str(HERE / 'runs'), help='checkpoints: <runs>/<run>/snapshot/<step>.pt')
    parser.add_argument('--checkpoint', default=None, help='load this one at start')
    parser.add_argument('--no-motion', action='store_true', help='preview only: never send enabled actions or home')
    parser.add_argument('--max-steps', type=int, default=600, help='episode length limit in policy steps')
    parser.add_argument('--port', type=int, default=47820, help='policy_port of gamepad_teleop')
    parser.add_argument('--teleop-config', default=str(ROOT / 'config' / 'teleop.conf'), help='for home_joints')
    parser.add_argument('--max-position-m', type=float, default=0.05, help='clip of the position action')
    parser.add_argument('--max-rotation-rad', type=float, default=0.3, help='clip of the rotation action')
    parser.add_argument('--camera-timeout-s', type=float, default=0.25)
    parser.add_argument('--out', default=str(Path.home() / 'franka_policy_rollouts'))
    parser.add_argument('--record', default='http://localhost:8765', metavar='URL',
                        help='recorder web UI used to record episodes ("" = none)')
    parser.add_argument('--safe-export', default=None, metavar='DIR', help='write failure-detection files here')
    parser.add_argument('--suite-name', default='real_franka_v1')
    parser.add_argument('--robot-id', default='franka_fr3')
    parser.add_argument('--camera-calibration', default='unversioned')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)  # also when started in the background (bash ignores SIGINT there)
    server = Server(args)
    if args.checkpoint:
        server.load(str(Path(args.checkpoint).resolve()))
    http = make_http(server, args.http_port)
    print('Policy server on http://127.0.0.1:%d (Policy panel on the recorder page)%s' % (
        args.http_port, '; NO-MOTION mode' if args.no_motion else ''), flush=True)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print('Stopping the policy server ...', flush=True)
    finally:
        server.stop_requested = True
        if server.worker is not None:
            server.worker.join(timeout=3.0)
        if not args.no_motion:
            server.link.hold()


if __name__ == '__main__':
    main()
