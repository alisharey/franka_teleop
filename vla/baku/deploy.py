#!/usr/bin/env python3
"""Runs a Franka BAKU snapshot on the robot.

    camera topics (record.sh --policy) -> BAKU -> UDP -> gamepad_teleop --input policy

The teleop executes each action exactly as it executed the demonstrations
(target = pose + action, tracked with the Quest's gains, speed limits, IK and
workspace fence; src/policy_input.h). Run it with vla/baku/deploy.sh, which
sets up ROS and the Python environment. The model code is BAKU's (vla/BAKU).

Keys (type, then Enter):
    Enter   start a rollout / stop the running rollout (then label it)
    h       move to the saved home pose (between rollouts)
    t N     switch to trained task N (see the list printed at start)
    q       quit

Per rollout it writes <out>/<id>/rollout.npz (actions sent, VQ-BeT entropies,
action-token features, the policy's input images) and, with --safe-export, the
failure-detection artifacts task<T>--ep<E>--succ<0|1>.{csv,pkl,meta.json}
(interpretability/monitor.py's format). With --record it also records each
rollout with the recorder (cameras + robot bag) and stores the outcome there.
"""

import argparse
import json
import math
import pickle
import queue
import socket
import struct
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# BAKU with the Franka add-ons: the vla/BAKU submodule (branch franka-teleop of alisharey/BAKU)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'BAKU' / 'baku'))
from franka.policy import FrankaPolicy  # noqa: E402

PACKET = struct.Struct('<4sIQQ7d')  # src/policy_input.h
ACTION_NAMES = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'dgripper']


class Cameras:
    """Latest color frame of each camera from the recording launch's topics."""

    def __init__(self, cameras):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        rclpy.init()
        self.node = Node('baku_policy')
        self.lock = threading.Lock()
        self.frames = {}
        for camera in cameras:
            topic = '/cameras/%s/color/image_raw' % camera
            self.node.create_subscription(Image, topic, lambda msg, c=camera: self._on_image(c, msg),
                                          qos_profile_sensor_data)
        self.thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.thread.start()

    def _on_image(self, camera, msg):
        if msg.encoding not in ('rgb8', 'bgr8'):
            return
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
        image = image.reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            image = image[:, :, ::-1]
        with self.lock:
            self.frames[camera] = (time.monotonic(), np.ascontiguousarray(image))

    def latest(self, max_age_s):
        with self.lock:
            frames = dict(self.frames)
        now = time.monotonic()
        stale = [c for c, (t, _) in frames.items() if now - t > max_age_s]
        return {c: image for c, (_, image) in frames.items()}, stale


class Recorder:
    """The recording web UI's API (franka_teleop_recorder/web.py)."""

    def __init__(self, url):
        self.url = url.rstrip('/')

    def _post(self, path, body):
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read())

    def start(self, task, name):
        self._post('/api/settings', {'task': task, 'next_name': name})
        return self._post('/api/record', {'action': 'start'})

    def stop(self):
        return self._post('/api/record', {'action': 'stop'})

    def label(self, episode_id, success, notes):
        return self._post('/api/episodes/%s' % episode_id, {'success': success, 'notes': notes})


class Link:
    def __init__(self, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address = ('127.0.0.1', port)
        self.sequence = 0
        self.home_requests = 0

    def send(self, action, enabled):
        self.sequence += 1
        self.socket.sendto(PACKET.pack(b'FPA1', 1 if enabled else 0, self.sequence, self.home_requests,
                                       *[float(v) for v in action]), self.address)


def keyboard(commands):
    for line in sys.stdin:
        commands.put(line.strip())
    commands.put('q')


def next_episode_index(folder, task_id):
    used = [int(p.name.split('--ep')[1].split('--')[0]) for p in folder.glob('task%d--ep*--succ*.csv' % task_id)]
    return max(used) + 1 if used else 0


def write_safe_export(folder, log, policy, args, task_id, success, intervention, notes):
    folder.mkdir(parents=True, exist_ok=True)
    episode = next_episode_index(folder, task_id)
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
            'robot_id': args.robot_id, 'control_hz': args.rate, 'camera_calibration_version': args.camera_calibration,
            'action_filter': 'deploy clip |pos| <= %g m, |rot| <= %g rad; gamepad_teleop input=policy: target '
                             'tracking (quest gains), speed caps, workspace fence, IK' % (args.max_position_m,
                                                                                          args.max_rotation_rad),
            'executed_action': 'the CSV holds the actions sent (after the clip); the robot tracked them through '
                               'the teleop controller (recorded in the bag with --record)',
            'intervention': bool(intervention), 'notes': notes, 'task_description': policy.task_text}
    Path(str(stem) + '.meta.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
    return stem


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('snapshot', help='<run>/snapshot/<step>.pt (with its .hydra folder)')
    parser.add_argument('--task', default='0', help='trained task index or text (default 0)')
    parser.add_argument('--rate', type=float, default=None, help='policy steps per second (default: dataset rate)')
    parser.add_argument('--max-steps', type=int, default=600, help='rollout length limit in policy steps')
    parser.add_argument('--port', type=int, default=47820, help='policy_port of gamepad_teleop')
    parser.add_argument('--max-position-m', type=float, default=0.05, help='clip of the position action')
    parser.add_argument('--max-rotation-rad', type=float, default=0.3, help='clip of the rotation action')
    parser.add_argument('--camera-timeout-s', type=float, default=0.25)
    parser.add_argument('--out', default=str(Path.home() / 'franka_policy_rollouts'))
    parser.add_argument('--record', default=None, metavar='URL',
                        help='recorder web UI to record each rollout (e.g. http://localhost:8765)')
    parser.add_argument('--safe-export', default=None, metavar='DIR', help='write failure-detection artifacts here')
    parser.add_argument('--suite-name', default='real_franka_v1')
    parser.add_argument('--robot-id', default='franka_fr3')
    parser.add_argument('--camera-calibration', default='unversioned')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    task = int(args.task) if args.task.isdigit() else args.task
    policy = FrankaPolicy(args.snapshot, device=args.device, max_steps=args.max_steps, task=task)
    args.rate = args.rate or policy.rate_hz
    print('Snapshot: %s\nTrained tasks:' % policy.snapshot)
    for index, text in enumerate(policy.info['task_texts']):
        print('  %d: %s' % (index, text))
    print('Task: %d (%s); %g Hz; images %s' % (policy.task_index, policy.task_text, args.rate,
                                              {k: '%s %s' % (policy.cameras[k], policy.info['dataset']['image_specs'][k][1])
                                               for k in policy.pixel_keys}))
    cameras = Cameras(sorted(set(policy.cameras.values())))
    link = Link(args.port)
    recorder = Recorder(args.record) if args.record else None
    commands = queue.Queue()
    threading.Thread(target=keyboard, args=(commands,), daemon=True).start()
    out = Path(args.out)

    # warm up (CUDA kernels) on a blank frame so the first real step is on time
    blank = {k: np.zeros((480, 640, 3), np.uint8) for k in policy.pixel_keys}
    policy.act(blank)
    policy.reset()

    print('\nEnter = start, h = home, t N = task, q = quit. Keep the robot\'s stop button in reach.')
    running = False
    log = None
    period = 1.0 / args.rate
    next_tick = time.monotonic()
    late = 0
    episode_record = None
    while True:
        try:
            command = commands.get_nowait()
        except queue.Empty:
            command = None
        if command == 'q':
            if running:
                link.send(np.zeros(7), False)
            break
        if command is not None and not running:
            if command == '':
                frames, stale = cameras.latest(args.camera_timeout_s)
                missing = [c for c in policy.cameras.values() if c not in frames or c in stale]
                if missing:
                    print('cameras %s have no recent frames; is record.sh --policy running?' % missing)
                    continue
                policy.reset()
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log = {'id': '%s_task%d' % (stamp, policy.task_index), 't': [], 'action': [], 'raw_action': [],
                       'entropy_avg': [], 'token_entropy': [], 'token_prob': [], 'feature': [], 'images': [],
                       'compute_ms': []}
                if recorder:
                    try:
                        reply = recorder.start(policy.task_text, 'policy_%s' % stamp)
                        episode_record = reply.get('id') if reply.get('ok') else None
                        if not episode_record:
                            print('recorder did not start: %s' % reply)
                    except OSError as error:
                        print('recorder unreachable (%s); continuing without it' % error)
                        episode_record = None
                running = True
                late = 0
                start = time.monotonic()
                next_tick = start
                print('rollout %s started (Enter stops)' % log['id'])
            elif command == 'h':
                link.home_requests += 1
                for _ in range(3):
                    link.send(np.zeros(7), False)
                print('home requested (the teleop prints when it arrives)')
            elif command.startswith('t '):
                try:
                    value = command[2:].strip()
                    policy.set_task(int(value) if value.isdigit() else value)
                    print('task %d: %s' % (policy.task_index, policy.task_text))
                except (ValueError, IndexError) as error:
                    print(error)
            else:
                print('Enter = start, h = home, t N = task, q = quit')
        elif command is not None and running:
            running = False  # any input stops the rollout
            link.send(np.zeros(7), False)
            finish(log, policy, args, recorder, episode_record, out, commands, late)
            print('\nEnter = start, h = home, t N = task, q = quit')
            continue

        if not running:
            time.sleep(0.02)
            continue

        frames, stale = cameras.latest(args.camera_timeout_s)
        if stale or any(c not in frames for c in policy.cameras.values()):
            link.send(np.zeros(7), False)
            print('camera frames stale (%s): holding' % stale)
        else:
            began = time.monotonic()
            raw = policy.act({k: frames[policy.cameras[k]] for k in policy.pixel_keys})
            compute_ms = (time.monotonic() - began) * 1000
            action = raw.copy()
            norm = np.linalg.norm(action[:3])
            if norm > args.max_position_m:
                action[:3] *= args.max_position_m / norm
            action[3:6] = np.clip(action[3:6], -args.max_rotation_rad, args.max_rotation_rad)
            link.send(action, True)
            entropy, prob = policy.token_columns()
            log['t'].append(began - start)
            log['action'].append(action)
            log['raw_action'].append(raw)
            log['entropy_avg'].append(policy.last.get('entropy_avg', math.nan))
            log['token_entropy'].append(entropy)
            log['token_prob'].append(prob)
            log['feature'].append(policy.last.get('feature', np.full(256, np.nan, np.float32)))
            log['images'].append(np.stack([policy.preprocess({k: frames[policy.cameras[k]]})[k]
                                           for k in policy.pixel_keys]))
            log['compute_ms'].append(compute_ms)
            if policy.step_index >= args.max_steps:
                running = False
                link.send(np.zeros(7), False)
                print('max steps reached')
                finish(log, policy, args, recorder, episode_record, out, commands, late)
                print('\nEnter = start, h = home, t N = task, q = quit')
                continue
        next_tick += period
        delay = next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            late += 1
            next_tick = time.monotonic()


def finish(log, policy, args, recorder, episode_record, out, commands, late):
    steps = len(log['action'])
    print('rollout stopped after %d steps (%d late ticks, compute %.1f ms median)' % (
        steps, late, float(np.median(log['compute_ms'])) if steps else 0.0))
    stopped_recording = None
    if recorder and episode_record:
        try:
            stopped_recording = recorder.stop().get('id', episode_record)
        except OSError as error:
            print('recorder stop failed: %s' % error)
    answer = None
    while answer not in ('s', 'f', 'x', 'i'):
        print('Outcome? s = success, f = failure, i = failure with intervention, x = discard: ', end='', flush=True)
        answer = (commands.get() or ' ')[0].lower()
    if answer == 'x' or steps == 0:
        print('discarded (rollout.npz kept)' if steps else 'nothing to save')
    success = answer == 's'
    notes = 'BAKU policy rollout, %s, task %d' % (policy.snapshot, policy.task_index)
    if steps:
        folder = out / log['id']
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'rollout.npz', **{k: np.array(v) for k, v in log.items() if k != 'id'},
                            success=success, discarded=answer == 'x', intervention=answer == 'i')
        (folder / 'rollout.json').write_text(json.dumps({
            'snapshot': policy.snapshot, 'task_index': policy.task_index, 'task': policy.task_text,
            'rate_hz': args.rate, 'steps': steps, 'outcome': answer, 'recorder_episode': stopped_recording,
            'clip': {'position_m': args.max_position_m, 'rotation_rad': args.max_rotation_rad}}, indent=1))
        print('saved %s' % folder)
        if args.safe_export and answer != 'x':
            stem = write_safe_export(Path(args.safe_export), log, policy, args, policy.task_index, success,
                                     answer == 'i', notes)
            print('failure-detection export: %s.{csv,pkl,meta.json}' % stem)
    if recorder and stopped_recording:
        try:
            recorder.label(stopped_recording, None if answer == 'x' else success,
                           notes + ('; operator intervened' if answer == 'i' else ''))
        except OSError as error:
            print('recorder label failed: %s' % error)


if __name__ == '__main__':
    main()
