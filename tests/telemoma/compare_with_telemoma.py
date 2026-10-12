"""Checks src/telemoma_policy.h against TeleMoMa's own OculusPolicy (third_party/telemoma).

Generates random controller readings (grip presses/releases, trigger pulls) and
robot poses, runs both implementations, and compares every action.
Usage: python3 -s compare_with_telemoma.py <telemoma_policy_test binary>
"""
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'third_party' / 'telemoma'))



class FakeReader:
    def __init__(self, run=False):
        self.events = iter(())

    def run(self):
        pass

    def get_transformations_and_buttons(self):
        return next(self.events)


sys.modules['oculus_reader'] = types.SimpleNamespace(OculusReader=FakeReader)
from telemoma.human_interface import oculus as telemoma_oculus  # noqa: E402
from telemoma.utils import transformations as telemoma_transformations  # noqa: E402

# TeleMoMa calls as_quat(canonical=True), which needs scipy >= 1.11. With an
# older scipy, run its helpers without that flag: it only picks the quaternion's
# sign, which changes no rotation and no Euler angle.
if 'canonical' not in (Rotation.as_quat.__doc__ or ''):
    telemoma_oculus.rmat_to_quat = lambda m: Rotation.from_matrix(m).as_quat()
    telemoma_oculus.quat_diff = lambda t, s: (Rotation.from_quat(t) * Rotation.from_quat(s).inv()).as_quat()

telemoma_oculus.time = types.SimpleNamespace(sleep=lambda *_: None, time=lambda: 0.0)


def random_pose(rng, base=None, step=0.05):
    pose = np.eye(4)
    rotation = Rotation.from_rotvec(rng.normal(size=3) * (0.3 if base is None else step))
    pose[:3, :3] = rotation.as_matrix() if base is None else rotation.as_matrix() @ base[:3, :3]
    pose[:3, 3] = (rng.normal(size=3) * 0.3) if base is None else base[:3, 3] + rng.normal(size=3) * step
    return pose


def make_events(seed, count=400):
    rng = np.random.default_rng(seed)
    controller = random_pose(rng)
    robot = random_pose(rng)
    robot[:3, 3] = [0.4, 0.0, 0.3]
    grip, trigger = False, 0.0
    events = []
    for _ in range(count):
        if rng.random() < 0.08:
            grip = not grip
        trigger = 1.0 if rng.random() < 0.1 else (0.0 if rng.random() < 0.5 else trigger)
        controller = random_pose(rng, controller)
        events.append(('R', controller.copy(), grip, False, trigger))
        robot = random_pose(rng, robot, step=0.02)
        # Gripper position held per sequence: TeleMoMa re-evaluates a pending
        # toggle on every action call until the next reading, which the port
        # does once (see src/telemoma_policy.h); with a steady gripper both agree.
        events.append(('A', robot.copy(), float(seed % 2)))
    return events


def run_telemoma(events):
    policy = telemoma_oculus.OculusPolicy()
    actions = []

    def stream():
        for event in events:
            if event[0] == 'R':
                _, pose, grip, click, trigger = event
                yield {'r': pose}, {'RG': grip, 'RJ': click, 'rightTrig': (trigger,), 'A': False, 'B': False}
            else:
                _, robot, gripper = event
                if policy._state['right']['poses'] is not None:
                    quat = Rotation.from_matrix(robot[:3, :3]).as_quat()
                    obs = {'cartesian_position': np.concatenate([robot[:3, 3], quat]), 'gripper_position': gripper}
                    actions.append(policy._calculate_action(obs, 'right'))
        # end of events: leave TeleMoMa's endless read loop
        raise StopIteration

    reader_events = stream()
    policy.oculus_reader.get_transformations_and_buttons = lambda: next(reader_events)
    try:
        policy._update_internal_state()
    except (StopIteration, RuntimeError):
        pass
    return np.array(actions)


def run_port(binary, events):
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as handle:
        for event in events:
            if event[0] == 'R':
                _, pose, grip, click, trigger = event
                handle.write('R %s %d %d %.12f\n' % (' '.join('%.12f' % v for v in pose.ravel()), grip, click, trigger))
            else:
                _, robot, gripper = event
                handle.write('A %s %.1f\n' % (' '.join('%.12f' % v for v in robot.ravel()), gripper))
        path = handle.name
    output = subprocess.run([binary, path], capture_output=True, text=True, check=True).stdout
    os.unlink(path)
    return np.array([[float(v) for v in line.split()] for line in output.splitlines()])


def main():
    binary = sys.argv[1]
    worst = 0.0
    total = 0
    for seed in range(20):
        events = make_events(seed)
        expected, actual = run_telemoma(events), run_port(binary, events)
        if expected.shape != actual.shape:
            print('seed %d: shape %s vs %s' % (seed, expected.shape, actual.shape))
            return 1
        worst = max(worst, float(np.abs(expected - actual).max()))
        total += len(expected)
    print('compared %d actions over 20 random sequences; largest difference %.2e' % (total, worst))
    return 0 if worst < 1e-6 else 1


if __name__ == '__main__':
    sys.exit(main())
