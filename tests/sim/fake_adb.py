#!/usr/bin/env python3
"""Stand-in for `adb` with a scripted Meta Quest (QUEST_ADB=tests/sim/fake_adb.py).

`logcat` streams oculus_reader lines at 72 Hz: both controllers present (the
Quest 3 app needs the left one on to send the right pose). The right controller
is held like a real one, tilted (pitched 40 deg down, rolled 10 deg, turned 30
deg), at p0 relative to a level headset (OpenXR axes: x right, y up, -z
forward; defaults quest_axis_reorder 3, 1, 2, 4 and quest_level_frame true):
  2-4 s    grip held, moves 8 cm straight up              -> robot +z 8 cm
           (up, not down: straight down from the sim's home pose needs the elbow
           past its limit, see tests/sim/README.md)
  4-5 s    grip held, still
  5-6.5 s  grip released, moves back down                 -> robot must not move
  6.5-9 s  grip held, moves 5 cm backwards along its       -> robot +x 5 cm
           horizontal heading (trigger pulled at 7.0-7.3 s: gripper toggle)
  9-11 s   grip held, still
  11-13 s  grip held, turns 20 deg about vertical          -> robot turns 20 deg about base +z
  13- s    grip held, still
"""
import math
import signal
import sys
import time

import numpy as np

args = sys.argv[1:]
if args[:1] == ['-s']:
    args = args[2:]
command = args[0] if args else ''
if command == 'devices':
    print('List of devices attached\nFAKEQUEST\tdevice')
elif command == 'connect':
    print('connected to %s' % args[1])
elif command == 'shell':
    if args[1:3] == ['pm', 'list']:
        print('package:com.rail.oculus.teleop')
    elif args[1] == 'pidof':
        print('4242')
elif command == 'logcat':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    def rot(axis, angle):
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)

    r0 = rot([0, 1, 0], math.radians(30)) @ rot([1, 0, 0], math.radians(-40)) @ rot([0, 0, 1], math.radians(10))
    p0 = np.array([0.10, -0.25, -0.40])
    # Level heading frame of the controller: forward = its -z flattened.
    forward = -r0[:, 2]
    forward[1] = 0.0
    forward /= np.linalg.norm(forward)
    level = np.column_stack([np.cross([0.0, 1.0, 0.0], -forward), [0.0, 1.0, 0.0], -forward])

    def smooth(t, start, end):
        s = min(max((t - start) / (end - start), 0.0), 1.0)
        return s * s * (3 - 2 * s)

    start = time.time()
    while True:
        t = time.time() - start
        local = np.array([0.0, 0.08 * smooth(t, 2, 4), 0.0])            # in the controller's own frame
        local += np.array([0.0, -0.08 * smooth(t, 5, 6.5), 0.0])        # back again, grip released
        local += np.array([0.0, 0.0, 0.05 * smooth(t, 6.5, 9)])
        angle = math.radians(20) * smooth(t, 11, 13)
        pose = np.eye(4)
        pose[:3, :3] = rot([0, 1, 0], angle) @ r0      # turn about vertical
        pose[:3, 3] = p0 + level @ local                # up / back along the heading
        grip = 2 <= t < 5 or t >= 6.5
        trigger = 1.0 if 7.0 <= t < 7.3 else 0.0
        values = ' '.join('%.6f' % v for v in pose.ravel())
        identity = ' '.join('%g' % v for v in np.eye(4).ravel())
        buttons = 'R,' + ('RG,' if grip else '') + ('RTr,' if trigger > 0.25 else '') + \
            'rightJS 0.000000 0.000000,rightTrig %.6f,rightGrip %.6f,L,leftJS 0 0,leftTrig 0,leftGrip 0' % (
                trigger, 1.0 if grip else 0.0)
        sys.stdout.write('10-12 17:00:00.000  4242  4243 I wE9ryARX: l:%s|r:%s &%s\n' % (identity, values, buttons))
        sys.stdout.flush()
        time.sleep(1 / 72)
