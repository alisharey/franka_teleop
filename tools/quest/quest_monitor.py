"""Meta Quest connection check and live controller readout.

Checks the USB/ADB connection, keeps the headset awake when it is not worn
(proximity-sensor override), installs and starts the oculus_reader app, then
shows the controller pose, buttons and triggers live in the terminal.

Readings come from the oculus_reader app (third_party/oculus_reader): it logs
each controller's pose relative to the headset, plus buttons, to the headset's
logcat, which this script reads over ADB. Only the standard library and numpy.

    python3 tools/quest/quest_monitor.py                         # USB; live readout, Ctrl-C quits
    python3 tools/quest/quest_monitor.py --connection 10.0.0.42  # Wi-Fi (see quest_wifi.sh)
    python3 tools/quest/quest_monitor.py --restore               # give the proximity sensor back
"""

import argparse
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

PACKAGE = 'com.rail.oculus.teleop'
ACTIVITY = PACKAGE + '/com.rail.oculus.teleop.MainActivity'
TAG = 'wE9ryARX'
APK = Path(__file__).resolve().parents[2] / 'third_party' / 'oculus_reader' / 'oculus_reader' / 'APK' / 'teleop-debug.apk'


SERIAL = []  # ['-s', serial] once the device is chosen


def adb(*args, timeout=10):
    result = subprocess.run(['adb', *SERIAL, *args], capture_output=True, text=True, timeout=timeout)
    return result.stdout.strip()


def parse(line):
    """oculus_reader line -> ({'r'|'l': 4x4}, buttons). Same format as oculus_reader.reader."""
    try:
        transforms_text, buttons_text = line.split('&')
    except ValueError:
        return None, None
    transforms = {}
    for pair in transforms_text.split('|'):
        side, _, values = pair.partition(':')
        numbers = [float(v) for v in values.split() if v]
        if side and len(numbers) == 16:
            transforms[side] = np.array(numbers).reshape(4, 4)
    buttons = {}
    for item in buttons_text.split(','):
        parts = item.split(' ')
        if len(parts) == 1 and parts[0]:
            buttons[parts[0]] = True
        elif len(parts) > 1:
            buttons[parts[0]] = tuple(float(v) for v in parts[1:] if v)
    return transforms, buttons


def roll_pitch_yaw(rotation):
    pitch = math.asin(-max(-1.0, min(1.0, rotation[2, 0])))
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return [math.degrees(v) for v in (roll, pitch, yaw)]


class LogReader(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.transforms, self.buttons = {}, {}
        self.last_time = None
        self.times = []
        self.process = subprocess.Popen(['adb', *SERIAL, 'logcat', '-T', '0', '-s', TAG + ':I'], stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True, errors='replace')

    def run(self):
        for line in self.process.stdout:
            if TAG not in line:
                continue
            transforms, buttons = parse(line.split(TAG + ': ', 1)[-1].strip())
            if transforms is None:
                continue
            now = time.time()
            with self.lock:
                self.transforms, self.buttons, self.last_time = transforms, buttons, now
                self.times = [t for t in self.times if now - t < 1.0] + [now]

    def snapshot(self):
        with self.lock:
            return dict(self.transforms), dict(self.buttons), self.last_time, len(self.times)


def keep_awake():
    adb('shell', 'input', 'keyevent', 'KEYCODE_WAKEUP')
    adb('shell', 'am', 'broadcast', '-a', 'com.oculus.vrpowermanager.prox_close')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--connection', default='usb',
                        help="'usb' (default) or the headset's IP for Wi-Fi ADB (run quest_wifi.sh once first)")
    parser.add_argument('--restore', action='store_true',
                        help='turn the proximity sensor back on (headset sleeps when taken off) and exit')
    args = parser.parse_args()

    if args.connection != 'usb':
        address = args.connection if ':' in args.connection else args.connection + ':5555'
        reply = adb('connect', address)
        if 'connected' not in reply:
            sys.exit('Cannot reach the Quest at %s (%s). Run tools/quest/quest_wifi.sh with the cable in first.'
                     % (address, reply))
        devices = [[address, 'device']]
    else:
        devices = [l.split() for l in adb('devices').splitlines()[1:] if l.strip() and ':' not in l.split()[0]]
        if not devices:
            sys.exit('No Quest found over USB. Check the cable and that Developer Mode is on.')
        if devices[0][1] != 'device':
            sys.exit('Quest is %s: put the headset on and accept "Allow USB debugging".' % devices[0][1])
    SERIAL[:] = ['-s', devices[0][0]]
    if args.restore:
        adb('shell', 'am', 'broadcast', '-a', 'com.oculus.vrpowermanager.automation_disable')
        print('Proximity sensor restored: the headset sleeps again when taken off.')
        return

    model = adb('shell', 'getprop', 'ro.product.model')
    battery = adb('shell', 'dumpsys', 'battery')
    level = next((l.split(':')[1].strip() for l in battery.splitlines() if 'level:' in l), '?')
    print('Connected: %s (%s), battery %s%%' % (model, devices[0][0], level))
    if PACKAGE not in adb('shell', 'pm', 'list', 'packages', PACKAGE):
        print('Installing the oculus_reader app...')
        print(adb('install', '-t', '-r', str(APK), timeout=120))
    keep_awake()
    reader = LogReader()
    reader.start()
    if not adb('shell', 'pidof', PACKAGE):  # launching it again while running makes it abort
        adb('shell', 'am', 'start', '-n', ACTIVITY)
    print('Proximity sensor overridden: the headset stays awake when not worn (--restore undoes it).')
    print('Waiting for controller data... (Ctrl-C quits)\n')

    last_wake = time.time()
    last_start = time.time()
    restarts = 0
    try:
        while True:
            time.sleep(0.1)
            transforms, buttons, last_time, rate = reader.snapshot()
            now = time.time()
            if now - last_wake > 20:  # the override can lapse; renew it
                keep_awake()
                last_wake = now
            # The Quest 3 build of the app aborts when its VR session is interrupted
            # (menu button, closing it, headset idle); start it again.
            stale = last_time is None or now - last_time > 2.0
            if stale and now - last_start > 5.0:
                if adb('shell', 'pidof', PACKAGE):  # alive but paused: stop it first (relaunching it aborts)
                    adb('shell', 'am', 'force-stop', PACKAGE)
                adb('shell', 'am', 'start', '-n', ACTIVITY)
                last_start = now
                restarts += 1
            lines = []
            if last_time is None:
                pid = adb('shell', 'pidof', PACKAGE)
                lines.append('No data yet. App %s. If it does not start, put the headset on and dismiss '
                             'any dialog.' % ('running' if pid else 'NOT running'))
            else:
                age = now - last_time
                if not transforms and not buttons.get('R'):
                    lines.append('Readings arrive but no controller is visible: press a button on both controllers.')
                lines.append('Data: %3d Hz, last %.2f s ago%s   controllers seen: %s   app restarts: %d' % (
                    rate, age, '   <-- STALE (app crashed or paused?)' if age > 0.5 else '',
                    ', '.join(sorted(transforms)) or 'none', restarts))
                for side in sorted(transforms):  # 'h' = headset (fixed app), 'l'/'r' = controllers
                    pose = transforms[side]
                    x, y, z = pose[:3, 3]
                    r, p, w = roll_pitch_yaw(pose[:3, :3])
                    tracked = not np.allclose(pose, np.eye(4))
                    lines.append('  %s: xyz [%7.3f %7.3f %7.3f] m   rpy [%7.1f %7.1f %7.1f] deg  %s' % (
                        side, x, y, z, r, p, w, '(rel. to headset)' if tracked else
                        '<-- NOT TRACKED (identity): headset cannot see the controller'))
                if buttons.get('R') and not any(side == 'r' and not np.allclose(m, np.eye(4))
                                                for side, m in transforms.items()):
                    lines.append('  right controller buttons arrive but no right pose: the Quest 3 reader app only '
                                 'sends it while the LEFT controller is on too')
                pressed = [k for k, v in buttons.items() if v is True]
                analog = {k: v for k, v in buttons.items() if isinstance(v, tuple)}
                lines.append('  pressed: %s' % (' '.join(pressed) or '-'))
                lines.append('  ' + '  '.join('%s %s' % (k, ' '.join('%.2f' % x for x in v))
                                              for k, v in sorted(analog.items())))
            sys.stdout.write('\033[J' + '\n'.join(lines) + '\033[%dA\r' % len(lines))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print('\n' * 6)
    finally:
        reader.process.terminate()


if __name__ == '__main__':
    main()
