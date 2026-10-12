"""Checks a simulated Quest session (run_quest_session.sh): the arm follows
TeleMoMa's mapping of the scripted controller motion (fake_adb.py)."""
import csv
import sys

rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(sys.argv[1]))]


def at(t):
    return min(rows, key=lambda r: abs(r['t'] - t))


def moved(t0, t1, axis):
    return (at(t1)[axis] - at(t0)[axis]) * 1000


failures = 0


def check(ok, text):
    global failures
    print('%s %s' % ('PASS' if ok else 'FAIL', text))
    failures += 0 if ok else 1


# The reader starts ~0.2 s before the control loop; allow a little slack.
dx = moved(1.5, 5.0, 'x'); dy = moved(1.5, 5.0, 'y'); dz = moved(1.5, 5.0, 'z')
print('grip + tilted controller 8 cm straight up: robot moved x %+.1f  y %+.1f  z %+.1f mm' % (dx, dy, dz))
check(abs(dz - 80) < 4 and abs(dx) < 4 and abs(dy) < 4, 'controller up -> robot up, 8 cm')
still = max(abs(moved(5.3, t, a)) for t in (5.6, 6.0, 6.3, 6.5) for a in 'xyz')
print('grip released, controller moving: robot moved up to %.2f mm' % still)
check(still < 1.0, 'arm holds while the grip is released')
dx = moved(6.4, 10.8, 'x'); dy = moved(6.4, 10.8, 'y'); dz = moved(6.4, 10.8, 'z')
print('grip + controller 5 cm back along its heading: robot moved x %+.1f  y %+.1f  z %+.1f mm' % (dx, dy, dz))
check(abs(dx - 50) < 4 and abs(dy) < 4 and abs(dz) < 4,
      'controller back (toward an operator facing the robot) -> robot +x, 5 cm (relative to the new grip press)')
dyaw = (at(14.5)['yaw'] - at(10.8)['yaw']) * 180 / 3.141592653589793
print('grip + controller turned 20 deg about vertical: tool heading changed %+.1f deg' % dyaw)
check(abs(dyaw - 20) < 2, 'turning the controller turns the gripper about base +z (20 deg)')
sys.exit(1 if failures else 0)
