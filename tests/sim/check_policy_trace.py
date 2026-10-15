"""Checks a simulated policy session (run_policy_session.sh, script in fake_policy.py)."""
import csv
import math
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


# The sender and the control loop start within ~0.2 s of each other.
dx, dy, dz = (moved(1.3, 4.4, a) for a in 'xyz')
print('one action dx = +3 cm: robot moved x %+.1f  y %+.1f  z %+.1f mm' % (dx, dy, dz))
check(abs(dx - 30) < 2 and abs(dy) < 2 and abs(dz) < 2, 'target = pose at arrival + action, reached and held')
still = max(abs(moved(4.6, t, a)) for t in (4.9, 5.2, 5.5) for a in 'xyz')
print('enabled = 0: robot moved up to %.2f mm' % still)
check(still < 0.5, 'actions with enabled = 0 do not move the arm')
dyaw = math.degrees(at(8.4)['yaw'] - at(5.5)['yaw'])
print('one action dyaw = +0.2 rad: tool heading changed %+.1f deg' % dyaw)
check(abs(dyaw - math.degrees(0.2)) < 1.0, 'Euler z of the action turns the gripper about base z')
rise = moved(8.5, 10.0, 'z')
print('10 Hz stream of dz = +1 cm for 1.5 s: robot rose %.1f mm' % rise)
check(rise > 20, 'a policy-rate stream keeps the arm moving')
after = max(abs(moved(10.6, t, 'z')) for t in (10.8, 11.0))
print('stream stopped at 10.0 s: robot moved %.2f mm after 10.6 s' % after)
check(after < 0.5, 'the arm stops when actions stop (policy_timeout_ms)')
home = [0.036286, -1.367388, -0.180505, -2.794799, -0.043301, 2.076045, 0.407056]
error = max(abs(rows[-1]['j%d' % (i + 1)] - home[i]) for i in range(7))
print('home request at 11 s: final joint error %.4f rad' % error)
check(error < 2e-3, 'home request moves the arm to home_joints')
sys.exit(1 if failures else 0)
