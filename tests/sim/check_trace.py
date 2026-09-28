"""Checks the simulated session trace against what each scripted phase should do.

Trace time 0 is when the control loop started (just after RB was first held),
so phases are shifted by the script's rb_down time. Windows skip the first
0.4 s of each phase to let the S-curve ramp.
"""
import csv
import sys

rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(sys.argv[1]))]
phases = {}
for line in open(sys.argv[2]):
    t, label = line.split()
    phases[label] = float(t)
offset = phases["rb_down"]
failures = []


def window(start, end):
    a, b = phases[start] - offset, phases[end] - offset
    return [r for r in rows if a <= r["t"] <= b]


def check(ok, text):
    print(("PASS " if ok else "FAIL ") + text)
    if not ok:
        failures.append(text)


def speed(rs, key, skip=0.4):
    rs = [r for r in rs if r["t"] >= rs[0]["t"] + skip]
    if len(rs) < 2:
        return 0.0
    # steady-state speed over the first second after the ramp
    first = rs[0]
    later = next((r for r in rs if r["t"] >= first["t"] + 1.0), rs[-1])
    return (later[key] - first[key]) / (later["t"] - first["t"])


down = window("down_start", "down_end")
z_min = min(r["z"] for r in rows)
print(f"descend: start z {down[0]['z']:.3f}, end z {down[-1]['z']:.3f}, lowest z {z_min:.3f}, "
      f"steady speed {speed(down, 'z'):+.3f} m/s")
check(speed(down, "z") < -0.02, "keeps descending while the stick is held (detours around the elbow limit)")
check(0.0 <= z_min and down[-1]["z"] <= 0.04, "reaches the floor fence (z 0.00 + 2 cm margin) and does not cross it")

up = window("up_start", "up_end")
print(f"climb: z {up[0]['z']:.3f} -> {up[-1]['z']:.3f}")
check(up[-1]["z"] > up[0]["z"] + 0.05, "climbs out of the low corner when pushed up")

base = window("toward_base_start", "toward_base_end")
x_min = min(r["x"] for r in rows)
print(f"toward base: start x {base[0]['x']:.3f}, end x {base[-1]['x']:.3f}, lowest x {x_min:.3f}, "
      f"steady speed {speed(base, 'x'):+.3f} m/s")
check(x_min >= 0.16, "does not cross the near fence (x 0.16)")
check(base[-1]["x"] <= 0.20, "gets to within 4 cm of the near fence (was stalling before)")

paused = window("paused_push_start", "paused_push_end")
drift = max(abs(r[k] - paused[0][k]) for r in paused for k in ("x", "y", "z"))
print(f"paused with stick pushed: TCP moved {drift * 1000:.2f} mm")
check(drift < 0.001, "arm holds still while RB is released")

side = window("sideways_start", "sideways_end")
print(f"sideways at 50%: steady speed {speed(side, 'y'):+.3f} m/s, y {side[0]['y']:.3f} -> {side[-1]['y']:.3f}")
check(abs(speed(side, "y") - 0.075) < 0.01, "D-pad halved speed: +Y at 0.075 m/s")

rot = window("rotate_start", "rotate_end")
moved = max(abs(r[k] - rot[0][k]) for r in rot for k in ("x", "y", "z"))
import math
turned = abs(math.remainder(rot[-1]["yaw"] - rot[0]["yaw"], 2 * math.pi))
print(f"rotate (L1 + right stick): TCP moved {moved * 1000:.1f} mm, gripper heading turned {turned:.2f} rad")
check(moved < 0.01 and turned > 0.2, "rotates about the tool point without translating")

# Never against the push: over any 100 ms while a stick is held, the gripper must
# not move backwards along the pushed axis by more than 0.5 mm.
for name, (start, end), key, sign in [("descend", ("down_start", "down_end"), "z", -1),
                                        ("climb", ("up_start", "up_end"), "z", +1),
                                        ("toward base", ("toward_base_start", "toward_base_end"), "x", -1),
                                        ("sideways", ("sideways_start", "sideways_end"), "y", +1)]:
    rs = window(start, end)
    worst_back = max((sign * (rs[i][key] - rs[i + 10][key]) for i in range(len(rs) - 10)), default=0.0)
    check(worst_back <= 0.0005, f"{name}: never moves against the push (worst {worst_back * 1000:.2f} mm back per 100 ms)")

box = {"x": (0.16, 0.70), "y": (-0.30, 0.35), "z": (0.0, 0.71)}
worst = min(min(r[k] - lo, hi - r[k]) for r in rows for k, (lo, hi) in box.items())
print(f"workspace box: closest the gripper came to (or past, if negative) a wall: {worst * 1000:.1f} mm")
check(worst >= 0.0, "gripper never leaves the workspace box, including any IK detour")

peak = max(r["max_abs_dq"] for r in rows)
closest = min(r["min_limit_dist"] for r in rows)
print(f"whole session: peak joint speed {peak:.3f} rad/s, closest to a joint limit {closest:.3f} rad")
check(peak <= 1.0 + 0.01, "joint speeds stay within the 1 rad/s teleop cap")
peak_acc = max(r["max_abs_ddq"] for r in rows)
peak_jerk = max(r["max_abs_jerk"] for r in rows)
print(f"whole session: peak joint acceleration {peak_acc:.2f} rad/s^2, peak jerk {peak_jerk:.0f} rad/s^3 (robot max 10 / 5000)")
check(peak_acc <= 5.0 + 1e-6 and peak_jerk <= 2500.0, "joint acceleration and jerk stay at or below half the robot's limits")

print("\nSIM SESSION: " + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED"))
sys.exit(1 if failures else 0)
