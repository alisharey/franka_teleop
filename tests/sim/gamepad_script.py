"""Scripted gamepad for the simulation: writes Linux js_event records to a FIFO.

Controller layout matches the xbox_android_standard profile in gamepad_teleop:
axes 0/1 left stick X/Y, 2/3 right stick X/Y, 4 L2, 7 D-pad vertical;
buttons 6 = L1, 7 = RB (deadman). Stick up / left is negative.
Prints the start time of each phase so check_trace.py can line them up.
"""
import os
import struct
import sys
import time

JS_EVENT_BUTTON, JS_EVENT_AXIS, JS_EVENT_INIT = 0x01, 0x02, 0x80
FULL = 32767

fifo = os.open(sys.argv[1], os.O_RDWR)  # RDWR: never blocks and never sees "no writer"
t0 = time.monotonic()


def send(kind, number, value, init=False):
    ms = int((time.monotonic() - t0) * 1000) & 0xFFFFFFFF
    os.write(fifo, struct.pack("<IhBB", ms, value, kind | (JS_EVENT_INIT if init else 0), number))


def axis(n, v):
    send(JS_EVENT_AXIS, n, v)


def button(n, pressed):
    send(JS_EVENT_BUTTON, n, 1 if pressed else 0)


def at(t, label):
    time.sleep(max(0.0, t - (time.monotonic() - t0)))
    print(f"{time.monotonic() - t0:.3f} {label}", flush=True)


at(0.1, "init")
for n in range(8):
    send(JS_EVENT_AXIS, n, 0, init=True)
for n in range(15):
    send(JS_EVENT_BUTTON, n, 0, init=True)
at(0.5, "rb_down")
button(7, True)
at(1.5, "down_start")          # 12 s: left stick down -> base -Z at half speed (z_down_speed_scale)
axis(1, FULL)
at(13.5, "down_end")
axis(1, 0)
at(14.0, "up_start")           # left stick up -> base +Z: climb back out of the low corner
axis(1, -FULL)
at(17.0, "up_end")
axis(1, 0)
at(17.5, "toward_base_start")  # right stick down -> base -X, toward the base (push up = +X)
axis(3, FULL)
at(21.0, "toward_base_end")
axis(3, 0)
at(21.5, "pause_start")        # RB released; stick pushed: arm must not move
button(7, False)
at(21.6, "paused_push_start")
axis(0, FULL)
at(22.6, "paused_push_end")
axis(0, 0)
at(23.0, "resume")
button(7, True)
axis(7, FULL)                  # D-pad down: 100% -> 50%
at(23.1, "dpad_release")
axis(7, 0)
at(23.5, "sideways_start")     # left stick right -> base +Y at 50%
axis(0, FULL)
at(26.5, "sideways_end")
axis(0, 0)
at(27.0, "rotate_start")       # L1 held + right stick X -> yaw
button(6, True)
axis(2, FULL)
at(29.0, "rotate_end")
axis(2, 0)
button(6, False)
at(29.5, "idle")
time.sleep(30)
