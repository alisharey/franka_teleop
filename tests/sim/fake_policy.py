#!/usr/bin/env python3
"""Scripted policy for run_policy_session.sh: sends TeleMoMa actions to
gamepad_teleop --input policy over UDP (src/policy_input.h), timed from start.

  1.5- 4.5 s  one action dx = +3 cm, re-sent (same sequence): the arm moves 3 cm and stops
  4.5- 5.5 s  dz = +2 cm with enabled = 0: no motion
  5.5- 8.5 s  one action dyaw = +0.2 rad (Euler z): the gripper turns 11.5 deg about base z
  8.5-10.0 s  a new action every 100 ms, dz = +1 cm (policy-like stream): the arm rises
 10.0-11.0 s  nothing sent: the arm stops (policy_timeout_ms)
 11.0 s       home request (enabled = 0): the arm returns to the start joints
"""
import socket
import struct
import sys
import time

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 47820


def packet(sequence, action, enabled=True, home=0):
    return struct.pack('<4sIQQ7d', b'FPA1', 1 if enabled else 0, sequence, home, *action)


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    start = time.monotonic()
    sequence = 10
    sent_stream = 0
    while True:
        t = time.monotonic() - start
        message = None
        if 1.5 <= t < 4.5:
            message = packet(1, [0.03, 0, 0, 0, 0, 0, 1])
        elif 4.5 <= t < 5.5:
            message = packet(2, [0, 0, 0.02, 0, 0, 0, 1], enabled=False)
        elif 5.5 <= t < 8.5:
            message = packet(3, [0, 0, 0, 0, 0, 0.2, 1])
        elif 8.5 <= t < 10.0:
            step = int((t - 8.5) / 0.1)
            if step >= sent_stream:
                sent_stream = step + 1
                sequence += 1
            message = packet(sequence, [0, 0, 0.01, 0, 0, 0, 1])
        elif 11.0 <= t < 12.0:
            message = packet(sequence + 1, [0, 0, 0, 0, 0, 0, 1], enabled=False, home=1)
        elif t >= 12.0:
            message = packet(sequence + 1, [0, 0, 0, 0, 0, 0, 1], enabled=False, home=1)
        if message is not None:
            sock.sendto(message, ('127.0.0.1', PORT))
        if t > 30:
            return
        time.sleep(0.02)


if __name__ == '__main__':
    main()
