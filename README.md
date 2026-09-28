# Franka gamepad teleoperation

Standalone, conservative Cartesian-velocity teleoperation for a Franka arm with the parallel gripper. It does not modify or depend on `Franka-SM` at runtime.

## Safety model

- `controller_probe` is input-only and cannot connect to the robot.
- The supplied configuration uses a fixed workspace box in the robot base frame (x 0.27–0.76 m, y −0.39–0.36 m, z 0.00–0.71 m; Frankastein's `configs/robots/fr3.yaml` box with z lowered from 0.06 to 0). The program refuses to start with the gripper outside it and slows motion near each wall so the arm stops about 2 cm inside the boundary. Franka Desk's active collision settings are preserved.
- `gamepad_teleop --check-config` validates text configuration only and cannot connect to hardware.
- Motion requires a valid configuration, `--enable-motion`, and a held deadman button before the program opens an FCI connection.
- Releasing the deadman pauses: the arm is brought smoothly to zero velocity and holds while the session stays open; pressing RB again resumes. Ctrl-C, joystick disconnect/read failure, missed input heartbeat, or loss of the gripper connection ends the arm-control loop with a zero Cartesian velocity. A single rejected or unsuccessful gripper command is reported and does not stop the arm. The program never retries after a stop; if the robot is still in reflex mode from a previous stop, it runs libfranka's error recovery once at the next startup. Commands are scaled down as any joint approaches within 0.1 rad of its FR3 limit (estimated from the robot Jacobian), and on a libfranka control error the program prints the joint positions against those limits.
- Cartesian velocity commands are in Franka's base frame `O`: linear velocities are metres/second and angular velocity is radians/second. The configured workspace is a fixed translation box in that frame.
- The project explicitly rate-limits Cartesian velocity, acceleration, and jerk, begins each control session with a 0.1 s zero-velocity hold, and does not change impedance gains or attempt to bypass robot collision/reflex behaviour.

Do not operate until the workcell is clear, the emergency stop is accessible, FCI is enabled, the operator has approved the configuration, and no other client controls the robot or parallel gripper.

## Build

```bash
cd /home/airocs/Desktop/franka_gamepad_teleop
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

`Franka::Franka` must be discoverable through CMake. This PC exposes it from the ROS 2 Humble installation.

## Controller mapping first

This does not open a robot connection:

```bash
build/controller_probe --device /dev/input/by-id/usb-Microsoft_Android-joystick
```

To test the final gamepad mapping without a robot, use the command preview:

```bash
build/controller_probe --device /dev/input/by-id/usb-Microsoft_Android-joystick --xbox-preview
```

It prints normalized Cartesian translation or rotation commands but contains no libfranka code and cannot reach the robot. This is an input preview, not a 3D physics simulator.

This is only needed to diagnose a controller problem. The `xbox_android_standard` profile is fixed to the currently detected controller: hold physical **RB** (reported as Linux button 7) to enable motion, right-stick vertical for base-frame X (push down = +X, away from the base), left-stick horizontal for base-frame Y (push right = +Y), left-stick vertical for base-frame Z (push up = +Z), and hold **L1** (Linux button 6) or **L2** to change the sticks into rotation mode (left stick roll/pitch, right-stick horizontal yaw). **A** opens and **B** closes the parallel gripper while RB is held. **D-pad up/down** switches speed between 100 %, 50 % and 25 % (the D-pad axis is `speed_toggle_axis`, 7 by default; confirm it with `controller_probe`). Downward Z moves at half speed, like Frankastein. The detected controller provides eight axes and fifteen buttons; use the stable `/dev/input/by-id` path, not `/dev/input/js0`, because `js0` can change after reconnecting devices.

## Choosing your own workspace limits

`pose_monitor` is read-only: it connects to the robot but never commands motion.
Run it, guide the arm by hand to the edges of the space you want, then press
Ctrl-C:

```bash
build/pose_monitor --robot-ip 192.168.3.200 --config config/teleop.conf
```

It shows live x/y/z of the end effector (base frame, metres, the same point the
teleop limits), marks an axis `OUT` when it is outside the configured limits,
and on Ctrl-C prints the min/max seen as `workspace_*_m` lines to paste into
`config/teleop.conf`. Only one program can be connected to the robot at a time,
so close `gamepad_teleop` first.

## Configure and validate

```bash
cp config/teleop.conf.example config/teleop.conf
build/gamepad_teleop --config config/teleop.conf --check-config
```

The supplied `config/teleop.conf` matches Frankastein's keyboard teleop: `0.15 m/s`, `0.40 rad/s`, `1.0 m/s²` / `10 m/s³` linear and `2.5 rad/s²` / `25 rad/s³` angular rate limits, a `0.1 s` zero-velocity startup hold, Frankastein's fixed workspace, and its gripper settings (open `0.08 m/s`, grasp `0.03 m/s` at `50 N`). It preserves the currently configured Franka Desk collision behavior.

The tool commands Cartesian velocity in the base frame `O`; a controlled, clear-workspace verification is still required before physical use because the operator's view may not be aligned with that frame.

## Kernel: PREEMPT_RT or lowlatency

The program picks the realtime mode itself. On a PREEMPT_RT kernel it enforces
realtime (libfranka refuses to run if it cannot). On any other kernel, such as
Ubuntu's lowlatency kernel, it runs without that requirement, the same default
Frankastein's libfranka helpers use. In both cases libfranka still asks for
the highest `SCHED_FIFO` priority for the 1 kHz control loop, so the user must
be allowed realtime priority:

```bash
ulimit -r   # must print a non-zero value, e.g. 99
```

If it prints 0, add the user to a group with realtime rights (Franka's setup
guide uses `realtime`):

```bash
sudo groupadd -f realtime && sudo usermod -aG realtime $USER
printf '@realtime soft rtprio 99\n@realtime hard rtprio 99\n@realtime soft memlock 102400\n@realtime hard memlock 102400\n' | sudo tee /etc/security/limits.d/99-realtime.conf
```

then log out and back in. On lowlatency, occasional
`communication_constraints_violation` errors mean the loop missed deadlines;
if they happen often, boot the PREEMPT_RT kernel.

## Hardware operation

After the configuration is approved and the no-hardware validation succeeds:

```bash
build/gamepad_teleop --config config/teleop.conf --enable-motion
```

The tool waits ten seconds for **RB (the right bumper)**. If RB is not held, it exits without opening an FCI connection. Once enabled, release RB to pause motion and press it again to resume. `Ctrl-C` ends the session; it is not a substitute for the physical emergency stop.

There has been no hardware test of the Frankastein-matched profile yet.
