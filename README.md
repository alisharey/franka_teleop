# Franka gamepad teleoperation

Standalone gamepad teleoperation for a Franka FR3 with the parallel gripper. The sticks command a Cartesian velocity of the gripper; an onboard differential-IK solver turns that into joint velocities at 1 kHz. It does not modify or depend on `Franka-SM` at runtime.

## How motion is computed

Each 1 ms control tick:

1. **Sticks → twist.** Stick deflection gives a base-frame Cartesian velocity (linear and, with L1/L2 held, angular), clamped so the gripper can always stop before the workspace walls.
2. **Cartesian S-curve.** The twist is shaped with jerk-limited acceleration so stick steps never cause overshoot or ringing.
3. **IK solver** (`src/diff_ik.h`, the approach DROID uses, in C++):
   - damped least squares, with damping only near singular poses;
   - per-joint velocity bounds from the robot's own position-dependent limits (`Robot::getUpper/LowerJointVelocityLimits`), evaluated 0.05 rad early and capped at 1 rad/s. A joint that reaches its bound is frozen there and the others are re-solved to keep the gripper on course (saturation in the null space);
   - while the sticks move, a gentle pull of the elbow toward Franka's ready pose and away from joint limits, which does not move the gripper.
4. **Fence on actual motion.** The resulting gripper motion (including any detour, below) is checked against the workspace box again, and the arm never moves against the push.
5. **Joint S-curve, then libfranka's rate limiter** with the robot's exact velocity, acceleration and jerk limits as the final safety clip.

The robot runs its joint impedance controller (`ControllerMode::kJointImpedance`).

**Reachability.** Some pushed directions are geometrically impossible from some poses with the gripper pointing down. For example, straight down from the home pose needs the elbow (joint 4) to fold past its limit, and low near the base only moving away from the base is possible. `allow_detour` in the config decides what happens then:

- `true` (default): the arm swings off the pushed line as needed to keep making progress (the descent from home to the table detours about 10 cm forward and takes about 11 s);
- `false`: the arm keeps the exact direction and slows or stops instead.

Either way the gripper stays inside the workspace box and never moves against the push.

## Safety model

- `controller_probe` is input-only and cannot connect to the robot.
- The supplied configuration uses a fixed workspace box in the robot base frame (`workspace_*_m` in `config/teleop.conf`). The program refuses to start with the gripper outside it and slows motion near each wall so the arm stops about 2 cm inside the boundary, including any IK detour. Franka Desk's active collision settings are preserved.
- `gamepad_teleop --check-config` validates text configuration only and cannot connect to hardware.
- Motion requires a valid configuration, `--enable-motion`, and a held deadman button before the program opens an FCI connection.
- Releasing the deadman pauses: the arm is brought smoothly to zero velocity and holds while the session stays open; pressing RB again resumes. Ctrl-C, joystick disconnect/read failure, missed input heartbeat, or loss of the gripper connection ends the arm-control loop at zero velocity. A single rejected or unsuccessful gripper command is reported and does not stop the arm. The program never retries after a stop; if the robot is still in reflex mode from a previous stop, it runs libfranka's error recovery once at the next startup. Joint velocities stay inside the robot's own position-dependent limits (evaluated 0.05 rad early), so joints come to rest before their limits; on a libfranka control error the program prints the joint positions against the FR3 limits.
- Stick commands are Cartesian velocities in Franka's base frame `O` (m/s and rad/s); the workspace is a fixed translation box in that frame.
- Motion is shaped twice (Cartesian and joint S-curves) and clipped by libfranka's joint rate limiter; each session begins with a 0.1 s zero-velocity hold. The project does not change impedance gains or attempt to bypass robot collision/reflex behaviour.

Do not operate until the workcell is clear, the emergency stop is accessible, FCI is enabled, the operator has approved the configuration, and no other client controls the robot or parallel gripper.

## Build

```bash
cd /home/airocs/Desktop/franka_gamepad_teleop
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

`Franka::Franka` must be discoverable through CMake. This PC exposes it from the ROS 2 Humble installation (libfranka 0.20.4; the IK uses `Robot::getUpper/LowerJointVelocityLimits`, available in recent libfranka versions).

`ctest` includes `diff_ik_test`, an off-robot simulation of the IK on an FR3 kinematic model (FK checked against a recorded robot pose). An end-to-end simulation that runs the real `gamepad_teleop` against a simulated FR3 with a scripted gamepad lives in `tests/sim/` (see its README).

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

The joint-velocity IK version has been tested in simulation only; the first run on the robot should use the 25 % speed level (D-pad down twice) in a clear workspace.
