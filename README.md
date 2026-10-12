# Franka gamepad teleoperation

Gamepad teleoperation for a Franka FR3 with the parallel gripper, plus an episode recorder for VLA data collection. The sticks command a Cartesian velocity of the gripper; an onboard differential-IK solver turns that into joint velocities at 1 kHz. While it runs, the teleop publishes its full state on ROS 2, and the recorder saves it together with two RealSense cameras (see [Recording episodes](#recording-episodes)).

It is a ROS 2 Humble package (`franka_teleop`) built with colcon.

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
- Motion requires a valid configuration and `--enable-motion`. The program then connects to the robot at once and holds the arm; it moves only while the deadman (RB) is held.
- Releasing the deadman pauses: the arm is brought smoothly to zero velocity and holds while the session stays open; pressing RB again resumes. Ctrl-C, joystick disconnect/read failure, missed input heartbeat, or loss of the gripper connection ends the arm-control loop at zero velocity. A single rejected or unsuccessful gripper command is reported and does not stop the arm. The program never retries after a stop; if the robot is still in reflex mode from a previous stop, it runs libfranka's error recovery once at the next startup. Joint velocities stay inside the robot's own position-dependent limits (evaluated 0.05 rad early), so joints come to rest before their limits; on a libfranka control error the program prints the joint positions against the FR3 limits.
- Stick commands are Cartesian velocities in Franka's base frame `O` (m/s and rad/s); the workspace is a fixed translation box in that frame.
- Motion is shaped twice (Cartesian and joint S-curves) and clipped by libfranka's joint rate limiter; each session begins with a 0.1 s zero-velocity hold. The project does not change impedance gains or attempt to bypass robot collision/reflex behaviour.

Do not operate until the workcell is clear, the emergency stop is accessible, FCI is enabled, the operator has approved the configuration, and no other client controls the robot or parallel gripper.

## Build

```bash
cd /home/airocs/Desktop/franka_gamepad_teleop
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -Wno-dev
colcon test && colcon test-result --verbose
source install/setup.bash
```

Every new terminal needs `source /opt/ros/humble/setup.bash && source install/setup.bash` (from the project folder) before the `ros2` commands below. The recorder additionally needs the RealSense driver and the MCAP bag plugin: `sudo apt install ros-humble-realsense2-camera ros-humble-rosbag2-storage-mcap`.

`Franka::Franka` must be discoverable through CMake. This PC exposes it from the ROS 2 Humble installation (libfranka 0.20.4; the IK uses `Robot::getUpper/LowerJointVelocityLimits`, available in recent libfranka versions).

The tests include `diff_ik_test`, an off-robot simulation of the IK on an FR3 kinematic model (FK checked against a recorded robot pose). An end-to-end simulation that runs the real `gamepad_teleop` against a simulated FR3 with a scripted gamepad lives in `tests/sim/` (see its README).

## Controller mapping first

This does not open a robot connection:

```bash
ros2 run franka_teleop controller_probe --device /dev/input/by-id/usb-Microsoft_Android-joystick
```

To test the final gamepad mapping without a robot, use the command preview:

```bash
ros2 run franka_teleop controller_probe --device /dev/input/by-id/usb-Microsoft_Android-joystick --xbox-preview
```

It prints normalized Cartesian translation or rotation commands but contains no libfranka code and cannot reach the robot. This is an input preview, not a 3D physics simulator.

This is only needed to diagnose a controller problem. The `xbox_android_standard` profile is fixed to the currently detected controller: hold physical **RB** (reported as Linux button 7) to enable motion, right-stick vertical for base-frame X (push up = +X, away from the base), left-stick horizontal for base-frame Y (push right = +Y), left-stick vertical for base-frame Z (push up = +Z), and hold **L1** (Linux button 6) or **L2** to change the sticks into rotation mode (left stick roll/pitch, right-stick horizontal yaw). **A** opens and **B** closes the parallel gripper while RB is held. **D-pad up/down** switches speed between 100 %, 50 % and 25 % (the D-pad axis is `speed_toggle_axis`, 7 by default; confirm it with `controller_probe`). Downward Z moves at half speed, like Frankastein. The detected controller provides eight axes and fifteen buttons; use the stable `/dev/input/by-id` path, not `/dev/input/js0`, because `js0` can change after reconnecting devices.

## Choosing your own workspace limits

`pose_monitor` is read-only: it connects to the robot but never commands motion.
Run it, guide the arm by hand to the edges of the space you want, then press
Ctrl-C:

```bash
ros2 run franka_teleop pose_monitor --robot-ip 192.168.3.200 --config config/teleop.conf
```

It shows live x/y/z of the end effector (base frame, metres, the same point the
teleop limits), marks an axis `OUT` when it is outside the configured limits,
and on Ctrl-C prints the min/max seen as `workspace_*_m` lines to paste into
`config/teleop.conf`. Only one program can be connected to the robot at a time,
so close `gamepad_teleop` first.

## Configure and validate

```bash
cp config/teleop.conf.example config/teleop.conf
ros2 run franka_teleop gamepad_teleop --config config/teleop.conf --check-config
```

The supplied `config/teleop.conf` uses `0.15 m/s` and `0.40 rad/s` top speeds, `1.0 m/s²` / `10 m/s³` linear and `2.5 rad/s²` / `25 rad/s³` angular S-curve limits, a `0.1 s` zero-velocity startup hold, the lab's fixed workspace box, and the parallel-gripper settings (open `0.08 m/s`, grasp `0.03 m/s` at `50 N`). Speeds and gripper settings were originally taken from Frankastein's keyboard teleop. It preserves the currently configured Franka Desk collision behavior.

Stick directions are in the robot base frame `O`; a controlled, clear-workspace check is still required before physical use because the operator's view may not be aligned with that frame.

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
ros2 run franka_teleop gamepad_teleop --config config/teleop.conf --enable-motion
```

The tool connects to the robot at once and holds the arm still. The arm moves only while **RB (the right bumper)** is held; release RB to pause motion and press it again to resume. `Ctrl-C` ends the session; it is not a substitute for the physical emergency stop.

The joint-velocity IK version has been tested in simulation only; the first run on the robot should use the 25 % speed level (D-pad down twice) in a clear workspace.

## Recording episodes

```bash
./record.sh
```

This starts both cameras and the recorder in the background (log in `log/`), opens the web UI at <http://localhost:8765>, and runs `gamepad_teleop` in the foreground. Ctrl-C ends only the teleop; you can then restart it or quit, and quitting saves a running episode first. `--no-teleop` runs cameras and recorder only, `--no-browser` skips opening the page. The pieces can also be run by hand: `ros2 launch franka_teleop recording.launch.py` and `ros2 run franka_teleop gamepad_teleop --config config/teleop.conf --enable-motion`.

The page shows both live camera views, the robot state, and the saved episodes. Type the task instruction once (it is stored with every new episode), then press **Y** on the gamepad, or **Record** on the page, to start; press it again to stop and save. Episodes can be renamed (click the name), marked success/fail, replayed as a side-by-side video of both cameras, or deleted (moved to `.trash` in the data folder). Stopping the launch with Ctrl-C saves a running episode first.

Setup lives in `config/recording.yaml`: data folder (default `~/franka_teleop_data`), web port, record button, camera names and serial numbers (rename `cam_1`/`cam_2` to where they are mounted), resolution (640×480 at 30 fps), depth and IMU on/off (IMU is off by default), and a hardware reset of the cameras at startup, which clears a stream that stopped delivering frames.

**What is published.** `gamepad_teleop` publishes, from a normal-priority thread fed by a lock-free buffer (the 1 kHz loop never waits on ROS):

- `/teleop/state` (`franka_teleop/msg/TeleopState`, every control tick): the complete `franka::RobotState`, the controller's internals (stick target twist, S-curve twist, IK solution, fence scale, commanded joint velocities), the raw gamepad, the gripper command and its measured width;
- `/teleop/joy` (`sensor_msgs/Joy`, 100 Hz and on every button change);
- `/teleop/session` (latched JSON): the config file text and the git commit the binary was built from.

The RealSense driver publishes each camera under `/cameras/<name>/` (color, depth aligned to color, camera_info, IMU, TF).

**What is saved,** one folder per episode (`<date>_<time>_<name>`):

| File | Contents |
|---|---|
| `meta.json` | name, task, success, notes, duration, per-stream frame counts and drops, camera intrinsics, teleop config and git commit, warnings |
| `bag/` | ROS 2 bag (MCAP, zstd) with every non-image topic: 1 kHz `/teleop/state`, `/teleop/joy`, camera_info, extrinsics, `/tf_static`, … |
| `<cam>_color.mp4` | H.264 color video, 30 fps (CRF 18, visually lossless) |
| `<cam>_depth.mkv` | 16-bit depth aligned to color, lossless FFV1 (millimetres) |
| `<cam>_<stream>_frames.csv` | per frame: index, camera timestamp, receive time (ns, ROS time) |
| `replay.mp4` | side-by-side replay of all color views |

Everything is recorded at its native rate; nothing is resampled. Pick the action representation (delta end-effector pose, joint velocity, …) and the rate when exporting to a training format, by matching camera frame timestamps to `/teleop/state` stamps. All timestamps use the same system clock. Color and depth of one camera share a timestamp (`enable_sync`); the two cameras are not hardware-synchronised, so match them by timestamp too.

Robot state is about 4 MB/s before compression. Video sizes depend on the scene; check the Size column after a first real episode. The recorder refuses to start with less than 2 GB free.

## Teleoperation with the Meta Quest 3

```bash
./record.sh --quest
```

The teleop reads the Quest's right controller itself (no separate bridge), like TeleMoMa/DROID: the headset runs the `oculus_reader` app ("Xr Controllers", from `third_party/oculus_reader`, the Quest 3 fork jborbik/oculus_reader), which streams controller poses and buttons over ADB; `src/quest_input.h` reads them in a background thread.

- **Hold the right GRIP** to move. On every grip press and release, TeleMoMa resets its origins (robot pose and controller pose); while held, the arm's target is the robot origin plus the controller's motion since, axes mapped to the robot base with a fixed `rmat_reorder` (see Axes below) relative to the controller's orientation at the first grip press (or the last B press). Release = the arm stops and holds.
- **TRIGGER** toggles the gripper (TeleMoMa). **A** starts/stops a recording (like Y on the gamepad). **B**, with the grip released, re-sets "forward" to how you hold the controller now (TeleMoMa's joystick click, which the Quest 3 app does not send).
- **Axes.** The Quest 3 app reports OpenXR controller axes (x right, y up, −z forward), not those of the original app TeleMoMa's `[-2, -1, -3]` was made for (with it, pushing forward moved the arm up). The default `quest_axis_reorder = 3, 1, 2, 4` is for an operator standing in front of the robot, facing it: forward → toward the base (−x), right → +y, up → +z. Standing behind the robot, facing its front: `-3, -1, 2, 4`. Set it in `config/teleop.conf`.
- **Level frame** (`quest_level_frame = true`, default). TeleMoMa takes the reference frame from the controller's full orientation at the reset, but the app reports the grip pose, whose axes follow the handle, tilted 30-40° forward when held naturally, so "down" came out as down-and-forward. With the level frame, the reset keeps only the controller's heading (where it points, flattened) and takes "up" from the headset, so the headset must sit **upright and level** (e.g. on a box or stand, visor facing you). The action math is TeleMoMa's either way; `false` restores TeleMoMa's exact frame.
- The arm is driven toward TeleMoMa's target (`current + action`) with a proportional twist (`quest_position_gain`, `quest_rotation_gain`, 5/s) through the same workspace braking, S-curves, IK and limits as the gamepad.

**TeleMoMa's action** is computed every control tick by `src/telemoma_policy.h`, a line-by-line port of `telemoma/human_interface/oculus.py` (`ctest` checks it against TeleMoMa's own Python on random sequences; they agree to 1e-11). It is the remaining offset to the grip-anchored target, `[dx, dy, dz, droll, dpitch, dyaw (scipy "xyz" Euler), gripper (1 open, 0 closed)]` clipped to ±1, not a step-to-step delta. It is recorded at 1 kHz on `/teleop/quest` (`franka_teleop/QuestState`: raw controller pose and buttons, action, target, origins), next to `/teleop/state`, so both this action and the delta pose of the arm's actual motion can be exported from the same episode (`franka_teleop_recorder.actions.telemoma_actions` / `delta_pose_actions`). One deliberate difference from TeleMoMa: a trigger pull toggles the gripper once (TeleMoMa re-evaluates a pending toggle on every action call, which at 1 kHz would make the gripper flicker).

**Headset setup.** Developer Mode on, USB cable (USB 2 is enough), USB debugging allowed. `python3 tools/quest/quest_monitor.py` checks the connection, installs and starts the app, keeps the headset awake when it is not worn (proximity-sensor override; `--restore` undoes it), and shows the controller live. Put the headset on a fixed spot facing where you stand: the poses are relative to the headset.

**Reader app on the headset:** the fixed fork `dreaming-anshu/oculus_reader` (`third_party/oculus_reader_fixed`, see [docs/quest_reader_app.md](docs/quest_reader_app.md)): only the right controller is needed, the stick click arrives (it and B re-set "forward" with the grip released), and poses are in the room's fixed, gravity-aligned frame, so the headset may move. It is a Debug build: it can abort when its VR session is interrupted or it is launched twice; the teleop and the monitor never relaunch a running app and restart a dead or stalled one. (The original Quest 3 build, `third_party/oculus_reader`, needed the left controller on and sent no stick click.)

**Wi-Fi instead of USB:** with the cable plugged in, run `tools/quest/quest_wifi.sh` (once after each headset restart), then set `quest_connection = <headset IP>` in `config/teleop.conf` (or `quest_monitor.py --connection <IP>`). The Quest and the PC must be able to reach each other on the network.

## Replaying an episode on the robot

```bash
./replay.sh episode_0002                          # accumulate mode, recorded speed
./replay.sh episode_0002 --mode from-current      # apply each action to the measured pose
./replay.sh episode_0002 --speed 0.5 --no-gripper
```

`scripts/episode_actions` turns the episode's 1 kHz state into delta end-effector pose actions at `--rate` Hz (default 15): per step, the gripper's position change and rotation change (rotation vector) in the base frame, plus open/closed. These are the actions a VLA would be trained on. `episode_replay` (independent of `gamepad_teleop`) then:

1. asks for Enter, then moves the arm in joint space to the episode's first joint positions (at most 0.3 rad/s per joint);
2. asks for Enter, then executes the actions at their rate: each moves a pose target, tracked with feed-forward plus a pose correction through the teleop's kind of pipeline (Cartesian S-curve, the same IK solver, workspace fence, joint S-curve, libfranka's rate limiter), with the gripper opened/closed when the action says so.

In both phases the arm moves only while **RB** is held; releasing RB stops it and freezes the replay clock, and Ctrl-C quits. Only one program can control the robot, so end `gamepad_teleop` first.

- `accumulate`: each target is the previous target plus the action, so the recorded path is replayed without drift. It measures how well the actions plus the controller reproduce the demonstration.
- `from-current`: each action is added to the pose the arm is actually at when its step begins, as a policy predicting deltas from the current observation would do. Tracking shortfalls are never corrected, so it drifts.

The actions file and a per-step CSV log (recorded vs replayed position, errors) are written to `<data_dir>/replays/<episode>/`; a summary is printed at the end. In simulation, replaying `episode_0002` (28 s) gave a mean distance to the recorded path of 2.7 mm (final 0 mm) with `accumulate`, and 10 mm (final 10 mm) with `from-current`.

The recorder runs under the system Python (`scripts/recorder` ignores `~/.local`, whose numpy 2.x breaks ROS Humble's Python modules).
