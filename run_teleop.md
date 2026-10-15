# Running the teleop (Meta Quest 3)

## Before you start

- Robot: stop button released, robot ready in Desk, FCI enabled.
- Wi-Fi adapter (Tenda U12) plugged into this PC. It starts the `franka-quest`
  hotspot automatically (password in `~/franka_quest_hotspot_password.txt`).
- Quest on the `franka-quest` Wi-Fi (headset IP `10.42.0.175`).
- **After every headset restart**: plug the USB cable in once and run
  `tools/quest/quest_wifi.sh` (Wi-Fi debugging turns off on every restart; the
  Quest does not allow keeping it on). Then the cable can come out.
- Wake the right controller (the left one is not needed).
- **After the robot is powered on**: home the gripper once (fingers close and
  open fully; keep them clear): `tools/home_gripper.sh` (or Desk: end effector >
  Homing). Until then it ignores open/close, and the teleop prints
  "The gripper is NOT HOMED".

## Start

```bash
cd ~/Desktop/franka_gamepad_teleop && ./record.sh --quest
```

Then:

1. Point the controller at the robot and click the stick (grip released) to set
   "forward".
2. Squeeze and hold the grip, then move slowly. First time: check that forward,
   up and right go the way you expect.

## Controls (right controller)

| Control | Does |
|---|---|
| **Grip** (side button, middle finger) | Hold to move the arm; release stops it. Each press re-zeroes the position (like lifting a mouse). |
| **Trigger** (index finger) | Toggles the gripper: pull once = close, pull again = open. Works while the grip is held (TeleMoMa); a pull with the grip released takes effect at the next grip press. |
| **Stick click** or **B** (grip released) | Re-sets "forward" to where the controller points now. |
| **A** | Start / stop a recording. |
| **Stick flicked down** (grip released) | The arm moves to the saved home pose (joint space, at most 0.3 rad/s per joint). Squeezing the grip, flicking again or Ctrl-C stops it. |

Ctrl-C in the terminal ends the teleop; you can then restart it (Enter) or quit
(q). The web page with the cameras and episodes is at http://localhost:8765.

## Settings (`config/teleop.conf`)

- `home_joints`: the home pose for the stick flick. To change it, move the arm
  to the new pose, close the teleop, and run `tools/save_home_pose.sh` (reads the
  joints, no motion).

- `quest_connection = 10.42.0.175` uses Wi-Fi; `quest_connection = usb` uses the cable.
- `quest_axis_reorder = 3, 1, 2, 4` (default) is for standing in front of the
  robot, facing it; behind it, facing its front: `-3, -1, 2, 4`.
- `quest_timeout_ms = 500`: readings older than this pause the arm (Wi-Fi has
  brief gaps).
- `quest_rotation_scale = 1.5`: the gripper turns 1.5x as far as your hand
  (1 = exactly like your hand, as in TeleMoMa). The recorded TeleMoMa action
  uses the scaled rotation.

## Connection modes

The teleop reads the Quest over ADB, either way:

| Mode | `config/teleop.conf` | Notes |
|---|---|---|
| **Wi-Fi** (current) | `quest_connection = 10.42.0.175` | Through the `franka-quest` hotspot. Needs the cable once after every headset restart (below). Readings ~66/s with occasional brief gaps. |
| **USB cable** | `quest_connection = usb` | Most reliable; USB 2 is enough. Nothing else to set up. |

To check either: `python3 tools/quest/quest_monitor.py` (USB) or
`python3 tools/quest/quest_monitor.py --connection 10.42.0.175` (Wi-Fi).

## The `franka-quest` hotspot

- Made by this PC through the USB Wi-Fi adapter (Tenda U12, RTL8812AU chip).
  Its driver is the community `8812au` driver, installed with DKMS
  (`third_party/8812au`; it rebuilds itself on kernel updates).
- Name `franka-quest`, 5 GHz channel 36, WPA2; the password is in
  `~/franka_quest_hotspot_password.txt`. The PC is `10.42.0.1`; devices get
  `10.42.0.x`. It is a separate network from the robot (`192.168.3.x`) and the
  university (`10.127.x`); it also passes internet through to the Quest.
- Starts automatically whenever the adapter is plugged in. By hand:
  `nmcli connection up franka-quest` / `nmcli connection down franka-quest`.
  Change the password: `nmcli connection modify franka-quest wifi-sec.psk "NEW"`
  then `nmcli connection up franka-quest`.
- The PC must boot kernel `6.8.0-142-lowlatency` (the default in GRUB). The
  realtime kernel `5.15` has no driver for the adapter (and no NVIDIA driver).
- Why not the university Wi-Fi or a phone hotspot: the PC can only reach the
  Quest if both are on the same network and allowed to talk to each other; the
  university network does not allow that, and a phone hotspot hides the Quest
  behind the phone.

## Setting up a Quest (new headset or new person)

Once per headset:

1. **Developer account**: sign in to the headset's Meta account at
   developer.meta.com and create an organization (any name; Meta may ask to
   verify the account).
2. **Developer Mode**: in the Meta Horizon phone app paired with the headset:
   Menu > Devices > the headset > Headset settings > Developer Mode on. Restart
   the headset.
3. **USB debugging**: connect the headset to this PC with a USB-C data cable, put
   it on and accept "Allow USB debugging" (tick "Always allow from this
   computer"). Check: `adb devices` lists it as `device`.
4. **Reader app**: install the fixed reader app (it shows as "Xr Controllers"):
   ```bash
   adb uninstall com.rail.oculus.teleop   # only if an older version is installed
   adb install -t third_party/oculus_reader_fixed/oculus_reader/APK/teleop-debug.apk
   ```
   You never need to open it yourself: the teleop and the monitor start it.
5. **Wi-Fi** (optional): on the headset, Settings > Wi-Fi > join `franka-quest`.
   With the cable in, run `tools/quest/quest_wifi.sh`, put the headset on and
   accept the debugging prompt for the Wi-Fi connection ("Always allow"). The
   script prints the headset's IP: put it in `config/teleop.conf` as
   `quest_connection = <IP>` (a different headset gets a different IP).

Every session: wake the right controller. After a headset restart (Wi-Fi mode):
cable in, `tools/quest/quest_wifi.sh`, cable out.

## Checks and troubleshooting

- Live controller readout (rate, pose, buttons):
  `python3 tools/quest/quest_monitor.py --connection 10.42.0.175`
- "User stopped" error: the robot's stop button is pressed; release it.
- Trigger does nothing: the gripper is not homed (`tools/home_gripper.sh`).
- "the Quest (...) is unauthorized": put the headset on and accept "Allow
  debugging" (tick "Always allow"). It can ask again over Wi-Fi.
- No readings / "Wi-Fi ADB" errors after a headset restart: cable in, run
  `tools/quest/quest_wifi.sh`, accept the debugging prompt in the headset.
- The reader app on the headset ("Xr Controllers", fixed fork, see
  `docs/quest_reader_app.md`) can crash if you open the headset menu; the teleop
  restarts it and the arm pauses meanwhile.
- Gamepad instead of the Quest: `./record.sh` (no `--quest`).
