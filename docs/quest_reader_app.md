# Quest 3 reader app: findings and a fixed version to try

Status (2026-10-12): **the dreaming-anshu fork is installed** on the headset
(over Wi-Fi ADB). Verified there: the right controller's pose arrives with only
the right controller on, the stick click (`RJ`) arrives, and poses are in the
room's fixed frame (plus the headset pose `h:`). It is still a Debug build, so
the reader keeps avoiding relaunches of a running app.

## The app that was on the headset before

`jborbik/oculus_reader` (the Quest 3 / OpenXR rewrite of rail-berkeley's
`oculus_reader`), APK `third_party/oculus_reader/oculus_reader/APK/teleop-debug.apk`,
shown in the headset as **"Xr Controllers"**, package `com.rail.oculus.teleop`.

Problems we hit, and their causes in its source:

| Problem | Cause |
|---|---|
| Right controller pose missing unless the left controller is on | `app/src/Remotes.h`, `remotes_to_string()`: `if (RightRemoteTracked)` writes `"l:" + LeftRemoteTransform`, `if (LeftRemoteTracked)` writes `"r:" + RightRemoteTransform` (swapped checks) |
| No joystick click (`RJ`/`LJ`), so TeleMoMa's axis re-set does not work | never emitted |
| Forward moved the arm up; down moved it down-and-forward | poses are OpenXR (x right, y up, -z forward), not the old VrApi axes TeleMoMa's `rmat_reorder [-2,-1,-3,4]` was made for; and it streams the GRIP pose (axes along the handle, tilted 30-60 deg) relative to the headset |
| App crashes when the menu opens, it is closed, or it is launched again while running | Debug build: `XrApp.cpp:131 HandleSessionStateChanges: assertion "Resumed" failed` |

Workarounds currently in our code: keep the left controller on; B (grip released)
re-sets the frame; `quest_axis_reorder = 3, 1, 2, 4`; `quest_level_frame = true`;
the reader never relaunches a running app and restarts a dead or paused one.

## A fork that fixes it: dreaming-anshu/oculus_reader

<https://github.com/dreaming-anshu/oculus_reader> (fork of jborbik's, last commit
`62aefd3`, 2026-08-07, "Quest hand-tracking APK: 26-joint hand stream + native
Jetson build"). Cloned to `third_party/oculus_reader_fixed/`.

Source changes (compare `jborbik/oculus_reader...dreaming-anshu:oculus_reader`):

- **Left/right fixed**: each side is gated on its own tracked flag.
- **Joystick clicks** sent as `RJ,` / `LJ,`.
- **World / tracking-space poses** instead of head-relative (gravity-aligned,
  y up), so the headset may move or be worn.
- **AIM pose** (controller tip, x right / y up / -z along the pointing ray, same
  for both hands) instead of the tilted GRIP pose.
- **Head pose** added as `h:<16 values>` (our parser ignores it).
- **Quest 3 interaction profile** fixed (misspelled `touch_controller_plus`
  paths; adds the core `/interaction_profiles/oculus/touch_controller`).
- `app/build.gradle`: a **release** build signed with the Android debug key,
  which compiles out the crashing asserts. Also a bare-hand 26-joint stream on a
  second logcat tag `hE9ryARX`, and `reader.py` filters logcat on the device.

Checked in its prebuilt APK (`oculus_reader/APK/teleop-debug.apk`, 14.8 MB): the
native library contains the new message strings (`|l:`, `|r:`, `RJ,`, `LJ,`, `h:`)
and the core touch-controller profile, so it is built from the fixed source.
**But it is a Debug native build** (`.cxx/Debug` paths), so the crash on
session changes may remain; only a release build from source removes it. It is a
prebuilt binary from an individual's repository (0 stars): the source is
reviewed, the binary cannot be proven to match it.

### Effect on our code

None needed: `src/quest_input.h` reads `r:` and ignores `h:`; buttons are the
same. `RJ` would then work as TeleMoMa's re-set (B keeps working too). With
world-frame, gravity-aligned poses the level frame uses true gravity instead of
assuming a level headset. Re-check the axis directions on the robot after
switching (aim instead of grip pose).

### How to install (headset on USB)

The APK is signed with a different key, so it cannot update the current app;
uninstall first (the old one can be reinstalled from
`third_party/oculus_reader/oculus_reader/APK/teleop-debug.apk`):

```bash
adb uninstall com.rail.oculus.teleop
adb install -t third_party/oculus_reader_fixed/oculus_reader/APK/teleop-debug.apk
python3 tools/quest/quest_monitor.py   # should now show 'r' with only the right controller on
```

Or build a release APK from `third_party/oculus_reader_fixed/app` (needs JDK 17,
Android SDK 32 + NDK 27, accepting the Android SDK license): this is the
verifiable option and removes the crash asserts.
