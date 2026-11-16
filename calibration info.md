# Camera calibration

Current values, copied from `config/camera_calibration.json` (written by the calibration page,
`tools/calibration/calibration_server.sh`; read that file in code, it is the source of truth).

- Session: `/home/airocs/franka_calibration/20261116_160720`
- Copied: 2026-11-16 16:39
- Target: ChArUco 5x7 squares, DICT_4X4_50, square 33.7 mm, marker 25.0 mm
- Snapshots: 16, hand-eye method: park

## Accuracy (read before using it for obstacle avoidance)

- Verdict: **CHECK THE SETUP**
  - snapshots disagree (board 8.2 mm apart, wrist reprojection 10.2 px; good: < 3 mm, < 2 px): check --square-size / --marker-size against a ruler, that the board is flat and did not move, and that the camera mount is tight
- Board position agreement between snapshots: 8.2 mm median, 16.8 mm max; 0.75 deg median
- Wrist camera reprojection: 10.2 px median
- Expect errors of about 1 cm in positions computed from these cameras until it is redone with the
  camera mount tightened (the snapshots disagree by about 0.9 deg in the camera's angle on the gripper).

## Frames and conventions

- `T_a_b` maps points from frame b into frame a (4 x 4, metres): `p_a = T_a_b @ p_b`.
- base: the robot base (libfranka `O`). ee: libfranka `O_T_EE` (the Franka Hand TCP, between the fingertips).
- camera: OpenCV optical frame of the colour sensor (x right, y down, z forward); depth is aligned to colour.
- Wrist camera (cam_1) in the base frame at any moment: `T_base_cam1 = O_T_EE @ T_ee_cam_cam_1`.
- Static camera (cam_2) in the base frame: `T_base_cam_cam_2` (fixed).
- A pixel (u, v) with depth z [m] (aligned depth PNG/topic is in mm): `p_cam = [(u - cx) z / fx, (v - cy) z / fy, z]`.

## Values

### Wrist camera (cam_1) on the gripper: `T_ee_cam_cam_1`

```
    [  0.00746,  -0.90972,  -0.41515,   0.06212]
    [  0.99997,   0.00614,   0.00451,  -0.00394]
    [ -0.00156,  -0.41517,   0.90974,  -0.07828]
    [  0.00000,   0.00000,   0.00000,   1.00000]
```

### Static camera (cam_2) in the base frame: `T_base_cam_cam_2`

```
    [  0.00876,   0.55052,  -0.83478,   1.21919]
    [  0.99885,  -0.04419,  -0.01866,   0.08726]
    [ -0.04716,  -0.83365,  -0.55027,   0.68459]
    [  0.00000,   0.00000,   0.00000,   1.00000]
```

### Calibration board (on the table) in the base frame: `T_base_tag`

```
    [ -0.99895,  -0.03301,   0.03163,   0.75055]
    [ -0.03368,   0.99922,  -0.02084,  -0.11100]
    [ -0.03092,  -0.02189,  -0.99928,   0.02841]
    [  0.00000,   0.00000,   0.00000,   1.00000]
```

Table "up" (normal) in the base frame: `[-0.0316, 0.0208, 0.9993]` (2.2 deg from the base z axis).

### Intrinsics (colour, 640 x 480, no distortion)

- cam_1: fx 614.85, fy 615.16, cx 322.13, cy 243.16; depth scale 0.0010000000474974513 m per unit
- cam_2: fx 606.27, fy 606.02, cx 324.70, cy 245.32; depth scale 0.0010000000474974513 m per unit
