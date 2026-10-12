# Meta Quest teleoperation, one message per 1 kHz control tick (input = quest).
# Joins /teleop/state by `tick` (same header stamp).
# All 4x4 matrices here are ROW-major (as oculus_reader and TeleMoMa use them),
# unlike TeleopState's libfranka matrices.

std_msgs/Header header
uint64 tick

# ---- Latest controller reading (oculus_reader app on the headset) ----
bool reading_valid            # a reading has arrived
uint64 reading_sequence       # increments per reading (~70 Hz)
float64 reading_age           # seconds since that reading arrived
builtin_interfaces/Time reading_stamp
bool right_present            # the app reports the right controller
bool pose_valid               # a real right pose arrived (see the left-controller note in the README)
float64[16] controller_pose   # right controller in the headset frame
bool button_a
bool button_b
bool trigger_pressed          # RTr (index trigger > 0.25)
bool grip_pressed             # RG (grip > 0.25): the clutch
bool joystick_click           # RJ (not sent by the Quest 3 app)
bool trigger_touch            # RT
float64 trigger               # 0..1
float64 grip                  # 0..1
float64[2] joystick           # x, y in -1..1

# ---- TeleMoMa action (src/telemoma_policy.h, = telemoma/human_interface/oculus.py) ----
bool policy_active            # the policy has a controller pose
bool movement_enabled         # grip held (TeleMoMa's movement_enabled)
float64[7] action             # [dx dy dz droll dpitch dyaw gripper], base frame, clipped to +-1;
                              # rotation as scipy "xyz" Euler; gripper 1 = open, 0 = closed
float64[16] target            # robot pose the action points to: position + action, R_euler(action) R_robot
float64[16] robot_origin      # robot pose at the last grip press/release
float64[16] vr_state          # controller pose in the robot-aligned VR frame (reorder @ vr_to_global @ pose)
float64[16] vr_origin         # vr_state at the last grip press/release
float64[16] vr_to_global      # inverse controller pose at the first grip press
