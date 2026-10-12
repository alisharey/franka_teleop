# One 1 kHz control tick of gamepad_teleop: the full libfranka robot state, the
# teleop controller's internal signals, the raw gamepad and the gripper.
# RobotState fields keep libfranka's names (franka::RobotState), in snake_case.
# Frames: O = robot base, F = flange, NE = nominal end effector, EE = end
# effector, K = stiffness frame. Matrices are 4x4 column-major, as in libfranka.

std_msgs/Header header        # stamp: system (ROS) time at the control tick
uint64 tick                   # control tick since the motion session started
float64 period                # time since the previous tick [s]
uint64 dropped                # ticks lost in the telemetry buffer so far (cumulative)

# ---- franka::RobotState ----
float64 robot_time            # RobotState::time [s], the robot's own clock
float64[16] o_t_ee
float64[16] o_t_ee_d
float64[16] f_t_ee
float64[16] f_t_ne
float64[16] ne_t_ee
float64[16] ee_t_k
float64 m_ee
float64[9] i_ee
float64[3] f_x_cee
float64 m_load
float64[9] i_load
float64[3] f_x_cload
float64 m_total
float64[9] i_total
float64[3] f_x_ctotal
float64[2] elbow
float64[2] elbow_d
float64[2] elbow_c
float64[2] delbow_c
float64[2] ddelbow_c
float64[7] tau_j
float64[7] tau_j_d
float64[7] dtau_j
float64[7] q
float64[7] q_d
float64[7] dq
float64[7] dq_d
float64[7] ddq_d
float64[7] joint_contact
float64[6] cartesian_contact
float64[7] joint_collision
float64[6] cartesian_collision
float64[7] tau_ext_hat_filtered
float64[6] o_f_ext_hat_k
float64[6] k_f_ext_hat_k
float64[6] o_dp_ee_d
float64[3] o_ddp_o
float64[16] o_t_ee_c
float64[6] o_dp_ee_c
float64[6] o_ddp_ee_c
float64[7] theta
float64[7] dtheta
float32[18] accelerometer_top      # 6 sensors x 3 axes, row-major
float32[18] accelerometer_bottom   # 6 sensors x 3 axes, row-major
string current_errors              # libfranka's "[name, ...]" list; "[]" when none
string last_motion_errors
float64 control_command_success_rate
uint8 robot_mode                   # franka::RobotMode, see constants
uint8 ROBOT_MODE_OTHER=0
uint8 ROBOT_MODE_IDLE=1
uint8 ROBOT_MODE_MOVE=2
uint8 ROBOT_MODE_GUIDING=3
uint8 ROBOT_MODE_REFLEX=4
uint8 ROBOT_MODE_USER_STOPPED=5
uint8 ROBOT_MODE_AUTOMATIC_ERROR_RECOVERY=6

# ---- Teleop controller ----
bool deadman                  # RB held
bool rotation_mode            # L1 or L2 held: sticks rotate
bool motion_enabled           # sticks are being applied (deadman, no stop, past startup hold)
float64 speed_scale           # D-pad speed level (0.25, 0.5 or 1.0)
float64[6] target_twist       # from the sticks after workspace braking; base frame O [m/s, rad/s]
float64[6] shaped_twist       # after the Cartesian S-curve: what the IK is asked for
float64[7] ik_dq              # IK joint velocities after the workspace fence [rad/s]
float64 fence_scale           # factor the fence applied to the IK solution (1 = none)
float64 nullspace_weight      # posture-pull activity (0..1)
bool ik_strict_used           # the exact-direction solution replaced the detour one
float64[7] commanded_dq       # joint velocities sent to the robot after shaping and rate limiting

# ---- Gamepad: raw Linux joystick values ----
int16[] axes                  # -32767..32767
bool[] buttons

# ---- Parallel gripper ----
uint8 gripper_command         # last requested action, see constants
uint8 GRIPPER_NONE=0
uint8 GRIPPER_OPEN=1
uint8 GRIPPER_GRASP=2
uint64 gripper_command_count  # increments on every open/grasp request
float64 gripper_width         # measured finger opening [m]; 0 until the first reading
float64 gripper_max_width
bool gripper_is_grasped
uint16 gripper_temperature
float64 gripper_time          # gripper's own clock [s] of that reading; 0 = no reading yet
