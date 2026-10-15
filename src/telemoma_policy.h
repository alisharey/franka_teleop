#pragma once

// TeleMoMa's Meta Quest action, ported line by line from
// telemoma/human_interface/oculus.py (OculusPolicy, itself adapted from DROID's
// oculus_controller.py), for one arm (the right controller). Rotations are kept
// as matrices instead of quaternions; every product is the same rotation.
//
//   on_reading()       = one pass of OculusPolicy._update_internal_state
//   calculate_action() = OculusPolicy._calculate_action (+ _process_reading)
//
// action = [dx, dy, dz, droll, dpitch, dyaw, gripper], clipped to +-1:
//   position: (vr - vr_origin) - (robot - robot_origin)
//   rotation: (R_vr R_vr_origin^T) (R_robot R_robot_origin^T)^T as scipy "xyz" Euler
//             (extrinsic x-y-z: R = Rz(yaw) Ry(pitch) Rx(roll))
//   gripper:  TeleMoMa's toggle target, 1 = open, 0 = closed
// so target = (robot position + action position, R_euler(action) R_robot), as
// TeleMoMa's robot side builds it. Origins reset on every grip press and release;
// the VR frame is the controller frame at the first grip press (the joystick click
// that re-arms it in TeleMoMa is not sent by the Quest 3 reader app).
//
// Optional (level_frame): TeleMoMa takes the VR frame from the controller's full
// orientation at the reset, so a controller held tilted (the Quest 3 app reports
// the grip pose, whose axes follow the handle, ~30-40 deg forward when held
// naturally) tilts every motion: "down" becomes down-and-forward. With
// level_frame the reset keeps only the controller's heading and takes "up" from
// the headset frame (headset upright and level). The action math is unchanged.
//
// Optional (rotation_scale): the controller's rotation since the grip press is
// scaled before it becomes the target offset (1 = TeleMoMa; 1.5 = the gripper
// turns 1.5x as far as the hand). Positions are unaffected.
//
// One deliberate difference: TeleMoMa keeps re-evaluating a pending gripper
// toggle on every action call until the next controller reading. At its 10 Hz
// that is harmless; called at 1 kHz it would flip the gripper back and forth, so
// here a toggle is applied once.
//
// Real-time safe: no allocation, no I/O.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <utility>

namespace telemoma {

using Mat4 = std::array<double, 16>;  // row-major
using Mat3 = std::array<double, 9>;   // row-major
using Vec3 = std::array<double, 3>;

inline Mat4 identity4() { return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}; }

inline Mat4 multiply(const Mat4& a, const Mat4& b) {
  Mat4 c{};
  for (int r = 0; r < 4; ++r)
    for (int k = 0; k < 4; ++k)
      for (int col = 0; col < 4; ++col) c[r * 4 + col] += a[r * 4 + k] * b[k * 4 + col];
  return c;
}

inline Mat3 multiply(const Mat3& a, const Mat3& b) {
  Mat3 c{};
  for (int r = 0; r < 3; ++r)
    for (int k = 0; k < 3; ++k)
      for (int col = 0; col < 3; ++col) c[r * 3 + col] += a[r * 3 + k] * b[k * 3 + col];
  return c;
}

inline Mat3 transpose(const Mat3& a) { return {a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]}; }

inline Mat3 rotation_of(const Mat4& m) { return {m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]}; }

// General 4x4 inverse (np.linalg.inv); false if singular.
inline bool invert(const Mat4& m, Mat4& out) {
  std::array<double, 32> a{};
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) a[r * 8 + c] = m[r * 4 + c];
    a[r * 8 + 4 + r] = 1.0;
  }
  for (int col = 0; col < 4; ++col) {
    int pivot = col;
    for (int r = col + 1; r < 4; ++r)
      if (std::abs(a[r * 8 + col]) > std::abs(a[pivot * 8 + col])) pivot = r;
    if (std::abs(a[pivot * 8 + col]) < 1e-12) return false;
    for (int c = 0; c < 8; ++c) std::swap(a[col * 8 + c], a[pivot * 8 + c]);
    const double scale = a[col * 8 + col];
    for (int c = 0; c < 8; ++c) a[col * 8 + c] /= scale;
    for (int r = 0; r < 4; ++r) {
      if (r == col) continue;
      const double factor = a[r * 8 + col];
      for (int c = 0; c < 8; ++c) a[r * 8 + c] -= factor * a[col * 8 + c];
    }
  }
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) out[r * 4 + c] = a[r * 8 + 4 + c];
  return true;
}

// vec_to_reorder_mat: row i has sign(vec[i]) in column |vec[i]| - 1.
inline Mat4 reorder_matrix(const std::array<int, 4>& vec) {
  Mat4 x{};
  for (int i = 0; i < 4; ++i) x[i * 4 + (std::abs(vec[i]) - 1)] = vec[i] > 0 ? 1.0 : -1.0;
  return x;
}

// scipy Rotation.as_euler("xyz") (extrinsic): R = Rz(c) Ry(b) Rx(a) -> (a, b, c).
inline Vec3 euler_xyz(const Mat3& r) {
  const double sine = std::clamp(-r[6], -1.0, 1.0);
  const double pitch = std::asin(sine);
  if (std::abs(sine) < 1.0 - 1e-9) {
    return {std::atan2(r[7], r[8]), pitch, std::atan2(r[3], r[0])};
  }
  // Gimbal lock: scipy sets the third angle to zero.
  return {std::atan2(-r[5], r[4]), pitch, 0.0};
}

inline Mat3 from_euler_xyz(const Vec3& e) {
  const double ca = std::cos(e[0]), sa = std::sin(e[0]), cb = std::cos(e[1]), sb = std::sin(e[1]),
               cc = std::cos(e[2]), sc = std::sin(e[2]);
  return {cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa,
          sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa,
          -sb,     cb * sa,                cb * ca};
}

// Rotation matrix of a rotation vector (Rodrigues).
inline Mat3 rotation_exp(const Vec3& w) {
  const double angle = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  if (angle < 1e-12) return {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const double x = w[0] / angle, y = w[1] / angle, z = w[2] / angle;
  const double s = std::sin(angle), c = std::cos(angle), t = 1.0 - c;
  return {c + x * x * t, x * y * t - z * s, x * z * t + y * s,
          y * x * t + z * s, c + y * y * t, y * z * t - x * s,
          z * x * t - y * s, z * y * t + x * s, c + z * z * t};
}

// Rotation vector of a rotation matrix (for the tracking controller).
inline Vec3 rotation_vector(const Mat3& r) {
  const double cosine = std::clamp((r[0] + r[4] + r[8] - 1.0) / 2.0, -1.0, 1.0);
  const double angle = std::acos(cosine);
  const Vec3 vee{r[7] - r[5], r[2] - r[6], r[3] - r[1]};
  const double factor = angle < 1e-7 ? 0.5 : angle / (2.0 * std::sin(angle));
  return {vee[0] * factor, vee[1] * factor, vee[2] * factor};
}

struct ControllerButtons {
  bool grip{};            // 'RG'
  bool joystick_click{};  // 'RJ'
  double trigger{};       // 'rightTrig'
};

class OculusPolicy {
 public:
  explicit OculusPolicy(const std::array<int, 4>& rmat_reorder = {-2, -1, -3, 4}, double spatial_coeff = 1.0,
                        bool level_frame = false, double rotation_scale = 1.0)
      : global_to_env_(reorder_matrix(rmat_reorder)),
        spatial_coeff_(spatial_coeff),
        level_frame_(level_frame),
        rotation_scale_(rotation_scale) {}

  // One controller reading (the right controller is in `poses`).
  void on_reading(const Mat4& pose, const ControllerButtons& buttons) {
    const bool toggled = movement_enabled_ != buttons.grip;
    update_sensor_ = update_sensor_ || buttons.grip;
    reset_orientation_ = reset_orientation_ || buttons.joystick_click;
    reset_origin_ = reset_origin_ || toggled;
    pose_ = pose;
    has_pose_ = true;
    movement_enabled_ = buttons.grip;
    const bool new_gripper = buttons.trigger > 0.5;
    gripper_toggle_ = ((!prev_gripper_) && new_gripper) || gripper_toggle_;
    prev_gripper_ = new_gripper;
    const bool stop_updating = buttons.joystick_click || movement_enabled_;
    if (reset_orientation_) {
      if (stop_updating) reset_orientation_ = false;
      Mat4 inverse;
      if (invert(level_frame_ ? leveled(pose_) : pose_, inverse)) {
        vr_to_global_ = inverse;
      } else {
        vr_to_global_ = identity4();
        reset_orientation_ = true;
      }
    }
  }

  // robot_pose: end-effector pose in the base frame (row-major 4x4);
  // gripper_position: 1 = open ... 0 = closed. Requires has_pose().
  std::array<double, 7> calculate_action(const Mat4& robot_pose, double gripper_position) {
    if (update_sensor_) {
      process_reading();
      update_sensor_ = false;
    }
    const Vec3 robot_pos{robot_pose[3], robot_pose[7], robot_pose[11]};
    const Mat3 robot_rot = rotation_of(robot_pose);
    if (reset_origin_) {
      robot_origin_pos_ = robot_pos;
      robot_origin_rot_ = robot_rot;
      vr_origin_pos_ = vr_pos_;
      vr_origin_rot_ = vr_rot_;
      reset_origin_ = false;
    }
    std::array<double, 7> action{};
    for (int i = 0; i < 3; ++i) {
      action[i] = (vr_pos_[i] - vr_origin_pos_[i]) - (robot_pos[i] - robot_origin_pos_[i]);
    }
    const Mat3 robot_offset = multiply(robot_rot, transpose(robot_origin_rot_));
    Mat3 target_offset = multiply(vr_rot_, transpose(vr_origin_rot_));
    if (rotation_scale_ != 1.0) {
      Vec3 w = rotation_vector(target_offset);
      for (double& value : w) value *= rotation_scale_;
      target_offset = rotation_exp(w);
    }
    const Vec3 euler = euler_xyz(multiply(target_offset, transpose(robot_offset)));
    action[3] = euler[0];
    action[4] = euler[1];
    action[5] = euler[2];
    if (vr_gripper_toggle_) {
      target_gripper_ = 1 - (gripper_position > 0.5 ? 1 : 0);
      vr_gripper_toggle_ = false;  // applied once (see the header comment)
      gripper_toggled_ = true;
    }
    action[6] = target_gripper_;
    for (double& value : action) value = std::clamp(value, -1.0, 1.0);
    return action;
  }

  // True once after each applied trigger toggle (consumed). The target can equal
  // the previous one (e.g. the gripper was already closed at start), so a
  // command must follow every toggle, not only a change of the target.
  bool take_gripper_toggled() {
    const bool toggled = gripper_toggled_;
    gripper_toggled_ = false;
    return toggled;
  }

  bool has_pose() const { return has_pose_; }
  bool movement_enabled() const { return movement_enabled_; }
  int target_gripper() const { return target_gripper_; }
  const Mat4& vr_to_global() const { return vr_to_global_; }
  Mat4 vr_state() const { return compose(vr_rot_, vr_pos_); }
  Mat4 vr_origin() const { return compose(vr_origin_rot_, vr_origin_pos_); }
  Mat4 robot_origin() const { return compose(robot_origin_rot_, robot_origin_pos_); }

 private:
  void process_reading() {
    const Mat4 rot_mat = multiply(multiply(global_to_env_, vr_to_global_), pose_);
    vr_pos_ = {spatial_coeff_ * rot_mat[3], spatial_coeff_ * rot_mat[7], spatial_coeff_ * rot_mat[11]};
    vr_rot_ = rotation_of(rot_mat);
    vr_gripper_toggle_ = gripper_toggle_;
    gripper_toggle_ = false;
  }

  // The controller pose with its rotation reduced to a heading about the headset's
  // up axis (+y): -z = the controller's forward direction projected onto the
  // horizontal plane, +y = up, +x = right (OpenXR axis convention).
  static Mat4 leveled(const Mat4& pose) {
    double fx = -pose[2], fz = -pose[10];  // controller -z, horizontal part (x, z)
    double length = std::hypot(fx, fz);
    if (length < 1e-3) {  // pointing straight up or down: use its +y instead
      fx = pose[1];
      fz = pose[9];
      length = std::hypot(fx, fz);
      if (length < 1e-9) return pose;
    }
    fx /= length;
    fz /= length;
    // Columns: x = right = (-fz, 0, fx), y = up, z = back = -forward.
    return {-fz, 0.0, -fx, pose[3],
            0.0, 1.0, 0.0, pose[7],
            fx,  0.0, -fz, pose[11],
            0.0, 0.0, 0.0, 1.0};
  }

  static Mat4 compose(const Mat3& r, const Vec3& p) {
    return {r[0], r[1], r[2], p[0], r[3], r[4], r[5], p[1], r[6], r[7], r[8], p[2], 0, 0, 0, 1};
  }

  Mat4 global_to_env_;
  double spatial_coeff_;
  bool level_frame_;
  double rotation_scale_;
  Mat4 vr_to_global_ = identity4();
  Mat4 pose_ = identity4();
  bool has_pose_ = false;
  bool movement_enabled_ = false;
  bool prev_gripper_ = false;
  bool gripper_toggle_ = false;
  bool update_sensor_ = true;
  bool reset_origin_ = true;
  bool reset_orientation_ = true;
  int target_gripper_ = 1;
  Vec3 vr_pos_{};
  Mat3 vr_rot_{1, 0, 0, 0, 1, 0, 0, 0, 1};
  bool vr_gripper_toggle_ = false;
  bool gripper_toggled_ = false;
  Vec3 vr_origin_pos_{};
  Mat3 vr_origin_rot_{1, 0, 0, 0, 1, 0, 0, 0, 1};
  Vec3 robot_origin_pos_{};
  Mat3 robot_origin_rot_{1, 0, 0, 0, 1, 0, 0, 0, 1};
};

}  // namespace telemoma
