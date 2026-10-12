// Replays a recorded episode on the robot from delta end-effector pose actions
// (written by scripts/episode_actions; run both through ./replay.sh).
//
//   1. Moves the arm in joint space to the episode's first joint positions.
//   2. Executes the actions at their rate. Each action moves the gripper target
//      by (dp, dr) in the base frame; the target is tracked with feed-forward
//      velocity plus a pose correction, through the same kind of pipeline as the
//      teleop: Cartesian S-curve -> differential IK (diff_ik.h) -> workspace
//      fence -> joint S-curve -> libfranka rate limiter.
//
// --mode accumulate    each target = previous target + action (open-loop plan)
// --mode from-current  each action is applied to the measured pose at the start
//                      of its step, like a policy predicting deltas from what it sees
//
// The arm moves only while RB is held (both phases); releasing RB pauses and
// freezes the replay clock. Ctrl-C stops. This program is independent of
// gamepad_teleop; the helpers copied from it are marked "as in gamepad_teleop".

#include <fcntl.h>
#include <linux/joystick.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <unistd.h>

#include <franka/control_types.h>
#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/lowpass_filter.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>

#include "diff_ik.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
volatile sig_atomic_t g_stop_requested = 0;
void request_stop(int) { g_stop_requested = 1; }

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<double, 9>;  // row-major
using Vec6 = std::array<double, 6>;
using Vec7 = std::array<double, 7>;

// ------------------------------------------------------------------ rotations

Mat3 multiply(const Mat3& a, const Mat3& b) {
  Mat3 c{};
  for (int r = 0; r < 3; ++r) {
    for (int k = 0; k < 3; ++k) {
      for (int col = 0; col < 3; ++col) {
        c[r * 3 + col] += a[r * 3 + k] * b[k * 3 + col];
      }
    }
  }
  return c;
}

Mat3 transpose(const Mat3& a) {
  return {a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]};
}

Mat3 rotation_exp(const Vec3& w) {
  const double angle = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  if (angle < 1e-12) {
    return {1, 0, 0, 0, 1, 0, 0, 0, 1};
  }
  const double x = w[0] / angle, y = w[1] / angle, z = w[2] / angle;
  const double s = std::sin(angle), c = std::cos(angle), t = 1.0 - c;
  return {c + x * x * t,     x * y * t - z * s, x * z * t + y * s,
          y * x * t + z * s, c + y * y * t,     y * z * t - x * s,
          z * x * t - y * s, z * y * t + x * s, c + z * z * t};
}

// Rotation vector; only used for small rotations (errors, deltas), well away from pi.
Vec3 rotation_log(const Mat3& r) {
  const double cosine = std::clamp((r[0] + r[4] + r[8] - 1.0) / 2.0, -1.0, 1.0);
  const double angle = std::acos(cosine);
  const Vec3 vee{r[7] - r[5], r[2] - r[6], r[3] - r[1]};
  const double factor = angle < 1e-7 ? 0.5 : angle / (2.0 * std::sin(angle));
  return {vee[0] * factor, vee[1] * factor, vee[2] * factor};
}

Mat3 from_quaternion(double x, double y, double z, double w) {
  return {1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
          2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
          2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)};
}

double norm(const Vec3& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

struct Pose {
  Vec3 p{};
  Mat3 R{1, 0, 0, 0, 1, 0, 0, 0, 1};
};

Pose pose_from(const std::array<double, 16>& t) {  // libfranka: column-major 4x4
  return Pose{{t[12], t[13], t[14]}, {t[0], t[4], t[8], t[1], t[5], t[9], t[2], t[6], t[10]}};
}

// base moved by fraction * (dp, dr), rotation applied in the base frame.
Pose apply(const Pose& base, const Vec3& dp, const Vec3& dr, double fraction) {
  Pose result;
  for (int i = 0; i < 3; ++i) {
    result.p[i] = base.p[i] + fraction * dp[i];
  }
  result.R = multiply(rotation_exp({dr[0] * fraction, dr[1] * fraction, dr[2] * fraction}), base.R);
  return result;
}

Vec3 rotation_error(const Mat3& desired, const Mat3& measured) {
  return rotation_log(multiply(desired, transpose(measured)));
}

// ------------------------------------------------------------------ actions file

struct Action {
  Vec3 dp{};
  Vec3 dr{};
  int gripper{};
  Vec3 recorded_p{};
  Mat3 recorded_R{};
};

struct Episode {
  std::map<std::string, std::string> header;
  std::vector<Action> rows;  // steps + 1 (the last row only carries the final recorded pose)

  const std::string& value(const std::string& key) const {
    const auto found = header.find(key);
    if (found == header.end()) {
      throw std::runtime_error("actions file is missing '" + key + "'");
    }
    return found->second;
  }
  double number(const std::string& key) const { return std::stod(value(key)); }
  template <size_t N>
  std::array<double, N> numbers(const std::string& key) const {
    std::array<double, N> result{};
    std::stringstream stream(value(key));
    std::string field;
    for (size_t i = 0; i < N; ++i) {
      if (!std::getline(stream, field, ',')) {
        throw std::runtime_error("'" + key + "' needs " + std::to_string(N) + " values");
      }
      result[i] = std::stod(field);
    }
    return result;
  }
  bool flag(const std::string& key) const { return value(key) == "true"; }
};

std::string trim(const std::string& text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  return first == std::string::npos ? "" : text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

Episode load_episode(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot open actions file '" + path + "'");
  }
  Episode episode;
  std::string line;
  while (std::getline(stream, line) && trim(line) != "[actions]") {
    line = trim(line.substr(0, line.find('#')));
    const auto separator = line.find('=');
    if (separator != std::string::npos) {
      episode.header[trim(line.substr(0, separator))] = trim(line.substr(separator + 1));
    }
  }
  std::getline(stream, line);  // column names
  while (std::getline(stream, line)) {
    if (trim(line).empty()) {
      continue;
    }
    std::stringstream fields(line);
    std::string field;
    std::vector<double> v;
    while (std::getline(fields, field, ',')) {
      v.push_back(std::stod(field));
    }
    if (v.size() != 15) {
      throw std::runtime_error("bad actions row: " + line);
    }
    Action action;
    action.dp = {v[1], v[2], v[3]};
    action.dr = {v[4], v[5], v[6]};
    action.gripper = static_cast<int>(v[7]);
    action.recorded_p = {v[8], v[9], v[10]};
    action.recorded_R = from_quaternion(v[11], v[12], v[13], v[14]);
    episode.rows.push_back(action);
  }
  if (episode.rows.size() != static_cast<size_t>(episode.number("steps")) + 1) {
    throw std::runtime_error("actions file row count does not match 'steps'");
  }
  return episode;
}

// ------------------------------------------------------------------ gamepad (RB only)

class Deadman {
 public:
  Deadman(const std::string& path, int timeout_ms) : timeout_ms_(timeout_ms) {
    descriptor_ = open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (descriptor_ < 0) {
      throw std::runtime_error("cannot open joystick '" + path + "': " + std::strerror(errno));
    }
    unsigned char buttons = 0;
    ioctl(descriptor_, JSIOCGBUTTONS, &buttons);
    if (buttons <= kRightBumper) {
      throw std::runtime_error("joystick has no RB button (button 7)");
    }
    heartbeat_ms_.store(now_ms());
    reader_ = std::thread([this] { run(); });
  }
  ~Deadman() {
    stopping_.store(true);
    reader_.join();
    close(descriptor_);
  }
  bool pressed() const { return pressed_.load(); }
  bool healthy() const { return !faulted_.load() && now_ms() - heartbeat_ms_.load() <= timeout_ms_; }

 private:
  static constexpr int kRightBumper = 7;  // as in gamepad_teleop (xbox_android_standard)
  static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  void run() {
    while (!stopping_.load()) {
      pollfd descriptor_poll{descriptor_, POLLIN, 0};
      const int result = poll(&descriptor_poll, 1, 50);
      heartbeat_ms_.store(now_ms());
      if (result < 0 && errno == EINTR) {
        continue;
      }
      if (result < 0 || (descriptor_poll.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        faulted_.store(true);
        return;
      }
      js_event event{};
      while (read(descriptor_, &event, sizeof(event)) == sizeof(event)) {
        if ((event.type & ~JS_EVENT_INIT) == JS_EVENT_BUTTON && event.number == kRightBumper) {
          pressed_.store(event.value != 0);
        }
      }
    }
  }
  int descriptor_{-1};
  int timeout_ms_;
  std::atomic<bool> pressed_{false};
  std::atomic<bool> stopping_{false};
  std::atomic<bool> faulted_{false};
  std::atomic<int64_t> heartbeat_ms_{0};
  std::thread reader_;
};

// ------------------------------------------------------------------ gripper

class GripperWorker {
 public:
  GripperWorker(const Episode& episode) : episode_(episode) {
    thread_ = std::thread([this] { run(); });
  }
  ~GripperWorker() {
    stopping_.store(true);
    thread_.join();
  }
  void request(int closed) {  // 1 = grasp, 0 = open
    desired_.store(closed);
    sequence_.fetch_add(1);
  }
  bool faulted() const { return faulted_.load(); }

 private:
  void run() {
    try {
      franka::Gripper gripper(episode_.value("robot_ip"));
      uint64_t done = 0;
      while (!stopping_.load()) {
        const uint64_t sequence = sequence_.load();
        if (sequence != done) {
          done = sequence;
          const bool close = desired_.load() == 1;
          try {
            const bool ok = close ? gripper.grasp(episode_.number("gripper_grasp_width_m"),
                                                  episode_.number("gripper_grasp_speed_m_s"),
                                                  episode_.number("gripper_grasp_force_N"),
                                                  episode_.number("gripper_grasp_epsilon_inner_m"),
                                                  episode_.number("gripper_grasp_epsilon_outer_m"))
                                  : gripper.move(episode_.number("gripper_open_width_m"),
                                                 episode_.number("gripper_open_speed_m_s"));
            if (!ok) {
              std::cerr << "gripper: " << (close ? "grasp" : "open") << " did not reach its target\n";
            }
          } catch (const franka::CommandException& error) {
            std::cerr << "gripper command rejected: " << error.what() << '\n';
          }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    } catch (const std::exception& error) {
      std::cerr << "gripper fault: " << error.what() << '\n';
      faulted_.store(true);
    }
  }
  const Episode& episode_;
  std::atomic<int> desired_{0};
  std::atomic<uint64_t> sequence_{0};
  std::atomic<bool> stopping_{false};
  std::atomic<bool> faulted_{false};
  std::thread thread_;
};

// ------------------------------------------------------------------ as in gamepad_teleop

constexpr Vec7 kJointLowerRad{-2.7501, -1.7918, -2.9065, -3.0481, -2.8101, 0.54092, -3.0196};
constexpr Vec7 kJointUpperRad{2.7501, 1.7918, 2.9065, -0.1458, 2.8101, 4.5205, 3.0196};
constexpr double kJointLimitMarginRad = 0.05;
constexpr double kMaximumJointSpeedRad_s = 1.0;
constexpr double kNullspaceFullActivity = 0.1;
constexpr double kWorkspaceMarginM = 0.02;
constexpr double kBrakingAccelerationFraction = 0.5;
constexpr double kShaperJerkMargin = 0.8;
constexpr double kJointShaperAccelerationRad_s2 = 5.0;
constexpr double kJointShaperJerkRad_s3 = 500.0;

struct Limits {
  double linear_speed, angular_speed;
  double linear_acceleration, linear_jerk, angular_acceleration, angular_jerk;
  Vec3 workspace_min, workspace_max;
};

bool kernel_is_realtime() {
  std::ifstream realtime("/sys/kernel/realtime");
  int value = 0;
  return static_cast<bool>(realtime >> value) && value == 1;
}

double stoppable_speed(double distance_m, double max_acceleration, double jerk) {
  const double usable_m = distance_m - kWorkspaceMarginM;
  if (usable_m <= 0.0) {
    return 0.0;
  }
  const double acceleration = max_acceleration * kBrakingAccelerationFraction;
  const double ramp_s = acceleration / jerk;
  return acceleration * (-ramp_s + std::sqrt(ramp_s * ramp_s + 2.0 * usable_m / acceleration));
}

void shape_group(const Vec6& target, const Vec6& last_velocity, const Vec6& last_acceleration, size_t first,
                 double max_acceleration, double max_jerk, Vec6& shaped, Vec6& shaped_acceleration) {
  constexpr double kDeltaT = 1e-3;
  Vec3 error{}, acceleration{};
  double error_norm = 0.0, acceleration_norm = 0.0;
  for (size_t axis = 0; axis < 3; ++axis) {
    error[axis] = target[first + axis] - last_velocity[first + axis];
    acceleration[axis] = last_acceleration[first + axis];
    error_norm += error[axis] * error[axis];
    acceleration_norm += acceleration[axis] * acceleration[axis];
  }
  error_norm = std::sqrt(error_norm);
  acceleration_norm = std::sqrt(acceleration_norm);
  if (error_norm < 5.0 * max_jerk * kDeltaT * kDeltaT && acceleration_norm < max_jerk * kDeltaT) {
    for (size_t axis = 0; axis < 3; ++axis) {
      shaped[first + axis] = target[first + axis];
      shaped_acceleration[first + axis] = 0.0;
    }
    return;
  }
  const double desired_magnitude =
      error_norm > 0.0 ? std::min(max_acceleration, std::sqrt(2.0 * kShaperJerkMargin * max_jerk * error_norm)) : 0.0;
  Vec3 jerk{};
  double jerk_norm = 0.0;
  for (size_t axis = 0; axis < 3; ++axis) {
    const double desired = error_norm > 0.0 ? error[axis] / error_norm * desired_magnitude : 0.0;
    jerk[axis] = (desired - acceleration[axis]) / kDeltaT;
    jerk_norm += jerk[axis] * jerk[axis];
  }
  jerk_norm = std::sqrt(jerk_norm);
  const double jerk_scale = jerk_norm > max_jerk ? max_jerk / jerk_norm : 1.0;
  double new_acceleration_norm = 0.0;
  for (size_t axis = 0; axis < 3; ++axis) {
    acceleration[axis] += jerk[axis] * jerk_scale * kDeltaT;
    new_acceleration_norm += acceleration[axis] * acceleration[axis];
  }
  new_acceleration_norm = std::sqrt(new_acceleration_norm);
  const double acceleration_scale =
      new_acceleration_norm > max_acceleration ? max_acceleration / new_acceleration_norm : 1.0;
  for (size_t axis = 0; axis < 3; ++axis) {
    shaped_acceleration[first + axis] = acceleration[axis] * acceleration_scale;
    shaped[first + axis] = last_velocity[first + axis] + shaped_acceleration[first + axis] * kDeltaT;
  }
}

class TwistShaper {
 public:
  Vec6 step(const Vec6& target, const Limits& limits) {
    Vec6 shaped{}, shaped_acceleration{};
    shape_group(target, velocity_, acceleration_, 0, limits.linear_acceleration, limits.linear_jerk, shaped,
                shaped_acceleration);
    shape_group(target, velocity_, acceleration_, 3, limits.angular_acceleration, limits.angular_jerk, shaped,
                shaped_acceleration);
    acceleration_ = shaped_acceleration;
    velocity_ = shaped;
    return shaped;
  }

 private:
  Vec6 velocity_{};
  Vec6 acceleration_{};
};

Vec7 shape_joint_velocity(const Vec7& target, const franka::RobotState& state) {
  constexpr double kDeltaT = 1e-3;
  constexpr double a_max = kJointShaperAccelerationRad_s2;
  constexpr double j_max = kJointShaperJerkRad_s3;
  Vec7 shaped{};
  for (size_t joint = 0; joint < 7; ++joint) {
    const double velocity = state.dq_d[joint];
    const double acceleration = state.ddq_d[joint];
    const double error = target[joint] - velocity;
    if (std::abs(error) < 2.0 * j_max * kDeltaT * kDeltaT && std::abs(acceleration) < j_max * kDeltaT) {
      shaped[joint] = target[joint];
      continue;
    }
    const double desired = std::copysign(std::min(a_max, std::sqrt(2.0 * kShaperJerkMargin * j_max * std::abs(error))), error);
    const double jerk = std::clamp((desired - acceleration) / kDeltaT, -j_max, j_max);
    const double new_acceleration = std::clamp(acceleration + jerk * kDeltaT, -a_max, a_max);
    shaped[joint] = velocity + new_acceleration * kDeltaT;
  }
  return shaped;
}

double fence_scale(const franka::RobotState& state, const Limits& limits, const std::array<double, 42>& jacobian,
                   const Vec7& joint_velocity) {
  double scale = 1.0;
  for (size_t axis = 0; axis < 3; ++axis) {
    double velocity = 0.0;
    for (size_t joint = 0; joint < 7; ++joint) {
      velocity += jacobian[joint * 6 + axis] * joint_velocity[joint];
    }
    const double position = state.O_T_EE[12 + axis];
    const double allowed =
        velocity > 0.0
            ? stoppable_speed(limits.workspace_max[axis] - position, limits.linear_acceleration, limits.linear_jerk)
            : stoppable_speed(position - limits.workspace_min[axis], limits.linear_acceleration, limits.linear_jerk);
    if (std::abs(velocity) > allowed) {
      scale = std::min(scale, allowed / std::abs(velocity));
    }
  }
  return scale;
}

template <size_t N>
bool near_zero(const std::array<double, N>& values, double tolerance = 1e-6) {
  return std::all_of(values.begin(), values.end(), [tolerance](double v) { return std::abs(v) < tolerance; });
}

// Robot velocity bounds evaluated kJointLimitMarginRad early, capped at `cap`.
void joint_bounds(franka::Robot& robot, const franka::RobotState& state, double cap, Vec7& lower, Vec7& upper) {
  Vec7 toward_upper{}, toward_lower{};
  for (size_t joint = 0; joint < 7; ++joint) {
    toward_upper[joint] = state.q_d[joint] + kJointLimitMarginRad;
    toward_lower[joint] = state.q_d[joint] - kJointLimitMarginRad;
  }
  upper = robot.getUpperJointVelocityLimits(toward_upper);
  lower = robot.getLowerJointVelocityLimits(toward_lower);
  for (size_t joint = 0; joint < 7; ++joint) {
    upper[joint] = std::min(upper[joint], cap);
    lower[joint] = std::max(lower[joint], -cap);
  }
}

Vec7 rate_limited(franka::Robot& robot, const franka::RobotState& state, const Vec7& target) {
  return franka::limitRate(robot.getUpperJointVelocityLimits(state.q_d), robot.getLowerJointVelocityLimits(state.q_d),
                           franka::kMaxJointAcceleration, franka::kMaxJointJerk, shape_joint_velocity(target, state),
                           state.dq_d, state.ddq_d);
}

// ------------------------------------------------------------------ phases

bool wait_for_enter(const std::string& prompt) {
  std::cout << prompt << std::flush;
  std::string line;
  const bool ok = static_cast<bool>(std::getline(std::cin, line));
  return ok && g_stop_requested == 0;
}

// Quintic joint-space move to q_goal, paced so no joint exceeds 0.3 rad/s or 0.5 rad/s^2.
void move_to_start(franka::Robot& robot, Deadman& deadman, const Vec7& q_goal, double duration_s) {
  constexpr double kGain = 3.0;  // rad/s per rad of tracking error
  constexpr double kCap = 0.5;
  Vec7 q_from{};
  bool first = true, settling = false;
  double profile_s = 0.0;
  robot.control(
      [&](const franka::RobotState& state, franka::Duration period) -> franka::JointVelocities {
        if (first) {
          q_from = state.q;
          first = false;
        }
        const bool stopping = g_stop_requested != 0 || !deadman.healthy();
        const bool moving = deadman.pressed() && !stopping;
        if (moving) {
          profile_s = std::min(profile_s + period.toSec(), duration_s);
        }
        const double tau = profile_s / duration_s;
        const double s = tau * tau * tau * (10.0 - 15.0 * tau + 6.0 * tau * tau);
        const double ds = 30.0 * tau * tau * (1.0 - tau) * (1.0 - tau) / duration_s;
        double error = 0.0;
        Vec7 target{};
        for (size_t joint = 0; joint < 7; ++joint) {
          const double q_desired = q_from[joint] + (q_goal[joint] - q_from[joint]) * s;
          target[joint] = (q_goal[joint] - q_from[joint]) * ds + kGain * (q_desired - state.q[joint]);
          error = std::max(error, std::abs(q_goal[joint] - state.q[joint]));
        }
        settling |= tau >= 1.0 && error < 1e-3;
        if (!moving || settling) {  // RB released = stop now (as in gamepad_teleop)
          target = {};
        }
        Vec7 lower{}, upper{};
        joint_bounds(robot, state, kCap, lower, upper);
        for (size_t joint = 0; joint < 7; ++joint) {
          target[joint] = std::clamp(target[joint], lower[joint], upper[joint]);
        }
        const Vec7 limited = rate_limited(robot, state, target);
        if ((stopping || settling) && near_zero(limited, 1e-4)) {
          return franka::MotionFinished(franka::JointVelocities(limited));
        }
        return franka::JointVelocities(limited);
      },
      franka::ControllerMode::kJointImpedance, false, franka::kMaxCutoffFrequency);
}

struct StepLog {
  double t{};
  int step{};
  Vec3 recorded{};
  Vec3 measured{};
  Vec3 target{};
  double position_error_m{};   // measured vs recorded
  double rotation_error_rad{};
  double tracking_error_m{};   // measured vs this step's target
  int gripper{};
};

void print_usage(const char* name) {
  std::cout << "Usage: " << name << " --actions FILE [--mode accumulate|from-current] [--speed F]\n"
            << "       [--log FILE] [--no-gripper]\n"
            << "Normally started through ./replay.sh. The arm moves only while RB is held.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string actions_path, mode = "accumulate", log_path;
    double speed = 1.0;
    bool use_gripper = true;
    for (int i = 1; i < argc; ++i) {
      const std::string argument(argv[i]);
      if (argument == "--help" || argument == "-h") {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
      } else if (argument == "--actions" && i + 1 < argc) {
        actions_path = argv[++i];
      } else if (argument == "--mode" && i + 1 < argc) {
        mode = argv[++i];
      } else if (argument == "--speed" && i + 1 < argc) {
        speed = std::stod(argv[++i]);
      } else if (argument == "--log" && i + 1 < argc) {
        log_path = argv[++i];
      } else if (argument == "--no-gripper") {
        use_gripper = false;
      } else {
        throw std::runtime_error("invalid argument '" + argument + "'; use --help");
      }
    }
    if (actions_path.empty() || (mode != "accumulate" && mode != "from-current") || !(speed > 0.0 && speed <= 1.0)) {
      throw std::runtime_error("need --actions, --mode accumulate|from-current and 0 < --speed <= 1");
    }
    const Episode episode = load_episode(actions_path);
    const double rate_hz = episode.number("rate_hz");
    const int steps = static_cast<int>(episode.rows.size()) - 1;
    const Vec7 q_start = episode.numbers<7>("q_start");
    const auto wx = episode.numbers<2>("workspace_x_m"), wy = episode.numbers<2>("workspace_y_m"),
               wz = episode.numbers<2>("workspace_z_m");
    const Limits limits{episode.number("max_linear_speed_m_s"), episode.number("max_angular_speed_rad_s"),
                        episode.number("max_linear_acceleration_m_s2"), episode.number("max_linear_jerk_m_s3"),
                        episode.number("max_angular_acceleration_rad_s2"), episode.number("max_angular_jerk_rad_s3"),
                        {wx[0], wy[0], wz[0]}, {wx[1], wy[1], wz[1]}};
    use_gripper = use_gripper && episode.flag("gripper_enabled");
    for (size_t joint = 0; joint < 7; ++joint) {
      if (q_start[joint] < kJointLowerRad[joint] + kJointLimitMarginRad ||
          q_start[joint] > kJointUpperRad[joint] - kJointLimitMarginRad) {
        throw std::runtime_error("the episode's start joint " + std::to_string(joint + 1) + " is at its limit");
      }
    }

    struct sigaction action {};
    action.sa_handler = request_stop;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);

    Deadman deadman(episode.value("joystick_path"), std::stoi(episode.value("input_timeout_ms")));
    const bool realtime = kernel_is_realtime();
    franka::Robot robot(episode.value("robot_ip"),
                        realtime ? franka::RealtimeConfig::kEnforce : franka::RealtimeConfig::kIgnore);
    franka::RobotState current = robot.readOnce();
    if (current.robot_mode == franka::RobotMode::kReflex) {
      std::cout << "Robot is in reflex mode from a previous stop; running error recovery.\n";
      robot.automaticErrorRecovery();
      current = robot.readOnce();
    }
    franka::Model model = robot.loadModel();
    std::optional<GripperWorker> gripper;
    if (use_gripper) {
      gripper.emplace(episode);
    }

    // ---- Phase 1: move to the episode's start joints.
    double largest = 0.0;
    for (size_t joint = 0; joint < 7; ++joint) {
      largest = std::max(largest, std::abs(q_start[joint] - current.q[joint]));
    }
    const double move_s = std::max({2.0, 1.875 * largest / 0.3, std::sqrt(5.7735 * largest / 0.5)});
    std::cout << std::fixed << std::setprecision(2) << "Episode " << episode.value("episode") << ": " << steps
              << " actions at " << rate_hz << " Hz (" << steps / rate_hz << " s at full speed), mode " << mode
              << ", speed x" << speed << (use_gripper ? "" : ", gripper off") << ".\n"
              << "Start pose: largest joint change " << largest << " rad, about " << move_s << " s.\n";
    if (!wait_for_enter("Press Enter, then HOLD RB to move to the start pose (release RB pauses, Ctrl-C quits): ")) {
      return EXIT_SUCCESS;
    }
    if (gripper) {
      gripper->request(episode.rows[0].gripper);
    }
    move_to_start(robot, deadman, q_start, move_s);
    if (g_stop_requested != 0) {
      std::cout << "Stopped.\n";
      return EXIT_SUCCESS;
    }
    if (!deadman.healthy()) {
      throw std::runtime_error("joystick fault");
    }
    const Pose start = pose_from(robot.readOnce().O_T_EE);
    std::cout << "At the start pose (" << std::setprecision(1)
              << norm({start.p[0] - episode.rows[0].recorded_p[0], start.p[1] - episode.rows[0].recorded_p[1],
                       start.p[2] - episode.rows[0].recorded_p[2]}) * 1000
              << " mm from the recorded one).\n";

    // ---- Phase 2: replay.
    if (!wait_for_enter("Press Enter, then HOLD RB to replay (release RB pauses, Ctrl-C stops): ")) {
      return EXIT_SUCCESS;
    }
    constexpr double kPositionGain = 4.0;  // 1/s
    constexpr double kRotationGain = 4.0;  // 1/s
    // Recorded motion reached slightly above the teleop caps (IK detours); allow headroom.
    const double linear_cap = 1.5 * limits.linear_speed, angular_cap = 1.5 * limits.angular_speed;
    diff_ik::Params ik;
    ik.q_lower = kJointLowerRad;
    ik.q_upper = kJointUpperRad;
    ik.max_task_error = episode.flag("allow_detour") ? 1e9 : 0.05;
    diff_ik::Params ik_strict = ik;
    ik_strict.max_task_error = 0.05;
    TwistShaper shaper;
    std::vector<StepLog> log;
    log.reserve(static_cast<size_t>(steps) + 2);
    double clock_steps = 0.0, elapsed = 0.0, end_time = -1.0;
    int step = -1;
    int gripper_state = episode.rows[0].gripper;
    Pose step_base;
    bool finishing = false;
    const bool accumulate = mode == "accumulate";

    auto record = [&](const Pose& measured) {
      const Action& row = episode.rows[static_cast<size_t>(step)];
      StepLog entry;
      entry.t = elapsed;
      entry.step = step;
      entry.recorded = row.recorded_p;
      entry.measured = measured.p;
      entry.target = step_base.p;
      entry.position_error_m = norm({measured.p[0] - row.recorded_p[0], measured.p[1] - row.recorded_p[1],
                                     measured.p[2] - row.recorded_p[2]});
      entry.rotation_error_rad = norm(rotation_error(row.recorded_R, measured.R));
      entry.tracking_error_m = norm({measured.p[0] - step_base.p[0], measured.p[1] - step_base.p[1],
                                     measured.p[2] - step_base.p[2]});
      entry.gripper = row.gripper;
      log.push_back(entry);  // capacity reserved: no allocation in the loop
    };

    robot.control(
        [&](const franka::RobotState& state, franka::Duration period) -> franka::JointVelocities {
          elapsed += period.toSec();
          const Pose measured = pose_from(state.O_T_EE);
          if (step < 0) {
            step = 0;
            step_base = measured;
            record(measured);
          }
          const bool stopping = g_stop_requested != 0 || !deadman.healthy() || (gripper && gripper->faulted());
          const bool running = deadman.pressed() && !stopping && !finishing;
          if (running && step < steps) {
            clock_steps = std::min(clock_steps + period.toSec() * rate_hz * speed, static_cast<double>(steps));
          }
          while (step < static_cast<int>(std::floor(clock_steps)) && step < steps) {
            const Action& done = episode.rows[static_cast<size_t>(step)];
            step_base = accumulate ? apply(step_base, done.dp, done.dr, 1.0) : measured;
            ++step;
            record(measured);
            const int wanted = episode.rows[static_cast<size_t>(step)].gripper;
            if (gripper && wanted != gripper_state) {
              gripper->request(wanted);
            }
            gripper_state = wanted;
          }

          // Target pose along the current action, with its velocity as feed-forward.
          Vec6 target_twist{};
          Pose desired = step_base;
          Vec6 feed_forward{};
          if (step < steps) {
            const Action& current = episode.rows[static_cast<size_t>(step)];
            desired = apply(step_base, current.dp, current.dr, clock_steps - step);
            if (running) {
              const double per_second = rate_hz * speed;
              feed_forward = {current.dp[0] * per_second, current.dp[1] * per_second, current.dp[2] * per_second,
                              current.dr[0] * per_second, current.dr[1] * per_second, current.dr[2] * per_second};
            }
          }
          const Vec3 rotation = rotation_error(desired.R, measured.R);
          const Vec3 translation{desired.p[0] - measured.p[0], desired.p[1] - measured.p[1],
                                 desired.p[2] - measured.p[2]};
          if (step >= steps && !finishing) {
            if (end_time < 0.0) {
              end_time = elapsed;
            }
            finishing = (norm(translation) < 1e-3 && norm(rotation) < 0.01) || elapsed - end_time > 3.0;
          }
          if (running) {  // RB released = stop now; the correction catches up after RB is pressed again
            for (size_t i = 0; i < 3; ++i) {
              target_twist[i] = feed_forward[i] + kPositionGain * translation[i];
              target_twist[3 + i] = feed_forward[3 + i] + kRotationGain * rotation[i];
            }
            const double linear = std::hypot(target_twist[0], target_twist[1], target_twist[2]);
            const double angular = std::hypot(target_twist[3], target_twist[4], target_twist[5]);
            for (size_t i = 0; i < 3; ++i) {
              target_twist[i] *= linear > linear_cap ? linear_cap / linear : 1.0;
              target_twist[3 + i] *= angular > angular_cap ? angular_cap / angular : 1.0;
            }
          }
          const Vec6 twist = shaper.step(target_twist, limits);

          // As in gamepad_teleop: IK with per-joint bounds, posture pull, fence, never against the push.
          Vec7 lower{}, upper{};
          joint_bounds(robot, state, kMaximumJointSpeedRad_s, lower, upper);
          const double activity = std::max(std::hypot(twist[0], twist[1], twist[2]) / limits.linear_speed,
                                           std::hypot(twist[3], twist[4], twist[5]) / limits.angular_speed);
          const double nullspace_weight = std::clamp(activity / kNullspaceFullActivity, 0.0, 1.0);
          Vec7 joint_velocity{};
          if (!near_zero(twist)) {
            const std::array<double, 42> jacobian = model.zeroJacobian(franka::Frame::kEndEffector, state);
            auto solve_fenced = [&](const diff_ik::Params& params) {
              Vec7 dq = diff_ik::solve(jacobian, twist, state.q, lower, upper, nullspace_weight, params).dq;
              const double scale = fence_scale(state, limits, jacobian, dq);
              for (double& value : dq) {
                value *= scale;
              }
              return std::pair{dq, scale};
            };
            auto progress = [&](const Vec7& dq) {
              double sum = 0.0;
              for (size_t joint = 0; joint < 7; ++joint) {
                for (size_t row = 0; row < 6; ++row) {
                  sum += jacobian[joint * 6 + row] * dq[joint] * twist[row];
                }
              }
              return sum;
            };
            auto [dq, scale] = solve_fenced(ik);
            if (ik.max_task_error > 1.0 && scale < 0.999) {
              auto strict = solve_fenced(ik_strict);
              if (progress(strict.first) > progress(dq)) {
                dq = strict.first;
              }
            }
            if (progress(dq) <= 0.0) {
              dq = {};
            }
            joint_velocity = dq;
          }
          const Vec7 limited = rate_limited(robot, state, joint_velocity);
          if ((stopping || finishing) && near_zero(twist) && near_zero(limited, 1e-4)) {
            return franka::MotionFinished(franka::JointVelocities(limited));
          }
          return franka::JointVelocities(limited);
        },
        franka::ControllerMode::kJointImpedance, false, franka::kMaxCutoffFrequency);

    // ---- Report.
    const Pose final_pose = pose_from(robot.readOnce().O_T_EE);
    double sum = 0.0, worst = 0.0, worst_rotation = 0.0, worst_tracking = 0.0;
    for (const StepLog& entry : log) {
      sum += entry.position_error_m;
      worst = std::max(worst, entry.position_error_m);
      worst_rotation = std::max(worst_rotation, entry.rotation_error_rad);
      worst_tracking = std::max(worst_tracking, entry.tracking_error_m);
    }
    const Action& last = episode.rows.back();
    const double final_error =
        norm({final_pose.p[0] - last.recorded_p[0], final_pose.p[1] - last.recorded_p[1], final_pose.p[2] - last.recorded_p[2]});
    std::cout << std::setprecision(1) << (step >= steps ? "Replay finished" : "Replay stopped") << " at step " << step
              << "/" << steps << ". Distance from the recorded path at each step: mean "
              << (log.empty() ? 0.0 : sum / static_cast<double>(log.size())) * 1000 << " mm, worst " << worst * 1000
              << " mm, worst orientation " << worst_rotation * 180.0 / kPi << " deg; final pose "
              << final_error * 1000 << " mm from the recorded end. Worst lag behind the step target "
              << worst_tracking * 1000 << " mm.\n";
    if (!log_path.empty()) {
      std::ofstream out(log_path);
      out << "t,step,rec_x,rec_y,rec_z,x,y,z,target_x,target_y,target_z,position_error_mm,rotation_error_deg,"
             "tracking_error_mm,gripper\n"
          << std::setprecision(6);
      for (const StepLog& e : log) {
        out << e.t << ',' << e.step << ',' << e.recorded[0] << ',' << e.recorded[1] << ',' << e.recorded[2] << ','
            << e.measured[0] << ',' << e.measured[1] << ',' << e.measured[2] << ',' << e.target[0] << ','
            << e.target[1] << ',' << e.target[2] << ',' << e.position_error_m * 1000 << ','
            << e.rotation_error_rad * 180.0 / kPi << ',' << e.tracking_error_m * 1000 << ',' << e.gripper << '\n';
      }
      std::cout << "Per-step log: " << log_path << '\n';
    }
    return EXIT_SUCCESS;
  } catch (const franka::ControlException& error) {
    std::cerr << "libfranka error: " << error.what() << '\n';
    return EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
