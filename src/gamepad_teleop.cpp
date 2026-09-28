#include <fcntl.h>
#include <linux/joystick.h>
#include <poll.h>
#include <signal.h>
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
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr size_t kMaximumAxes = 16;
constexpr size_t kMaximumButtons = 32;
constexpr int kLeftStickX = 0;
constexpr int kLeftStickY = 1;
constexpr int kRightStickX = 2;
constexpr int kRightStickY = 3;
constexpr int kLeftTrigger = 4;
// The connected Microsoft Android HID device reports its physical RB as button 7.
constexpr int kRightBumper = 7;
// Android HID layout: physical LB is Linux button 6 (RB is 7). Confirm with controller_probe.
constexpr int kLeftBumper = 6;
constexpr int kButtonA = 0;
constexpr int kButtonB = 1;
volatile sig_atomic_t g_stop_requested = 0;

void request_stop(int) { g_stop_requested = 1; }

[[nodiscard]] int64_t monotonic_time_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

[[nodiscard]] std::string trim(const std::string& value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}

[[nodiscard]] double parse_double(const std::string& key, const std::string& value) {
  size_t consumed = 0;
  const double parsed = std::stod(value, &consumed);
  if (consumed != value.size() || !std::isfinite(parsed)) {
    throw std::runtime_error("'" + key + "' must be a finite number");
  }
  return parsed;
}

[[nodiscard]] int parse_nonnegative_integer(const std::string& key, const std::string& value) {
  size_t consumed = 0;
  const long parsed = std::stol(value, &consumed);
  if (consumed != value.size() || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
    throw std::runtime_error("'" + key + "' must be a non-negative integer");
  }
  return static_cast<int>(parsed);
}

template <size_t Count>
[[nodiscard]] std::array<double, Count> parse_array(const std::string& key,
                                                      const std::string& value) {
  std::array<double, Count> result{};
  std::stringstream stream(value);
  std::string field;
  size_t index = 0;
  while (std::getline(stream, field, ',')) {
    if (index >= Count) {
      throw std::runtime_error("'" + key + "' must contain exactly " + std::to_string(Count) +
                               " comma-separated values");
    }
    result[index++] = parse_double(key, trim(field));
  }
  if (index != Count) {
    throw std::runtime_error("'" + key + "' must contain exactly " + std::to_string(Count) +
                             " comma-separated values");
  }
  return result;
}

struct Config {
  std::string robot_ip;
  std::string joystick_path;
  std::string controller_profile;
  double deadzone{};
  double max_linear_speed_m_s{};
  double max_angular_speed_rad_s{};
  double max_linear_acceleration_m_s2{};
  double max_linear_jerk_m_s3{};
  double max_angular_acceleration_rad_s2{};
  double max_angular_jerk_rad_s3{};
  double initial_zero_hold_s{};
  // Fixed workspace in the robot base frame O, as [min, max] metres per axis.
  std::array<double, 2> workspace_x_m{};
  std::array<double, 2> workspace_y_m{};
  std::array<double, 2> workspace_z_m{};
  // Downward Z speed as a fraction of max_linear_speed_m_s (Frankastein: 0.5).
  double z_down_speed_scale{};
  // Joystick axis for the D-pad vertical direction; up = faster, down = slower.
  int speed_toggle_axis{};
  // true: the IK may swing the arm off the pushed line to get around unreachable
  // straight-line paths (e.g. straight down near the base). false: it keeps the
  // exact direction and slows or stops instead.
  bool allow_detour{};
  int input_timeout_ms{};
  bool gripper_enabled{};
  double gripper_open_width_m{};
  double gripper_open_speed_m_s{};
  double gripper_grasp_width_m{};
  double gripper_grasp_speed_m_s{};
  double gripper_grasp_force_N{};
  double gripper_grasp_epsilon_inner_m{};
  double gripper_grasp_epsilon_outer_m{};
};

[[nodiscard]] std::unordered_map<std::string, std::string> read_key_values(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot open config '" + path + "'");
  }
  const std::array<std::string, 26> allowed_keys{
      "robot_ip", "joystick_path", "controller_profile", "deadzone", "max_linear_speed_m_s",
      "max_angular_speed_rad_s", "max_linear_acceleration_m_s2", "max_linear_jerk_m_s3",
      "max_angular_acceleration_rad_s2", "max_angular_jerk_rad_s3", "initial_zero_hold_s",
      "workspace_x_m", "workspace_y_m", "workspace_z_m", "z_down_speed_scale",
      "speed_toggle_axis", "allow_detour", "input_timeout_ms", "gripper_enabled", "gripper_open_width_m",
      "gripper_open_speed_m_s", "gripper_grasp_width_m", "gripper_grasp_speed_m_s",
      "gripper_grasp_force_N", "gripper_grasp_epsilon_inner_m", "gripper_grasp_epsilon_outer_m"};
  std::unordered_map<std::string, std::string> values;
  std::string line;
  size_t line_number = 0;
  while (std::getline(stream, line)) {
    ++line_number;
    const std::string uncommented = trim(line.substr(0, line.find('#')));
    if (uncommented.empty()) {
      continue;
    }
    const size_t separator = uncommented.find('=');
    if (separator == std::string::npos || uncommented.find('=', separator + 1) != std::string::npos) {
      throw std::runtime_error("config line " + std::to_string(line_number) + " must be key = value");
    }
    const std::string key = trim(uncommented.substr(0, separator));
    const std::string value = trim(uncommented.substr(separator + 1));
    if (std::find(allowed_keys.begin(), allowed_keys.end(), key) == allowed_keys.end()) {
      throw std::runtime_error("unknown config key '" + key + "'");
    }
    if (value.empty() || value == "REQUIRED") {
      throw std::runtime_error("config key '" + key + "' must be set by the operator");
    }
    if (!values.emplace(key, value).second) {
      throw std::runtime_error("duplicate config key '" + key + "'");
    }
  }
  return values;
}

[[nodiscard]] const std::string& required(const std::unordered_map<std::string, std::string>& values,
                                           const std::string& key) {
  const auto found = values.find(key);
  if (found == values.end()) {
    throw std::runtime_error("missing required config key '" + key + "'");
  }
  return found->second;
}

[[nodiscard]] bool parse_boolean(const std::string& key, const std::string& value) {
  if (value == "true") {
    return true;
  }
  if (value == "false") {
    return false;
  }
  throw std::runtime_error("'" + key + "' must be true or false");
}

[[nodiscard]] Config load_config(const std::string& path) {
  const auto values = read_key_values(path);
  Config config{
      .robot_ip = required(values, "robot_ip"),
      .joystick_path = required(values, "joystick_path"),
      .controller_profile = required(values, "controller_profile"),
      .deadzone = parse_double("deadzone", required(values, "deadzone")),
      .max_linear_speed_m_s = parse_double("max_linear_speed_m_s", required(values, "max_linear_speed_m_s")),
      .max_angular_speed_rad_s = parse_double("max_angular_speed_rad_s", required(values, "max_angular_speed_rad_s")),
      .max_linear_acceleration_m_s2 = parse_double("max_linear_acceleration_m_s2", required(values, "max_linear_acceleration_m_s2")),
      .max_linear_jerk_m_s3 = parse_double("max_linear_jerk_m_s3", required(values, "max_linear_jerk_m_s3")),
      .max_angular_acceleration_rad_s2 = parse_double("max_angular_acceleration_rad_s2", required(values, "max_angular_acceleration_rad_s2")),
      .max_angular_jerk_rad_s3 = parse_double("max_angular_jerk_rad_s3", required(values, "max_angular_jerk_rad_s3")),
      .initial_zero_hold_s = parse_double("initial_zero_hold_s", required(values, "initial_zero_hold_s")),
      .workspace_x_m = parse_array<2>("workspace_x_m", required(values, "workspace_x_m")),
      .workspace_y_m = parse_array<2>("workspace_y_m", required(values, "workspace_y_m")),
      .workspace_z_m = parse_array<2>("workspace_z_m", required(values, "workspace_z_m")),
      .z_down_speed_scale = parse_double("z_down_speed_scale", required(values, "z_down_speed_scale")),
      .speed_toggle_axis = parse_nonnegative_integer("speed_toggle_axis", required(values, "speed_toggle_axis")),
      .allow_detour = parse_boolean("allow_detour", required(values, "allow_detour")),
      .input_timeout_ms = parse_nonnegative_integer("input_timeout_ms", required(values, "input_timeout_ms")),
      .gripper_enabled = parse_boolean("gripper_enabled", required(values, "gripper_enabled")),
      .gripper_open_width_m = parse_double("gripper_open_width_m", required(values, "gripper_open_width_m")),
      .gripper_open_speed_m_s = parse_double("gripper_open_speed_m_s", required(values, "gripper_open_speed_m_s")),
      .gripper_grasp_width_m = parse_double("gripper_grasp_width_m", required(values, "gripper_grasp_width_m")),
      .gripper_grasp_speed_m_s = parse_double("gripper_grasp_speed_m_s", required(values, "gripper_grasp_speed_m_s")),
      .gripper_grasp_force_N = parse_double("gripper_grasp_force_N", required(values, "gripper_grasp_force_N")),
      .gripper_grasp_epsilon_inner_m = parse_double("gripper_grasp_epsilon_inner_m", required(values, "gripper_grasp_epsilon_inner_m")),
      .gripper_grasp_epsilon_outer_m = parse_double("gripper_grasp_epsilon_outer_m", required(values, "gripper_grasp_epsilon_outer_m"))};
  if (config.controller_profile != "xbox_android_standard") {
    throw std::runtime_error("controller_profile must be 'xbox_android_standard'");
  }
  if (config.deadzone < 0.0 || config.deadzone >= 1.0 || config.max_linear_speed_m_s <= 0.0 ||
      config.max_angular_speed_rad_s <= 0.0 ||
      config.max_linear_acceleration_m_s2 <= 0.0 || config.max_linear_jerk_m_s3 <= 0.0 ||
      config.max_angular_acceleration_rad_s2 <= 0.0 || config.max_angular_jerk_rad_s3 <= 0.0 ||
      config.initial_zero_hold_s < 0.0 || config.input_timeout_ms == 0 ||
      config.z_down_speed_scale <= 0.0 || config.z_down_speed_scale > 1.0) {
    throw std::runtime_error("deadzone must be in [0, 1); z_down_speed_scale in (0, 1]; limits and input_timeout_ms must be positive");
  }
  for (const auto& [name, bounds] : {std::pair{"workspace_x_m", config.workspace_x_m},
                                     std::pair{"workspace_y_m", config.workspace_y_m},
                                     std::pair{"workspace_z_m", config.workspace_z_m}}) {
    if (bounds[0] >= bounds[1]) {
      throw std::runtime_error(std::string("'") + name + "' must be 'min, max' with min < max");
    }
  }
  if (config.gripper_enabled &&
      (config.gripper_open_width_m <= 0.0 || config.gripper_grasp_width_m < 0.0 ||
       config.gripper_grasp_width_m >= config.gripper_open_width_m ||
       config.gripper_open_speed_m_s <= 0.0 || config.gripper_grasp_speed_m_s <= 0.0 ||
       config.gripper_grasp_force_N <= 0.0 || config.gripper_grasp_epsilon_inner_m < 0.0 ||
       config.gripper_grasp_epsilon_outer_m < 0.0)) {
    throw std::runtime_error("enabled gripper widths, speeds, force, and tolerances must define a valid bounded command");
  }
  return config;
}

class JoystickInput {
 public:
  explicit JoystickInput(const Config& config) : config_(config) {}
  ~JoystickInput() { stop(); }
  JoystickInput(const JoystickInput&) = delete;
  JoystickInput& operator=(const JoystickInput&) = delete;

  void start() {
    descriptor_ = open(config_.joystick_path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (descriptor_ < 0) {
      throw std::runtime_error("cannot open joystick '" + config_.joystick_path + "': " + std::strerror(errno));
    }
    ioctl(descriptor_, JSIOCGAXES, &axis_count_);
    ioctl(descriptor_, JSIOCGBUTTONS, &button_count_);
    if (axis_count_ > kMaximumAxes || button_count_ > kMaximumButtons ||
        kLeftStickX >= axis_count_ || kLeftStickY >= axis_count_ || kRightStickX >= axis_count_ ||
        kRightStickY >= axis_count_ || kLeftTrigger >= axis_count_ || kRightBumper >= button_count_ ||
        kLeftBumper >= button_count_ ||
        static_cast<size_t>(config_.speed_toggle_axis) >= axis_count_ ||
        (config_.gripper_enabled && (kButtonA >= button_count_ || kButtonB >= button_count_))) {
      throw std::runtime_error("configured axis/button index is not provided by joystick");
    }
    heartbeat_ms_.store(monotonic_time_ms());
    reader_ = std::thread([this] { read_loop(); });
  }

  void stop() noexcept {
    stopping_.store(true);
    if (reader_.joinable()) {
      reader_.join();
    }
    if (descriptor_ >= 0) {
      close(descriptor_);
      descriptor_ = -1;
    }
  }

  [[nodiscard]] bool deadman_pressed() const { return buttons_[kRightBumper].load(); }
  [[nodiscard]] bool rotation_modifier_pressed() const {
    // Hold L1 (left bumper) or L2 (left trigger) to rotate.
    return buttons_[kLeftBumper].load() || normalized_axis(kLeftTrigger) > 0.5;
  }
  [[nodiscard]] bool healthy() const {
    return !faulted_.load() && monotonic_time_ms() - heartbeat_ms_.load() <= config_.input_timeout_ms;
  }
  [[nodiscard]] double axis(const size_t index) const {
    const double normalized = normalized_axis(index);
    const double magnitude = std::abs(normalized);
    if (magnitude <= config_.deadzone) {
      return 0.0;
    }
    return std::copysign((magnitude - config_.deadzone) / (1.0 - config_.deadzone), normalized);
  }
  // Speed multiplier chosen with the D-pad (like Frankastein's [ / ] step keys).
  [[nodiscard]] double speed_scale() const { return kSpeedLevels[speed_level_.load()]; }
  [[nodiscard]] uint64_t take_gripper_action() { return gripper_action_sequence_.exchange(0); }
  [[nodiscard]] int gripper_action() const { return gripper_action_.load(); }
  [[nodiscard]] std::string fault_reason() const { return fault_reason_; }

 private:
  [[nodiscard]] double normalized_axis(const size_t index) const {
    return static_cast<double>(axes_[index].load()) / 32767.0;
  }
  void fail(const std::string& reason) {
    fault_reason_ = reason;
    faulted_.store(true);
  }

  // One level per D-pad press: up (negative) = faster, down (positive) = slower.
  void update_speed_level(const int16_t value, const bool initial_state) {
    constexpr int16_t kPressThreshold = 16000;
    const int direction = value <= -kPressThreshold ? 1 : (value >= kPressThreshold ? -1 : 0);
    if (direction != 0 && speed_toggle_direction_ == 0 && !initial_state) {
      const int level = std::clamp(speed_level_.load() + direction, 0,
                                   static_cast<int>(kSpeedLevels.size()) - 1);
      speed_level_.store(level);
      std::cout << "speed " << static_cast<int>(kSpeedLevels[level] * 100.0) << "%\n" << std::flush;
    }
    speed_toggle_direction_ = direction;
  }

  void read_loop() {
    while (!stopping_.load()) {
      pollfd descriptor_poll{descriptor_, POLLIN, 0};
      const int result = poll(&descriptor_poll, 1, 50);
      heartbeat_ms_.store(monotonic_time_ms());
      if (result < 0 && errno == EINTR) {
        continue;
      }
      if (result < 0 || (descriptor_poll.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        fail("joystick disconnected or poll failed");
        return;
      }
      if ((descriptor_poll.revents & POLLIN) == 0) {
        continue;
      }
      js_event event{};
      while (read(descriptor_, &event, sizeof(event)) == sizeof(event)) {
        const unsigned char event_type = event.type & static_cast<unsigned char>(~JS_EVENT_INIT);
        if (event_type == JS_EVENT_AXIS && event.number < kMaximumAxes) {
          axes_[event.number].store(event.value);
          if (event.number == config_.speed_toggle_axis) {
            update_speed_level(event.value, (event.type & JS_EVENT_INIT) != 0);
          }
        }
        if (event_type == JS_EVENT_BUTTON && event.number < kMaximumButtons) {
          const bool pressed = event.value != 0;
          buttons_[event.number].store(pressed);
          if (pressed && deadman_pressed() && config_.gripper_enabled) {
            if (event.number == kButtonA) {
              gripper_action_.store(1);
              gripper_action_sequence_.fetch_add(1);
            } else if (event.number == kButtonB) {
              gripper_action_.store(2);
              gripper_action_sequence_.fetch_add(1);
            }
          }
        }
      }
      if (errno != EAGAIN && errno != EWOULDBLOCK) {
        fail("joystick read failed");
        return;
      }
    }
  }

  const Config& config_;
  int descriptor_{-1};
  unsigned char axis_count_{};
  unsigned char button_count_{};
  std::array<std::atomic<int16_t>, kMaximumAxes> axes_{};
  std::array<std::atomic<bool>, kMaximumButtons> buttons_{};
  std::atomic<bool> stopping_{false};
  std::atomic<bool> faulted_{false};
  std::atomic<int64_t> heartbeat_ms_{0};
  std::atomic<int> gripper_action_{0};
  std::atomic<uint64_t> gripper_action_sequence_{0};
  static constexpr std::array<double, 3> kSpeedLevels{0.25, 0.5, 1.0};
  std::atomic<int> speed_level_{2};
  int speed_toggle_direction_{0};  // Reader thread only.
  std::thread reader_;
  std::string fault_reason_{""};
};

// libfranka's own check: only PREEMPT_RT kernels expose /sys/kernel/realtime = 1.
[[nodiscard]] bool kernel_is_realtime() {
  std::ifstream realtime("/sys/kernel/realtime");
  int value = 0;
  return static_cast<bool>(realtime >> value) && value == 1;
}

// Enforce realtime on a PREEMPT_RT kernel; otherwise (e.g. lowlatency) run with
// kIgnore. In both modes libfranka still tries to put the 1 kHz control thread
// on the highest SCHED_FIFO priority; kIgnore only stops it aborting when the
// kernel is not RT. Warn if this user is not allowed realtime priority at all.
[[nodiscard]] franka::RealtimeConfig choose_realtime_config() {
  const bool realtime_kernel = kernel_is_realtime();
  rlimit realtime_priority{};
  const bool may_use_fifo = geteuid() == 0 ||
                            (getrlimit(RLIMIT_RTPRIO, &realtime_priority) == 0 &&
                             realtime_priority.rlim_cur > 0);
  if (realtime_kernel) {
    std::cout << "PREEMPT_RT kernel detected: realtime scheduling enforced.\n";
  } else {
    std::cout << "Non-RT kernel (e.g. lowlatency) detected: running without the RT-kernel "
                 "requirement; the control loop still requests realtime priority.\n";
  }
  if (!may_use_fifo) {
    std::cout << "warning: this user has no realtime priority limit (ulimit -r is 0), so the "
                 "control loop runs at normal priority and may miss 1 kHz deadlines. See README.\n";
  }
  return realtime_kernel ? franka::RealtimeConfig::kEnforce : franka::RealtimeConfig::kIgnore;
}

void print_usage(const char* name) {
  std::cout << "Usage: " << name << " --config PATH [--check-config | --enable-motion]\n"
            << "\n"
            << "--check-config validates configuration only and never contacts hardware.\n"
            << "--enable-motion requires a held deadman before it opens the robot connection.\n";
}

// Fixed box in the robot base frame O (same values as Frankastein's
// configs/robots/fr3.yaml robot.limits.workspace). It does not move with the
// start pose.
struct Workspace {
  std::array<double, 3> minimum_m{};
  std::array<double, 3> maximum_m{};
};

[[nodiscard]] Workspace workspace_from_config(const Config& config) {
  return Workspace{
      .minimum_m = {config.workspace_x_m[0], config.workspace_y_m[0], config.workspace_z_m[0]},
      .maximum_m = {config.workspace_x_m[1], config.workspace_y_m[1], config.workspace_z_m[1]}};
}

void require_inside_workspace(const franka::RobotState& state, const Workspace& workspace) {
  constexpr std::array<const char*, 3> kAxisNames{"x", "y", "z"};
  for (size_t index = 0; index < 3; ++index) {
    const double position = state.O_T_EE[12 + index];
    if (position < workspace.minimum_m[index] || position > workspace.maximum_m[index]) {
      std::ostringstream message;
      message << "end effector " << kAxisNames[index] << "=" << position
              << " m is outside the workspace [" << workspace.minimum_m[index] << ", "
              << workspace.maximum_m[index] << "] m; move the arm inside before teleop";
      throw std::runtime_error(message.str());
    }
  }
}

// Fastest speed toward a wall `distance_m` away that still stops before it:
// v^2/(2a) + v*(a/j) <= d, i.e. braking distance plus the extra travel while
// deceleration ramps up under the jerk limit. Braking is planned at half the
// acceleration limit and aims kWorkspaceMarginM short of the wall, because the
// jerk-limited rate limiter lags a moving target; simulating that limiter at
// every speed level showed this keeps the arm inside the box.
constexpr double kWorkspaceMarginM = 0.02;
constexpr double kBrakingAccelerationFraction = 0.5;

[[nodiscard]] double stoppable_speed(const double distance_m, const double max_acceleration,
                                     const double jerk) {
  const double usable_m = distance_m - kWorkspaceMarginM;
  if (usable_m <= 0.0) {
    return 0.0;
  }
  const double acceleration = max_acceleration * kBrakingAccelerationFraction;
  const double ramp_s = acceleration / jerk;
  return acceleration * (-ramp_s + std::sqrt(ramp_s * ramp_s + 2.0 * usable_m / acceleration));
}

[[nodiscard]] std::array<double, 6> bounded_velocity(const franka::RobotState& state,
                                                      const Config& config, const Workspace& workspace,
                                                      const JoystickInput& input) {
  const double scale = input.speed_scale();
  const double linear = config.max_linear_speed_m_s * scale;
  const double angular = config.max_angular_speed_rad_s * scale;
  std::array<double, 6> velocity{};
  if (input.rotation_modifier_pressed()) {
    velocity = {0.0,
                0.0,
                0.0,
                input.axis(kLeftStickX) * angular,
                -input.axis(kLeftStickY) * angular,
                input.axis(kRightStickX) * angular};
  } else {
    // Right stick up/down -> base X (push down = +X, away from the base);
    // left stick left/right -> base Y (push right = +Y);
    // left stick up/down -> base Z (push up = +Z).
    velocity = {input.axis(kRightStickY) * linear,
                input.axis(kLeftStickX) * linear,
                -input.axis(kLeftStickY) * linear,
                0.0,
                0.0,
                0.0};
    if (velocity[2] < 0.0) {
      velocity[2] *= config.z_down_speed_scale;  // Frankastein moves down at half speed.
    }
  }
  for (size_t index = 0; index < 3; ++index) {
    const double position = state.O_T_EE[12 + index];
    const double to_minimum = position - workspace.minimum_m[index];
    const double to_maximum = workspace.maximum_m[index] - position;
    const double max_down = stoppable_speed(to_minimum, config.max_linear_acceleration_m_s2,
                                            config.max_linear_jerk_m_s3);
    const double max_up = stoppable_speed(to_maximum, config.max_linear_acceleration_m_s2,
                                          config.max_linear_jerk_m_s3);
    velocity[index] = std::clamp(velocity[index], -max_down, max_up);
  }
  return velocity;
}

// FR3 joint position limits from the robot's URDF (libfranka test/fr3.urdf).
// The robot enforces position-dependent velocity limits derived from these;
// the control loop reads those exact limits from libfranka each tick.
constexpr std::array<double, 7> kJointLowerRad{-2.7501, -1.7918, -2.9065, -3.0481,
                                               -2.8101, 0.54092, -3.0196};
constexpr std::array<double, 7> kJointUpperRad{2.7501, 1.7918, 2.9065, -0.1458,
                                               2.8101, 4.5205, 3.0196};
// Joints are brought to rest this far before the robot's own limit (the robot's
// velocity bound is evaluated this much closer to the limit than the joint is).
constexpr double kJointLimitMarginRad = 0.05;
// Teleop cap on any joint's speed; well under the FR3's 2.6-5.3 rad/s.
constexpr double kMaximumJointSpeedRad_s = 1.0;
// Null-space (elbow) motion fades in between 0 and this fraction of full stick,
// so the elbow never moves by itself while the sticks are idle.
constexpr double kNullspaceFullActivity = 0.1;

[[nodiscard]] diff_ik::Params ik_params(const Config& config) {
  diff_ik::Params params;  // Posture: Franka ready pose; see diff_ik.h for gains.
  params.q_lower = kJointLowerRad;
  params.q_upper = kJointUpperRad;
  // Strict mode: fall back to the exact direction (slower) if the per-joint
  // saturation step would stray more than 5 % from the pushed direction.
  params.max_task_error = config.allow_detour ? 1e9 : 0.05;
  return params;
}

void print_joint_positions(const std::array<double, 7>& q) {
  std::cerr << "joint positions (rad) vs FR3 limits:\n";
  for (size_t joint = 0; joint < 7; ++joint) {
    const double to_limit = std::min(q[joint] - kJointLowerRad[joint], kJointUpperRad[joint] - q[joint]);
    std::cerr << "  joint " << joint + 1 << ": " << q[joint] << "  [" << kJointLowerRad[joint] << ", "
              << kJointUpperRad[joint] << "]  distance to nearest limit " << to_limit
              << (to_limit < kJointLimitMarginRad ? "  <-- AT LIMIT" : "") << '\n';
  }
}

// Turns the raw stick target into a smooth jerk-limited (S-curve) twist before
// the IK. Feeding raw stick steps to a jerk-limited rate limiter makes it chase
// at full jerk, overshoot and ring (simulated: about +/-5 cm/s back-and-forth
// after releasing the stick at 1 m/s^2 and 10 m/s^3, felt as vibration). Here
// the acceleration toward the target is capped at sqrt(2 * jerk * error), so it
// reaches zero exactly as the velocity arrives and nothing overshoots. Linear
// and angular parts are shaped separately, as vectors, starting from the last
// shaped velocity and acceleration.
constexpr double kShaperJerkMargin = 0.8;  // Plan with 80 % of the jerk limit.

void shape_group(const std::array<double, 6>& target, const std::array<double, 6>& last_velocity,
                 const std::array<double, 6>& last_acceleration, size_t first, double max_acceleration,
                 double max_jerk, std::array<double, 6>& shaped,
                 std::array<double, 6>& shaped_acceleration) {
  constexpr double kDeltaT = 1e-3;
  std::array<double, 3> error{};
  std::array<double, 3> acceleration{};
  double error_norm = 0.0;
  double acceleration_norm = 0.0;
  for (size_t axis = 0; axis < 3; ++axis) {
    error[axis] = target[first + axis] - last_velocity[first + axis];
    acceleration[axis] = last_acceleration[first + axis];
    error_norm += error[axis] * error[axis];
    acceleration_norm += acceleration[axis] * acceleration[axis];
  }
  error_norm = std::sqrt(error_norm);
  acceleration_norm = std::sqrt(acceleration_norm);
  // Snap window: a few ticks' worth of jerk-limited change (5e-5 m/s linear,
  // 1.25e-4 rad/s angular at the configured limits). Closer than that, the
  // discrete S-curve would hop back and forth around the target instead.
  if (error_norm < 5.0 * max_jerk * kDeltaT * kDeltaT && acceleration_norm < max_jerk * kDeltaT) {
    // Arrived: hold the target exactly with zero acceleration (otherwise the
    // leftover acceleration of the snap step starts a tiny limit cycle).
    for (size_t axis = 0; axis < 3; ++axis) {
      shaped[first + axis] = target[first + axis];
      shaped_acceleration[first + axis] = 0.0;
    }
    return;
  }
  const double desired_magnitude =
      error_norm > 0.0
          ? std::min(max_acceleration, std::sqrt(2.0 * kShaperJerkMargin * max_jerk * error_norm))
          : 0.0;
  std::array<double, 3> jerk{};
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

// Keeps the shaped twist and its acceleration between control ticks. (In joint
// velocity mode the robot does not report a commanded Cartesian velocity.)
class TwistShaper {
 public:
  [[nodiscard]] std::array<double, 6> step(const std::array<double, 6>& target, const Config& config) {
    std::array<double, 6> shaped{};
    std::array<double, 6> shaped_acceleration{};
    shape_group(target, velocity_, acceleration_, 0, config.max_linear_acceleration_m_s2,
                config.max_linear_jerk_m_s3, shaped, shaped_acceleration);
    shape_group(target, velocity_, acceleration_, 3, config.max_angular_acceleration_rad_s2,
                config.max_angular_jerk_rad_s3, shaped, shaped_acceleration);
    acceleration_ = shaped_acceleration;
    velocity_ = shaped;
    return shaped;
  }

 private:
  std::array<double, 6> velocity_{};
  std::array<double, 6> acceleration_{};
};

// Joint-space S-curve between the IK and libfranka's limiter. The IK output can
// step when a joint freezes at or leaves its bound; shaped here at half the
// FR3's acceleration and a tenth of its jerk limit, those steps become smooth
// ramps, so the robot-limit clip afterwards never has to act.
constexpr double kJointShaperAccelerationRad_s2 = 5.0;
constexpr double kJointShaperJerkRad_s3 = 500.0;

[[nodiscard]] std::array<double, 7> shape_joint_velocity(const std::array<double, 7>& target,
                                                         const franka::RobotState& state) {
  constexpr double kDeltaT = 1e-3;
  constexpr double a_max = kJointShaperAccelerationRad_s2;
  constexpr double j_max = kJointShaperJerkRad_s3;
  std::array<double, 7> shaped{};
  for (size_t joint = 0; joint < 7; ++joint) {
    const double velocity = state.dq_d[joint];
    const double acceleration = state.ddq_d[joint];
    const double error = target[joint] - velocity;
    // Snap window of two ticks' jerk-limited change keeps the arrival step (and
    // its jerk spike, <= ~1500 rad/s^3) well under the FR3's 5000 rad/s^3.
    if (std::abs(error) < 2.0 * j_max * kDeltaT * kDeltaT && std::abs(acceleration) < j_max * kDeltaT) {
      shaped[joint] = target[joint];
      continue;
    }
    const double desired =
        std::copysign(std::min(a_max, std::sqrt(2.0 * kShaperJerkMargin * j_max * std::abs(error))), error);
    const double jerk = std::clamp((desired - acceleration) / kDeltaT, -j_max, j_max);
    const double new_acceleration = std::clamp(acceleration + jerk * kDeltaT, -a_max, a_max);
    shaped[joint] = velocity + new_acceleration * kDeltaT;
  }
  return shaped;
}

// Fence check on the arm's actual motion (J * dq), not just the commanded twist:
// a detour or IK approximation can move the gripper sideways, and that must not
// carry it through the workspace box either. Returns the factor (0..1) to scale
// the joint velocities by so that every axis can still stop before its wall.
[[nodiscard]] double fence_scale(const franka::RobotState& state, const Config& config,
                                 const Workspace& workspace, const std::array<double, 42>& jacobian,
                                 const std::array<double, 7>& joint_velocity) {
  double scale = 1.0;
  for (size_t axis = 0; axis < 3; ++axis) {
    double velocity = 0.0;
    for (size_t joint = 0; joint < 7; ++joint) {
      velocity += jacobian[joint * 6 + axis] * joint_velocity[joint];
    }
    const double position = state.O_T_EE[12 + axis];
    const double allowed =
        velocity > 0.0
            ? stoppable_speed(workspace.maximum_m[axis] - position, config.max_linear_acceleration_m_s2,
                              config.max_linear_jerk_m_s3)
            : stoppable_speed(position - workspace.minimum_m[axis], config.max_linear_acceleration_m_s2,
                              config.max_linear_jerk_m_s3);
    if (std::abs(velocity) > allowed) {
      scale = std::min(scale, allowed / std::abs(velocity));
    }
  }
  return scale;
}

template <size_t N>
[[nodiscard]] bool near_zero(const std::array<double, N>& velocity, double tolerance = 1e-6) {
  return std::all_of(velocity.begin(), velocity.end(), [tolerance](const double value) {
    return std::abs(value) < tolerance;
  });
}

void run_gripper_worker(std::stop_token stop_token, const Config& config, JoystickInput& input,
                        std::atomic<bool>& stop_requested, std::atomic<bool>& faulted) {
  try {
    franka::Gripper gripper(config.robot_ip);
    while (!stop_token.stop_requested() && !stop_requested.load()) {
      if (input.take_gripper_action() != 0U) {
        const int action = input.gripper_action();
        try {
          // Wide outer tolerance (Frankastein: 0.08 m) makes a grasp on any object
          // width up to fully open count as success instead of a failure.
          const bool completed =
              action == 1 ? gripper.move(config.gripper_open_width_m, config.gripper_open_speed_m_s)
                          : gripper.grasp(config.gripper_grasp_width_m, config.gripper_grasp_speed_m_s,
                                          config.gripper_grasp_force_N,
                                          config.gripper_grasp_epsilon_inner_m,
                                          config.gripper_grasp_epsilon_outer_m);
          if (!completed) {
            std::cerr << "gripper: " << (action == 1 ? "open" : "grasp")
                      << " did not reach its target (arm teleop continues)\n";
          }
        } catch (const franka::CommandException& error) {
          // A rejected single command is reported, not treated as a fault, like
          // Frankastein's keyboard teleop. Connection loss still faults below.
          std::cerr << "gripper command rejected: " << error.what() << '\n';
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  } catch (const std::exception& error) {
    std::cerr << "gripper fault: " << error.what() << '\n';
    faulted.store(true);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::optional<std::string> config_path;
    bool check_config = false;
    bool enable_motion = false;
    for (int index = 1; index < argc; ++index) {
      const std::string argument(argv[index]);
      if (argument == "--help" || argument == "-h") {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
      }
      if (argument == "--config" && index + 1 < argc) {
        config_path = argv[++index];
      } else if (argument == "--check-config") {
        check_config = true;
      } else if (argument == "--enable-motion") {
        enable_motion = true;
      } else {
        throw std::runtime_error("invalid argument '" + argument + "'; use --help");
      }
    }
    if (!config_path.has_value() || check_config == enable_motion) {
      throw std::runtime_error("supply --config and exactly one of --check-config or --enable-motion");
    }
    const Config config = load_config(*config_path);
    std::cout << "configuration is valid; robot_ip=" << config.robot_ip
              << ", joystick_path=" << config.joystick_path
              << ", gripper_enabled=" << (config.gripper_enabled ? "true" : "false") << '\n';
    if (check_config) {
      return EXIT_SUCCESS;
    }
    struct sigaction action {};
    action.sa_handler = request_stop;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);

    JoystickInput input(config);
    input.start();
    std::cout << "Hold RB (the right bumper) to arm motion. Releasing RB pauses (the arm holds);\n"
                 "press RB again to resume. Ctrl-C ends control. D-pad up/down changes speed.\n";
    const int64_t deadline_ms = monotonic_time_ms() + 10000;
    while (!input.deadman_pressed() && input.healthy() && g_stop_requested == 0 &&
           monotonic_time_ms() < deadline_ms) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!input.deadman_pressed()) {
      throw std::runtime_error("RB (the right bumper) was not held within 10 seconds; robot connection was not opened");
    }
    if (!input.healthy()) {
      throw std::runtime_error("joystick fault before arm connection: " + input.fault_reason());
    }

    franka::Robot robot(config.robot_ip, choose_realtime_config());
    franka::RobotState initial_state = robot.readOnce();
    if (initial_state.robot_mode == franka::RobotMode::kReflex) {
      // A previous session ended in a reflex; clear it once, as libfranka's
      // examples do at startup, so the operator does not have to use Desk.
      std::cout << "Robot is in reflex mode from a previous stop; running error recovery.\n";
      robot.automaticErrorRecovery();
      initial_state = robot.readOnce();
    }
    bool starts_near_joint_limit = false;
    for (size_t joint = 0; joint < 7; ++joint) {
      starts_near_joint_limit |=
          std::min(initial_state.q[joint] - kJointLowerRad[joint],
                   kJointUpperRad[joint] - initial_state.q[joint]) < kJointLimitMarginRad;
    }
    if (starts_near_joint_limit) {
      std::cerr << "note: a joint starts within " << kJointLimitMarginRad
                << " rad of its limit; motion that pushes it further is blocked.\n";
      print_joint_positions(initial_state.q);
    }
    franka::Model model = robot.loadModel();
    const Workspace workspace = workspace_from_config(config);
    require_inside_workspace(initial_state, workspace);

    std::atomic<bool> stop_requested{false};
    std::atomic<bool> gripper_faulted{false};
    std::optional<std::jthread> gripper_worker;
    if (config.gripper_enabled) {
      gripper_worker.emplace(run_gripper_worker, std::cref(config), std::ref(input),
                             std::ref(stop_requested), std::ref(gripper_faulted));
    }
    std::atomic<int> stop_reason{0};
    double control_elapsed_s = 0.0;
    const diff_ik::Params ik = ik_params(config);
    diff_ik::Params ik_strict = ik;
    ik_strict.max_task_error = 0.05;
    TwistShaper shaper;
    std::cout << "Motion enabled. Sticks command base-frame Cartesian velocity; joint velocities come from\n"
                 "the onboard IK solver. Desk collision behavior is preserved.\n";
    robot.control(
        [&](const franka::RobotState& state, franka::Duration period) -> franka::JointVelocities {
          control_elapsed_s += period.toSec();
          bool stopping = false;
          if (g_stop_requested != 0) {
            stop_reason.store(1);
            stopping = true;
          }
          if (!stopping && !input.healthy()) {
            stop_reason.store(2);
            stopping = true;
          }
          if (!stopping && gripper_faulted.load()) {
            stop_reason.store(4);
            stopping = true;
          }
          // RB released = pause: command zero velocity (the S-curve brings the arm
          // smoothly to rest) but keep the session open so RB resumes instantly.
          const bool paused = !input.deadman_pressed();
          std::array<double, 6> target_twist{};
          if (!stopping && !paused && control_elapsed_s >= config.initial_zero_hold_s) {
            target_twist = bounded_velocity(state, config, workspace, input);
          }
          // 1. Smooth S-curve in Cartesian space.
          const std::array<double, 6> twist = shaper.step(target_twist, config);

          // 2. Per-joint velocity bounds: the robot's own position-dependent limits,
          //    evaluated kJointLimitMarginRad closer to each limit, capped for teleop.
          std::array<double, 7> q_toward_upper{};
          std::array<double, 7> q_toward_lower{};
          for (size_t joint = 0; joint < 7; ++joint) {
            q_toward_upper[joint] = state.q_d[joint] + kJointLimitMarginRad;
            q_toward_lower[joint] = state.q_d[joint] - kJointLimitMarginRad;
          }
          std::array<double, 7> upper = robot.getUpperJointVelocityLimits(q_toward_upper);
          std::array<double, 7> lower = robot.getLowerJointVelocityLimits(q_toward_lower);
          for (size_t joint = 0; joint < 7; ++joint) {
            upper[joint] = std::min(upper[joint], kMaximumJointSpeedRad_s);
            lower[joint] = std::max(lower[joint], -kMaximumJointSpeedRad_s);
          }

          // 3. IK: twist -> joint velocities, with posture pull while the sticks move.
          const double activity =
              std::max(std::hypot(twist[0], twist[1], twist[2]) / config.max_linear_speed_m_s,
                       std::hypot(twist[3], twist[4], twist[5]) / config.max_angular_speed_rad_s);
          const double nullspace_weight = std::clamp(activity / kNullspaceFullActivity, 0.0, 1.0);
          std::array<double, 7> joint_velocity{};
          if (!near_zero(twist)) {
            const std::array<double, 42> jacobian = model.zeroJacobian(franka::Frame::kEndEffector, state);
            // 3b. Keep the resulting gripper motion (including any detour) inside the box.
            auto solve_fenced = [&](const diff_ik::Params& params) {
              std::array<double, 7> dq =
                  diff_ik::solve(jacobian, twist, state.q, lower, upper, nullspace_weight, params).dq;
              const double scale = fence_scale(state, config, workspace, jacobian, dq);
              for (double& value : dq) {
                value *= scale;
              }
              return std::pair{dq, scale};
            };
            // Progress along the pushed twist (dot product of achieved and requested).
            auto progress = [&](const std::array<double, 7>& dq) {
              double sum = 0.0;
              for (size_t joint = 0; joint < 7; ++joint) {
                for (size_t row = 0; row < 6; ++row) {
                  sum += jacobian[joint * 6 + row] * dq[joint] * twist[row];
                }
              }
              return sum;
            };
            auto [dq, scale] = solve_fenced(ik);
            if (config.allow_detour && scale < 0.999) {
              // The detour runs into a wall (e.g. in a corner): the exact-direction
              // solution may still make progress where the detour cannot.
              auto [strict_dq, strict_scale] = solve_fenced(ik_strict);
              (void)strict_scale;
              if (progress(strict_dq) > progress(dq)) {
                dq = strict_dq;
              }
            }
            // Never move against the push: when the push is impossible from this pose,
            // the best-effort solution can come out partly backwards. Hold instead.
            if (progress(dq) <= 0.0) {
              dq = {};
            }
            joint_velocity = dq;
          }

          // 4. Joint-space S-curve, then the hard safety clip with the robot's
          //    exact velocity limits and max acceleration/jerk.
          const std::array<double, 7> limited = franka::limitRate(
              robot.getUpperJointVelocityLimits(state.q_d), robot.getLowerJointVelocityLimits(state.q_d),
              franka::kMaxJointAcceleration, franka::kMaxJointJerk, shape_joint_velocity(joint_velocity, state),
              state.dq_d, state.ddq_d);
          // libfranka's joint limiter settles into a +/-1e-5 rad/s dither around zero.
          if (stopping && near_zero(twist) && near_zero(limited, 1e-4)) {
            return franka::MotionFinished(franka::JointVelocities(limited));
          }
          return franka::JointVelocities(limited);
        },
        franka::ControllerMode::kJointImpedance, false, franka::kMaxCutoffFrequency);
    stop_requested.store(true);
    if (gripper_worker.has_value()) {
      gripper_worker->request_stop();
      gripper_worker->join();
    }
    if (gripper_faulted.load()) {
      throw std::runtime_error("stopped after a parallel-gripper fault");
    }
    if (!input.healthy()) {
      throw std::runtime_error("stopped after joystick fault: " + input.fault_reason());
    }
    if (stop_reason.load() == 1) {
      std::cout << "Motion ended safely: software stop requested.\n";
    } else {
      std::cout << "Motion ended safely.\n";
    }
    return EXIT_SUCCESS;
  } catch (const franka::ControlException& error) {
    std::cerr << "libfranka error: " << error.what() << '\n';
    if (!error.log.empty()) {
      const franka::RobotState& last = error.log.back().state;
      print_joint_positions(last.q);
      std::cerr << "end effector position (m): x=" << last.O_T_EE[12] << " y=" << last.O_T_EE[13]
                << " z=" << last.O_T_EE[14] << '\n';
    }
    return EXIT_FAILURE;
  } catch (const franka::Exception& error) {
    std::cerr << "libfranka error: " << error.what() << '\n';
    return EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
