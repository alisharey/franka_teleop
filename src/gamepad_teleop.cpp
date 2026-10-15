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
#include "policy_input.h"
#include "quest_input.h"
#include "telemetry.h"
#include "telemoma_policy.h"

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
#include <iterator>
#include <limits>
#include <memory>
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
  // Input device (optional keys): "gamepad" (default) or "quest", the Meta Quest
  // right controller with TeleMoMa's action (src/telemoma_policy.h).
  std::string input{"gamepad"};
  std::string quest_connection{"usb"};       // "usb" or the headset's IP for Wi-Fi ADB
  double quest_position_gain{5.0};           // twist = gain * TeleMoMa action [1/s]
  double quest_rotation_gain{5.0};
  // TeleMoMa's rmat_reorder, controller axes -> robot base axes. TeleMoMa's own
  // -2, -1, -3 was made for the original (VrApi) oculus_reader app; the Quest 3
  // app reports OpenXR controller axes (x right, y up, -z forward), for which
  // 3, 1, 2 maps forward -> -x (toward the base), right -> +y, up -> +z for an
  // operator standing in front of the robot, facing it. Behind it: -3, -1, 2.
  std::array<int, 4> quest_axis_reorder{3, 1, 2, 4};
  double quest_position_scale{1.0};          // TeleMoMa spatial_coeff
  double quest_rotation_scale{1.0};          // gripper rotation = this x the hand's (1 = TeleMoMa)
  // Level the VR frame at the reset: keep the controller's heading, take "up"
  // from the headset (upright and level). false = TeleMoMa exactly (full
  // controller orientation, so a tilted controller tilts every motion).
  bool quest_level_frame{true};
  // Readings older than this pause the arm. Wi-Fi has occasional 100-250 ms
  // gaps; within this window the arm keeps tracking the last target.
  int quest_timeout_ms{500};
  // Home pose (optional, tools/save_home_pose.sh): a downward flick of the
  // Quest stick (grip released) moves the arm there in joint space.
  bool has_home{false};
  std::array<double, 7> home_joints{};
  // input = policy: TeleMoMa actions from a learned policy (vla/baku/baku/franka/deploy.py)
  // over UDP on this PC, executed with the Quest's gains so the arm responds as it did
  // while recording the demonstrations.
  std::string policy_host{"127.0.0.1"};
  int policy_port{47820};
  int policy_timeout_ms{300};  // no action for this long pauses the arm
};

[[nodiscard]] std::unordered_map<std::string, std::string> read_key_values(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot open config '" + path + "'");
  }
  const std::array<std::string, 39> allowed_keys{
      "robot_ip", "joystick_path", "controller_profile", "deadzone", "max_linear_speed_m_s",
      "max_angular_speed_rad_s", "max_linear_acceleration_m_s2", "max_linear_jerk_m_s3",
      "max_angular_acceleration_rad_s2", "max_angular_jerk_rad_s3", "initial_zero_hold_s",
      "workspace_x_m", "workspace_y_m", "workspace_z_m", "z_down_speed_scale",
      "speed_toggle_axis", "allow_detour", "input_timeout_ms", "gripper_enabled", "gripper_open_width_m",
      "gripper_open_speed_m_s", "gripper_grasp_width_m", "gripper_grasp_speed_m_s",
      "gripper_grasp_force_N", "gripper_grasp_epsilon_inner_m", "gripper_grasp_epsilon_outer_m",
      "input", "quest_connection", "quest_position_gain", "quest_rotation_gain", "quest_axis_reorder",
      "quest_position_scale", "quest_level_frame", "quest_timeout_ms", "home_joints", "quest_rotation_scale",
      "policy_host", "policy_port", "policy_timeout_ms"};
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

void validate_input(const Config& config) {
  if (config.input != "gamepad" && config.input != "quest" && config.input != "policy") {
    throw std::runtime_error("input must be 'gamepad', 'quest' or 'policy'");
  }
  if (config.policy_port < 1 || config.policy_port > 65535 || config.policy_timeout_ms <= 0) {
    throw std::runtime_error("policy_port must be 1..65535 and policy_timeout_ms positive");
  }
  if (config.quest_position_gain <= 0.0 || config.quest_rotation_gain <= 0.0 || config.quest_position_scale <= 0.0 ||
      config.quest_rotation_scale <= 0.0 || config.quest_rotation_scale > 3.0) {
    throw std::runtime_error("quest gains and quest_position_scale must be positive; quest_rotation_scale in (0, 3]");
  }
  std::array<bool, 4> used{};
  for (const int entry : config.quest_axis_reorder) {
    const int axis = std::abs(entry);
    if (axis < 1 || axis > 4 || used[static_cast<size_t>(axis - 1)]) {
      throw std::runtime_error("quest_axis_reorder must be a signed permutation of 1, 2, 3, 4 (e.g. 3, 1, 2, 4)");
    }
    used[static_cast<size_t>(axis - 1)] = true;
  }
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
  const auto optional = [&values](const std::string& key) -> const std::string* {
    const auto found = values.find(key);
    return found == values.end() ? nullptr : &found->second;
  };
  if (const auto* value = optional("input")) config.input = *value;
  if (const auto* value = optional("quest_connection")) config.quest_connection = *value;
  if (const auto* value = optional("quest_position_gain")) config.quest_position_gain = parse_double("quest_position_gain", *value);
  if (const auto* value = optional("quest_rotation_gain")) config.quest_rotation_gain = parse_double("quest_rotation_gain", *value);
  if (const auto* value = optional("quest_position_scale")) config.quest_position_scale = parse_double("quest_position_scale", *value);
  if (const auto* value = optional("quest_rotation_scale")) config.quest_rotation_scale = parse_double("quest_rotation_scale", *value);
  if (const auto* value = optional("quest_level_frame")) config.quest_level_frame = parse_boolean("quest_level_frame", *value);
  if (const auto* value = optional("quest_timeout_ms")) config.quest_timeout_ms = parse_nonnegative_integer("quest_timeout_ms", *value);
  if (const auto* value = optional("policy_host")) config.policy_host = *value;
  if (const auto* value = optional("policy_port")) config.policy_port = parse_nonnegative_integer("policy_port", *value);
  if (const auto* value = optional("policy_timeout_ms")) config.policy_timeout_ms = parse_nonnegative_integer("policy_timeout_ms", *value);
  if (const auto* value = optional("home_joints")) {
    config.home_joints = parse_array<7>("home_joints", *value);
    config.has_home = true;
  }
  if (const auto* value = optional("quest_axis_reorder")) {
    const auto reorder = parse_array<4>("quest_axis_reorder", *value);
    for (size_t index = 0; index < 4; ++index) config.quest_axis_reorder[index] = static_cast<int>(reorder[index]);
  }
  validate_input(config);
  return config;
}

// The teleop's input device: the gamepad or the Meta Quest right controller.
// Everything is called from the control loop except start().
class TeleopInput {
 public:
  virtual ~TeleopInput() = default;
  virtual void start() = 0;
  // Once at the start of every control tick (real-time: must not block).
  virtual void tick(const franka::RobotState& /*state*/, double /*gripper_open_fraction*/) {}
  [[nodiscard]] virtual bool healthy() const = 0;
  [[nodiscard]] virtual std::string fault_reason() const = 0;
  [[nodiscard]] virtual bool deadman_pressed() const = 0;
  [[nodiscard]] virtual bool rotation_modifier_pressed() const = 0;
  [[nodiscard]] virtual double speed_scale() const = 0;
  // Base-frame twist [m/s, rad/s], before workspace braking.
  [[nodiscard]] virtual std::array<double, 6> command_twist(const Config& config) const = 0;
  [[nodiscard]] virtual uint64_t take_gripper_action() = 0;
  [[nodiscard]] virtual int gripper_action() const = 0;  // 0 none, 1 open, 2 grasp
  [[nodiscard]] virtual uint64_t gripper_command_count() const = 0;
  // Counts "go to the home pose" requests (Quest: downward stick flick).
  [[nodiscard]] virtual uint64_t home_requests() const { return 0; }
  virtual void snapshot(telemetry::Sample& sample) const = 0;
};

class JoystickInput final : public TeleopInput {
 public:
  explicit JoystickInput(const Config& config) : config_(config) {}
  ~JoystickInput() override { stop(); }
  JoystickInput(const JoystickInput&) = delete;
  JoystickInput& operator=(const JoystickInput&) = delete;

  void start() override {
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

  [[nodiscard]] bool deadman_pressed() const override { return buttons_[kRightBumper].load(); }
  [[nodiscard]] bool rotation_modifier_pressed() const override {
    // Hold L1 (left bumper) or L2 (left trigger) to rotate.
    return buttons_[kLeftBumper].load() || normalized_axis(kLeftTrigger) > 0.5;
  }
  [[nodiscard]] bool healthy() const override {
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
  [[nodiscard]] double speed_scale() const override { return kSpeedLevels[speed_level_.load()]; }
  [[nodiscard]] uint64_t take_gripper_action() override { return gripper_action_sequence_.exchange(0); }
  [[nodiscard]] int gripper_action() const override { return gripper_action_.load(); }
  [[nodiscard]] uint64_t gripper_command_count() const override { return gripper_command_count_.load(); }
  // Sticks -> base-frame twist.
  [[nodiscard]] std::array<double, 6> command_twist(const Config& config) const override {
    const double scale = speed_scale();
    const double linear = config.max_linear_speed_m_s * scale;
    const double angular = config.max_angular_speed_rad_s * scale;
    std::array<double, 6> velocity{};
    if (rotation_modifier_pressed()) {
      velocity = {0.0,
                  0.0,
                  0.0,
                  axis(kLeftStickX) * angular,
                  -axis(kLeftStickY) * angular,
                  axis(kRightStickX) * angular};
    } else {
      // Right stick up/down -> base X (push up = +X, away from the base);
      // left stick left/right -> base Y (push right = +Y);
      // left stick up/down -> base Z (push up = +Z).
      velocity = {-axis(kRightStickY) * linear,
                  axis(kLeftStickX) * linear,
                  -axis(kLeftStickY) * linear,
                  0.0,
                  0.0,
                  0.0};
      if (velocity[2] < 0.0) {
        velocity[2] *= config.z_down_speed_scale;  // Frankastein moves down at half speed.
      }
    }
    return velocity;
  }
  // Raw axis values and button states, for telemetry.
  void snapshot(telemetry::Sample& sample) const override {
    sample.axis_count = std::min<uint8_t>(axis_count_, telemetry::kMaxAxes);
    sample.button_count = std::min<uint8_t>(button_count_, telemetry::kMaxButtons);
    for (size_t index = 0; index < sample.axis_count; ++index) {
      sample.axes[index] = axes_[index].load(std::memory_order_relaxed);
    }
    for (size_t index = 0; index < sample.button_count; ++index) {
      sample.buttons[index] = buttons_[index].load(std::memory_order_relaxed);
    }
  }
  [[nodiscard]] std::string fault_reason() const override { return fault_reason_; }

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
              gripper_command_count_.fetch_add(1);
            } else if (event.number == kButtonB) {
              gripper_action_.store(2);
              gripper_action_sequence_.fetch_add(1);
              gripper_command_count_.fetch_add(1);
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
  std::atomic<uint64_t> gripper_command_count_{0};
  static constexpr std::array<double, 3> kSpeedLevels{0.25, 0.5, 1.0};
  std::atomic<int> speed_level_{2};
  int speed_toggle_direction_{0};  // Reader thread only.
  std::thread reader_;
  std::string fault_reason_{""};
};

// Meta Quest right controller (input = quest), TeleMoMa's way: hold the grip
// to move, the trigger toggles the gripper, B (grip released) re-sets "forward"
// to the controller's current orientation. Every tick, TeleMoMa's action
// (src/telemoma_policy.h) is computed against the measured end-effector pose,
// and the arm is driven toward the target it implies (current + action) with a
// proportional twist, through the same braking, shaping and IK as the gamepad.
// Readings that stop arriving pause the arm (not fatal); the reader restarts
// the headset app. A on the controller is the recorder's start/stop button.
class QuestTeleopInput final : public TeleopInput {
 public:
  explicit QuestTeleopInput(const Config& config)
      : config_(config),
        reader_(config.quest_connection),
        policy_(config.quest_axis_reorder, config.quest_position_scale, config.quest_level_frame,
                config.quest_rotation_scale) {}

  void start() override {
    reader_.start();
    std::cout << "Meta Quest connected (" << reader_.serial() << ").\n";
  }

  void tick(const franka::RobotState& state, double gripper_open_fraction) override {
    quest::Reading reading;
    if (reader_.latest(reading)) {  // keeps the previous one if the reader holds the lock
      reading_ = reading;
    }
    const int64_t now = quest::steady_ns();
    age_s_ = reading_.sequence == 0 ? 0.0 : static_cast<double>(now - reading_.receive_steady_ns) * 1e-9;
    fresh_ = reading_.sequence != 0 && age_s_ * 1000.0 <= config_.quest_timeout_ms;
    if (reading_.sequence != last_sequence_) {
      last_sequence_ = reading_.sequence;
      // Downward flick of the stick, grip released: request the home pose.
      const bool stick_down = reading_.joystick_y < -0.7;
      if (stick_down && !stick_was_down_ && !reading_.grip_pressed) {
        ++home_requests_;
      }
      stick_was_down_ = stick_down || (stick_was_down_ && reading_.joystick_y < -0.3);
      if (reading_.pose_valid) {  // TeleMoMa: `if controller_id not in poses: continue`
        // TeleMoMa re-sets "forward" on a joystick click (sent by the fixed reader
        // app, docs/quest_reader_app.md; the original Quest 3 build never sends
        // it); B does the same. Only with the grip released: a re-set while
        // moving would make the target jump.
        const bool reset_forward = (reading_.joystick_click || reading_.b) && !reading_.grip_pressed;
        policy_.on_reading(reading_.pose, {reading_.grip_pressed, reset_forward, reading_.trigger});
      }
    }
    if (!policy_.has_pose()) {
      return;
    }
    robot_pose_ = row_major(state.O_T_EE);
    action_ = policy_.calculate_action(robot_pose_, gripper_open_fraction);
    if (policy_.take_gripper_toggled()) {  // every trigger toggle sends TeleMoMa's target
      const int target = policy_.target_gripper();
      gripper_action_.store(target == 1 ? 1 : 2);
      gripper_action_sequence_.fetch_add(1);
      gripper_command_count_.fetch_add(1);
    }
  }

  [[nodiscard]] bool healthy() const override { return true; }
  [[nodiscard]] std::string fault_reason() const override { return ""; }
  [[nodiscard]] uint64_t home_requests() const override { return home_requests_; }
  [[nodiscard]] bool deadman_pressed() const override {
    return fresh_ && reading_.pose_valid && policy_.has_pose() && policy_.movement_enabled();
  }
  [[nodiscard]] bool rotation_modifier_pressed() const override { return false; }
  [[nodiscard]] double speed_scale() const override { return 1.0; }

  // Toward target = current + action: twist = gain * action (rotation as a rotation vector).
  [[nodiscard]] std::array<double, 6> command_twist(const Config& config) const override {
    if (!deadman_pressed()) {
      return {};
    }
    const telemoma::Vec3 rotation =
        telemoma::rotation_vector(telemoma::from_euler_xyz({action_[3], action_[4], action_[5]}));
    std::array<double, 6> twist{};
    for (size_t axis = 0; axis < 3; ++axis) {
      twist[axis] = config.quest_position_gain * action_[axis];
      twist[3 + axis] = config.quest_rotation_gain * rotation[axis];
    }
    const double linear = std::hypot(twist[0], twist[1], twist[2]);
    const double angular = std::hypot(twist[3], twist[4], twist[5]);
    for (size_t axis = 0; axis < 3; ++axis) {
      if (linear > config.max_linear_speed_m_s) twist[axis] *= config.max_linear_speed_m_s / linear;
      if (angular > config.max_angular_speed_rad_s) twist[3 + axis] *= config.max_angular_speed_rad_s / angular;
    }
    return twist;
  }

  [[nodiscard]] uint64_t take_gripper_action() override { return gripper_action_sequence_.exchange(0); }
  [[nodiscard]] int gripper_action() const override { return gripper_action_.load(); }
  [[nodiscard]] uint64_t gripper_command_count() const override { return gripper_command_count_.load(); }

  void snapshot(telemetry::Sample& sample) const override {
    sample.input_kind = 1;
    telemetry::QuestSample& q = sample.quest;
    q.reading_valid = reading_.sequence != 0;
    q.reading_sequence = reading_.sequence;
    q.reading_age_s = age_s_;
    q.reading_stamp_ns = reading_.receive_ns;
    q.right_present = reading_.right_present;
    q.pose_valid = reading_.pose_valid;
    q.controller_pose = reading_.pose;
    q.button_a = reading_.a;
    q.button_b = reading_.b;
    q.trigger_pressed = reading_.trigger_pressed;
    q.grip_pressed = reading_.grip_pressed;
    q.joystick_click = reading_.joystick_click;
    q.trigger_touch = reading_.trigger_touch;
    q.trigger = reading_.trigger;
    q.grip = reading_.grip;
    q.joystick = {reading_.joystick_x, reading_.joystick_y};
    q.policy_active = policy_.has_pose();
    q.movement_enabled = policy_.movement_enabled();
    q.action = action_;
    q.target = target_pose();
    q.robot_origin = policy_.robot_origin();
    q.vr_state = policy_.vr_state();
    q.vr_origin = policy_.vr_origin();
    q.vr_to_global = policy_.vr_to_global();
    // /teleop/joy and TeleopState.axes/buttons carry the Quest's in quest mode:
    // axes [joystick x, joystick y, trigger, grip] (x32767), buttons [A, B, RTr, RG, RJ, RT].
    const std::array<double, 4> axes{reading_.joystick_x, reading_.joystick_y, reading_.trigger, reading_.grip};
    sample.axis_count = static_cast<uint8_t>(axes.size());
    for (size_t index = 0; index < axes.size(); ++index) {
      sample.axes[index] = static_cast<int16_t>(std::lround(std::clamp(axes[index], -1.0, 1.0) * 32767.0));
    }
    const std::array<bool, 6> buttons{reading_.a, reading_.b, reading_.trigger_pressed, reading_.grip_pressed,
                                      reading_.joystick_click, reading_.trigger_touch};
    sample.button_count = static_cast<uint8_t>(buttons.size());
    for (size_t index = 0; index < buttons.size(); ++index) {
      sample.buttons[index] = buttons[index];
    }
  }

 private:
  static telemoma::Mat4 row_major(const std::array<double, 16>& column_major) {
    telemoma::Mat4 out{};
    for (size_t row = 0; row < 4; ++row) {
      for (size_t col = 0; col < 4; ++col) {
        out[row * 4 + col] = column_major[col * 4 + row];
      }
    }
    return out;
  }

  // TeleMoMa's robot side: position + action, rotation R_euler(action) R_robot.
  telemoma::Mat4 target_pose() const {
    const telemoma::Mat3 rotation = telemoma::multiply(telemoma::from_euler_xyz({action_[3], action_[4], action_[5]}),
                                                       telemoma::rotation_of(robot_pose_));
    return {rotation[0], rotation[1], rotation[2], robot_pose_[3] + action_[0],
            rotation[3], rotation[4], rotation[5], robot_pose_[7] + action_[1],
            rotation[6], rotation[7], rotation[8], robot_pose_[11] + action_[2],
            0.0, 0.0, 0.0, 1.0};
  }

  const Config& config_;
  quest::Reader reader_;
  telemoma::OculusPolicy policy_;
  quest::Reading reading_;
  uint64_t last_sequence_{0};
  bool fresh_{false};
  double age_s_{0.0};
  bool stick_was_down_{false};
  uint64_t home_requests_{0};
  telemoma::Mat4 robot_pose_ = telemoma::identity4();
  std::array<double, 7> action_{0, 0, 0, 0, 0, 0, 1};
  std::atomic<int> gripper_action_{0};
  std::atomic<uint64_t> gripper_action_sequence_{0};
  std::atomic<uint64_t> gripper_command_count_{0};
};

// Learned policy (input = policy): TeleMoMa actions over UDP (src/policy_input.h).
// Each new action sets a target the way TeleMoMa's robot side builds it, from the
// pose when the action arrives: position + action, R_euler(action) R_robot. Every
// tick the arm is driven toward that target with the Quest's gains and limits, the
// same tracking that executed the demonstrations (there the target moved with the
// hand; here it moves at the policy rate).
class PolicyTeleopInput final : public TeleopInput {
 public:
  explicit PolicyTeleopInput(const Config& config)
      : config_(config), receiver_(config.policy_host, config.policy_port) {}

  void start() override {
    receiver_.start();
    std::cout << "Waiting for policy actions on udp://" << config_.policy_host << ':' << config_.policy_port
              << " (vla/baku/baku/franka/deploy.py).\n";
  }

  void tick(const franka::RobotState& state, double /*gripper_open_fraction*/) override {
    policy_link::Packet packet;
    if (receiver_.latest(packet)) {  // keeps the previous one if the reader holds the lock
      packet_ = packet;
    }
    for (size_t row = 0; row < 3; ++row) {
      for (size_t col = 0; col < 3; ++col) robot_rotation_[row * 3 + col] = state.O_T_EE[col * 4 + row];
      robot_position_[row] = state.O_T_EE[12 + row];
    }
    age_s_ = packet_.sequence == 0
                 ? 0.0
                 : static_cast<double>(policy_link::steady_ns() - packet_.receive_steady_ns) * 1e-9;
    fresh_ = packet_.sequence != 0 && age_s_ * 1000.0 <= config_.policy_timeout_ms;
    if (packet_.sequence != last_sequence_) {
      last_sequence_ = packet_.sequence;
      // Home requests count up per sender; a restarted sender (lower count) re-syncs silently.
      if (packet_.home_requests > sender_home_requests_) {
        home_requests_ += packet_.home_requests - sender_home_requests_;
      }
      sender_home_requests_ = packet_.home_requests;
      const auto& a = packet_.action;
      for (size_t axis = 0; axis < 3; ++axis) target_position_[axis] = robot_position_[axis] + a[axis];
      target_rotation_ = telemoma::multiply(telemoma::from_euler_xyz({a[3], a[4], a[5]}), robot_rotation_);
      has_target_ = true;
      // Gripper target (1 open, 0 closed) with hysteresis: temporal aggregation
      // averages it, so it passes 0.5 gradually.
      if (packet_.enabled) {
        const int current = gripper_action_.load();
        const int wanted = a[6] >= 0.65 ? 1 : a[6] <= 0.35 ? 2 : current;
        if (wanted != 0 && wanted != current) {
          gripper_action_.store(wanted);
          gripper_action_sequence_.fetch_add(1);
          gripper_command_count_.fetch_add(1);
        }
      }
    }
    if (!has_target_) {
      return;
    }
    // The tracking error in TeleMoMa's action form (what /teleop/quest records).
    const telemoma::Mat3 error = telemoma::multiply(target_rotation_, telemoma::transpose(robot_rotation_));
    const telemoma::Vec3 euler = telemoma::euler_xyz(error);
    rotation_error_ = telemoma::rotation_vector(error);
    for (size_t axis = 0; axis < 3; ++axis) {
      action_[axis] = std::clamp(target_position_[axis] - robot_position_[axis], -1.0, 1.0);
      action_[3 + axis] = std::clamp(euler[axis], -1.0, 1.0);
    }
    action_[6] = packet_.action[6];
  }

  [[nodiscard]] bool healthy() const override { return true; }
  [[nodiscard]] std::string fault_reason() const override { return ""; }
  [[nodiscard]] uint64_t home_requests() const override { return home_requests_; }
  [[nodiscard]] bool deadman_pressed() const override { return fresh_ && packet_.enabled && has_target_; }
  [[nodiscard]] bool rotation_modifier_pressed() const override { return false; }
  [[nodiscard]] double speed_scale() const override { return 1.0; }

  [[nodiscard]] std::array<double, 6> command_twist(const Config& config) const override {
    if (!deadman_pressed()) {
      return {};
    }
    std::array<double, 6> twist{};
    for (size_t axis = 0; axis < 3; ++axis) {
      twist[axis] = config.quest_position_gain * action_[axis];
      twist[3 + axis] = config.quest_rotation_gain * rotation_error_[axis];
    }
    const double linear = std::hypot(twist[0], twist[1], twist[2]);
    const double angular = std::hypot(twist[3], twist[4], twist[5]);
    for (size_t axis = 0; axis < 3; ++axis) {
      if (linear > config.max_linear_speed_m_s) twist[axis] *= config.max_linear_speed_m_s / linear;
      if (angular > config.max_angular_speed_rad_s) twist[3 + axis] *= config.max_angular_speed_rad_s / angular;
    }
    return twist;
  }

  [[nodiscard]] uint64_t take_gripper_action() override { return gripper_action_sequence_.exchange(0); }
  [[nodiscard]] int gripper_action() const override { return gripper_action_.load(); }
  [[nodiscard]] uint64_t gripper_command_count() const override { return gripper_command_count_.load(); }

  void snapshot(telemetry::Sample& sample) const override {
    sample.input_kind = 2;
    telemetry::QuestSample& q = sample.quest;
    q = {};
    q.reading_valid = packet_.sequence != 0;
    q.reading_sequence = packet_.sequence;
    q.reading_age_s = age_s_;
    q.reading_stamp_ns = packet_.receive_ns;
    q.policy_active = has_target_;
    q.movement_enabled = deadman_pressed();
    q.action = action_;
    q.target = {target_rotation_[0], target_rotation_[1], target_rotation_[2], target_position_[0],
                target_rotation_[3], target_rotation_[4], target_rotation_[5], target_position_[1],
                target_rotation_[6], target_rotation_[7], target_rotation_[8], target_position_[2],
                0.0, 0.0, 0.0, 1.0};
    sample.axis_count = 0;
    sample.button_count = 0;
  }

 private:
  const Config& config_;
  policy_link::Receiver receiver_;
  policy_link::Packet packet_;
  uint64_t last_sequence_{0};
  uint64_t sender_home_requests_{0};
  uint64_t home_requests_{0};
  bool fresh_{false};
  bool has_target_{false};
  double age_s_{0.0};
  telemoma::Vec3 robot_position_{};
  telemoma::Mat3 robot_rotation_{1, 0, 0, 0, 1, 0, 0, 0, 1};
  telemoma::Vec3 target_position_{};
  telemoma::Mat3 target_rotation_{1, 0, 0, 0, 1, 0, 0, 0, 1};
  telemoma::Vec3 rotation_error_{};
  std::array<double, 7> action_{0, 0, 0, 0, 0, 0, 1};
  std::atomic<int> gripper_action_{0};
  std::atomic<uint64_t> gripper_action_sequence_{0};
  std::atomic<uint64_t> gripper_command_count_{0};
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
  std::cout << "Usage: " << name << " --config PATH [--input gamepad|quest|policy] [--check-config | --enable-motion]\n"
            << "\n"
            << "--check-config validates configuration only and never contacts hardware.\n"
            << "--enable-motion connects to the robot; the arm moves only while RB (gamepad) or the\n"
            << "right grip (Meta Quest) is held, or (policy) while enabled actions arrive over UDP.\n"
            << "--input overrides the config's input setting.\n"
            << "\n"
            << "While running it publishes ROS 2 topics for recording: /teleop/state (every\n"
            << "control tick), /teleop/joy and /teleop/session.\n";
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
                                                      const TeleopInput& input) {
  std::array<double, 6> velocity = input.command_twist(config);
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
// Home move: peak joint speed and acceleration of the quintic profile, and the
// correction gain toward it (rad/s per rad).
constexpr double kHomeSpeedRad_s = 0.3;
constexpr double kHomeAccelerationRad_s2 = 0.5;
constexpr double kHomeGain = 3.0;

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

// Latest gripper reading, shared lock-free with the control loop's telemetry.
struct GripperReading {
  std::atomic<double> width{0.0};
  std::atomic<double> max_width{0.0};
  std::atomic<bool> is_grasped{false};
  std::atomic<uint16_t> temperature{0};
  std::atomic<double> time_s{0.0};
};

void run_gripper_worker(std::stop_token stop_token, const Config& config, TeleopInput& input,
                        std::atomic<bool>& stop_requested, std::atomic<bool>& faulted,
                        GripperReading& reading) {
  try {
    franka::Gripper gripper(config.robot_ip);
    // After a power cycle the hand reports max_width ~0 until it is homed, and
    // it rejects move/grasp commands until then.
    if (gripper.readOnce().max_width < 0.01) {
      std::cerr << "\n*** The gripper is NOT HOMED (max width ~0): open/close commands will be rejected.\n"
                   "*** Home it once (Desk: end effector > Homing, or tools/home_gripper.sh), then restart.\n\n";
    }
    // Continuous state reads alongside the (blocking) move/grasp commands on the
    // same connection, as Franka's own gripper node does. Read failures only end
    // the readings; a lost connection still faults through the commands below.
    std::jthread reader([&gripper, &reading, &stop_requested](std::stop_token reader_stop) {
      try {
        while (!reader_stop.stop_requested() && !stop_requested.load()) {
          const franka::GripperState state = gripper.readOnce();
          reading.width.store(state.width);
          reading.max_width.store(state.max_width);
          reading.is_grasped.store(state.is_grasped);
          reading.temperature.store(state.temperature);
          reading.time_s.store(state.time.toSec());
        }
      } catch (const std::exception& error) {
        std::cerr << "gripper state reading stopped: " << error.what() << '\n';
      }
    });
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
    std::optional<std::string> input_override;
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
      } else if (argument == "--input" && index + 1 < argc) {
        input_override = argv[++index];
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
    Config loaded = load_config(*config_path);
    if (input_override.has_value()) {
      loaded.input = *input_override;
      validate_input(loaded);
    }
    const Config config = loaded;
    std::cout << "configuration is valid; robot_ip=" << config.robot_ip << ", input=" << config.input
              << (config.input == "quest"    ? ", quest_connection=" + config.quest_connection
                  : config.input == "policy" ? ", policy=udp://" + config.policy_host + ":" +
                                                   std::to_string(config.policy_port)
                                             : ", joystick_path=" + config.joystick_path)
              << ", gripper_enabled=" << (config.gripper_enabled ? "true" : "false") << '\n';
    if (check_config) {
      return EXIT_SUCCESS;
    }
    struct sigaction action {};
    action.sa_handler = request_stop;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);

    // ROS 2 telemetry for recording (/teleop/state, /teleop/joy, /teleop/session).
    std::ifstream config_stream(*config_path);
    const std::string config_text((std::istreambuf_iterator<char>(config_stream)),
                                  std::istreambuf_iterator<char>());
    const std::unique_ptr<telemetry::Publisher> telemetry_publisher =
        telemetry::make_publisher({.config_path = *config_path, .config_text = config_text});

    std::unique_ptr<TeleopInput> input;
    if (config.input == "quest") {
      input = std::make_unique<QuestTeleopInput>(config);
    } else if (config.input == "policy") {
      input = std::make_unique<PolicyTeleopInput>(config);
    } else {
      input = std::make_unique<JoystickInput>(config);
    }
    input->start();
    if (config.input == "quest") {
      std::cout << "The arm moves only while the right controller's GRIP is held (TeleMoMa: the target is the\n"
                   "grip-press pose plus the controller's motion since); releasing it pauses. TRIGGER toggles\n"
                   "the gripper. B (grip released) re-sets \"forward\" to how you hold the controller now.\n"
                   "Ctrl-C ends control.\n";
    } else if (config.input == "policy") {
      std::cout << "The arm follows the policy's actions while they arrive with 'enabled' set; it pauses\n"
                   "when they stop for policy_timeout_ms. Keep the robot's stop button in reach.\n"
                   "Ctrl-C ends control.\n";
    } else {
      std::cout << "The arm moves only while RB (the right bumper) is held; releasing RB pauses (the\n"
                   "arm holds). Ctrl-C ends control. D-pad up/down changes speed.\n";
    }
    if (g_stop_requested != 0) {
      return EXIT_SUCCESS;
    }
    if (!input->healthy()) {
      throw std::runtime_error("input fault before arm connection: " + input->fault_reason());
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
    GripperReading gripper_reading;
    std::optional<std::jthread> gripper_worker;
    if (config.gripper_enabled) {
      gripper_worker.emplace(run_gripper_worker, std::cref(config), std::ref(*input),
                             std::ref(stop_requested), std::ref(gripper_faulted),
                             std::ref(gripper_reading));
    }
    if (config.has_home) {
      for (size_t joint = 0; joint < 7; ++joint) {
        if (config.home_joints[joint] < kJointLowerRad[joint] + kJointLimitMarginRad ||
            config.home_joints[joint] > kJointUpperRad[joint] - kJointLimitMarginRad) {
          throw std::runtime_error("home_joints: joint " + std::to_string(joint + 1) + " is at or beyond its limit");
        }
      }
    }
    // Home move (stick flick): joint-space quintic to home_joints, at most
    // kHomeSpeedRad_s per joint; the grip, another flick or Ctrl-C cancels it.
    struct HomeMove {
      bool active{false};
      std::array<double, 7> from{};
      double duration_s{0.0};
      double elapsed_s{0.0};
    } home;
    uint64_t home_requests_seen = input->home_requests();
    std::atomic<int> home_event{0};  // 1 started, 2 arrived, 3 cancelled, 4 no home saved
    std::atomic<uint64_t> home_event_count{0};
    std::atomic<double> home_duration_s{0.0};
    auto announce = [&](int event) {
      home_event.store(event);
      home_event_count.fetch_add(1);
    };
    std::optional<std::jthread> announcer;  // the control loop must not print
    if (config.input == "quest" || config.input == "policy") {
      announcer.emplace([&home_event, &home_event_count, &home_duration_s,
                        quest = config.input == "quest"](const std::stop_token& stop) {
        uint64_t seen = 0;
        while (!stop.stop_requested()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          if (const uint64_t count = home_event_count.load(); count != seen) {
            seen = count;
            switch (home_event.load()) {
              case 1: std::cout << "home: moving to the saved home pose (" << std::setprecision(2) << std::fixed
                                << home_duration_s.load() << " s)"
                                << (quest ? "; squeeze the grip or flick again to stop" : "") << '\n'; break;
              case 2: std::cout << "home: arrived\n"; break;
              case 3: std::cout << "home: stopped\n"; break;
              case 4: std::cout << "home: no home pose saved; run tools/save_home_pose.sh\n"; break;
              default: break;
            }
            std::cout << std::flush;
          }
        }
      });
    }
    std::atomic<int> stop_reason{0};
    double control_elapsed_s = 0.0;
    const diff_ik::Params ik = ik_params(config);
    diff_ik::Params ik_strict = ik;
    ik_strict.max_task_error = 0.05;
    TwistShaper shaper;
    uint64_t tick = 0;
    telemetry::Sample sample;  // Reused every tick; filled and copied into the ring.
    std::cout << "Motion enabled. The input commands a base-frame Cartesian velocity; joint velocities come\n"
                 "from the onboard IK solver. Desk collision behavior is preserved.\n";
    robot.control(
        [&](const franka::RobotState& state, franka::Duration period) -> franka::JointVelocities {
          control_elapsed_s += period.toSec();
          sample.stamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count();
          // TeleMoMa's policy compares against the commanded gripper state
          // (1 = open, 0 = closed); before any command, the measured opening.
          const int gripper_command = input->gripper_action();
          const double max_width = gripper_reading.max_width.load(std::memory_order_relaxed);
          const double gripper_open_fraction =
              gripper_command == 2 ? 0.0
              : gripper_command == 1 ? 1.0
              : max_width > 0.0 ? gripper_reading.width.load(std::memory_order_relaxed) / max_width
                                : 1.0;
          input->tick(state, gripper_open_fraction);
          bool stopping = false;
          if (g_stop_requested != 0) {
            stop_reason.store(1);
            stopping = true;
          }
          if (!stopping && !input->healthy()) {
            stop_reason.store(2);
            stopping = true;
          }
          if (!stopping && gripper_faulted.load()) {
            stop_reason.store(4);
            stopping = true;
          }
          // RB released = pause: command zero velocity (the S-curve brings the arm
          // smoothly to rest) but keep the session open so RB resumes instantly.
          const bool paused = !input->deadman_pressed();
          std::array<double, 6> target_twist{};
          const bool motion_enabled = !stopping && !paused && control_elapsed_s >= config.initial_zero_hold_s;
          if (motion_enabled) {
            target_twist = bounded_velocity(state, config, workspace, *input);
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
          double applied_fence_scale = 1.0;
          bool strict_used = false;
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
              if (progress(strict_dq) > progress(dq)) {
                dq = strict_dq;
                scale = strict_scale;
                strict_used = true;
              }
            }
            // Never move against the push: when the push is impossible from this pose,
            // the best-effort solution can come out partly backwards. Hold instead.
            if (progress(dq) <= 0.0) {
              dq = {};
            }
            joint_velocity = dq;
            applied_fence_scale = scale;
          }

          // 3c. Home move (stick flick, grip released): replaces the teleop command.
          if (const uint64_t requests = input->home_requests(); requests != home_requests_seen) {
            home_requests_seen = requests;
            if (home.active) {
              home.active = false;
              announce(3);
            } else if (!config.has_home) {
              announce(4);
            } else if (!stopping) {
              double largest = 0.0;
              for (size_t joint = 0; joint < 7; ++joint) {
                largest = std::max(largest, std::abs(config.home_joints[joint] - state.q[joint]));
              }
              home = {true, state.q,
                      std::max({1.0, 1.875 * largest / kHomeSpeedRad_s, std::sqrt(5.7735 * largest / kHomeAccelerationRad_s2)}),
                      0.0};
              home_duration_s.store(home.duration_s);
              announce(1);
            }
          }
          if (home.active && (stopping || input->deadman_pressed())) {
            home.active = false;
            announce(3);
          }
          if (home.active) {
            home.elapsed_s = std::min(home.elapsed_s + period.toSec(), home.duration_s);
            const double tau = home.elapsed_s / home.duration_s;
            const double s = tau * tau * tau * (10.0 - 15.0 * tau + 6.0 * tau * tau);
            const double ds = 30.0 * tau * tau * (1.0 - tau) * (1.0 - tau) / home.duration_s;
            double remaining = 0.0;
            for (size_t joint = 0; joint < 7; ++joint) {
              const double span = config.home_joints[joint] - home.from[joint];
              const double desired = home.from[joint] + span * s;
              joint_velocity[joint] = std::clamp(span * ds + kHomeGain * (desired - state.q[joint]),
                                                 std::max(lower[joint], -2.0 * kHomeSpeedRad_s),
                                                 std::min(upper[joint], 2.0 * kHomeSpeedRad_s));
              remaining = std::max(remaining, std::abs(config.home_joints[joint] - state.q[joint]));
            }
            if (tau >= 1.0 && remaining < 1e-3) {
              home.active = false;
              joint_velocity = {};
              announce(2);
            }
          }

          // 4. Joint-space S-curve, then the hard safety clip with the robot's
          //    exact velocity limits and max acceleration/jerk.
          const std::array<double, 7> limited = franka::limitRate(
              robot.getUpperJointVelocityLimits(state.q_d), robot.getLowerJointVelocityLimits(state.q_d),
              franka::kMaxJointAcceleration, franka::kMaxJointJerk, shape_joint_velocity(joint_velocity, state),
              state.dq_d, state.ddq_d);

          // 5. Telemetry: copy this tick into the lock-free ring (never blocks).
          sample.tick = tick++;
          sample.period_s = period.toSec();
          sample.state = state;
          sample.deadman = !paused;
          sample.rotation_mode = input->rotation_modifier_pressed();
          sample.motion_enabled = motion_enabled;
          sample.speed_scale = input->speed_scale();
          sample.target_twist = target_twist;
          sample.shaped_twist = twist;
          sample.ik_dq = joint_velocity;
          sample.fence_scale = applied_fence_scale;
          sample.nullspace_weight = nullspace_weight;
          sample.ik_strict_used = strict_used;
          sample.commanded_dq = limited;
          input->snapshot(sample);
          sample.gripper_command = static_cast<uint8_t>(input->gripper_action());
          sample.gripper_command_count = input->gripper_command_count();
          sample.gripper_width = gripper_reading.width.load(std::memory_order_relaxed);
          sample.gripper_max_width = gripper_reading.max_width.load(std::memory_order_relaxed);
          sample.gripper_is_grasped = gripper_reading.is_grasped.load(std::memory_order_relaxed);
          sample.gripper_temperature = gripper_reading.temperature.load(std::memory_order_relaxed);
          sample.gripper_time_s = gripper_reading.time_s.load(std::memory_order_relaxed);
          telemetry_publisher->push(sample);

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
    if (!input->healthy()) {
      throw std::runtime_error("stopped after joystick fault: " + input->fault_reason());
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
