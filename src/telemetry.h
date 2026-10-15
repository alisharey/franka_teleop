#pragma once

// Telemetry out of the 1 kHz control loop, with no ROS types in this header so
// the control code (and the off-robot simulation) builds without ROS.
//
// The control callback fills a Sample each tick and push()es it into a
// single-producer/single-consumer ring: a plain copy, no lock, no allocation,
// no system call. A separate, normal-priority thread drains the ring and
// publishes (src/ros_telemetry.cpp). If that thread falls behind, samples are
// dropped and counted, never waited for.

#include <franka/robot_state.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace telemetry {

constexpr size_t kMaxAxes = 16;
constexpr size_t kMaxButtons = 32;

// Meta Quest input (input = quest); see msg/QuestState.msg. Matrices row-major.
struct QuestSample {
  bool reading_valid{};
  uint64_t reading_sequence{};
  double reading_age_s{};
  int64_t reading_stamp_ns{};
  bool right_present{};
  bool pose_valid{};
  std::array<double, 16> controller_pose{};
  bool button_a{}, button_b{}, trigger_pressed{}, grip_pressed{}, joystick_click{}, trigger_touch{};
  double trigger{}, grip{};
  std::array<double, 2> joystick{};
  bool policy_active{};
  bool movement_enabled{};
  std::array<double, 7> action{};
  std::array<double, 16> target{}, robot_origin{}, vr_state{}, vr_origin{}, vr_to_global{};
};

struct Sample {
  int64_t stamp_ns{};  // system clock (ROS time) at the control tick
  uint64_t tick{};
  double period_s{};
  franka::RobotState state{};
  // Teleop controller.
  bool deadman{};
  bool rotation_mode{};
  bool motion_enabled{};
  bool ik_strict_used{};
  double speed_scale{};
  double fence_scale{1.0};
  double nullspace_weight{};
  std::array<double, 6> target_twist{};
  std::array<double, 6> shaped_twist{};
  std::array<double, 7> ik_dq{};
  std::array<double, 7> commanded_dq{};
  // Input device: 0 = gamepad, 1 = Meta Quest (then `quest` is filled and the
  // axes/buttons below carry the Quest's, see ros_telemetry.cpp), 2 = policy
  // (quest.action/target/movement_enabled/reading_* only, no axes/buttons).
  uint8_t input_kind{};
  QuestSample quest{};
  // Raw gamepad.
  uint8_t axis_count{};
  uint8_t button_count{};
  std::array<int16_t, kMaxAxes> axes{};
  std::array<bool, kMaxButtons> buttons{};
  // Gripper: last command and latest reading.
  uint8_t gripper_command{};
  uint64_t gripper_command_count{};
  double gripper_width{};
  double gripper_max_width{};
  bool gripper_is_grasped{};
  uint16_t gripper_temperature{};
  double gripper_time_s{};
};

// Fixed-capacity single-producer/single-consumer queue. Storage is allocated
// once, before the control loop starts.
class SampleRing {
 public:
  explicit SampleRing(size_t capacity) : slots_(capacity + 1) {}

  // Producer (control loop). Returns false, without blocking, when full.
  bool push(const Sample& sample) noexcept {
    const size_t head = head_.load(std::memory_order_relaxed);
    const size_t next = (head + 1) % slots_.size();
    if (next == tail_.load(std::memory_order_acquire)) {
      return false;
    }
    slots_[head] = sample;
    head_.store(next, std::memory_order_release);
    return true;
  }

  // Consumer (publisher thread).
  bool pop(Sample& sample) noexcept {
    const size_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_.load(std::memory_order_acquire)) {
      return false;
    }
    sample = slots_[tail];
    tail_.store((tail + 1) % slots_.size(), std::memory_order_release);
    return true;
  }

 private:
  std::vector<Sample> slots_;
  alignas(64) std::atomic<size_t> head_{0};
  alignas(64) std::atomic<size_t> tail_{0};
};

// Published once (latched) so every recording can store the exact setup.
struct SessionInfo {
  std::string config_path;
  std::string config_text;
};

class Publisher {
 public:
  virtual ~Publisher() = default;
  // Real-time safe: copies into the ring; drops and counts when it is full.
  virtual void push(const Sample& sample) noexcept = 0;
};

// Starts the publisher and its thread. Implemented by src/ros_telemetry.cpp
// (ROS 2) or tests/sim/null_telemetry.cpp (simulation, publishes nothing).
std::unique_ptr<Publisher> make_publisher(const SessionInfo& info);

}  // namespace telemetry
