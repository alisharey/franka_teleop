// ROS 2 side of the telemetry (see telemetry.h). Runs entirely outside the
// 1 kHz control loop: a normal-priority thread drains the sample ring and
// publishes
//   /teleop/state    franka_teleop/TeleopState, every control tick (1 kHz)
//   /teleop/joy      sensor_msgs/Joy, normalized gamepad, 100 Hz and on every button change
//   /teleop/session  std_msgs/String (latched JSON): config path and text, git commit
//   /teleop/quest    franka_teleop/QuestState, every control tick, with input = quest

#include "telemetry.h"

#include "git_version.h"

#include <franka_teleop/msg/quest_state.hpp>
#include <franka_teleop/msg/teleop_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <std_msgs/msg/string.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stop_token>
#include <thread>

namespace telemetry {
namespace {

constexpr size_t kRingCapacity = 4096;  // 4 s of ticks
constexpr int64_t kJoyPeriodNs = 10'000'000;

std::string json_escape(const std::string& text) {
  std::string escaped;
  escaped.reserve(text.size() + 16);
  for (const char character : text) {
    switch (character) {
      case '"': escaped += "\\\""; break;
      case '\\': escaped += "\\\\"; break;
      case '\n': escaped += "\\n"; break;
      case '\r': escaped += "\\r"; break;
      case '\t': escaped += "\\t"; break;
      default:
        if (static_cast<unsigned char>(character) < 0x20) {
          char code[8];
          std::snprintf(code, sizeof(code), "\\u%04x", character);
          escaped += code;
        } else {
          escaped += character;
        }
    }
  }
  return escaped;
}

std::string errors_text(const franka::Errors& errors) {
  return errors ? static_cast<std::string>(errors) : std::string("[]");
}

template <size_t N>
std::array<float, N * 3> flatten(const std::array<std::array<float, 3>, N>& rows) {
  std::array<float, N * 3> flat{};
  for (size_t row = 0; row < N; ++row) {
    std::copy(rows[row].begin(), rows[row].end(), flat.begin() + static_cast<std::ptrdiff_t>(row * 3));
  }
  return flat;
}

class RosPublisher final : public Publisher {
 public:
  explicit RosPublisher(const SessionInfo& info) : ring_(kRingCapacity) {
    // The teleop keeps its own SIGINT/SIGTERM handling (Ctrl-C stops the arm).
    rclcpp::init(0, nullptr, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
    node_ = std::make_shared<rclcpp::Node>("gamepad_teleop");
    state_publisher_ = node_->create_publisher<franka_teleop::msg::TeleopState>(
        "/teleop/state", rclcpp::QoS(rclcpp::KeepLast(1000)).reliable());
    joy_publisher_ = node_->create_publisher<sensor_msgs::msg::Joy>("/teleop/joy", rclcpp::QoS(100));
    quest_publisher_ = node_->create_publisher<franka_teleop::msg::QuestState>(
        "/teleop/quest", rclcpp::QoS(rclcpp::KeepLast(1000)).reliable());
    session_publisher_ = node_->create_publisher<std_msgs::msg::String>(
        "/teleop/session", rclcpp::QoS(1).reliable().transient_local());
    std_msgs::msg::String session;
    session.data = std::string("{\"config_path\": \"") + json_escape(info.config_path) +
                   "\", \"config_text\": \"" + json_escape(info.config_text) +
                   "\", \"git_commit\": \"" + kGitCommit + "\", \"git_dirty\": " +
                   (kGitDirty ? "true" : "false") + "}";
    session_publisher_->publish(session);
    thread_ = std::jthread([this](std::stop_token stop_token) { run(stop_token); });
  }

  ~RosPublisher() override {
    thread_.request_stop();
    thread_.join();
    rclcpp::shutdown();
  }

  void push(const Sample& sample) noexcept override {
    if (!ring_.push(sample)) {
      dropped_.fetch_add(1, std::memory_order_relaxed);
    }
  }

 private:
  void run(const std::stop_token& stop_token) {
    Sample sample;
    while (true) {
      bool published = false;
      while (ring_.pop(sample)) {
        publish(sample);
        published = true;
      }
      if (stop_token.stop_requested()) {
        while (ring_.pop(sample)) {
          publish(sample);
        }
        return;
      }
      if (!published) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }

  void publish(const Sample& sample) {
    const franka::RobotState& s = sample.state;
    auto& m = message_;
    m.header.stamp = rclcpp::Time(sample.stamp_ns, RCL_SYSTEM_TIME);
    m.header.frame_id = "base";
    m.tick = sample.tick;
    m.period = sample.period_s;
    m.dropped = dropped_.load(std::memory_order_relaxed);
    m.robot_time = s.time.toSec();
    m.o_t_ee = s.O_T_EE;
    m.o_t_ee_d = s.O_T_EE_d;
    m.f_t_ee = s.F_T_EE;
    m.f_t_ne = s.F_T_NE;
    m.ne_t_ee = s.NE_T_EE;
    m.ee_t_k = s.EE_T_K;
    m.m_ee = s.m_ee;
    m.i_ee = s.I_ee;
    m.f_x_cee = s.F_x_Cee;
    m.m_load = s.m_load;
    m.i_load = s.I_load;
    m.f_x_cload = s.F_x_Cload;
    m.m_total = s.m_total;
    m.i_total = s.I_total;
    m.f_x_ctotal = s.F_x_Ctotal;
    m.elbow = s.elbow;
    m.elbow_d = s.elbow_d;
    m.elbow_c = s.elbow_c;
    m.delbow_c = s.delbow_c;
    m.ddelbow_c = s.ddelbow_c;
    m.tau_j = s.tau_J;
    m.tau_j_d = s.tau_J_d;
    m.dtau_j = s.dtau_J;
    m.q = s.q;
    m.q_d = s.q_d;
    m.dq = s.dq;
    m.dq_d = s.dq_d;
    m.ddq_d = s.ddq_d;
    m.joint_contact = s.joint_contact;
    m.cartesian_contact = s.cartesian_contact;
    m.joint_collision = s.joint_collision;
    m.cartesian_collision = s.cartesian_collision;
    m.tau_ext_hat_filtered = s.tau_ext_hat_filtered;
    m.o_f_ext_hat_k = s.O_F_ext_hat_K;
    m.k_f_ext_hat_k = s.K_F_ext_hat_K;
    m.o_dp_ee_d = s.O_dP_EE_d;
    m.o_ddp_o = s.O_ddP_O;
    m.o_t_ee_c = s.O_T_EE_c;
    m.o_dp_ee_c = s.O_dP_EE_c;
    m.o_ddp_ee_c = s.O_ddP_EE_c;
    m.theta = s.theta;
    m.dtheta = s.dtheta;
    m.accelerometer_top = flatten(s.accelerometer_top);
    m.accelerometer_bottom = flatten(s.accelerometer_bottom);
    m.current_errors = errors_text(s.current_errors);
    m.last_motion_errors = errors_text(s.last_motion_errors);
    m.control_command_success_rate = s.control_command_success_rate;
    m.robot_mode = static_cast<uint8_t>(s.robot_mode);

    m.deadman = sample.deadman;
    m.rotation_mode = sample.rotation_mode;
    m.motion_enabled = sample.motion_enabled;
    m.speed_scale = sample.speed_scale;
    m.target_twist = sample.target_twist;
    m.shaped_twist = sample.shaped_twist;
    m.ik_dq = sample.ik_dq;
    m.fence_scale = sample.fence_scale;
    m.nullspace_weight = sample.nullspace_weight;
    m.ik_strict_used = sample.ik_strict_used;
    m.commanded_dq = sample.commanded_dq;

    m.axes.assign(sample.axes.begin(), sample.axes.begin() + sample.axis_count);
    m.buttons.assign(sample.buttons.begin(), sample.buttons.begin() + sample.button_count);

    m.gripper_command = sample.gripper_command;
    m.gripper_command_count = sample.gripper_command_count;
    m.gripper_width = sample.gripper_width;
    m.gripper_max_width = sample.gripper_max_width;
    m.gripper_is_grasped = sample.gripper_is_grasped;
    m.gripper_temperature = sample.gripper_temperature;
    m.gripper_time = sample.gripper_time_s;
    state_publisher_->publish(m);

    const bool buttons_changed = m.buttons != last_joy_buttons_;
    if (buttons_changed || sample.stamp_ns - last_joy_ns_ >= kJoyPeriodNs) {
      sensor_msgs::msg::Joy joy;
      joy.header = m.header;
      joy.header.frame_id = sample.input_kind == 1 ? "quest" : "gamepad";
      joy.axes.reserve(sample.axis_count);
      for (size_t axis = 0; axis < sample.axis_count; ++axis) {
        joy.axes.push_back(static_cast<float>(sample.axes[axis]) / 32767.0F);
      }
      joy.buttons.assign(m.buttons.begin(), m.buttons.end());
      joy_publisher_->publish(joy);
      last_joy_ns_ = sample.stamp_ns;
      last_joy_buttons_ = m.buttons;
    }
    if (sample.input_kind == 1) {
      publish_quest(sample);
    }
  }

  void publish_quest(const Sample& sample) {
    const QuestSample& q = sample.quest;
    auto& m = quest_message_;
    m.header = message_.header;
    m.tick = sample.tick;
    m.reading_valid = q.reading_valid;
    m.reading_sequence = q.reading_sequence;
    m.reading_age = q.reading_age_s;
    m.reading_stamp = rclcpp::Time(q.reading_stamp_ns, RCL_SYSTEM_TIME);
    m.right_present = q.right_present;
    m.pose_valid = q.pose_valid;
    m.controller_pose = q.controller_pose;
    m.button_a = q.button_a;
    m.button_b = q.button_b;
    m.trigger_pressed = q.trigger_pressed;
    m.grip_pressed = q.grip_pressed;
    m.joystick_click = q.joystick_click;
    m.trigger_touch = q.trigger_touch;
    m.trigger = q.trigger;
    m.grip = q.grip;
    m.joystick = q.joystick;
    m.policy_active = q.policy_active;
    m.movement_enabled = q.movement_enabled;
    m.action = q.action;
    m.target = q.target;
    m.robot_origin = q.robot_origin;
    m.vr_state = q.vr_state;
    m.vr_origin = q.vr_origin;
    m.vr_to_global = q.vr_to_global;
    quest_publisher_->publish(m);
  }

  SampleRing ring_;
  std::atomic<uint64_t> dropped_{0};
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<franka_teleop::msg::TeleopState>::SharedPtr state_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Joy>::SharedPtr joy_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr session_publisher_;
  rclcpp::Publisher<franka_teleop::msg::QuestState>::SharedPtr quest_publisher_;
  franka_teleop::msg::QuestState quest_message_;  // Reused; publisher thread only.
  franka_teleop::msg::TeleopState message_;  // Reused; publisher thread only.
  int64_t last_joy_ns_{0};
  std::vector<bool> last_joy_buttons_;
  std::jthread thread_;
};

}  // namespace

std::unique_ptr<Publisher> make_publisher(const SessionInfo& info) {
  return std::make_unique<RosPublisher>(info);
}

}  // namespace telemetry
