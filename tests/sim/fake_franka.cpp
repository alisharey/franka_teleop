// Simulated FR3 standing in for libfranka's Robot/Model/Gripper, so the real
// gamepad_teleop binary can run end to end off-robot. Each 1 ms tick it applies
// the checks the real robot enforces on a joint-velocity command and throws a
// ControlException (like a reflex) on any violation:
//   * |dq| within the position-dependent velocity bounds (libfranka formula),
//   * joint acceleration <= 10 rad/s^2 and jerk <= 5000 rad/s^3,
//   * joint positions inside the URDF limits,
//   * finite values, and zero velocity when the motion finishes.
// It also writes a CSV trace (env SIM_TRACE) and a summary on stderr.
// Test-only; never linked into the robot build.
#include <franka/active_control_base.h>
#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>
#include <sys/ioctl.h>
#include <linux/joystick.h>

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include "../fr3_sim.h"

namespace franka {

class Robot::Impl {
 public:
  fr3_sim::Vec7 q{0.036286, -1.367388, -0.180505, -2.794799, -0.043301, 2.076045, 0.407056};
};

class Network {};

Robot::Robot(const std::string&, RealtimeConfig, size_t) : impl_(std::make_shared<Impl>()) {}
Robot::~Robot() noexcept = default;

static RobotState make_state(const fr3_sim::Vec7& q) {
  RobotState state{};
  state.q = q;
  state.q_d = q;
  const auto f = fr3_sim::frames(q);
  state.O_T_EE = f[8];
  state.robot_mode = RobotMode::kIdle;
  return state;
}

RobotState Robot::readOnce() { return make_state(impl_->q); }
void Robot::automaticErrorRecovery() {}

auto Robot::getUpperJointVelocityLimits(const std::array<double, kNumJoints>& q) -> std::array<double, kNumJoints> {
  return fr3_sim::upper_bound(q, 0.0);
}
auto Robot::getLowerJointVelocityLimits(const std::array<double, kNumJoints>& q) -> std::array<double, kNumJoints> {
  return fr3_sim::lower_bound(q, 0.0);
}

Model Robot::loadModel() { return Model(std::unique_ptr<RobotModelBase>{}); }
Model::Model(std::unique_ptr<RobotModelBase> robot_model) : robot_model_(std::move(robot_model)) {}
Model::Model(Model&&) noexcept = default;
Model::~Model() noexcept = default;
std::array<double, 42> Model::zeroJacobian(Frame, const RobotState& state) const {
  return fr3_sim::jacobian(state.q);
}

void Robot::control(std::function<JointVelocities(const RobotState&, franka::Duration)> callback,
                    ControllerMode, bool, double) {
  using clock = std::chrono::steady_clock;
  const char* trace_path = std::getenv("SIM_TRACE");
  FILE* trace = trace_path ? std::fopen(trace_path, "w") : nullptr;
  if (trace) std::fprintf(trace, "t,x,y,z,yaw,max_abs_dq,max_abs_ddq,max_abs_jerk,min_limit_dist,j1,j2,j3,j4,j5,j6,j7\n");
  fr3_sim::Vec7 q = impl_->q;
  fr3_sim::Vec7 dq{}, ddq{};
  double peak_speed = 0, peak_acc = 0, peak_jerk = 0, min_dist = 1e9;
  const auto start = clock::now();
  auto fail = [&](const std::string& why) {
    if (trace) std::fclose(trace);
    impl_->q = q;
    throw ControlException("simulated reflex: " + why);
  };
  for (uint64_t tick = 0;; ++tick) {
    std::this_thread::sleep_until(start + std::chrono::milliseconds(tick));  // real-time pacing
    RobotState state = make_state(q);
    state.dq = dq;
    state.dq_d = dq;
    state.ddq_d = ddq;
    state.robot_mode = RobotMode::kMove;
    const JointVelocities command = callback(state, Duration(tick == 0 ? 0 : 1));
    const auto upper = fr3_sim::upper_bound(q, 0.0), lower = fr3_sim::lower_bound(q, 0.0);
    double speed = 0, acc_max = 0, jerk_max = 0;
    fr3_sim::Vec7 new_ddq{};
    for (size_t i = 0; i < 7; ++i) {
      const double v = command.dq[i];
      if (!std::isfinite(v)) fail("non-finite joint velocity");
      if (v > std::max(upper[i], 0.0) + 1e-9 || v < std::min(lower[i], 0.0) - 1e-9)
        fail("joint_velocity_violation on joint " + std::to_string(i + 1));
      new_ddq[i] = (v - dq[i]) / fr3_sim::kDt;
      const double jerk = (new_ddq[i] - ddq[i]) / fr3_sim::kDt;
      if (std::abs(new_ddq[i]) > 10.0 + 1e-6) fail("joint_acceleration discontinuity on joint " + std::to_string(i + 1));
      if (std::abs(jerk) > 5000.0 + 1e-3) fail("joint_jerk discontinuity on joint " + std::to_string(i + 1));
      speed = std::max(speed, std::abs(v));
      acc_max = std::max(acc_max, std::abs(new_ddq[i]));
      jerk_max = std::max(jerk_max, std::abs(jerk));
    }
    for (size_t i = 0; i < 7; ++i) {
      dq[i] = command.dq[i];
      ddq[i] = new_ddq[i];
      q[i] += dq[i] * fr3_sim::kDt;
      const double d = std::min(fr3_sim::kQUpper[i] - q[i], q[i] - fr3_sim::kQLower[i]);
      if (d < 0) fail("joint_position_limits_violation on joint " + std::to_string(i + 1));
      min_dist = std::min(min_dist, d);
    }
    peak_speed = std::max(peak_speed, speed);
    peak_acc = std::max(peak_acc, acc_max);
    peak_jerk = std::max(peak_jerk, jerk_max);
    if (trace && tick % 10 == 0) {
      const auto p = fr3_sim::position(q);
      const auto tcp = fr3_sim::frames(q)[8];
      const double yaw = std::atan2(tcp[1], tcp[0]);  // heading of the tool x-axis in the base frame
      std::fprintf(trace, "%.3f,%.5f,%.5f,%.5f,%.5f,%.5f,%.4f,%.2f,%.4f", tick * 1e-3, p[0], p[1], p[2], yaw, speed,
                   acc_max, jerk_max, min_dist);
      for (double v : q) std::fprintf(trace, ",%.4f", v);
      std::fprintf(trace, "\n");
    }
    if (command.motion_finished) {
      if (speed > 1e-3) fail("motion finished with non-zero velocity");
      break;
    }
    if (tick > 180000) fail("simulation timeout");
  }
  if (trace) std::fclose(trace);
  impl_->q = q;
  const auto p = fr3_sim::position(q);
  std::fprintf(stderr,
               "[sim] finished cleanly. peak joint speed %.3f rad/s, peak accel %.2f rad/s^2, peak jerk %.1f rad/s^3, "
               "closest to a joint limit %.3f rad, final TCP (%.3f, %.3f, %.3f)\n",
               peak_speed, peak_acc, peak_jerk, min_dist, p[0], p[1], p[2]);
}

// Unused virtual entry points (the teleop only uses control()).
using MoveMode = research_interface::robot::Move::ControllerMode;
std::unique_ptr<ActiveControlBase> Robot::startTorqueControl() { throw std::runtime_error("sim: unused"); }
std::unique_ptr<ActiveControlBase> Robot::startJointPositionControl(const MoveMode&) { throw std::runtime_error("sim: unused"); }
std::unique_ptr<ActiveControlBase> Robot::startAsyncJointPositionControl(const MoveMode&, const std::optional<std::vector<double>>&) { throw std::runtime_error("sim: unused"); }
std::unique_ptr<ActiveControlBase> Robot::startJointVelocityControl(const MoveMode&) { throw std::runtime_error("sim: unused"); }
std::unique_ptr<ActiveControlBase> Robot::startCartesianPoseControl(const MoveMode&) { throw std::runtime_error("sim: unused"); }
std::unique_ptr<ActiveControlBase> Robot::startCartesianVelocityControl(const MoveMode&) { throw std::runtime_error("sim: unused"); }

Gripper::Gripper(const std::string&) { throw std::runtime_error("no gripper in simulation"); }
Gripper::~Gripper() noexcept = default;
bool Gripper::grasp(double, double, double, double, double) const { return false; }
bool Gripper::move(double, double) const { return false; }

}  // namespace franka

// The simulated joystick is a FIFO; answer the joystick ioctls it cannot.
extern "C" int __real_ioctl(int fd, unsigned long request, ...);
extern "C" int __wrap_ioctl(int fd, unsigned long request, ...) {
  va_list args;
  va_start(args, request);
  void* argument = va_arg(args, void*);
  va_end(args);
  if (request == JSIOCGAXES) {
    *static_cast<unsigned char*>(argument) = 8;
    return 0;
  }
  if (request == JSIOCGBUTTONS) {
    *static_cast<unsigned char*>(argument) = 15;
    return 0;
  }
  return __real_ioctl(fd, request, argument);
}
