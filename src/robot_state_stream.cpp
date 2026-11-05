// Read-only robot state stream for tools/calibration (camera calibration page).
// It never sends motion commands: like pose_monitor it only reads the state
// (robot.read), which works while the arm is hand-guided in Programming mode.
// Prints one JSON line per sample, at most RATE_HZ per second:
//   {"t": robot time [s], "mode": "...", "q": [7], "dq_max": rad/s, "O_T_EE": [16, column-major]}
// Usage: robot_state_stream ROBOT_IP [RATE_HZ]
#include <signal.h>

#include <franka/exception.h>
#include <franka/robot.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

namespace {

volatile sig_atomic_t g_stop_requested = 0;
void request_stop(int) { g_stop_requested = 1; }

const char* mode_name(franka::RobotMode mode) {
  switch (mode) {
    case franka::RobotMode::kOther: return "other";
    case franka::RobotMode::kIdle: return "idle";
    case franka::RobotMode::kMove: return "move";
    case franka::RobotMode::kGuiding: return "guiding";
    case franka::RobotMode::kReflex: return "reflex";
    case franka::RobotMode::kUserStopped: return "user_stopped";
    case franka::RobotMode::kAutomaticErrorRecovery: return "error_recovery";
  }
  return "unknown";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3 || std::string(argv[1]) == "--help") {
    std::cout << "Usage: robot_state_stream ROBOT_IP [RATE_HZ]\n"
                 "Prints the robot state as JSON lines (read-only, no motion).\n";
    return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 1;
  }
  const double rate_hz = argc == 3 ? std::max(1.0, std::stod(argv[2])) : 20.0;
  signal(SIGINT, request_stop);
  signal(SIGTERM, request_stop);
  try {
    franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);
    double last_print = -1.0;
    robot.read([&](const franka::RobotState& state) {
      const double t = state.time.toSec();
      if (last_print < 0.0 || t - last_print >= 1.0 / rate_hz) {
        last_print = t;
        double dq_max = 0.0;
        for (const double v : state.dq) dq_max = std::max(dq_max, std::abs(v));
        std::printf("{\"t\": %.4f, \"mode\": \"%s\", \"dq_max\": %.5f, \"q\": [", t, mode_name(state.robot_mode), dq_max);
        for (size_t i = 0; i < 7; ++i) std::printf("%s%.6f", i ? ", " : "", state.q[i]);
        std::printf("], \"O_T_EE\": [");
        for (size_t i = 0; i < 16; ++i) std::printf("%s%.7f", i ? ", " : "", state.O_T_EE[i]);
        std::printf("]}\n");
        std::fflush(stdout);
      }
      return g_stop_requested == 0;
    });
    return 0;
  } catch (const franka::Exception& error) {
    std::printf("{\"error\": \"%s\"}\n", error.what());
    std::fflush(stdout);
    return 1;
  }
}
