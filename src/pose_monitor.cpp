// Read-only end-effector position monitor for choosing workspace limits.
// It never sends motion commands: it only reads the robot state (robot.read),
// the same O_T_EE position that gamepad_teleop checks against its workspace.
#include <signal.h>

#include <franka/exception.h>
#include <franka/robot.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

volatile sig_atomic_t g_stop_requested = 0;
void request_stop(int) { g_stop_requested = 1; }

struct Limits {
  std::array<double, 3> minimum{};
  std::array<double, 3> maximum{};
};

// Reads workspace_x_m / _y_m / _z_m from a teleop config, if present.
std::optional<Limits> read_limits(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("cannot open config '" + path + "'");
  }
  const std::array<std::string, 3> keys{"workspace_x_m", "workspace_y_m", "workspace_z_m"};
  Limits limits{};
  std::array<bool, 3> found{};
  std::string line;
  while (std::getline(stream, line)) {
    line = line.substr(0, line.find('#'));
    const size_t separator = line.find('=');
    if (separator == std::string::npos) {
      continue;
    }
    std::string key = line.substr(0, separator);
    key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
    for (size_t axis = 0; axis < 3; ++axis) {
      if (key == keys[axis]) {
        std::string value = line.substr(separator + 1);
        std::replace(value.begin(), value.end(), ',', ' ');
        std::istringstream values(value);
        if (values >> limits.minimum[axis] >> limits.maximum[axis]) {
          found[axis] = true;
        }
      }
    }
  }
  if (!(found[0] && found[1] && found[2])) {
    return std::nullopt;
  }
  return limits;
}

void print_usage(const char* name) {
  std::cout << "Usage: " << name << " --robot-ip IP [--config config/teleop.conf]\n\n"
            << "Read-only: never moves the robot. Guide the arm by hand through the space you\n"
            << "want; live x/y/z (robot base frame, metres) and the min/max seen are shown.\n"
            << "With --config, each axis is marked OUT when outside the configured limits.\n"
            << "Ctrl-C prints the min/max seen as workspace lines for config/teleop.conf.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string robot_ip;
    std::optional<Limits> configured;
    for (int index = 1; index < argc; ++index) {
      const std::string argument(argv[index]);
      if (argument == "--help" || argument == "-h") {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
      }
      if (argument == "--robot-ip" && index + 1 < argc) {
        robot_ip = argv[++index];
      } else if (argument == "--config" && index + 1 < argc) {
        configured = read_limits(argv[++index]);
        if (!configured) {
          throw std::runtime_error("config has no complete workspace_x_m/_y_m/_z_m lines");
        }
      } else {
        throw std::runtime_error("invalid argument '" + argument + "'; use --help");
      }
    }
    if (robot_ip.empty()) {
      throw std::runtime_error("supply --robot-ip; use --help");
    }

    struct sigaction action {};
    action.sa_handler = request_stop;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);

    // kIgnore: reading state needs no realtime kernel.
    franka::Robot robot(robot_ip, franka::RealtimeConfig::kIgnore);
    std::cout << "Connected (read-only). Guide the arm by hand; Ctrl-C to finish.\n";
    if (configured) {
      std::printf("Configured limits: x [%.3f, %.3f]  y [%.3f, %.3f]  z [%.3f, %.3f]\n",
                  configured->minimum[0], configured->maximum[0], configured->minimum[1],
                  configured->maximum[1], configured->minimum[2], configured->maximum[2]);
    }

    Limits seen{};
    seen.minimum.fill(std::numeric_limits<double>::infinity());
    seen.maximum.fill(-std::numeric_limits<double>::infinity());
    uint64_t sample = 0;
    constexpr std::array<const char*, 3> kNames{"x", "y", "z"};
    robot.read([&](const franka::RobotState& state) {
      for (size_t axis = 0; axis < 3; ++axis) {
        const double position = state.O_T_EE[12 + axis];
        seen.minimum[axis] = std::min(seen.minimum[axis], position);
        seen.maximum[axis] = std::max(seen.maximum[axis], position);
      }
      if (sample++ % 100 == 0) {  // State arrives at 1 kHz; print at 10 Hz.
        std::string line;
        char field[96];
        for (size_t axis = 0; axis < 3; ++axis) {
          const double position = state.O_T_EE[12 + axis];
          const bool outside = configured && (position < configured->minimum[axis] ||
                                              position > configured->maximum[axis]);
          std::snprintf(field, sizeof(field), "%s=%+.3f%s  ", kNames[axis], position,
                        outside ? " OUT" : "    ");
          line += field;
        }
        std::snprintf(field, sizeof(field), "| seen x[%+.3f,%+.3f] y[%+.3f,%+.3f] z[%+.3f,%+.3f]",
                      seen.minimum[0], seen.maximum[0], seen.minimum[1], seen.maximum[1],
                      seen.minimum[2], seen.maximum[2]);
        line += field;
        std::cout << '\r' << line << "   " << std::flush;
      }
      return g_stop_requested == 0;
    });

    std::cout << "\n\nMin/max seen. Paste into config/teleop.conf (the teleop stops about 2 cm\n"
                 "inside these, so extend each side a little if you need to reach the edge):\n";
    std::printf("workspace_x_m = %.3f, %.3f\n", seen.minimum[0], seen.maximum[0]);
    std::printf("workspace_y_m = %.3f, %.3f\n", seen.minimum[1], seen.maximum[1]);
    std::printf("workspace_z_m = %.3f, %.3f\n", seen.minimum[2], seen.maximum[2]);
    return EXIT_SUCCESS;
  } catch (const franka::Exception& error) {
    std::cerr << "\nlibfranka error: " << error.what() << '\n';
    return EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "\nerror: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
