// Reads the arm's current joint positions (no motion) and saves them as
// `home_joints` in the teleop config: the pose the Quest stick-flick returns to.
// Usage: save_home_pose ROBOT_IP CONFIG   (see tools/save_home_pose.sh)
#include <franka/exception.h>
#include <franka/robot.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "Usage: save_home_pose ROBOT_IP CONFIG\nSaves the arm's current joints as home_joints in CONFIG.\n";
    return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 1;
  }
  try {
    franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);
    const franka::RobotState state = robot.readOnce();
    char line[256];
    std::snprintf(line, sizeof(line), "home_joints = %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f", state.q[0], state.q[1],
                  state.q[2], state.q[3], state.q[4], state.q[5], state.q[6]);
    std::ifstream in(argv[2]);
    if (!in) {
      std::cerr << "cannot read " << argv[2] << '\n';
      return 1;
    }
    std::vector<std::string> lines;
    std::string text;
    bool replaced = false;
    while (std::getline(in, text)) {
      if (text.rfind("home_joints", 0) == 0) {
        lines.push_back(line);
        replaced = true;
      } else {
        lines.push_back(text);
      }
    }
    in.close();
    if (!replaced) {
      lines.push_back("");
      lines.push_back("# Home pose for the Quest stick-flick (saved by tools/save_home_pose.sh).");
      lines.push_back(line);
    }
    std::ofstream out(argv[2]);
    for (const auto& entry : lines) out << entry << '\n';
    std::cout << line << "\n(end effector at x " << state.O_T_EE[12] << ", y " << state.O_T_EE[13] << ", z "
              << state.O_T_EE[14] << " m) saved to " << argv[2] << '\n';
    return 0;
  } catch (const franka::Exception& error) {
    std::cerr << "libfranka error: " << error.what() << '\n';
    return 1;
  }
}
