// Feeds an event file through the C++ port (src/telemoma_policy.h) and prints one
// action per "A" event. compare_with_telemoma.py runs TeleMoMa's own Python
// OculusPolicy on the same events and compares.
//   R <16 pose values, row-major> <grip 0/1> <joystick click 0/1> <trigger>
//   A <16 robot pose values, row-major> <gripper_position>
#include "../../src/telemoma_policy.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: telemoma_policy_test EVENTS\n";
    return 2;
  }
  std::ifstream input(argv[1]);
  telemoma::OculusPolicy policy;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream fields(line);
    char kind = 0;
    fields >> kind;
    telemoma::Mat4 m{};
    for (double& v : m) fields >> v;
    if (kind == 'R') {
      int grip = 0, click = 0;
      double trigger = 0;
      fields >> grip >> click >> trigger;
      policy.on_reading(m, {grip != 0, click != 0, trigger});
    } else if (kind == 'A') {
      double gripper = 0;
      fields >> gripper;
      const auto action = policy.calculate_action(m, gripper);
      for (double v : action) std::printf("%.12f ", v);
      std::printf("\n");
    }
  }
  return 0;
}
