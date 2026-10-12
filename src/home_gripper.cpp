// Homes the Franka Hand (fingers close and open fully once). See tools/home_gripper.sh.
#include <franka/exception.h>
#include <franka/gripper.h>

#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2 || std::string(argv[1]) == "--help") {
    std::cout << "Usage: home_gripper ROBOT_IP\nHomes the Franka Hand: the fingers close and open fully once.\n";
    return argc == 2 ? 0 : 1;
  }
  try {
    franka::Gripper gripper(argv[1]);
    std::cout << "Before: max width " << gripper.readOnce().max_width << " m. Homing (fingers will move)...\n";
    gripper.homing();
    const franka::GripperState state = gripper.readOnce();
    std::cout << "Done: max width " << state.max_width << " m, width " << state.width << " m.\n";
    return state.max_width > 0.01 ? 0 : 1;
  } catch (const franka::Exception& error) {
    std::cerr << "gripper error: " << error.what() << '\n';
    return 1;
  }
}
