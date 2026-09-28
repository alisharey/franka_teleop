cmake_minimum_required(VERSION 3.22)

project(franka_gamepad_teleop LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Franka CONFIG REQUIRED HINTS
  "$ENV{Franka_DIR}"
  "/opt/ros/humble/lib/x86_64-linux-gnu/cmake/Franka"
)

add_executable(controller_probe src/controller_probe.cpp)
add_executable(gamepad_teleop src/gamepad_teleop.cpp)
target_link_libraries(gamepad_teleop PRIVATE Franka::Franka)
add_executable(pose_monitor src/pose_monitor.cpp)
target_link_libraries(pose_monitor PRIVATE Franka::Franka)
target_link_options(pose_monitor PRIVATE "-Wl,--disable-new-dtags")
# The ROS-packaged libfranka has transitive libraries in this directory without
# its own RUNPATH. DT_RPATH keeps the build executable runnable without relying
# on a user's shell having sourced ROS 2.
target_link_options(gamepad_teleop PRIVATE "-Wl,--disable-new-dtags")

foreach(target controller_probe gamepad_teleop pose_monitor)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
endforeach()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME controller_probe_help COMMAND controller_probe --help)
  add_test(NAME gamepad_teleop_help COMMAND gamepad_teleop --help)
  add_test(NAME pose_monitor_help COMMAND pose_monitor --help)
  add_test(NAME gamepad_teleop_validates_nonhardware_config
    COMMAND gamepad_teleop --config ${CMAKE_CURRENT_SOURCE_DIR}/tests/valid.conf --check-config)
endif()
