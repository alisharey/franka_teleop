cmake_minimum_required(VERSION 3.22)

project(franka_teleop LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Franka CONFIG REQUIRED HINTS
  "$ENV{Franka_DIR}"
  "/opt/ros/humble/lib/x86_64-linux-gnu/cmake/Franka"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TeleopState.msg"
  "msg/QuestState.msg"
  DEPENDENCIES std_msgs builtin_interfaces
)
rosidl_get_typesupport_target(teleop_msgs_cpp ${PROJECT_NAME} rosidl_typesupport_cpp)

# Commit of the source tree, refreshed on every build (cmake/git_version.cmake).
add_custom_target(git_version
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
          -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/generated/git_version.h
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_version.cmake
  BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/generated/git_version.h
)

add_executable(controller_probe src/controller_probe.cpp)
add_executable(gamepad_teleop src/gamepad_teleop.cpp src/ros_telemetry.cpp)
add_dependencies(gamepad_teleop git_version)
target_include_directories(gamepad_teleop PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(gamepad_teleop PRIVATE
  Franka::Franka rclcpp::rclcpp ${std_msgs_TARGETS} ${sensor_msgs_TARGETS} "${teleop_msgs_cpp}")
add_executable(episode_replay src/episode_replay.cpp)
target_link_libraries(episode_replay PRIVATE Franka::Franka)
target_link_options(episode_replay PRIVATE "-Wl,--disable-new-dtags")
add_executable(home_gripper src/home_gripper.cpp)
target_link_libraries(home_gripper PRIVATE Franka::Franka)
target_link_options(home_gripper PRIVATE "-Wl,--disable-new-dtags")
add_executable(pose_monitor src/pose_monitor.cpp)
target_link_libraries(pose_monitor PRIVATE Franka::Franka)
# The ROS-packaged libfranka has transitive libraries in this directory without
# its own RUNPATH. DT_RPATH keeps the executables runnable without relying on a
# user's shell having sourced ROS 2.
target_link_options(pose_monitor PRIVATE "-Wl,--disable-new-dtags")
target_link_options(gamepad_teleop PRIVATE "-Wl,--disable-new-dtags")

foreach(target controller_probe gamepad_teleop pose_monitor episode_replay home_gripper)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
endforeach()

install(TARGETS controller_probe gamepad_teleop pose_monitor episode_replay home_gripper DESTINATION lib/${PROJECT_NAME})

# Recorder: Python node + web UI.
ament_python_install_package(franka_teleop_recorder)
install(PROGRAMS scripts/recorder scripts/episode_actions DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config web DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  enable_testing()
  add_test(NAME controller_probe_help COMMAND controller_probe --help)
  add_test(NAME gamepad_teleop_help COMMAND gamepad_teleop --help)
  add_test(NAME pose_monitor_help COMMAND pose_monitor --help)
  add_test(NAME episode_replay_help COMMAND episode_replay --help)
  add_executable(diff_ik_test tests/diff_ik_test.cpp)
  target_compile_options(diff_ik_test PRIVATE -O2 -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME diff_ik_simulation COMMAND diff_ik_test)
  add_test(NAME gamepad_teleop_validates_nonhardware_config
    COMMAND gamepad_teleop --config ${CMAKE_CURRENT_SOURCE_DIR}/tests/valid.conf --check-config)
  add_executable(telemoma_policy_test tests/telemoma/telemoma_policy_test.cpp)
  target_compile_options(telemoma_policy_test PRIVATE -O2 -Wall -Wextra -Wpedantic -Werror)
  # The C++ port against TeleMoMa's own Python OculusPolicy (needs third_party/telemoma).
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/telemoma/telemoma/human_interface/oculus.py)
    add_test(NAME telemoma_policy_matches_telemoma
      COMMAND python3 -s ${CMAKE_CURRENT_SOURCE_DIR}/tests/telemoma/compare_with_telemoma.py
              $<TARGET_FILE:telemoma_policy_test>)
  endif()
  add_test(NAME recorder_unit_tests
    COMMAND python3 -s -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/tests/recorder -v)
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
