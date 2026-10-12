// Simulation stand-in for src/ros_telemetry.cpp: accepts samples and publishes
// nothing, so the sim builds without ROS 2.
#include "../../src/telemetry.h"

namespace telemetry {
namespace {
class NullPublisher final : public Publisher {
 public:
  void push(const Sample&) noexcept override {}
};
}  // namespace

std::unique_ptr<Publisher> make_publisher(const SessionInfo&) { return std::make_unique<NullPublisher>(); }
}  // namespace telemetry
