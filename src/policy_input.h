#pragma once
// Actions from a learned policy (vla/baku/baku/franka/deploy.py) over UDP on this PC.
//
// One datagram per policy step, 80 bytes, little-endian:
//   char     magic[4]        "FPA1"
//   uint32   flags           bit 0: enabled (apply the action; 0 = hold still)
//   uint64   sequence        increments per policy step
//   uint64   home_requests   increments to request the saved home pose
//   double   action[7]       TeleMoMa action [dx dy dz droll dpitch dyaw gripper]:
//                            base frame, target - current position [m], scipy "xyz"
//                            Euler of R_target R_current^T [rad], gripper 1 = open
//
// A reader thread receives; the control loop only calls latest(), which never
// blocks (try_lock), as quest_input.h does.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace policy_link {

inline constexpr char kMagic[4] = {'F', 'P', 'A', '1'};
inline constexpr size_t kPacketSize = 80;

struct Packet {
  uint64_t sequence{};  // 0 = nothing received yet
  uint64_t home_requests{};
  bool enabled{};
  std::array<double, 7> action{};
  int64_t receive_steady_ns{};
  int64_t receive_ns{};  // system clock, for telemetry
};

inline int64_t steady_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

inline int64_t system_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// Parses one datagram; false if it is not a valid packet.
inline bool parse(const unsigned char* data, size_t size, Packet& out) {
  static_assert(sizeof(double) == 8);
  if (size != kPacketSize || std::memcmp(data, kMagic, 4) != 0) {
    return false;
  }
  uint32_t flags = 0;
  std::memcpy(&flags, data + 4, 4);
  std::memcpy(&out.sequence, data + 8, 8);
  std::memcpy(&out.home_requests, data + 16, 8);
  std::memcpy(out.action.data(), data + 24, 56);
  out.enabled = (flags & 1U) != 0;
  for (const double value : out.action) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

class Receiver {
 public:
  Receiver(std::string host, int port) : host_(std::move(host)), port_(port) {}
  ~Receiver() { stop(); }
  Receiver(const Receiver&) = delete;
  Receiver& operator=(const Receiver&) = delete;

  void start() {
    socket_ = ::socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (socket_ < 0) {
      throw std::runtime_error(std::string("policy input: socket: ") + std::strerror(errno));
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port_));
    if (inet_pton(AF_INET, host_.c_str(), &address.sin_addr) != 1) {
      throw std::runtime_error("policy input: policy_host must be an IPv4 address, got '" + host_ + "'");
    }
    if (::bind(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
      throw std::runtime_error("policy input: cannot listen on " + host_ + ":" + std::to_string(port_) + ": " +
                               std::strerror(errno));
    }
    reader_ = std::jthread([this](const std::stop_token& stop) { read_loop(stop); });
  }

  void stop() {
    reader_.request_stop();
    if (reader_.joinable()) reader_.join();
    if (socket_ >= 0) {
      ::close(socket_);
      socket_ = -1;
    }
  }

  // Non-blocking copy of the newest packet; false if none yet or the writer holds the lock.
  bool latest(Packet& out) const {
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock() || latest_.sequence == 0) {
      return false;
    }
    out = latest_;
    return true;
  }

  uint64_t rejected() const { return rejected_.load(); }

 private:
  void read_loop(const std::stop_token& stop) {
    std::array<unsigned char, 512> buffer{};
    while (!stop.stop_requested()) {
      pollfd descriptor{socket_, POLLIN, 0};
      if (::poll(&descriptor, 1, 100) <= 0) {
        continue;
      }
      const ssize_t size = ::recv(socket_, buffer.data(), buffer.size(), 0);
      Packet packet;
      if (size <= 0 || !parse(buffer.data(), static_cast<size_t>(size), packet) || packet.sequence == 0) {
        rejected_.fetch_add(1);
        continue;
      }
      packet.receive_steady_ns = steady_ns();
      packet.receive_ns = system_ns();
      std::lock_guard lock(mutex_);
      latest_ = packet;
    }
  }

  std::string host_;
  int port_;
  int socket_{-1};
  mutable std::mutex mutex_;
  Packet latest_;
  std::atomic<uint64_t> rejected_{0};
  std::jthread reader_;
};

}  // namespace policy_link
