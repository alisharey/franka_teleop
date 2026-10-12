#pragma once

// Reads the Meta Quest right controller through ADB, like oculus_reader's
// OculusReader: the oculus_reader app on the headset ("Xr Controllers",
// third_party/oculus_reader) logs each controller pose (relative to the
// headset, row-major 4x4) and the buttons to logcat under tag wE9ryARX; this
// runs `adb logcat` in a background thread and keeps the latest reading.
//
// A maintenance thread keeps the headset awake when it is not worn (proximity
// override), and restarts the app when readings stop and it is not running (the
// Quest 3 build aborts when its VR session is interrupted, and also when it is
// launched again while already running, so it is only started when absent).
//
// The control loop only calls latest(), which never blocks (try_lock).
//
// Connection: "usb" (the first USB device) or "IP[:PORT]" (Wi-Fi ADB; run
// `adb tcpip 5555` once over USB first, see tools/quest/quest_wifi.sh).
// QUEST_ADB overrides the adb executable (the simulation uses a fake one).

#include "telemoma_policy.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

extern char** environ;

namespace quest {

constexpr const char* kPackage = "com.rail.oculus.teleop";
constexpr const char* kActivity = "com.rail.oculus.teleop/com.rail.oculus.teleop.MainActivity";
constexpr const char* kTag = "wE9ryARX";

struct Reading {
  uint64_t sequence{};          // increments with every received line
  int64_t receive_ns{};         // system clock (ROS time)
  int64_t receive_steady_ns{};  // for staleness
  bool right_present{};         // the app reported the right controller ("R")
  bool pose_valid{};            // a real right pose ("r:", not identity)
  telemoma::Mat4 pose = telemoma::identity4();  // right controller in the headset frame, row-major
  bool a{}, b{}, trigger_pressed{}, grip_pressed{}, joystick_click{}, trigger_touch{};
  double trigger{}, grip{}, joystick_x{}, joystick_y{};
};

inline int64_t steady_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

inline int64_t system_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// One logcat payload (the text after "wE9ryARX: "), oculus_reader's format:
//   l:<16 values>|r:<16 values>&R,A,RTr,rightJS x y,rightTrig v,rightGrip v
inline bool parse_payload(const std::string& text, Reading& out) {
  const auto separator = text.find('&');
  if (separator == std::string::npos) {
    return false;
  }
  out.pose_valid = false;
  std::stringstream poses(text.substr(0, separator));
  std::string pair;
  while (std::getline(poses, pair, '|')) {
    if (pair.size() < 3 || pair[0] != 'r' || pair[1] != ':') {
      continue;
    }
    std::stringstream values(pair.substr(2));
    telemoma::Mat4 pose{};
    size_t count = 0;
    while (count < 16 && values >> pose[count]) {
      ++count;
    }
    if (count == 16) {
      out.pose = pose;
      out.pose_valid = pose != telemoma::identity4();
    }
  }
  out.right_present = out.a = out.b = out.trigger_pressed = out.grip_pressed = false;
  out.joystick_click = out.trigger_touch = false;
  std::stringstream buttons(text.substr(separator + 1));
  std::string item;
  while (std::getline(buttons, item, ',')) {
    std::stringstream fields(item);
    std::string key;
    fields >> key;
    if (key == "R") out.right_present = true;
    else if (key == "A") out.a = true;
    else if (key == "B") out.b = true;
    else if (key == "RTr") out.trigger_pressed = true;
    else if (key == "RG") out.grip_pressed = true;
    else if (key == "RJ") out.joystick_click = true;
    else if (key == "RT") out.trigger_touch = true;
    else if (key == "rightTrig") fields >> out.trigger;
    else if (key == "rightGrip") fields >> out.grip;
    else if (key == "rightJS") fields >> out.joystick_x >> out.joystick_y;
  }
  return true;
}

class Reader {
 public:
  explicit Reader(std::string connection) : connection_(std::move(connection)) {
    const char* adb = std::getenv("QUEST_ADB");
    adb_ = adb != nullptr && *adb != '\0' ? adb : "adb";
    if (connection_ != "usb" && !std::regex_match(connection_, std::regex(R"(\d{1,3}(\.\d{1,3}){3}(:\d{1,5})?)"))) {
      throw std::runtime_error("quest_connection must be 'usb' or an IP address like 192.168.1.42[:5555]");
    }
  }
  ~Reader() { stop(); }
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;

  // Finds the headset, keeps it awake and starts the reader app. Throws if no headset.
  void start() {
    if (connection_ == "usb") {
      std::stringstream devices(run({"devices"}));
      std::string line;
      std::getline(devices, line);  // "List of devices attached"
      while (std::getline(devices, line)) {
        std::stringstream fields(line);
        std::string serial, state;
        fields >> serial >> state;
        if (!serial.empty() && serial.find(':') == std::string::npos && state == "device") {
          serial_ = serial;
          break;
        }
        if (!serial.empty() && state == "unauthorized") {
          throw std::runtime_error("Quest is connected but USB debugging is not allowed yet: put the headset on and "
                                   "accept \"Allow USB debugging\"");
        }
      }
      if (serial_.empty()) {
        throw std::runtime_error("no Meta Quest found over USB (adb devices); check the cable and Developer Mode");
      }
    } else {
      serial_ = connection_.find(':') == std::string::npos ? connection_ + ":5555" : connection_;
      const std::string reply = run({"connect", serial_});
      if (reply.find("connected") == std::string::npos) {
        throw std::runtime_error("cannot reach the Quest at " + serial_ + " (" + reply +
                                 "); enable Wi-Fi ADB once over USB with tools/quest/quest_wifi.sh");
      }
    }
    if (run({"-s", serial_, "shell", "pm", "list", "packages", kPackage}).find(kPackage) == std::string::npos) {
      throw std::runtime_error("the oculus_reader app is not installed on the Quest; run tools/quest/quest_monitor.py once");
    }
    keep_awake();
    // Only when not running: re-launching a running instance makes the Quest 3
    // build abort (its "Resumed" assertion).
    if (run({"-s", serial_, "shell", "pidof", kPackage}).empty()) {
      run({"-s", serial_, "shell", "am", "start", "-n", kActivity});
    }
    last_start_ns_ = steady_ns();
    reader_ = std::jthread([this](std::stop_token stop) { read_loop(stop); });
    maintenance_ = std::jthread([this](std::stop_token stop) { maintain(stop); });
  }

  void stop() {
    reader_.request_stop();
    maintenance_.request_stop();
    if (reader_.joinable()) reader_.join();
    if (maintenance_.joinable()) maintenance_.join();
  }

  // Non-blocking copy of the newest reading; false if none yet or the writer holds the lock.
  bool latest(Reading& out) const {
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock() || latest_.sequence == 0) {
      return false;
    }
    out = latest_;
    return true;
  }

  const std::string& serial() const { return serial_; }

 private:
  // Runs adb and returns its stdout (one-shot commands from non-real-time threads).
  std::string run(const std::vector<std::string>& arguments) const {
    int pipe_fds[2];
    if (pipe(pipe_fds) != 0) {
      return "";
    }
    const pid_t pid = spawn(arguments, pipe_fds);
    close(pipe_fds[1]);
    std::string output;
    char buffer[4096];
    ssize_t count;
    while ((count = read(pipe_fds[0], buffer, sizeof(buffer))) > 0) {
      output.append(buffer, static_cast<size_t>(count));
    }
    close(pipe_fds[0]);
    if (pid > 0) {
      int status = 0;
      waitpid(pid, &status, 0);
    }
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) output.pop_back();
    return output;
  }

  pid_t spawn(const std::vector<std::string>& arguments, int pipe_fds[2]) const {
    std::vector<std::string> all{adb_};
    all.insert(all.end(), arguments.begin(), arguments.end());
    std::vector<char*> argv;
    for (auto& argument : all) argv.push_back(const_cast<char*>(argument.c_str()));
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, pipe_fds[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipe_fds[0]);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    pid_t pid = -1;
    if (posix_spawnp(&pid, adb_.c_str(), &actions, nullptr, argv.data(), environ) != 0) {
      pid = -1;
    }
    posix_spawn_file_actions_destroy(&actions);
    return pid;
  }

  void keep_awake() const {
    run({"-s", serial_, "shell", "input", "keyevent", "KEYCODE_WAKEUP"});
    run({"-s", serial_, "shell", "am", "broadcast", "-a", "com.oculus.vrpowermanager.prox_close"});
  }

  // `adb logcat` for the reader tag; restarted if it exits (e.g. a USB hiccup).
  void read_loop(const std::stop_token& stop) {
    const std::string marker = std::string(kTag) + ": ";
    while (!stop.stop_requested()) {
      int pipe_fds[2];
      if (pipe(pipe_fds) != 0) {
        return;
      }
      const pid_t pid = spawn({"-s", serial_, "logcat", "-T", "0", "-s", std::string(kTag) + ":I"}, pipe_fds);
      close(pipe_fds[1]);
      std::string pending;
      char buffer[8192];
      while (!stop.stop_requested()) {
        pollfd descriptor{pipe_fds[0], POLLIN, 0};
        const int ready = poll(&descriptor, 1, 100);
        if (ready < 0 && errno == EINTR) continue;
        if (ready <= 0) continue;
        const ssize_t count = read(pipe_fds[0], buffer, sizeof(buffer));
        if (count <= 0) break;  // logcat ended
        pending.append(buffer, static_cast<size_t>(count));
        size_t newline;
        while ((newline = pending.find('\n')) != std::string::npos) {
          const std::string line = pending.substr(0, newline);
          pending.erase(0, newline + 1);
          const auto start = line.find(marker);
          if (start == std::string::npos) continue;
          Reading reading;
          if (!parse_payload(line.substr(start + marker.size()), reading)) continue;
          reading.receive_ns = system_ns();
          reading.receive_steady_ns = steady_ns();
          std::lock_guard lock(mutex_);
          reading.sequence = latest_.sequence + 1;
          latest_ = reading;
        }
      }
      close(pipe_fds[0]);
      if (pid > 0) {
        kill(pid, SIGTERM);
        waitpid(pid, nullptr, 0);
      }
      for (int i = 0; i < 10 && !stop.stop_requested(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
  }

  // Keeps the headset awake, restarts the app if it died, reports problems once.
  void maintain(const std::stop_token& stop) {
    int64_t last_wake = steady_ns();
    enum class Problem { kNone, kNoData, kNoPose } reported = Problem::kNone;
    while (!stop.stop_requested()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      Reading reading;
      bool have = false;
      {
        std::lock_guard lock(mutex_);
        reading = latest_;
        have = latest_.sequence != 0;
      }
      const int64_t now = steady_ns();
      if (now - last_wake > 20'000'000'000LL) {
        keep_awake();
        last_wake = now;
      }
      const bool stale = !have || now - reading.receive_steady_ns > 2'000'000'000LL;
      Problem problem = Problem::kNone;
      if (stale) {
        problem = Problem::kNoData;
        // Not running, or alive but paused (e.g. after the cable was unplugged):
        // stop it first if needed (launching a running instance makes it abort).
        if (now - last_start_ns_ > 5'000'000'000LL) {
          if (!run({"-s", serial_, "shell", "pidof", kPackage}).empty()) {
            run({"-s", serial_, "shell", "am", "force-stop", kPackage});
          }
          run({"-s", serial_, "shell", "am", "start", "-n", kActivity});
          last_start_ns_ = now;
          std::cerr << "quest: restarted the reader app on the headset\n";
        }
      } else if (reading.right_present && !reading.pose_valid) {
        problem = Problem::kNoPose;
      }
      if (problem != reported) {
        if (problem == Problem::kNoData) {
          std::cerr << "quest: no controller readings (headset asleep, app closed, or cable unplugged); arm paused\n";
        } else if (problem == Problem::kNoPose) {
          std::cerr << "quest: right controller buttons arrive but no pose. The Quest 3 reader app only sends the "
                       "right pose while the LEFT controller is on too: switch it on.\n";
        } else {
          std::cerr << "quest: readings OK\n";
        }
        reported = problem;
      }
    }
  }

  std::string connection_;
  std::string adb_;
  std::string serial_;
  mutable std::mutex mutex_;
  Reading latest_;
  int64_t last_start_ns_{0};
  std::jthread reader_;
  std::jthread maintenance_;
};

}  // namespace quest
