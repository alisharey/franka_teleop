#include <fcntl.h>
#include <linux/joystick.h>
#include <poll.h>
#include <signal.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

volatile sig_atomic_t g_stop_requested = 0;

void request_stop(int) { g_stop_requested = 1; }

void print_usage(const char* name) {
  std::cout << "Usage: " << name << " [--device /dev/input/js0] [--xbox-preview]\n"
            << "\n"
            << "Prints Linux joystick axis/button events only. It never opens a Franka connection.\n"
            << "--xbox-preview also renders the built-in Xbox-style Cartesian command mapping.\n"
            << "Press Ctrl-C to exit.\n";
}

int open_joystick(const std::string& path) {
  const int descriptor = open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
  if (descriptor < 0) {
    throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
  }
  return descriptor;
}

double deadzoned_axis(const int16_t raw) {
  constexpr double deadzone = 0.10;
  const double normalized = static_cast<double>(raw) / 32767.0;
  const double magnitude = std::abs(normalized);
  if (magnitude <= deadzone) {
    return 0.0;
  }
  return std::copysign((magnitude - deadzone) / (1.0 - deadzone), normalized);
}

void print_xbox_preview(const std::array<int16_t, 8>& axes, const std::array<bool, 15>& buttons) {
  const bool rotation_mode = buttons[6] || static_cast<double>(axes[4]) / 32767.0 > 0.5;
  std::cout << "preview: deadman(RB)=" << (buttons[7] ? "held" : "released")
            << ", gripper A=" << (buttons[0] ? "pressed" : "released")
            << ", B=" << (buttons[1] ? "pressed" : "released") << '\n';
  if (rotation_mode) {
    std::cout << "  rotation mode (L1 or L2 held): roll=" << deadzoned_axis(axes[0])
              << " pitch=" << -deadzoned_axis(axes[1])
              << " yaw=" << deadzoned_axis(axes[2]) << " [normalized]\n";
  } else {
    std::cout << "  translation mode: base_x=" << deadzoned_axis(axes[3])
              << " base_y=" << deadzoned_axis(axes[0])
              << " base_z=" << -deadzoned_axis(axes[1]) << " [normalized]\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string device = "/dev/input/js0";
    bool xbox_preview = false;
    for (int index = 1; index < argc; ++index) {
      const std::string argument(argv[index]);
      if (argument == "--help" || argument == "-h") {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
      }
      if (argument == "--device" && index + 1 < argc) {
        device = argv[++index];
        continue;
      }
      if (argument == "--xbox-preview") {
        xbox_preview = true;
        continue;
      }
      throw std::runtime_error("invalid argument '" + argument + "'; use --help");
    }

    const int descriptor = open_joystick(device);
    std::array<char, 128> name{};
    unsigned char axes = 0;
    unsigned char buttons = 0;
    ioctl(descriptor, JSIOCGAXES, &axes);
    ioctl(descriptor, JSIOCGBUTTONS, &buttons);
    ioctl(descriptor, JSIOCGNAME(name.size()), name.data());
    std::cout << "device: " << device << "\nname: " << name.data() << "\naxes: "
              << static_cast<int>(axes) << "\nbuttons: " << static_cast<int>(buttons) << "\n"
              << "Move one control at a time; use the printed numbers in config/teleop.conf.\n";
    if (xbox_preview && (axes < 5 || buttons < 6)) {
      throw std::runtime_error("xbox preview requires at least 5 axes and 6 buttons");
    }
    std::array<int16_t, 8> preview_axes{};
    std::array<bool, 15> preview_buttons{};

    struct sigaction action {};
    action.sa_handler = request_stop;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);

    while (g_stop_requested == 0) {
      pollfd descriptor_poll{descriptor, POLLIN, 0};
      const int result = poll(&descriptor_poll, 1, 250);
      if (result < 0 && errno == EINTR) {
        continue;
      }
      if (result < 0) {
        throw std::runtime_error("poll failed: " + std::string(std::strerror(errno)));
      }
      if ((descriptor_poll.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        throw std::runtime_error("joystick disconnected or became unavailable");
      }
      if ((descriptor_poll.revents & POLLIN) == 0) {
        continue;
      }
      js_event event{};
      while (read(descriptor, &event, sizeof(event)) == sizeof(event)) {
        const unsigned char event_type = event.type & static_cast<unsigned char>(~JS_EVENT_INIT);
        if (event_type == JS_EVENT_AXIS) {
          std::cout << "axis " << static_cast<int>(event.number) << " = " << event.value << '\n';
          if (xbox_preview && event.number < preview_axes.size()) {
            preview_axes[event.number] = event.value;
          }
        } else if (event_type == JS_EVENT_BUTTON) {
          std::cout << "button " << static_cast<int>(event.number) << " = "
                    << (event.value == 0 ? "released" : "pressed") << '\n';
          if (xbox_preview && event.number < preview_buttons.size()) {
            preview_buttons[event.number] = event.value != 0;
          }
        }
        if (xbox_preview) {
          print_xbox_preview(preview_axes, preview_buttons);
        }
      }
      if (errno != EAGAIN && errno != EWOULDBLOCK) {
        throw std::runtime_error("read failed: " + std::string(std::strerror(errno)));
      }
    }
    close(descriptor);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
