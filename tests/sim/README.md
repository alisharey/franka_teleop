# End-to-end simulation

Runs the real `src/gamepad_teleop.cpp` against a simulated FR3
(`fake_franka.cpp`) driven by a scripted gamepad (`gamepad_script.py`, a FIFO
emitting Linux joystick events), then checks the recorded trace
(`check_trace.py`). The simulated robot rejects, like the real one, any joint
velocity outside the FR3's position-dependent bounds, acceleration above
10 rad/s², jerk above 5000 rad/s³, or a joint leaving its position limits.

The scripted session: hold RB; descend 12 s; climb; drive toward the base;
release RB and push a stick (must not move); resume at 50 % speed; move
sideways; rotate with L1; Ctrl-C.

It is not part of the CMake build because it replaces libfranka's `Robot`,
`Model` and `Gripper` and needs libfranka's source for its real rate limiter:

```bash
git clone --depth 1 --branch 0.20.4 https://github.com/frankarobotics/libfranka.git /tmp/lf
git -C /tmp/lf submodule update --init --depth 1 common
# Eigen headers, e.g. sudo apt install libeigen3-dev (then use /usr/include/eigen3)
INC="-I/opt/ros/humble/include -isystem /tmp/lf/common/include -isystem /usr/include/eigen3 -I/tmp/lf/src"
mkdir -p /tmp/simbuild
for f in control_types duration errors exception rate_limiting control_tools; do
  g++ -std=c++17 -O2 -w $INC -c /tmp/lf/src/$f.cpp -o /tmp/simbuild/$f.o
done
g++ -std=c++20 -O2 $INC -c src/gamepad_teleop.cpp -o /tmp/simbuild/teleop.o
g++ -std=c++20 -O2 $INC -c tests/sim/fake_franka.cpp -o /tmp/simbuild/fake.o
g++ /tmp/simbuild/*.o -o /tmp/sim_teleop -Wl,--wrap=ioctl -lpthread
tests/sim/run_sim_session.sh /tmp/sim_teleop /tmp/simrun   # about 30 s, real time
```
