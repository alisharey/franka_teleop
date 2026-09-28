// Off-robot simulation of src/diff_ik.h on an FR3 kinematic model.
// Checks: FK matches a recorded robot pose; the solver tracks commanded
// twists; joints never exceed the FR3's position-based velocity bounds
// (same formula and constants libfranka uses, from its test/fr3.urdf); and
// how far the arm gets when driven to the workspace edges compared with the
// old behaviour (uniformly scaling the whole motion when any joint nears a limit).
#include "../src/diff_ik.h"
#include "fr3_sim.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

namespace {

using namespace fr3_sim;
using diff_ik::Jacobian;
using diff_ik::Vec6;
using diff_ik::Vec7;

diff_ik::Params params() {
  diff_ik::Params p;
  p.q_lower = kQLower;
  p.q_upper = kQUpper;
  return p;
}

int g_failures = 0;
void check(bool ok, const std::string& what) {
  if (!ok) {
    ++g_failures;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

// Old behaviour for comparison: DLS solution with no null-space term, scaled
// uniformly until every joint is within bounds (a single joint near its limit
// slows or stops the whole arm).
Vec7 uniform_scaling(const Jacobian& j, const Vec6& twist, const Vec7& q, double margin) {
  diff_ik::Params p = params();
  p.posture_gain = 0.0;
  p.limit_gain = 0.0;
  Vec7 unbounded_low{}, unbounded_up{};
  unbounded_low.fill(-100.0);
  unbounded_up.fill(100.0);
  Vec7 dq = diff_ik::solve(j, twist, q, unbounded_low, unbounded_up, 0.0, p).dq;
  const Vec7 lo = lower_bound(q, margin), up = upper_bound(q, margin);
  double scale = 1.0;
  for (size_t i = 0; i < 7; ++i) {
    if (dq[i] > up[i]) scale = std::min(scale, std::max(0.0, up[i]) / dq[i]);
    if (dq[i] < lo[i]) scale = std::min(scale, std::min(0.0, lo[i]) / dq[i]);
  }
  for (double& v : dq) v *= scale;
  return dq;
}

struct DriveResult {
  double reached_fraction;  // progress toward target, 1 = arrived
  double seconds;
  double max_violation;     // worst |dq| beyond the robot's bound, rad/s
  double min_limit_distance;
  double worst_tracking_error;  // while unconstrained, relative
};

// Drive the TCP toward `target` at `speed` m/s, holding orientation.
DriveResult drive(Vec7 q, const std::array<double, 3>& target, double speed, bool new_solver,
                  double seconds_limit = 20.0) {
  const auto start = position(q);
  const double total = std::hypot(target[0] - start[0], target[1] - start[1], target[2] - start[2]);
  DriveResult r{0.0, 0.0, 0.0, 1e9, 0.0};
  const double margin = 0.05;
  const diff_ik::Params p = params();
  int steps = static_cast<int>(seconds_limit / kDt);
  for (int step = 0; step < steps; ++step) {
    const auto pos = position(q);
    Vec6 twist{};
    const double ex = target[0] - pos[0], ey = target[1] - pos[1], ez = target[2] - pos[2];
    const double dist = std::hypot(ex, ey, ez);
    if (dist < 0.002) {
      r.seconds = step * kDt;
      break;
    }
    const double v = std::min(speed, 2.0 * dist);
    twist = {ex / dist * v, ey / dist * v, ez / dist * v, 0, 0, 0};
    const Jacobian j = jacobian(q);
    Vec7 dq;
    if (new_solver) {
      Vec7 lo = lower_bound(q, margin), up = upper_bound(q, margin);
      for (size_t i = 0; i < 7; ++i) {  // teleop joint speed cap, as in gamepad_teleop
        up[i] = std::min(up[i], 1.0);
        lo[i] = std::max(lo[i], -1.0);
      }
      const auto res = diff_ik::solve(j, twist, q, lo, up, 1.0, p);
      dq = res.dq;
      // Accuracy is only promised where the solver is not deliberately constrained:
      // no joint at its bound, no scaling, and outside the damped (near-singular) region.
      if (res.saturated == 0 && res.task_scale > 0.999 &&
          diff_ik::detail::manipulability(j) > p.manipulability_threshold) {
        const Vec6 got = diff_ik::detail::multiply(j, dq);
        double err = 0, norm = 0;
        for (int k = 0; k < 6; ++k) {
          err += (got[k] - twist[k]) * (got[k] - twist[k]);
          norm += twist[k] * twist[k];
        }
        if (norm > 1e-8) r.worst_tracking_error = std::max(r.worst_tracking_error, std::sqrt(err / norm));
      }
    } else {
      dq = uniform_scaling(j, twist, q, margin);
    }
    const Vec7 up = upper_bound(q, 0.0), lo = lower_bound(q, 0.0);  // robot's own bound
    for (size_t i = 0; i < 7; ++i) {
      r.max_violation = std::max(r.max_violation, std::max(dq[i] - std::max(up[i], 0.0), std::min(lo[i], 0.0) - dq[i]));
      q[i] += dq[i] * kDt;
      r.min_limit_distance = std::min(r.min_limit_distance, std::min(kQUpper[i] - q[i], q[i] - kQLower[i]));
    }
    r.seconds = (step + 1) * kDt;
  }
  const auto end = position(q);
  const double left = std::hypot(target[0] - end[0], target[1] - end[1], target[2] - end[2]);
  r.reached_fraction = total > 0 ? std::clamp(1.0 - left / total, 0.0, 1.0) : 1.0;
  return r;
}

}  // namespace

int main() {
  // 1. Forward kinematics against Frankastein's recorded FR3 home pose.
  const Vec7 home{0.036286, -1.367388, -0.180505, -2.794799, -0.043301, 2.076045, 0.407056};
  const auto p = position(home);
  const double fk_error = std::hypot(p[0] - 0.273185, p[1] + 0.035352, p[2] - 0.495227);
  std::printf("FK at Frankastein home: (%.4f, %.4f, %.4f), recorded (0.2732, -0.0354, 0.4952), error %.1f mm\n",
              p[0], p[1], p[2], fk_error * 1000);
  check(fk_error < 0.005, "FK matches the recorded home pose within 5 mm");

  // 2. Tracking accuracy on random unconstrained configurations.
  std::mt19937 rng(7);
  double worst = 0;
  int tested = 0;
  for (int t = 0; t < 2000; ++t) {
    Vec7 q{};
    for (size_t i = 0; i < 7; ++i) {
      std::uniform_real_distribution<double> u(kQLower[i] + 0.4, kQUpper[i] - 0.4);
      q[i] = u(rng);
    }
    const Jacobian j = jacobian(q);
    if (diff_ik::detail::manipulability(j) < 0.03) continue;  // away from singularities
    std::uniform_real_distribution<double> v(-0.1, 0.1);
    Vec6 twist{v(rng), v(rng), v(rng), 3 * v(rng), 3 * v(rng), 3 * v(rng)};
    Vec7 lo{}, up{};
    lo.fill(-10);
    up.fill(10);
    const auto res = diff_ik::solve(j, twist, q, lo, up, 1.0, params());
    const Vec6 got = diff_ik::detail::multiply(j, res.dq);
    double err = 0, norm = 0;
    for (int k = 0; k < 6; ++k) {
      err += (got[k] - twist[k]) * (got[k] - twist[k]);
      norm += twist[k] * twist[k];
    }
    worst = std::max(worst, std::sqrt(err / norm));
    ++tested;
  }
  std::printf("Tracking on %d random poses (incl. null-space posture term): worst error %.3f%%\n", tested, worst * 100);
  check(worst < 0.01, "tracks commanded twist within 1% away from limits/singularities");

  // 3. Drive to the workspace edges from Frankastein home, new vs old behaviour.
  struct Case { const char* name; std::array<double, 3> target; };
  const Case cases[] = {
      {"toward base  x=0.16", {0.16, p[1], p[2]}},
      {"far reach    x=0.70", {0.70, p[1], p[2]}},
      {"down         z=0.02", {p[0], p[1], 0.02}},
      {"up           z=0.71", {p[0], p[1], 0.71}},
      {"left         y=0.35", {p[0], 0.35, p[2]}},
      {"right        y=-0.30", {p[0], -0.30, p[2]}},
      {"near corner  (0.16,-0.30,0.05)", {0.16, -0.30, 0.05}},
      {"far corner   (0.70, 0.35,0.05)", {0.70, 0.35, 0.05}},
      {"low & close  (0.20, 0.00,0.02)", {0.20, 0.0, 0.02}},
  };
  std::printf("\n%-34s | %-30s | %-30s\n", "drive from home at 0.15 m/s", "NEW solver: reached / time", "OLD scaling: reached / time");
  for (const auto& c : cases) {
    const auto n = drive(home, c.target, 0.15, true);
    const auto o = drive(home, c.target, 0.15, false);
    std::printf("%-34s | %5.1f%% %5.1fs  viol %.4f     | %5.1f%% %5.1fs  viol %.4f\n", c.name,
                n.reached_fraction * 100, n.seconds, n.max_violation, o.reached_fraction * 100, o.seconds,
                o.max_violation);
    check(n.max_violation <= 1e-9, std::string("new solver respects robot velocity bounds: ") + c.name);
    check(n.min_limit_distance > 0.0, std::string("new solver stays inside position limits: ") + c.name);
    check(n.worst_tracking_error < 0.02, std::string("new solver tracks when unconstrained: ") + c.name);
  }

  // 4. Long random stick session: random direction changes every 1.5 s for 60 s.
  {
    Vec7 q = home;
    std::mt19937 r2(11);
    std::uniform_real_distribution<double> u(-1, 1);
    double max_violation = 0, min_distance = 1e9;
    int stalled_ms = 0, commanded_ms = 0;
    Vec6 twist{};
    const diff_ik::Params pr = params();
    for (int step = 0; step < 60000; ++step) {
      if (step % 1500 == 0) {
        twist = {0.15 * u(r2), 0.15 * u(r2), 0.15 * u(r2), 0.4 * u(r2), 0.4 * u(r2), 0.4 * u(r2)};
      }
      // Keep the TCP inside the workspace box by zeroing outward components.
      const auto pos = position(q);
      const std::array<double, 3> lo{0.16, -0.30, 0.02}, hi{0.70, 0.35, 0.71};
      Vec6 cmd = twist;
      for (int k = 0; k < 3; ++k) {
        if ((pos[k] <= lo[k] && cmd[k] < 0) || (pos[k] >= hi[k] && cmd[k] > 0)) cmd[k] = 0;
      }
      const auto res = diff_ik::solve(jacobian(q), cmd, q, lower_bound(q, 0.05), upper_bound(q, 0.05), 1.0, pr);
      const Vec7 up = upper_bound(q, 0.0), lo_b = lower_bound(q, 0.0);
      double speed = 0;
      for (size_t i = 0; i < 7; ++i) {
        max_violation = std::max(max_violation, std::max(res.dq[i] - std::max(up[i], 0.0), std::min(lo_b[i], 0.0) - res.dq[i]));
        q[i] += res.dq[i] * kDt;
        min_distance = std::min(min_distance, std::min(kQUpper[i] - q[i], q[i] - kQLower[i]));
      }
      const Vec6 got = diff_ik::detail::multiply(jacobian(q), res.dq);
      double cmd_norm = 0, got_norm = 0;
      for (int k = 0; k < 6; ++k) { cmd_norm += cmd[k] * cmd[k]; got_norm += got[k] * got[k]; }
      (void)speed;
      if (cmd_norm > 1e-6) {
        ++commanded_ms;
        if (std::sqrt(got_norm) < 0.05 * std::sqrt(cmd_norm)) ++stalled_ms;
      }
    }
    std::printf("\n60 s random stick session: worst bound violation %.2e rad/s, closest to a position limit %.3f rad, "
                "stalled %.1f%% of commanded time\n", max_violation, min_distance, 100.0 * stalled_ms / std::max(1, commanded_ms));
    check(max_violation <= 1e-9, "random session respects robot velocity bounds");
    check(min_distance > 0.0, "random session stays inside position limits");
  }

  std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED", g_failures);
  return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
