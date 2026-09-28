// Differential inverse kinematics for a 7-joint arm (FR3), used at 1 kHz.
//
// Converts a desired end-effector twist into joint velocities:
//   * damped least squares, with damping that rises only near singular poses;
//   * per-joint velocity bounds handled by saturation in the null space (SNS):
//     a joint that would exceed its bound is frozen at that bound and the
//     remaining joints are re-solved so the end effector still follows the
//     twist. Only if too few joints remain is the whole motion scaled down;
//   * a null-space term (posture pull + push away from joint limits) that moves
//     the elbow without moving the end effector.
//
// No allocations and no external libraries: 6x6 solves by Gaussian
// elimination on std::array. The Jacobian layout matches libfranka's
// zeroJacobian(): 6x7 column-major, rows (vx, vy, vz, wx, wy, wz) in base frame.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace diff_ik {

using Vec6 = std::array<double, 6>;
using Vec7 = std::array<double, 7>;
using Jacobian = std::array<double, 42>;  // (row, col) = [col * 6 + row]

struct Params {
  // Damping lambda^2 = max_damping^2 * (1 - w / manipulability_threshold)^2 when the
  // manipulability w = sqrt(det(J J^T)) drops below the threshold, else min_damping^2.
  // Threshold 0.01 is about the 10th percentile of FR3 poses in the lab workspace
  // (gripper roughly down), so ordinary poses are solved undamped.
  double min_damping{1e-3};
  double max_damping{0.05};
  double manipulability_threshold{0.01};
  // Null-space posture pull toward `posture` (rad/s per rad).
  Vec7 posture{0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398};  // Franka ready pose
  double posture_gain{0.4};
  // Push away from position limits inside this zone (rad), up to limit_gain rad/s at the limit.
  Vec7 q_lower{};
  Vec7 q_upper{};
  double limit_zone{0.3};
  double limit_gain{0.8};
  // Cap on each joint's null-space speed (rad/s).
  double max_nullspace_speed{0.3};
  // Weight of the rotation rows relative to translation (1 = equal). Below 1,
  // when the exact twist is impossible (e.g. a joint is at its limit), the
  // unavoidable error goes into a small gripper tilt instead of position drift.
  // When the twist is achievable it is still tracked exactly.
  double rotation_weight{1.0};
  // Largest relative twist error accepted from the per-joint saturation step
  // (|J dq - twist| / |twist|). Above it the solver falls back to following the
  // exact direction at reduced speed (strict mode, e.g. 0.05). Infinity accepts
  // any error, letting the arm detour around unreachable straight-line paths.
  double max_task_error{1e9};
};

struct Result {
  Vec7 dq{};
  double task_scale{1.0};  // < 1 when the twist had to be slowed down
  int saturated{0};        // joints frozen at their velocity bound
};

namespace detail {

// Solves (J W J^T + lambda2 I) y = rhs, W = diag(active). Returns false if singular.
inline bool solve_damped(const Jacobian& jacobian, const std::array<bool, 7>& active, double lambda2,
                         const Vec6& rhs, Vec6& y) {
  std::array<std::array<double, 7>, 6> system{};
  for (size_t row = 0; row < 6; ++row) {
    for (size_t col = 0; col < 6; ++col) {
      double sum = row == col ? lambda2 : 0.0;
      for (size_t joint = 0; joint < 7; ++joint) {
        if (active[joint]) {
          sum += jacobian[joint * 6 + row] * jacobian[joint * 6 + col];
        }
      }
      system[row][col] = sum;
    }
    system[row][6] = rhs[row];
  }
  for (size_t pivot = 0; pivot < 6; ++pivot) {
    size_t best = pivot;
    for (size_t row = pivot + 1; row < 6; ++row) {
      if (std::abs(system[row][pivot]) > std::abs(system[best][pivot])) {
        best = row;
      }
    }
    if (std::abs(system[best][pivot]) < 1e-12) {
      return false;
    }
    std::swap(system[pivot], system[best]);
    for (size_t row = pivot + 1; row < 6; ++row) {
      const double factor = system[row][pivot] / system[pivot][pivot];
      for (size_t col = pivot; col < 7; ++col) {
        system[row][col] -= factor * system[pivot][col];
      }
    }
  }
  for (size_t row = 6; row-- > 0;) {
    double sum = system[row][6];
    for (size_t col = row + 1; col < 6; ++col) {
      sum -= system[row][col] * y[col];
    }
    y[row] = sum / system[row][row];
  }
  return true;
}

// sqrt(det(J W J^T)), W = diag(active), via LU on the undamped 6x6 product.
inline double manipulability(const Jacobian& jacobian, const std::array<bool, 7>& active) {
  std::array<std::array<double, 6>, 6> m{};
  for (size_t row = 0; row < 6; ++row) {
    for (size_t col = 0; col < 6; ++col) {
      double sum = 0.0;
      for (size_t joint = 0; joint < 7; ++joint) {
        if (active[joint]) {
          sum += jacobian[joint * 6 + row] * jacobian[joint * 6 + col];
        }
      }
      m[row][col] = sum;
    }
  }
  double det = 1.0;
  for (size_t pivot = 0; pivot < 6; ++pivot) {
    size_t best = pivot;
    for (size_t row = pivot + 1; row < 6; ++row) {
      if (std::abs(m[row][pivot]) > std::abs(m[best][pivot])) {
        best = row;
      }
    }
    if (std::abs(m[best][pivot]) < 1e-300) {
      return 0.0;
    }
    if (best != pivot) {
      std::swap(m[pivot], m[best]);
      det = -det;
    }
    det *= m[pivot][pivot];
    for (size_t row = pivot + 1; row < 6; ++row) {
      const double factor = m[row][pivot] / m[pivot][pivot];
      for (size_t col = pivot; col < 6; ++col) {
        m[row][col] -= factor * m[pivot][col];
      }
    }
  }
  return std::sqrt(std::max(0.0, det));
}

inline double manipulability(const Jacobian& jacobian) {
  std::array<bool, 7> all{};
  all.fill(true);
  return manipulability(jacobian, all);
}

inline Vec6 multiply(const Jacobian& jacobian, const Vec7& dq) {
  Vec6 out{};
  for (size_t joint = 0; joint < 7; ++joint) {
    for (size_t row = 0; row < 6; ++row) {
      out[row] += jacobian[joint * 6 + row] * dq[joint];
    }
  }
  return out;
}

}  // namespace detail

// Desired null-space joint velocity: posture pull plus limit repulsion, capped.
inline Vec7 nullspace_velocity(const Vec7& q, const Params& params) {
  Vec7 z{};
  for (size_t joint = 0; joint < 7; ++joint) {
    double value = params.posture_gain * (params.posture[joint] - q[joint]);
    const double to_upper = params.q_upper[joint] - q[joint];
    const double to_lower = q[joint] - params.q_lower[joint];
    if (to_upper < params.limit_zone) {
      const double depth = (params.limit_zone - std::max(0.0, to_upper)) / params.limit_zone;
      value -= params.limit_gain * depth * depth;
    }
    if (to_lower < params.limit_zone) {
      const double depth = (params.limit_zone - std::max(0.0, to_lower)) / params.limit_zone;
      value += params.limit_gain * depth * depth;
    }
    z[joint] = std::clamp(value, -params.max_nullspace_speed, params.max_nullspace_speed);
  }
  return z;
}

inline Result solve_weighted(const Jacobian& jacobian, const Vec6& twist, const Vec7& q,
                             const Vec7& lower_velocity, const Vec7& upper_velocity,
                             double nullspace_weight, double manipulability, const Params& params);

// Joint velocities that produce `twist`, stay within [lower_velocity, upper_velocity]
// per joint (lower <= 0 <= upper), and use leftover freedom for `nullspace_weight`
// (0..1) times the null-space velocity.
inline Result solve(const Jacobian& jacobian, const Vec6& twist, const Vec7& q,
                    Vec7 lower_velocity, Vec7 upper_velocity, double nullspace_weight,
                    const Params& params) {
  // libfranka's bounds include a small tolerance and can cross zero at a limit;
  // standing still must always be allowed.
  for (size_t joint = 0; joint < 7; ++joint) {
    upper_velocity[joint] = std::max(upper_velocity[joint], 0.0);
    lower_velocity[joint] = std::min(lower_velocity[joint], 0.0);
  }
  // Task weighting: scale the rotation rows of J and the twist (weighted least squares).
  Jacobian weighted = jacobian;
  Vec6 weighted_twist = twist;
  for (size_t joint = 0; joint < 7; ++joint) {
    for (size_t row = 3; row < 6; ++row) {
      weighted[joint * 6 + row] *= params.rotation_weight;
    }
  }
  for (size_t row = 3; row < 6; ++row) {
    weighted_twist[row] *= params.rotation_weight;
  }
  // Damping is judged on the true (unweighted) Jacobian.
  return solve_weighted(weighted, weighted_twist, q, lower_velocity, upper_velocity, nullspace_weight,
                        detail::manipulability(jacobian), params);
}

// Core solver on an already-weighted Jacobian and twist.
inline Result solve_weighted(const Jacobian& jacobian, const Vec6& twist, const Vec7& q,
                             const Vec7& lower_velocity, const Vec7& upper_velocity,
                             double nullspace_weight, double manipulability, const Params& params) {
  // Damping from how close to singular the joints being solved with are. With
  // all joints active that is the true arm (`manipulability`); after freezing
  // joints it is the reduced set, which can be near-singular even when the arm
  // is not, and would otherwise demand sudden large swings of the others.
  const double full_w = manipulability;
  const double scale_w = full_w / std::max(detail::manipulability(jacobian), 1e-12);  // undo row weighting
  auto damping_for = [&](double w) {
    double lambda = params.min_damping;
    if (w < params.manipulability_threshold) {
      lambda = std::max(params.min_damping, params.max_damping * (1.0 - w / params.manipulability_threshold));
    }
    return lambda * lambda;
  };
  double lambda2 = damping_for(full_w);

  Vec7 z = nullspace_velocity(q, params);
  for (double& value : z) {
    value *= nullspace_weight;
  }

  std::array<bool, 7> active{};
  active.fill(true);
  Vec7 saturated_dq{};  // Fixed velocities of frozen joints (zero for active ones).
  Result result{};

  for (int iteration = 0; iteration < 7; ++iteration) {
    if (iteration > 0) {
      lambda2 = damping_for(detail::manipulability(jacobian, active) * scale_w);
    }
    // Task part: active joints track what the frozen joints do not already provide.
    const Vec6 frozen_twist = detail::multiply(jacobian, saturated_dq);
    Vec6 residual{};
    for (size_t row = 0; row < 6; ++row) {
      residual[row] = twist[row] - frozen_twist[row];
    }
    Vec6 y{};
    if (!detail::solve_damped(jacobian, active, lambda2, residual, y)) {
      break;
    }
    Vec7 dq = saturated_dq;
    for (size_t joint = 0; joint < 7; ++joint) {
      if (active[joint]) {
        for (size_t row = 0; row < 6; ++row) {
          dq[joint] += jacobian[joint * 6 + row] * y[row];
        }
      }
    }
    // Null-space part, restricted to active joints: u - W J^T (J W J^T + l^2 I)^-1 J u.
    Vec7 u{};
    for (size_t joint = 0; joint < 7; ++joint) {
      u[joint] = active[joint] ? z[joint] : 0.0;
    }
    Vec6 yu{};
    if (detail::solve_damped(jacobian, active, lambda2, detail::multiply(jacobian, u), yu)) {
      for (size_t joint = 0; joint < 7; ++joint) {
        if (active[joint]) {
          double projected = u[joint];
          for (size_t row = 0; row < 6; ++row) {
            projected -= jacobian[joint * 6 + row] * yu[row];
          }
          dq[joint] += projected;
        }
      }
    }
    // Find the active joint that overshoots its bound the most.
    int worst = -1;
    double worst_excess = 1e-9;
    for (size_t joint = 0; joint < 7; ++joint) {
      if (!active[joint]) {
        continue;
      }
      const double excess =
          std::max(dq[joint] - upper_velocity[joint], lower_velocity[joint] - dq[joint]);
      if (excess > worst_excess) {
        worst_excess = excess;
        worst = static_cast<int>(joint);
      }
    }
    result.dq = dq;
    if (worst < 0) {
      break;  // Everything within bounds.
    }
    // Freeze that joint at its bound and re-solve with the others.
    const auto joint = static_cast<size_t>(worst);
    saturated_dq[joint] = std::clamp(dq[joint], lower_velocity[joint], upper_velocity[joint]);
    active[joint] = false;
    ++result.saturated;
    size_t remaining = 0;
    for (bool is_active : active) {
      remaining += is_active ? 1 : 0;
    }
    if (remaining == 0) {
      break;
    }
  }

  // Strict mode: if the saturated solution strays too far from the requested
  // twist, use the unconstrained minimum-norm solution instead; the uniform
  // scaling below then keeps its direction exactly (slower, or stopped).
  if (params.max_task_error < 1e8) {
    const Vec6 achieved = detail::multiply(jacobian, result.dq);
    double error = 0.0;
    double norm = 0.0;
    for (size_t row = 0; row < 6; ++row) {
      error += (achieved[row] - twist[row]) * (achieved[row] - twist[row]);
      norm += twist[row] * twist[row];
    }
    if (norm > 1e-12 && std::sqrt(error / norm) > params.max_task_error) {
      std::array<bool, 7> all{};
      all.fill(true);
      Vec6 y{};
      if (detail::solve_damped(jacobian, all, lambda2, twist, y)) {
        result.dq = {};
        for (size_t joint = 0; joint < 7; ++joint) {
          for (size_t row = 0; row < 6; ++row) {
            result.dq[joint] += jacobian[joint * 6 + row] * y[row];
          }
        }
      }
    }
  }

  // Scale the whole joint motion uniformly (keeps its direction) until every
  // joint is inside its bound.
  double scale = 1.0;
  for (size_t joint = 0; joint < 7; ++joint) {
    const double value = result.dq[joint];
    const double bound = value >= 0.0 ? upper_velocity[joint] : lower_velocity[joint];
    if (std::abs(value) > std::abs(bound) + 1e-12) {
      scale = std::min(scale, std::abs(bound) / std::abs(value));  // bound shares value's sign
    }
  }
  for (double& value : result.dq) {
    value *= scale;
  }
  result.task_scale = scale;
  return result;
}

}  // namespace diff_ik
