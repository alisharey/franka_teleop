// FR3 kinematic model for off-robot tests: forward kinematics and Jacobian at
// the Franka Hand TCP (Craig modified DH), and the FR3's position-based joint
// velocity bounds (same formula and constants libfranka uses, from its
// test/fr3.urdf). Checked against a recorded robot pose in diff_ik_test.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "../src/diff_ik.h"

namespace fr3_sim {

using diff_ik::Jacobian;
using diff_ik::Vec6;
using diff_ik::Vec7;
using Mat4 = std::array<double, 16>;  // column-major like libfranka O_T_EE

constexpr double kPi = 3.14159265358979323846;
constexpr double kDt = 1e-3;

// FR3 limits from libfranka test/fr3.urdf.
constexpr Vec7 kQLower{-2.7501, -1.7918, -2.9065, -3.0481, -2.8101, 0.54092, -3.0196};
constexpr Vec7 kQUpper{2.7501, 1.7918, 2.9065, -0.1458, 2.8101, 4.5205, 3.0196};
constexpr Vec7 kVMax{2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26};
constexpr Vec7 kOffset{0.30, 0.20, 0.20, 0.30, 0.35, 0.35, 0.35};
constexpr Vec7 kDecel{6.0, 2.585, 3.50, 4.00, 17.0, 5.5, 17.0};
constexpr double kTolerance = 1e-3;  // libfranka kJointVelocityLimitsTolerance

inline Vec7 upper_bound(const Vec7& q, double margin) {
  Vec7 out{};
  for (size_t i = 0; i < 7; ++i) {
    out[i] = std::min(kVMax[i], std::max(0.0, -kOffset[i] + std::sqrt(std::max(
                                                  0.0, 2.0 * kDecel[i] * (kQUpper[i] - (q[i] + margin)))))) -
             kTolerance;
  }
  return out;
}
inline Vec7 lower_bound(const Vec7& q, double margin) {
  Vec7 out{};
  for (size_t i = 0; i < 7; ++i) {
    out[i] = std::max(-kVMax[i], std::min(0.0, kOffset[i] - std::sqrt(std::max(
                                                   0.0, 2.0 * kDecel[i] * (-kQLower[i] + (q[i] - margin)))))) +
             kTolerance;
  }
  return out;
}

inline Mat4 multiply(const Mat4& a, const Mat4& b) {
  Mat4 c{};
  for (int r = 0; r < 4; ++r)
    for (int col = 0; col < 4; ++col)
      for (int k = 0; k < 4; ++k) c[col * 4 + r] += a[k * 4 + r] * b[col * 4 + k];
  return c;
}
// Craig modified DH: RotX(alpha) TransX(a) RotZ(theta) TransZ(d).
inline Mat4 dh(double a, double d, double alpha, double theta) {
  const double ca = std::cos(alpha), sa = std::sin(alpha), ct = std::cos(theta), st = std::sin(theta);
  return {ct, st * ca, st * sa, 0, -st, ct * ca, ct * sa, 0, 0, -sa, ca, 0, a, -d * sa, d * ca, 1};
}
// Frames 1..7, then flange, then hand TCP (Franka Hand: 0.1034 m, -45 deg about z).
inline std::array<Mat4, 9> frames(const Vec7& q) {
  constexpr std::array<double, 7> a{0, 0, 0, 0.0825, -0.0825, 0, 0.088};
  constexpr std::array<double, 7> d{0.333, 0, 0.316, 0, 0.384, 0, 0};
  constexpr std::array<double, 7> alpha{0, -kPi / 2, kPi / 2, kPi / 2, -kPi / 2, kPi / 2, kPi / 2};
  std::array<Mat4, 9> out{};
  Mat4 t{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  for (size_t i = 0; i < 7; ++i) {
    t = multiply(t, dh(a[i], d[i], alpha[i], q[i]));
    out[i] = t;
  }
  out[7] = multiply(t, dh(0, 0.107, 0, 0));
  out[8] = multiply(out[7], dh(0, 0.1034, 0, -kPi / 4));
  return out;
}
inline Jacobian jacobian(const Vec7& q) {
  const auto f = frames(q);
  const double px = f[8][12], py = f[8][13], pz = f[8][14];
  Jacobian j{};
  for (size_t i = 0; i < 7; ++i) {
    const double zx = f[i][8], zy = f[i][9], zz = f[i][10];  // joint axis = frame z
    const double rx = px - f[i][12], ry = py - f[i][13], rz = pz - f[i][14];
    j[i * 6 + 0] = zy * rz - zz * ry;
    j[i * 6 + 1] = zz * rx - zx * rz;
    j[i * 6 + 2] = zx * ry - zy * rx;
    j[i * 6 + 3] = zx;
    j[i * 6 + 4] = zy;
    j[i * 6 + 5] = zz;
  }
  return j;
}
inline std::array<double, 3> position(const Vec7& q) {
  const auto f = frames(q);
  return {f[8][12], f[8][13], f[8][14]};
}

}  // namespace fr3_sim
