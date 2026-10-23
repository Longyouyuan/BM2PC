#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

namespace trajectory_utils {

// Generate N points with exponentially dense spacing in the first portion,
// then uniform spacing for the remainder.
inline std::vector<double> halfDenseThenUniform(int N,
                                                double ratio = 0.5,
                                                double sharpness = 2.0,
                                                double start = 0.0,
                                                double end = 2.0) {
  std::vector<double> x;
  x.reserve(N);

  if (N <= 1) {
    x.push_back(start);
    return x;
  }

  int n_dense = static_cast<int>(N * ratio);
  n_dense = std::max(2, std::min(n_dense, N));
  int n_uniform = N - n_dense;

  const double base = std::exp(sharpness);

  std::vector<double> x_dense;
  x_dense.reserve(n_dense);
  for (int i = 0; i < n_dense; ++i) {
    double t = (n_dense == 1) ? 0.0 : static_cast<double>(i) / static_cast<double>(n_dense - 1);
    double val = ratio * (std::pow(base, t) - 1.0) / (base - 1.0);
    x_dense.push_back(val);
  }

  const double final_spacing =
      (n_dense >= 2) ? (x_dense[n_dense - 1] - x_dense[n_dense - 2]) : ratio;

  std::vector<double> x_uniform;
  x_uniform.reserve(n_uniform);
  for (int i = 1; i <= n_uniform; ++i) {
    x_uniform.push_back(x_dense.back() + final_spacing * static_cast<double>(i));
  }

  x.insert(x.end(), x_dense.begin(), x_dense.end());
  x.insert(x.end(), x_uniform.begin(), x_uniform.end());

  const double scale = (end - start) / x.back();
  for (double& v : x) {
    v = v * scale + start;
  }

  return x;
}

// Extend a trajectory by repeating the last point N times.
inline std::vector<Eigen::Vector3d> extendLastPoint(const std::vector<Eigen::Vector3d>& in, int N) {
  std::vector<Eigen::Vector3d> out = in;
  if (!out.empty()) {
    for (int i = 0; i < N; ++i) out.push_back(out.back());
  }
  return out;
}

// Generate a figure-eight (lemniscate) trajectory.
inline std::vector<Eigen::Vector3d> eightTrajectory(const std::vector<double>& points,
                                                     double scale_x = 0.25,
                                                     double scale_y = 0.45,
                                                     double z0 = 0.2,
                                                     int loops = 1,
                                                     const Eigen::Vector3d& bias = Eigen::Vector3d::Zero()) {
  std::vector<Eigen::Vector3d> traj;
  traj.reserve(points.size());

  if (points.empty()) return traj;

  const double last = points.back();
  const double a = -0.2;

  for (double p : points) {
    double u = p / last * (2.0 * M_PI * static_cast<double>(loops));
    double t = u + a * std::sin(2.0 * u);

    double sin_t = std::sin(t);
    double cos_t = std::cos(t);
    double denom = 1.0 + sin_t * sin_t;

    double y = -scale_x * cos_t / denom;
    double x =  scale_y * (sin_t * cos_t) / denom;
    double z = z0;

    traj.emplace_back(x, y, z);
  }

  Eigen::Vector3d start_shift = traj.front();
  start_shift.z() = 0.0;

  for (auto& p : traj) {
    p -= start_shift;
    p += bias;
  }

  return traj;
}

// Generate a continuous, repeating figure-eight trajectory.
// Smooth ramp-up over the first `ramp_fraction` of N points, then constant
// angular speed for the remainder, covering `loops` full figure-eight loops.
inline std::vector<Eigen::Vector3d> eightTrajectoryContinuous(
    int N,
    int loops = 5,
    double ramp_fraction = 0.1,
    double scale_x = 0.25,
    double scale_y = 0.45,
    double z0 = 0.2,
    const Eigen::Vector3d& bias = Eigen::Vector3d::Zero()) {
  std::vector<Eigen::Vector3d> traj;
  if (N <= 0) return traj;
  traj.reserve(N);

  const double total_angle = 2.0 * M_PI * static_cast<double>(loops);
  int n_ramp = std::max(1, static_cast<int>(N * ramp_fraction));

  // Build cumulative parameter u[i] with half-cosine speed ramp then cruise.
  // During ramp (i in [0, n_ramp)): speed = v_cruise * 0.5*(1 - cos(pi*i/n_ramp))
  // During cruise (i >= n_ramp):    speed = v_cruise
  // We first compute the unnormalized cumulative sum, then scale to total_angle.

  std::vector<double> u(N, 0.0);
  // speed profile (unnormalized, v_cruise = 1)
  double cum = 0.0;
  for (int i = 1; i < N; ++i) {
    double speed;
    if (i < n_ramp) {
      speed = 0.5 * (1.0 - std::cos(M_PI * static_cast<double>(i) / static_cast<double>(n_ramp)));
    } else {
      speed = 1.0;
    }
    cum += speed;
    u[i] = cum;
  }

  // Scale so u goes from 0 to total_angle
  if (cum > 0.0) {
    double s = total_angle / cum;
    for (int i = 0; i < N; ++i) u[i] *= s;
  }

  // Map parameter to figure-eight (lemniscate of Bernoulli)
  const double a = -0.2;  // same shape distortion as eightTrajectory
  for (int i = 0; i < N; ++i) {
    double t = u[i] + a * std::sin(2.0 * u[i]);

    double sin_t = std::sin(t);
    double cos_t = std::cos(t);
    double denom = 1.0 + sin_t * sin_t;

    double y = -scale_x * cos_t / denom;
    double x =  scale_y * (sin_t * cos_t) / denom;
    double z = z0;

    traj.emplace_back(x, y, z);
  }

  // Shift so trajectory starts at origin (in XY), then add bias
  Eigen::Vector3d start_shift = traj.front();
  start_shift.z() = 0.0;

  for (auto& p : traj) {
    p -= start_shift;
    p += bias;
  }

  return traj;
}

// Generate an egg-shaped trajectory.
inline std::vector<Eigen::Vector3d> eggTrajectory(const std::vector<double>& points,
                                                   double scale_x = 0.25,
                                                   double scale_y = 0.35,
                                                   double k = 0.3,
                                                   double z0 = 0.2,
                                                   const Eigen::Vector3d& bias = Eigen::Vector3d::Zero()) {
  std::vector<Eigen::Vector3d> traj;
  traj.reserve(points.size());
  if (points.empty()) return traj;

  const double p0 = points.front();
  const double pn = points.back();
  const double range = pn - p0;

  for (double p : points) {
    double t = (range > 0.0) ? (p - p0) / range : 0.0;
    double theta = t * 2.0 * M_PI;
    double theta_shift = theta - M_PI / 2.0;

    double r = 1.0 - k * std::cos(theta_shift);

    double x = scale_x * r * std::cos(theta_shift);
    double y = scale_y * r * std::sin(theta_shift);
    double z = z0;

    traj.emplace_back(x, y, z);
  }

  auto min_y_it = std::min_element(
      traj.begin(), traj.end(),
      [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
        return a.y() < b.y();
      });
  std::rotate(traj.begin(), min_y_it, traj.end());

  // Shift so first point is at (0, 0, z0)
  Eigen::Vector3d start_shift = traj.front();
  start_shift.z() = 0.0;

  for (auto& p : traj) {
    p -= start_shift;
    p += bias;
  }

  return traj;
}

// Generate a sinusoidal trajectory.
inline std::vector<Eigen::Vector3d> sinTrajectory(const std::vector<double>& points,
                                                   double width = 0.45,
                                                   double length = 2.0,
                                                   double z0 = 0.2,
                                                   const Eigen::Vector3d& bias = Eigen::Vector3d::Zero()) {
  std::vector<Eigen::Vector3d> traj;
  traj.reserve(points.size());
  if (points.empty()) return traj;

  const double p0 = points.front();
  const double pn = points.back();
  const double range = (pn - p0 > 0.0) ? (pn - p0) : 1.0;

  for (double p : points) {
    double t = (p - p0) / range;  // normalized to [0, 1]
    double x = std::sin(3.0 * p) * width;
    double y = t * length;
    double z = z0;

    traj.emplace_back(x, y, z);
  }

  // Shift so first point is at origin in XY
  Eigen::Vector3d start_shift = traj.front();
  start_shift.z() = 0.0;

  for (auto& pt : traj) {
    pt -= start_shift;
    pt += bias;
  }

  return traj;
}

}  // namespace trajectory_utils
