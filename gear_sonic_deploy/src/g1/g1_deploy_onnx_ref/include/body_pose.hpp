#pragma once

#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sstream>

// G1 body29 hardware order, rad. Bounds match g1_29dof_rh56dfx.xml;
// the offline scene test checks every bound against that model.
inline constexpr std::array<double, 29> body_pose_lower = {
    -2.5307, -.5236, -2.7576, -.087267, -.87267, -.2618,
    -2.5307, -2.9671, -2.7576, -.087267, -.87267, -.2618,
    -2.618, -.52, -.52,
    -3.0892, -1.5882, -2.618, -1.0472, -1.97222, -1.61443, -1.61443,
    -3.0892, -2.2515, -2.618, -1.0472, -1.97222, -1.61443, -1.61443};
inline constexpr std::array<double, 29> body_pose_upper = {
    2.8798, 2.9671, 2.7576, 2.8798, .5236, .2618,
    2.8798, .5236, 2.7576, 2.8798, .5236, .2618,
    2.618, .52, .52,
    2.6704, 2.2515, 2.618, 2.0944, 1.97222, 1.61443, 1.61443,
    2.6704, 1.5882, 2.618, 2.0944, 1.97222, 1.61443, 1.61443};

// Hardware body29 order -> planner waist/interleaved-arm order.
inline constexpr std::array<int, 17> ready_pose_upper_order = {
    12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28};

inline std::array<double, 29> ReadBodyPose(std::istream& input, const std::string& label) {
  std::array<double, 29> pose{};
  for (size_t i = 0; i < pose.size(); ++i) {
    if (!(input >> pose[i]) || !std::isfinite(pose[i]) ||
        pose[i] < body_pose_lower[i] || pose[i] > body_pose_upper[i]) {
      throw std::runtime_error("invalid " + label + " at body joint " + std::to_string(i));
    }
  }
  std::string extra;
  if (input >> extra) throw std::runtime_error(label + " requires exactly 29 angles");
  return pose;
}

inline std::array<double, 29> LoadSimInitialPose(
    const std::string& path, const std::string& network_interface) {
  if (network_interface != "lo" && network_interface != "lo0") {
    throw std::runtime_error("--sim-initial-pose requires loopback lo/lo0");
  }
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot open --sim-initial-pose: " + path);
  return ReadBodyPose(file, "sim initial pose");
}

// Independent preparation targets. These never change policy default_angles.
inline std::array<double, 29> ParseReadyPose(const std::string& values) {
  std::istringstream input(values);
  return ReadBodyPose(input, "ready pose");
}
