#ifndef BENCHMARK_FUNCTION_HPP
#define BENCHMARK_FUNCTION_HPP

#include <omp.h>

#include <cstdint>
#include <random>
#include <vector>

#include "counter.hpp"

namespace shirose {

/// @brief Generates random points in a region of [0,1]x[0,1] by using STL
/// random library.
/// @param[in] numPoints Number of points to generate
/// @returns A list of points
///
/// Requires OpenMP 2.0 or higher
std::vector<Point> generatePoints(std::size_t numPoints) noexcept {
  std::vector<Point> points(numPoints);
  const auto n = static_cast<std::int64_t>(numPoints);

#pragma omp parallel
  {
    std::random_device rnd;
    std::mt19937 engine(rnd());
    std::uniform_real_distribution<> dist(0.0, 1.0);

#pragma omp for
    for (std::int64_t i = 0; i < n; ++i) {
      auto& point = points[i];
      point.x = dist(engine);
      point.y = dist(engine);
    }
  }

  return points;
}

/// @brief Computes pi
/// @tparam Counter Functor which counts the number of points in a circle.
/// @param[in] p Pointer to the beginning of a list of points in a region of [0,1]x[0,1]
/// @param[in] numPoints Number of points
/// @param[in] counter Point counter
/// @returns Pi
template <typename Counter>
double calcPi(const Point* p, std::size_t numPoints, Counter&& counter) noexcept {
  const auto numPointsInCircle = counter(p, numPoints);
  return 4.0 * numPointsInCircle / numPoints;
}

}  // namespace shirose

#endif  // BENCHMARK_FUNCTION_HPP