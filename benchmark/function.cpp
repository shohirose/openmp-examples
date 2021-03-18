#include "function.hpp"

#include <omp.h>

#include <cstdint>
#include <random>

namespace shirose {

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
      auto& point = points[static_cast<std::size_t>(i)];
      point.x = dist(engine);
      point.y = dist(engine);
    }
  }

  return points;
}

}  // namespace shirose
