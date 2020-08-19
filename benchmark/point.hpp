#ifndef BENCHMARK_POINT_HPP
#define BENCHMARK_POINT_HPP

namespace shirose {

/// @brief Point in two dimensions
struct Point {
  double x;
  double y;

  Point() = default;
  Point(double t_x, double t_y) : x{t_x}, y{t_y} {}
  Point(const Point &) = default;
  Point(Point &&) = default;
  Point &operator=(const Point &) = default;
  Point &operator=(Point &&) = default;

  double rsqr() const noexcept { return x * x + y * y; }
};


} // namespace shirose

#endif  // BENCHMARK_POINT_HPP