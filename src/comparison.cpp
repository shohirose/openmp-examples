#include <omp.h>

// Microsoft PPL is only available for MSVC
#ifdef _MSC_VER
#include <ppl.h>  // concurrency::parallel_for_each, combinable
#endif            // _MSC_VER

#include <algorithm>  // std::count_if
#include <chrono>     // std::chrono
#include <execution>  // std::execution
#include <iomanip>    // std::setw
#include <iostream>   // std::cout, endl
#include <random>     // std::random_device, mt19937, uniform_real_distribution

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

class PointGenerator {
 public:
  PointGenerator(std::size_t numberOfPoints)
      : numberOfPoints_{numberOfPoints} {}

  /// @brief Generates random points in a region of [0,1]x[0,1] by using STL
  /// random library.
  /// @returns A list of points
  std::vector<Point> operator()() const noexcept {
    std::vector<Point> points(numberOfPoints_);
    const auto n = static_cast<std::int64_t>(numberOfPoints_);

#pragma omp parallel
    {
      std::random_device rnd;
      std::mt19937 engine(rnd());
      std::uniform_real_distribution<> dist(0.0, 1.0);

#pragma omp for
      for (std::int64_t i = 0; i < n; ++i) {
        auto &point = points[i];
        point.x = dist(engine);
        point.y = dist(engine);
      }
    }

    return points;
  }

 private:
  std::size_t numberOfPoints_;
};

/// @brief Calculates pi by using the Monte Carlo Method.
class PiCalculator {
 public:
  /// @param[in] points Points in a region of [0,1]x[0,1]
  PiCalculator(const std::vector<Point> &points) : points_{points} {}

  /// @brief Computes pi
  /// @tparam Counter Functor which counts the number of points in a circle.
  /// @param[in] counter Point counter
  /// @returns Pi
  template <typename Counter>
  double operator()(Counter &&counter) const noexcept {
    const auto numberOfPointsInCircle = counter(points_);
    const auto totalNumberOfPoints = points_.size();
    return 4.0 * numberOfPointsInCircle / totalNumberOfPoints;
  }

 private:
  const std::vector<Point> &points_;
};

/// @brief Counts the number of points in a circle in serial.
struct SerialCounter {
  size_t operator()(const std::vector<Point> &points) const noexcept {
    return std::count_if(begin(points), end(points), [](const Point &point) {
      return point.rsqr() <= 1.0;
    });
  }
};

#ifdef _MSC_VER
/// @brief Counts the number of points in a circle by using Microsoft PPL.
struct MicrosoftPPLCounter {
  size_t operator()(const std::vector<Point> &points) const noexcept {
    using concurrency::combinable;
    using concurrency::parallel_for_each;
    combinable<size_t> count;
    parallel_for_each(begin(points), end(points), [&count](const Point &point) {
      if (point.rsqr() <= 1.0) count.local() += 1;
    });
    return count.combine(std::plus<size_t>());
  }
};
#endif  // _MSC_VER

/// @brief Count the number of points in a circle by using OpenMP
struct OpenMPCounter {
  size_t operator()(const std::vector<Point> &points) const noexcept {
    std::size_t numberOfPointsInCircle = 0;
    const auto numberOfPoints = static_cast<std::int64_t>(points.size());

#pragma omp parallel for reduction(+ : numberOfPointsInCircle)
    for (std::int64_t i = 0; i < numberOfPoints; ++i) {
      if (points[i].rsqr() <= 1.0) ++numberOfPointsInCircle;
    }

    return numberOfPointsInCircle;
  }
};

/// @brief Counts the number of points in a circle by using Parallel STL.
struct ParallelSTLCounter {
  size_t operator()(const std::vector<Point> &points) const noexcept {
    using std::execution::par_unseq;
    return std::count_if(
        par_unseq, begin(points), end(points),
        [](const Point &point) { return point.rsqr() <= 1.0; });
  }
};

using std::chrono::system_clock;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::microseconds;

/// @brief Measures the execution time of PointGenerator.
std::pair<microseconds, std::vector<Point>> measureExecTime(
    const PointGenerator &f) noexcept {
  const auto start = system_clock::now();
  const auto points = f();
  const auto end = system_clock::now();
  return {duration_cast<microseconds>(end - start), points};
}

/// @brief Measures the execution time of PiCalculator.
template <typename Counter>
std::pair<microseconds, double> measureExecTime(const PiCalculator &f,
                                                Counter &&counter) noexcept {
  const auto start = system_clock::now();
  const auto pi = f(std::forward<Counter>(counter));
  const auto end = system_clock::now();
  return {duration_cast<microseconds>(end - start), pi};
}

int main() {
  size_t numberOfPoints = 100'000'000;

  std::cout << "Number of points: " << numberOfPoints << std::endl;

  const auto [time0, points] = measureExecTime(PointGenerator(numberOfPoints));
  std::cout << "Points Generation: time = " << time0.count() << " usec"
            << std::endl;

  const PiCalculator calculator(points);
  const auto [time1, pi1] = measureExecTime(calculator, SerialCounter{});
#ifdef _MSC_VER
  const auto [time2, pi2] = measureExecTime(calculator, MicrosoftPPLCounter{});
#endif  // _MSC_VER
  const auto [time3, pi3] = measureExecTime(calculator, OpenMPCounter{});
  const auto [time4, pi4] = measureExecTime(calculator, ParallelSTLCounter{});

  std::cout << "Serial: time = " << std::setw(10) << time1.count()
            << " usec, pi = " << pi1
#ifdef _MSC_VER
            << "\nPPL   : time = " << std::setw(10) << time2.count()
            << " usec, pi = " << pi2
#endif  // _MSC_VER
            << "\nOpenMP: time = " << std::setw(10) << time3.count()
            << " usec, pi = " << pi3 << "\nSTL   : time = " << std::setw(10)
            << time4.count() << " usec, pi = " << pi4 << std::endl;
}