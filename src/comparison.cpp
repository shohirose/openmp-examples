#include <omp.h>

#ifdef _MSC_VER
#include <ppl.h>
#endif  // _MSC_VER

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

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
  using result_type = std::vector<Point>;

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

/// @brief Calculates Pi using Monte Carlo method in serial.
class SerialPiCalculator {
 public:
  using result_type = double;

  /// @param[in] points Points in a region of [0,1]x[0,1]
  SerialPiCalculator(const std::vector<Point> &points) : points_{points} {}

  /// @brief Computes Pi
  double operator()() const {
    size_t numberOfPointsInCircle = 0;
    const auto numberOfPoints = points_.size();
    for (auto &&point : points_) {
      if (point.rsqr() <= 1.0) ++numberOfPointsInCircle;
    }
    return 4 * static_cast<double>(numberOfPointsInCircle) / numberOfPoints;
  }

 private:
  const std::vector<Point> &points_;
};

#ifdef _MSC_VER
/// @brief Calculates Pi using Monte Carlo method by using Microsoft PPL
struct PplPiCalculator {
 public:
  using result_type = double;

  /// @param[in] points Points in a region of [0,1]x[0,1]
  PplPiCalculator(const std::vector<Point> &points) : points_{points} {}

  /// @brief Computes Pi
  double operator()() const {
    using concurrency::combinable;
    using concurrency::parallel_for_each;
    combinable<size_t> count;
    parallel_for_each(begin(points_), end(points_),
                      [&count](const Point &point) {
                        if (point.rsqr() <= 1.0) count.local() += 1;
                      });
    const auto numberOfPointsInCircle = count.combine(std::plus<size_t>());
    return 4 * static_cast<double>(numberOfPointsInCircle) / points_.size();
  }

 private:
  const std::vector<Point> &points_;
};
#endif  // _MSC_VER

/// @brief Calculates Pi using Monte Carlo method by using OpenMP
struct OmpPiCalculator {
 public:
  using result_type = double;

  /// @param[in] points Points in a region of [0,1]x[0,1]
  OmpPiCalculator(const std::vector<Point> &points) : points_{points} {}

  /// @brief Computes Pi
  double operator()() const {
    std::int64_t numberOfPointsInCircle = 0;
    const auto numberOfPoints = static_cast<std::int64_t>(points_.size());

#pragma omp parallel for reduction(+ : numberOfPointsInCircle)
    for (std::int64_t i = 0; i < numberOfPoints; ++i) {
      if (points_[i].rsqr() <= 1.0) ++numberOfPointsInCircle;
    }

    return 4 * static_cast<double>(numberOfPointsInCircle) / numberOfPoints;
  }

 private:
  const std::vector<Point> &points_;
};

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::microseconds;

/// @brief Measures the execution time of a given functor.
template <typename F>
std::pair<microseconds, typename F::result_type> measureExecTime(F &&f) {
  using std::chrono::system_clock;
  const auto start = system_clock::now();
  const auto result = f();
  const auto end = system_clock::now();
  return {duration_cast<microseconds>(end - start), result};
}

int main() {
  size_t numberOfPoints = 100'000'000;

  std::cout << "Number of points: " << numberOfPoints << std::endl;

  const auto result0 = measureExecTime(PointGenerator(numberOfPoints));
  const auto &points = result0.second;
  std::cout << "Points Generation: time = " << result0.first.count() << " usec"
            << std::endl;

  const auto result1 = measureExecTime(SerialPiCalculator(points));
#ifdef _MSC_VER
  const auto result2 = measureExecTime(PplPiCalculator(points));
#endif  // _MSC_VER
  const auto result3 = measureExecTime(OmpPiCalculator(points));

  std::cout << "Serial: time = " << std::setw(10) << result1.first.count()
            << " usec, pi = " << result1.second
#ifdef _MSC_VER
            << "\nPPL   : time = " << std::setw(10) << result2.first.count()
            << " usec, pi = " << result2.second
#endif  // _MSC_VER
            << "\nOpenMP: time = " << std::setw(10) << result3.first.count()
            << " usec, pi = " << result3.second << std::endl;
}