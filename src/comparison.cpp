#include <omp.h>

#ifdef _MSC_VER
#include <ppl.h>
#endif  // _MSC_VER

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

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

std::vector<Point> generatePoints(size_t numberOfPoints) {
  std::random_device generator;
  std::mt19937_64 engine(generator());
  std::uniform_real_distribution<> dist(0.0, 1.0);
  std::vector<Point> points(numberOfPoints);
  const auto n = static_cast<std::int64_t>(numberOfPoints);

#pragma omp parallel for
  for (std::int64_t i = 0; i < n; ++i) {
    auto &point = points[i];
    point.x = dist(engine);
    point.y = dist(engine);
  }

  return points;
}

class SerialPiCalculator {
 public:
  SerialPiCalculator(const std::vector<Point> &points) : points_{points} {}

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
struct PplPiCalculator {
 public:
  PplPiCalculator(const std::vector<Point> &points) : points_{points} {}

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

struct OpenMpPiCalculator {
 public:
  OpenMpPiCalculator(const std::vector<Point> &points) : points_{points} {}

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

struct Result {
  microseconds time;
  double pi;
};

template <typename F>
Result measureExecTime(F &&f) {
  using std::chrono::system_clock;
  const auto start = system_clock::now();
  const auto pi = f();
  const auto end = system_clock::now();
  return {duration_cast<microseconds>(end - start), pi};
}

int main() {
  size_t numberOfPoints = 10'000'000;
  std::cout << "Number of points: " << numberOfPoints << std::endl;
  const auto points = generatePoints(numberOfPoints);

  const auto result1 = measureExecTime(SerialPiCalculator(points));
#ifdef _MSC_VER
  const auto result2 = measureExecTime(PplPiCalculator(points));
#endif  // _MSC_VER
  const auto result3 = measureExecTime(OpenMpPiCalculator(points));

  std::cout << "Serial: time = " << std::setw(10) << result1.time.count()
            << " usec, pi = " << result1.pi
#ifdef _MSC_VER
            << "\nPPL   : time = " << std::setw(10) << result2.time.count()
            << " usec, pi = " << result2.pi
#endif  // _MSC_VER
            << "\nOpenMP: time = " << std::setw(10) << result3.time.count()
            << " usec, pi = " << result3.pi << std::endl;
}