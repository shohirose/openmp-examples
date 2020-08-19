#ifndef BENCHMARK_COUNTER_HPP
#define BENCHMARK_COUNTER_HPP

#ifdef _MSC_VER
#include <ppl.h>  // concurrency::parallel_for_each, combinable, static_partitioner
#endif            // _MSC_VER

#include <algorithm>  // std::count_if
#include <execution>  // std::exection::par, par_unseq
#include <thread>     // std::thread::hardware_concurrency

#include "point.hpp"

namespace shirose {

/// @brief Counts the number of points in a circle sequentially using STL.
struct SequentialSTLCounter {
  /// @param[in] p Pointer to the begging of a list of points
  /// @param[in] n Number of points
  std::size_t operator()(const Point *p, std::size_t n) const noexcept {
    return std::count_if(
        p, p + n, [](const Point &point) { return point.rsqr() <= 1.0; });
  }
};

#ifdef _MSC_VER

/// @brief Counts the number of points in a circle by using Microsoft PPL.
struct MicrosoftPPLCounter {
  /// @param[in] p Pointer to the begging of a list of points
  /// @param[in] n Number of points
  std::size_t operator()(const Point *p, std::size_t n) const noexcept {
    concurrency::combinable<size_t> count;
    concurrency::parallel_for_each(p, p + n, [&count](const Point &point) {
      if (point.rsqr() <= 1.0) count.local()++;
    });
    return count.combine(std::plus<size_t>());
  }
};

/// @brief Counts the number of points in a circle by using Microsoft PPL and
/// manually dividing a range into chunks.
struct ChunkedMicrosoftPPLCounter {
  /// @param[in] p Pointer to the begging of a list of points
  /// @param[in] n Number of points
  std::size_t operator()(const Point *p, std::size_t n) const noexcept {
    const auto numThreads = std::thread::hardware_concurrency();

    // A list of indices representing the range of each thread
    std::vector<std::size_t> range(numThreads + 1, 0);
    const auto chunkSize = n / numThreads;
    for (unsigned int i = 1; i < numThreads; ++i) {
      range[i] = chunkSize * i;
    }
    range.back() = n;

    concurrency::combinable<std::size_t> count;
    concurrency::parallel_for(
        0u, numThreads,
        [p, &count, &range](auto i) {
          std::size_t localCount = 0;
          const auto begin = range[i];
          const auto end = range[i + 1];
          for (auto j = begin; j < end; ++j) {
            if (p[j].rsqr() <= 1.0) ++localCount;
          }
          count.local() += localCount;
        },
        concurrency::static_partitioner{});
    return count.combine(std::plus<std::size_t>());
  }
};

#endif  // _MSC_VER

/// @brief Count the number of points in a circle by using OpenMP
struct OpenMPCounter {
  /// @param[in] p Pointer to the begging of a list of points
  /// @param[in] n Number of points
  std::size_t operator()(const Point *p, std::size_t n) const noexcept {
    std::size_t numPointsInCircle = 0;
    const auto numPoints = static_cast<std::int64_t>(n);

#pragma omp parallel for reduction(+ : numPointsInCircle)
    for (std::int64_t i = 0; i < numPoints; ++i) {
      if (p[i].rsqr() <= 1.0) ++numPointsInCircle;
    }

    return numPointsInCircle;
  }
};

/// @brief Counts the number of points in a circle by using Parallel STL with
/// std::execution::par.
struct ParallelSTLCounter {
  /// @param[in] p Pointer to the begging of a list of points
  /// @param[in] n Number of points
  std::size_t operator()(const Point *p, std::size_t n) const noexcept {
    return std::count_if(std::execution::par, p, p + n, [](const Point &point) {
      return point.rsqr() <= 1.0;
    });
  }
};

/// @brief Counts the number of points in a circle by using Parallel STL with
/// std::execution::par_unseq.
struct ParallelOrVectorizedSTLCounter {
  /// @param[in] p Pointer to the begging of a list of points
  /// @param[in] n Number of points
  std::size_t operator()(const Point *p, std::size_t n) const noexcept {
    return std::count_if(
        std::execution::par_unseq, p, p + n,
        [](const Point &point) { return point.rsqr() <= 1.0; });
  }
};

}  // namespace shirose

#endif  // BENCHMARK_COUNTER_HPP