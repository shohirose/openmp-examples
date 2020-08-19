#include <benchmark/benchmark.h>

#include "function.hpp"

using namespace shirose;

std::vector<Point>& getPoints() {
  static std::vector<Point> points = generatePoints(10'000'000);
  return points;
}

template <typename Counter>
void BM_PiCalculation(benchmark::State& state) {
  const auto& points = getPoints();
  for (auto _ : state) {
    const auto numPoints = state.range(0);
    const auto pi = calcPi(points.data(), numPoints, Counter{});
    benchmark::DoNotOptimize(pi);
  }
}

BENCHMARK_TEMPLATE(BM_PiCalculation, SequentialSTLCounter)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22);

BENCHMARK_TEMPLATE(BM_PiCalculation, OpenMPCounter)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22);

BENCHMARK_TEMPLATE(BM_PiCalculation, MicrosoftPPLCounter)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22);

BENCHMARK_TEMPLATE(BM_PiCalculation, ChunkedMicrosoftPPLCounter)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22);

BENCHMARK_TEMPLATE(BM_PiCalculation, ParallelSTLCounter)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22);

BENCHMARK_TEMPLATE(BM_PiCalculation, ParallelOrVectorizedSTLCounter)
    ->RangeMultiplier(4)
    ->Range(1 << 10, 1 << 22);

BENCHMARK_MAIN();