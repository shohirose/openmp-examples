cmake_minimum_required(VERSION 3.8)
project(openmp-examples CXX)

find_package(OpenMP REQUIRED)

add_subdirectory(src)

option(BENCHMARK_ENABLE_TESTING "Enables google-benchmark tests" OFF)
option(BENCHMARK_ENABLE_INSTALL "Enables google-benchmark installation" OFF)
add_subdirectory(third-party/benchmark)
add_subdirectory(benchmark)