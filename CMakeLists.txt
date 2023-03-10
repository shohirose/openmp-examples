cmake_minimum_required(VERSION 3.14)

include(FetchContent)
FetchContent_Declare(
    benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.7.1.zip
    DOWNLOAD_EXTRACT_TIMESTAMP false
)
option(BENCHMARK_ENABLE_TESTING "Enables google-benchmark tests" OFF)
option(BENCHMARK_ENABLE_INSTALL "Enables google-benchmark installation" OFF)
FetchContent_MakeAvailable(benchmark)

project(openmp-examples CXX)

find_package(OpenMP REQUIRED)

add_subdirectory(src)
add_subdirectory(benchmark)