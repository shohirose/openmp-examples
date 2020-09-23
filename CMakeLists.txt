cmake_minimum_required(VERSION 3.8)
project(openmp-examples CXX)

find_package(OpenMP REQUIRED)
find_package(benchmark REQUIRED)

add_subdirectory(src)
add_subdirectory(benchmark)