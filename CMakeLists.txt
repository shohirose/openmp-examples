cmake_minimum_required(VERSION 3.8)
project(openmp-examples CXX)

find_package(OpenMP REQUIRED)

function(add_example name)
  add_executable(${name} src/${name}.cpp)
  target_compile_features(${name} PUBLIC cxx_std_11)
  target_compile_options(${name} PUBLIC ${OpenMP_CXX_FLAGS})
endfunction()

add_example(example1)