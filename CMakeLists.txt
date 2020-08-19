cmake_minimum_required(VERSION 3.8)
project(openmp-examples CXX)

find_package(OpenMP REQUIRED)

function(add_example name)
  add_executable(${name} src/${name}.cpp)
  target_compile_features(${name}
    PUBLIC
      cxx_std_11
    )
  target_compile_options(${name}
    PUBLIC
      $<$<CXX_COMPILER_ID:MSVC>:/W4>
      $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
      $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
      $<$<CXX_COMPILER_ID:Intel>:$<IF:$<PLATFORM_ID:Windows>,/warn:all,-warn all>>
    )
  target_compile_definitions(${name}
    PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX>
    )
  target_link_libraries(${name}
    PRIVATE
      OpenMP::OpenMP_CXX
    )
endfunction()

add_example(for_loop)
add_example(gather)
add_example(reduction)

find_package(benchmark REQUIRED)

add_executable(parallel_benchmark benchmark/parallel_benchmark.cpp)
target_compile_features(parallel_benchmark
  PRIVATE
    cxx_std_17
  )
target_compile_options(parallel_benchmark
  PRIVATE
    # /EHsc: For Microsoft PPL library
    # (Ref: https://docs.microsoft.com/ja-jp/cpp/parallel/concrt/parallel-algorithms?view=vs-2019#the-parallel_for_each-algorithm)
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /EHsc>
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Intel>:$<IF:$<PLATFORM_ID:Windows>,/warn:all,-warn all>>
  )
target_compile_definitions(parallel_benchmark
  PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX>
  )
target_link_libraries(parallel_benchmark
  PRIVATE
    OpenMP::OpenMP_CXX
    benchmark::benchmark
    $<$<CXX_COMPILER_ID:MSVC>:shlwapi>
  )