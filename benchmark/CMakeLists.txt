add_executable(parallel_benchmark
    parallel_benchmark.cpp
    function.cpp
  )
target_compile_features(parallel_benchmark
  PRIVATE
    cxx_std_17
  )
target_include_directories(parallel_benchmark
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
target_compile_options(parallel_benchmark
  PRIVATE
    # /EHsc: For Microsoft PPL library
    # (Ref: https://docs.microsoft.com/ja-jp/cpp/parallel/concrt/parallel-algorithms?view=vs-2019#the-parallel_for_each-algorithm)
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /EHsc>
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
  )