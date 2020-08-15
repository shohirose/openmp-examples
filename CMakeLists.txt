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
      ${OpenMP_CXX_FLAGS}
      $<$<CXX_COMPILER_ID:MSVC>:/W4>
      $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
      $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
      $<$<CXX_COMPILER_ID:Intel>:$<IF:$<PLATFORM_ID:Windows>,/warn:all,-warn all>>
    )
  target_compile_definitions(${name}
    PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX>
    )
endfunction()

add_example(for_loop)
add_example(gather)
add_example(reduction)

add_executable(comparison src/comparison.cpp)
target_compile_features(comparison
  PRIVATE
    cxx_std_17
  )
target_compile_options(comparison
  PRIVATE
    ${OpenMP_CXX_FLAGS}
    # /EHsc: MSVC PPL library
    # /Zc:__cplusplus: Update __cplusplus value according to c++ version
    # (Ref: https://docs.microsoft.com/ja-jp/cpp/build/reference/zc-cplusplus?view=vs-2019)
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /EHsc /Zc:__cplusplus>
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Intel>:$<IF:$<PLATFORM_ID:Windows>,/warn:all,-warn all>>
  )
target_compile_definitions(comparison
  PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX>
  )