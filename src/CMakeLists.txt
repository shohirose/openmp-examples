function(add_example name)
  add_executable(${name}
    ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp
  )
  target_compile_features(${name}
    PUBLIC
      cxx_std_17
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