# OpenMP Examples

This repository provides c++ code examples of OpenMP.

# How to compile

Examles can be compiled by using CMake.

```terminal
$ cmake -B build -S .
$ cmake --build build
```

# How to run

Examples and benchmarks can be run by

```terminal
$ ./build/src/for_loop.exe
$ ./build/src/gather.exe
$ ./build/src/reduction.exe
$ ./build/benchmark/parallel_benchmark.exe [--benchmark_filter=<regex>]
```