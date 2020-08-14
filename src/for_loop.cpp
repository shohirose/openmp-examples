#include <omp.h>

#include <cstdio>
#include <iostream>
#include <sstream>

void SingleForLoop();
void DoubleForLoops();

int main() {
  SingleForLoop();
  DoubleForLoops();
}

void SingleForLoop() {
  const int n = omp_get_max_threads();
  printf("Single for-loop with %d threads:\n", n);

#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    std::stringstream ss;
    ss << '[' << omp_get_thread_num() << "] " << i << '\n';
    std::cout << ss.str();
  }
}

void DoubleForLoops() {
  const int n = omp_get_max_threads();
  printf("Double for-loops with %d threads:\n", n);

#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    std::stringstream ss;
    ss << '[' << omp_get_thread_num() << ']';
    for (int j = 0; j < 4; ++j) {
      ss << ' ' << j;
    }
    ss << '\n';
    std::cout << ss.str();
  }
}