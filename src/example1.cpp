#include <cstdio>
#include <omp.h>

void SingleForLoopWithMaxNumThreads(int numLoops);
void SingleForLoopWithTwoThreads(int numLoops);
void DoubleForLoopWithMaxNumThreads(int numOuterLoops, int numInnerLoops);

int main() {
  printf("Maximum number of threads = %d\n\n", omp_get_max_threads());

  SingleForLoopWithMaxNumThreads(10);
  SingleForLoopWithTwoThreads(10);
  DoubleForLoopWithMaxNumThreads(10, 2);
}

void SingleForLoopWithMaxNumThreads(int numOuterLoops) {
  printf("[Single for-loop with %d threads]\n", omp_get_max_threads());

#pragma omp parallel for
  for (int i = 0; i < numOuterLoops; ++i) {
    printf("thread = %d, loop counter = %d\n", omp_get_thread_num(), i);
  }
  printf("\n");
}

void SingleForLoopWithTwoThreads(int numOuterLoops) {
  printf("[Single for-loop with %d threads]\n", 2);

#pragma omp parallel for num_threads(2)
  for (int i = 0; i < numOuterLoops; ++i) {
    printf("thread = %d, loop counter = %d\n", omp_get_thread_num(), i); 
  }
  printf("\n");
}

void DoubleForLoopWithMaxNumThreads(int numOuterLoops, int numInnerLoops) {
  printf("[Double for-loops with %d threads]\n", omp_get_max_threads());

#pragma omp parallel for
  for (int i = 0; i < numOuterLoops; ++i) {
    const auto id = omp_get_thread_num();
    for (int j = 0; j < numInnerLoops; ++j) {
      printf("thread = %d, outer loop counter = %d, inner loop counter = %d\n", omp_get_thread_num(), i, j);
    }
  }
  printf("\n");
}