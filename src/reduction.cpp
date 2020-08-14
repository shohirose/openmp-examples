#include <omp.h>

#include <iostream>

void sum();
void subtract();
void multiply();

// Reduction operations of max and min are available for OpenMP 3.0 or higher
#if _OPENMP >= 200805
void max();
void min();
#endif

int main() {
  std::cout << "Number of processes: " << omp_get_num_procs() << '\n';

  sum();
  subtract();
  multiply();

#if _OPENMP >= 200805
  max();
  min();
#endif
}

void sum() {
  int sum = 1;

#pragma omp parallel reduction(+ : sum)
  { ++sum; }

  std::cout << "Sum: " << sum << std::endl;
}

void subtract() {
  int sum = 10;

#pragma omp parallel reduction(- : sum)
  { --sum; }

  std::cout << "Subtract: " << sum << std::endl;
}

void multiply() {
  int mult = 10;

#pragma omp parallel reduction(* : mult)
  { ++mult; }

  std::cout << "Multiply: " << mult << std::endl;
}

#if _OPENMP >= 200805
void max() {
  int number = 0;

#pragma omp parallel reduction(max : number)
  { number = omp_get_thread_num(); }

  std::cout << "Max: " << number << std::endl;
}

void min() {
  int number = 0;

#pragma omp parallel reduction(min : number)
  { number = omp_get_thread_num(); }

  std::cout << "Min: " << number << std::endl;
}
#endif