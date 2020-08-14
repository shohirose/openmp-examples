#include <omp.h>

#include <iostream>

int main() {
  int sum = 0;
#pragma omp parallel reduction(+ : sum)
  { ++sum; }

  std::cout << "Number of processes: " << omp_get_num_procs() << '\n'
            << "Sum: " << sum << std::endl;
}