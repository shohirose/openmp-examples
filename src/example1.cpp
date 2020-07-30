#include <iostream>
#include <omp.h>

using std::cout;
using std::endl;

int main() {
  cout << "Maximum number of threads: " << omp_get_max_threads() << endl;

#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
    cout << "Thread = " << omp_get_thread_num() << ", i = " << i << endl;
  }
}