#include <omp.h>

#include <iostream>
#include <sstream>
#include <vector>

// Gathers slave vectors into the master vector.
int main() {
  std::vector<int> master;

  std::cout << "slave vectors:\n";

#pragma omp parallel
  {
    std::vector<int> slave;
    const int numProcs = omp_get_num_procs() * 2;
    const int threadId = omp_get_thread_num();

#pragma omp for
    for (int i = 0; i < numProcs; ++i) {
      slave.push_back(threadId);
    }

#pragma omp critical
    {
      // Prints slaves
      std::stringstream ss;
      ss << '[' << omp_get_thread_num() << ']';
      for (auto value : slave) {
        ss << ' ' << value;
      }
      ss << '\n';
      std::cout << ss.str();

      // Gathers slaves into master
      master.insert(master.end(),                            //
                    std::make_move_iterator(slave.begin()),  //
                    std::make_move_iterator(slave.end()));
    }
  }

  std::cout << "master vector\n";
  for (auto value : master) {
    std::cout << value << ' ';
  }
  std::cout << '\n' << std::flush;
}