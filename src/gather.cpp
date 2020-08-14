#include <omp.h>

#include <iostream>
#include <sstream>
#include <vector>

void gatherStdVector();

int main() { gatherStdVector(); }

void gatherStdVector() {
  std::vector<int> master;
  const int n = omp_get_num_procs() * 2;
  std::cout << "slave std::vectors:\n";

#pragma omp parallel
  {
    std::vector<int> slave;
#pragma omp for
    for (int i = 0; i < n; ++i) {
      slave.push_back(omp_get_thread_num());
    }

#pragma omp critical
    {
      // Prints slaves
      std::stringstream ss;
      ss << '[' << omp_get_thread_num() << ']';
      for (auto value : slave) ss << ' ' << value;
      ss << '\n';
      std::cout << ss.str();

      // Gathers slaves into master
      master.insert(master.end(),                            //
                    std::make_move_iterator(slave.begin()),  //
                    std::make_move_iterator(slave.end()));
    }
  }

  std::cout << "master std::vector\n";
  for (auto value : master) std::cout << value << ' ';
  std::cout << '\n' << std::flush;
}