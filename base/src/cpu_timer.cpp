#include "cpu_timer.h"

namespace hpc {
void cpu_timer::start() {
  using namespace std::chrono;
  start_t = high_resolution_clock::now();
}

void cpu_timer::end() {
  using namespace std::chrono;
  end_t = high_resolution_clock::now();
}

double cpu_timer::dur() const {
  using namespace std::chrono;
  return duration<double>(end_t - start_t).count();
}
} // namespace hpc