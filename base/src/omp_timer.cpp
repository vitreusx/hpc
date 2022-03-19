#include "omp_timer.h"

namespace hpc {
void omp_timer::start() { start_t = omp_get_wtime(); }
void omp_timer::end() { end_t = omp_get_wtime(); }
double omp_timer::dur() const { return end_t - start_t; }
} // namespace hpc