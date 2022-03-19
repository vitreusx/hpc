#pragma once
#include "timer.h"
#include <omp.h>

namespace hpc {
class omp_timer : public timer {
public:
  void start() override;
  void end() override;
  double dur() const override;

private:
  using time_point_t = decltype(omp_get_wtime());
  time_point_t start_t, end_t;
};
} // namespace hpc