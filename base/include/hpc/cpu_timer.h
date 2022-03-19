#pragma once
#include "timer.h"
#include <chrono>

namespace hpc {
class cpu_timer : public timer {
public:
  void start() override;
  void end() override;
  double dur() const override;

private:
  using time_point_t = decltype(std::chrono::high_resolution_clock::now());
  time_point_t start_t, end_t;
};
} // namespace hpc