#pragma once
#include "timer.h"

namespace hpc {
class cuda_timer: public timer {
public:
  cuda_timer();
  ~cuda_timer();

public:
  void start() override;
  void end() override;
  double dur() const override;

private:
  cudaEvent_t start_t, end_t;
};
} // namespace hpc