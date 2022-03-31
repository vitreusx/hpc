#pragma once
#include "timer.h"

namespace hpc {
class cuda_timer : public timer {
public:
  explicit cuda_timer(cudaStream_t stream = nullptr);
  ~cuda_timer();

public:
  void start() override;
  void end() override;
  double dur() const override;

private:
  cudaStream_t stream;
  cudaEvent_t start_t, end_t;
};
} // namespace hpc