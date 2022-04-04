#pragma once
#include "stream.cuh"
#include "timer.h"
#include <memory>

namespace hpc {
class cuda_timer : public timer {
public:
  explicit cuda_timer();
  explicit cuda_timer(std::shared_ptr<stream> timer_stream);
  ~cuda_timer() override;

public:
  void start() override;
  void end() override;
  double dur() const override;

private:
  std::shared_ptr<stream> timer_stream;
  cudaEvent_t start_t, end_t;
};
} // namespace hpc