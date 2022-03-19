#include "cuda_timer.cuh"

namespace hpc {
cuda_timer::cuda_timer() {
  cudaEventCreate(&start_t);
  cudaEventCreate(&end_t);
}

cuda_timer::~cuda_timer() {
  cudaEventDestroy(end_t);
  cudaEventDestroy(start_t);
}

void cuda_timer::start() { cudaEventRecord(start_t); }

void cuda_timer::end() { cudaEventRecord(end_t); }

double cuda_timer::dur() const {
  cudaEventSynchronize(end_t);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start_t, end_t);
  return ms / 1000.0f;
}
} // namespace hpc