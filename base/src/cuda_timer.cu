#include "config.cuh"
#include "cuda_timer.cuh"

namespace hpc {
cuda_timer::cuda_timer() : cuda_timer(nullptr) {}

cuda_timer::cuda_timer(std::shared_ptr<stream> timer_stream) {
  this->timer_stream = std::move(timer_stream);
  cudaCheck(cudaEventCreate(&start_t));
  cudaCheck(cudaEventCreate(&end_t));
}

cuda_timer::~cuda_timer() {
  cudaCheck(cudaEventDestroy(end_t));
  cudaCheck(cudaEventDestroy(start_t));
}

void cuda_timer::start() {
  cudaStream_t stream = timer_stream ? (cudaStream_t)*timer_stream : nullptr;
  cudaCheck(cudaEventRecord(start_t, stream));
}

void cuda_timer::end() {
  cudaStream_t stream = timer_stream ? (cudaStream_t)*timer_stream : nullptr;
  cudaCheck(cudaEventRecord(end_t, stream));
}

double cuda_timer::dur() const {
  cudaCheck(cudaEventSynchronize(end_t));
  float ms = 0.0f;
  cudaCheck(cudaEventElapsedTime(&ms, start_t, end_t));
  return ms / 1000.0f;
}
} // namespace hpc