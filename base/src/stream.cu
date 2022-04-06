#include "config.cuh"
#include "stream.cuh"

namespace hpc {
stream::stream() { cudaCheck(cudaStreamCreate(&handle)); }
stream::stream(cudaStream_t handle) { this->handle = handle; }
stream::~stream() { cudaCheck(cudaStreamDestroy(handle)); }

cudaStream_t stream::from_ptr(std::shared_ptr<stream> &stream_ptr) {
  return stream_ptr ? (cudaStream_t)*stream_ptr : nullptr;
}

stream::operator cudaStream_t &() { return handle; }
stream::operator cudaStream_t const &() const { return handle; }
} // namespace hpc