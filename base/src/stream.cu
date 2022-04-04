#include "config.cuh"
#include "stream.cuh"

namespace hpc {
stream::stream() { cudaCheck(cudaStreamCreate(&handle)); }
stream::stream(cudaStream_t handle) { this->handle = handle; }
stream::~stream() { cudaCheck(cudaStreamDestroy(handle)); }

stream::operator cudaStream_t &() { return handle; }
stream::operator cudaStream_t const &() const { return handle; }
} // namespace hpc