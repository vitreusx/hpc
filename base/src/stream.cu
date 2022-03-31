#include "stream.cuh"

namespace hpc {
stream::stream() { cudaStreamCreate(&handle); }
stream::~stream() { cudaStreamDestroy(handle); }

stream::operator cudaStream_t &() { return handle; }
stream::operator cudaStream_t const &() const { return handle; }
} // namespace hpc