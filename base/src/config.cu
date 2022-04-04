#include "config.cuh"
#include <iostream>

namespace hpc {
void cudaCheck_(cudaError_t ret, const char *file, int line) {
  if (ret != cudaSuccess) {
    std::cout << "CUDA error \"" << cudaGetErrorString(ret) << " at " << file
              << ":" << line << '\n';
    exit(EXIT_FAILURE);
  }
}
} // namespace hpc