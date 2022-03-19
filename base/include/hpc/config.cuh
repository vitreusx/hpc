#pragma once

#include <iostream>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

namespace hpc {
void cudaCheck_(cudaError_t ret, const char *file, int line) {
#ifndef NDEBUG
  if (ret != cudaSuccess) {
    std::cout << "CUDA error \"" << cudaGetErrorString(ret) << " at " << file
              << ":" << line << '\n';
    exit(EXIT_FAILURE);
  }
#endif
}

#define cudaCheck(expr) (hpc::cudaCheck_(expr, __FILE__, __LINE__))
} // namespace hpc