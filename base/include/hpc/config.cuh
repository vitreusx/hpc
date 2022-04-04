#pragma once

#include <iostream>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

namespace hpc {
void cudaCheck_(cudaError_t ret, const char *file, int line);

#define cudaCheck(expr) (hpc::cudaCheck_(expr, __FILE__, __LINE__))
} // namespace hpc