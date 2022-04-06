#pragma once

namespace hpc {
__host__ __device__ int div_round_up(int p, int q) { return (p + q - 1) / q; }
} // namespace hpc