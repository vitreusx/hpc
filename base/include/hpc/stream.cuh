#pragma once

namespace hpc {
class stream {
public:
  stream();
  ~stream();

  operator cudaStream_t&();
  operator cudaStream_t const&() const;

private:
  cudaStream_t handle;
};
} // namespace hpc