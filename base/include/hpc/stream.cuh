#pragma once
#include <memory>

namespace hpc {
class stream {
public:
  stream();
  explicit stream(cudaStream_t handle);
  ~stream();

  static cudaStream_t from_ptr(std::shared_ptr<stream>& stream_ptr);

  stream(stream const &) = delete;
  stream &operator=(stream const &) = delete;
  stream(stream &&) = default;
  stream &operator=(stream &&) = default;

  operator cudaStream_t &();
  operator cudaStream_t const &() const;

private:
  cudaStream_t handle;
};
} // namespace hpc