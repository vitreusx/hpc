#pragma once
#include "dev_buffer.cuh"
#include "host_buffer.cuh"

namespace hpc {
template <typename U>
inline void copy_(U const *first, U const *last, U *dest, cudaMemcpyKind kind,
                  cudaStream_t stream) {
  if (last != first) {
    if (stream != nullptr) {
      cudaCheck(cudaMemcpyAsync(dest, first, (last - first) * sizeof(U), kind,
                                stream));
    } else {
      cudaCheck(cudaMemcpy(dest, first, (last - first) * sizeof(U), kind));
    }
  }
}

template <typename U>
void copy(host_iter<const U> first, host_iter<const U> last, host_iter<U> dest,
          cudaStream_t stream = nullptr) {
  copy_<U>(first, last, dest, cudaMemcpyHostToHost, stream);
}

template <typename U>
void copy(host_iter<const U> first, host_iter<const U> last, dev_iter<U> dest,
          cudaStream_t stream = nullptr) {
  copy_<U>(first, last, dest, cudaMemcpyHostToDevice, stream);
}

template <typename U>
void copy(dev_iter<const U> first, dev_iter<const U> last, host_iter<U> dest,
          cudaStream_t stream = nullptr) {
  copy_<U>(first, last, dest, cudaMemcpyDeviceToHost, stream);
}

template <typename U>
void copy(dev_iter<const U> first, dev_iter<const U> last, dev_iter<U> dest,
          cudaStream_t stream = nullptr) {
  copy_<U>(first, last, dest, cudaMemcpyDeviceToDevice, stream);
}
} // namespace hpc