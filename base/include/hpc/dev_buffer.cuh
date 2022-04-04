#pragma once
#include "config.cuh"
#include <type_traits>

namespace hpc {
template <typename T> struct dev_iter {
  dev_iter(T *ptr) : ptr{ptr} {}

  template <typename U = T,
            std::enable_if_t<!std::is_const<U>::value, bool> = true>
  operator dev_iter<U const>() const {
    return ptr;
  }

  operator T *() const { return ptr; }

  dev_iter<T> operator+(int offset) const { return dev_iter<T>(ptr + offset); }

  T *ptr;
};

template <typename T> class dev_buffer {
public:
  explicit dev_buffer(int size) {
    cudaCheck(cudaMalloc((void **)&ptr, size * sizeof(T)));
    size_ = size;
  }

  dev_buffer(dev_buffer const &other) = delete;
  dev_buffer(dev_buffer &&other) = delete;

  ~dev_buffer() {
    cudaCheck(cudaFree(ptr));
    ptr = nullptr;
    size_ = 0;
  }

  int size() const { return size_; }

  operator T *() { return ptr; }
  operator T const *() const { return ptr; }

  T &operator[](int idx) { return ptr[idx]; }
  T const &operator[](int idx) const { return ptr[idx]; }

  dev_iter<T> begin() { return ptr; }
  dev_iter<T const> begin() const { return ptr; }

  operator dev_iter<T>() { return ptr; }
  operator dev_iter<T const>() const { return ptr; }

  dev_iter<T> end() { return ptr + size_; }
  dev_iter<T const> end() const { return ptr + size_; }

  dev_iter<T> operator+(int off) { return begin() + off; }
  dev_iter<T const> operator+(int off) const { return begin() + off; }

private:
  T *ptr;
  int size_;
};
} // namespace hpc