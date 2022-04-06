#pragma once
#include "config.cuh"
#include <type_traits>

namespace hpc {
template <typename T> struct host_iter {
  host_iter(T *ptr) : ptr{ptr} {}

  template <typename U = T,
            std::enable_if_t<!std::is_const<U>::value, bool> = true>
  operator host_iter<U const>() const {
    return ptr;
  }

  operator T *() const { return ptr; }

  host_iter<T> operator++() { return host_iter(++ptr); }

  host_iter<T> operator++(int) { return host_iter(ptr++); }

  bool operator!=(host_iter<T> const &other) { return ptr != other.ptr; }

  host_iter<T> operator+(int offset) const {
    return host_iter<T>(ptr + offset);
  }

  T *ptr;
};

template <typename T> class host_buffer {
public:
  host_buffer() : host_buffer(0) {}

  explicit host_buffer(int size) {
    if (size > 0)
      cudaCheck(cudaMallocHost((void **)&ptr, size * sizeof(T)));
    else
      ptr = nullptr;
    size_ = size;
  }

  host_buffer(host_buffer const &) = delete;
  host_buffer &operator=(host_buffer const &) = delete;

  host_buffer(host_buffer &&other) {
    ptr = other.ptr;
    size_ = other.size_;
    other.reset();
  }

  host_buffer &operator=(host_buffer &&other) {
    this->~host_buffer();
    ptr = other.ptr;
    size_ = other.size_;
    other.reset();
    return *this;
  }

  ~host_buffer() {
    if (ptr)
      cudaCheck(cudaFreeHost(ptr));
    reset();
  }

  int size() const { return size_; }

  operator T *() { return ptr; }
  operator T const *() const { return ptr; }

  T &operator[](int idx) { return ptr[idx]; }
  T const &operator[](int idx) const { return ptr[idx]; }

  host_iter<T> begin() { return ptr; }
  host_iter<T const> begin() const { return ptr; }

  operator host_iter<T>() { return ptr; }
  operator host_iter<T const>() const { return ptr; }

  host_iter<T> end() { return ptr + size_; }
  host_iter<T const> end() const { return ptr + size_; }

  host_iter<T> operator+(int off) { return begin() + off; }
  host_iter<T const> operator+(int off) const { return begin() + off; }

private:
  T *ptr;
  int size_;

  void reset() {
    ptr = nullptr;
    size_ = 0;
  }
};
} // namespace hpc