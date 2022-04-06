#pragma once
#include "copy.cuh"
#include "dev_buffer.cuh"
#include "host_buffer.cuh"
#include "stream.cuh"

namespace hpc {
template <typename T> class dev_var {
public:
  explicit dev_var(T init = T(), std::shared_ptr<stream> stream_ptr = nullptr) {
    var_buf = dev_buffer<T>(1);
    staging = host_buffer<T>(1);

    auto str = stream::from_ptr(stream_ptr);
    staging[0] = std::move(init);
    copy<T>(staging.begin(), staging.end(), var_buf, str);
  }

  dev_var(dev_var const &) = delete;
  dev_var &operator=(dev_var const &) = delete;

  dev_var(dev_var &&) = default;
  dev_var &operator=(dev_var &&) = default;

  dev_var &operator=(T const &val) {
    staging[0] = val;
    auto str = stream::from_ptr(stream_ptr);
    copy<T>(staging.begin(), staging.end(), var_buf, str);
    return *this;
  }

  dev_var &operator=(T &&val) {
    staging[0] = std::move(val);
    auto str = stream::from_ptr(stream_ptr);
    copy<T>(staging.begin(), staging.end(), var_buf, str);
    return *this;
  }

  operator T() {
    auto str = stream::from_ptr(stream_ptr);
    copy<T>(var_buf.begin(), var_buf.end(), staging, str);
    return staging[0];
  }

  T *get() { return var_buf; }

private:
  std::shared_ptr<stream> stream_ptr;
  dev_buffer<T> var_buf;
  host_buffer<T> staging;
};
} // namespace hpc