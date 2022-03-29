#pragma once
#include <functional>
#include <ostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

struct bitmap_view {
  unsigned char *pixels;
  int w, h;

  inline __host__ __device__ int pix_off(int x, int y) const {
    return 4 * (x + y * w);
  }
};

struct cpu_bitmap {
  thrust::host_vector<unsigned char> pixels;
  int w, h;

  explicit cpu_bitmap(int w, int h);
  int pix_off(int x, int y) const;

  friend std::ostream &operator<<(std::ostream &os, cpu_bitmap const &bitmap);
  operator bitmap_view();
};

struct gpu_bitmap {
  thrust::device_vector<unsigned char> pixels;
  int w, h;

  explicit gpu_bitmap(int w, int h);
  int pix_off(int x, int y) const;

  operator bitmap_view();
};
