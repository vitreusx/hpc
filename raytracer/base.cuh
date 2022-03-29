#pragma once
#include "bitmap.h"
#include <hpc/config.cuh>
#include <hpc/cpu_timer.h>
#include <hpc/cuda_timer.cuh>
#include <hpc/experiment.h>
#include <iostream>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

#define inf 2e10f

struct sphere {
  float red, green, blue;
  float radius;
  float x, y, z;

  __host__ __device__ float hit(float bitmap_x, float bitmap_y,
                                float *color_falloff) const {
    float dist_x = bitmap_x - x, dist_y = bitmap_y - y;
    float dist_z_sq = radius * radius - dist_x * dist_x - dist_y * dist_y;
    if (dist_z_sq > 0.0f) {
      float dist_z = sqrtf(dist_z_sq);
      *color_falloff = dist_z / radius;
      return dist_z + z;
    }
    return -inf;
  }
};

__host__ __device__ unsigned char pix_int(float x) {
  int pv = 256.0f * x;
  if (pv > 255)
    pv = 255;
  return pv;
}

__host__ __device__ int div_round_up(int p, int q) { return (p + q - 1) / q; }

void assert_equal(std::string const &name, cpu_bitmap const &cpu,
                  cpu_bitmap const &gpu) {
  auto check_channel = [&](std::string const &type, int type_off) -> void {
    float mean_cpu = 0.0f, mean_gpu = 0.0f;
    float N = cpu.w * cpu.h;

    for (int y = 0; y < cpu.h; ++y) {
      for (int x = 0; x < cpu.w; ++x) {
        auto idx = cpu.pix_off(x, y) + type_off;
        mean_cpu += (float)cpu.pixels[idx];
        mean_gpu += (float)gpu.pixels[idx];
      }
    }

    mean_cpu /= N, mean_gpu /= N;

    float sd_cpu = 0.0f, sd_gpu = 0.0f, cov = 0.0f;
    for (int y = 0; y < cpu.h; ++y) {
      for (int x = 0; x < cpu.w; ++x) {
        auto idx = cpu.pix_off(x, y) + type_off;
        float cpu_val = cpu.pixels[idx], gpu_val = gpu.pixels[idx];
        sd_cpu += (cpu_val - mean_cpu) * (cpu_val - mean_cpu);
        sd_gpu += (gpu_val - mean_gpu) * (gpu_val - mean_gpu);
        cov += (cpu_val - mean_cpu) * (gpu_val - mean_gpu);
      }
    }

    cov /= N;
    sd_cpu = sqrt(sd_cpu / N), sd_gpu = sqrt(sd_gpu / N);

    float rho = cov / (sd_cpu * sd_gpu);
    if (abs(rho - 1.0f) > 1.0e-4f) {
      std::stringstream error_ss;
      error_ss << "for kernel " << name << ", " << type
               << " channels have insufficient correlation (rho = " << rho
               << ")";
      throw std::runtime_error(error_ss.str());
    }
  };

  check_channel("red", 0);
  check_channel("green", 1);
  check_channel("blue", 2);
}

struct random_sphere_source {
  thrust::default_random_engine eng;
  thrust::uniform_real_distribution<float> dist;
  float w, h;

  random_sphere_source(bitmap_view const &bm)
      : eng(time(nullptr)), w(bm.w), h(bm.h) {}

  __host__ __device__ sphere operator()(int) {
    sphere sph;
    sph.red = dist(eng);
    sph.green = dist(eng);
    sph.blue = dist(eng);
    sph.x = w * (dist(eng) - 0.5f);
    sph.y = h * (dist(eng) - 0.5f);
    sph.z = 1000.0f * (dist(eng) - 0.5f);
    sph.radius = 20.0f + 100.0f * dist(eng);
    return sph;
  }
};