#pragma once
#include "base.cuh"

#define MAX_SPHERES 128
__device__ __constant__ sphere v1_spheres[MAX_SPHERES];

__global__ void v1_kernel(int num_spheres, bitmap_view bitmap) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  if (x >= bitmap.w || y >= bitmap.h)
    return;

  float bitmap_x = (float)x - (float)bitmap.w / 2.0f;
  float bitmap_y = (float)y - (float)bitmap.h / 2.0f;

  float red = 0.0f, green = 0.0f, blue = 0.0f;
  float max_depth = -inf;

  for (int sphere_idx = 0; sphere_idx < num_spheres; ++sphere_idx) {
    auto const &sphere = v1_spheres[sphere_idx];
    float color_falloff;
    float depth = sphere.hit(bitmap_x, bitmap_y, &color_falloff);
    if (depth > max_depth) {
      red = sphere.red * color_falloff;
      green = sphere.green * color_falloff;
      blue = sphere.blue * color_falloff;
      max_depth = depth;
    }
  }

  int off = bitmap.pix_off(x, y);
  bitmap.pixels[off] = pix_int(red);
  bitmap.pixels[off + 1] = pix_int(green);
  bitmap.pixels[off + 2] = pix_int(blue);
  bitmap.pixels[off + 3] = 0xff;
}

void v1_raytracer(dim3 block, thrust::host_vector<sphere> const &spheres,
                  cpu_bitmap &bitmap, hpc::experiment &xp) {
  auto total = xp.measure<hpc::cuda_timer>("total");

  auto *spheres_ptr = thrust::raw_pointer_cast(spheres.data());
  cudaMemcpyToSymbolAsync(v1_spheres, spheres_ptr,
                          spheres.size() * sizeof(sphere), 0,
                          cudaMemcpyHostToDevice);

  gpu_bitmap dev_bitmap(bitmap.w, bitmap.h);

  {
    auto kernel = xp.measure<hpc::cuda_timer>("kernel");
    dim3 grid(div_round_up(bitmap.w, block.x),
              div_round_up(bitmap.h, block.y));
    v1_kernel<<<grid, block>>>(spheres.size(), dev_bitmap);
  }

  thrust::copy(dev_bitmap.pixels.begin(), dev_bitmap.pixels.end(),
               bitmap.pixels.begin());
}