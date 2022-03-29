#pragma once
#include "base.cuh"

void cpu_raytracer_(sphere const *spheres, int num_spheres,
                    bitmap_view bitmap) {
  for (int y = 0; y < bitmap.h; ++y) {
    float bitmap_y = (float)y - (float)bitmap.h / 2.0f;
    for (int x = 0; x < bitmap.w; ++x) {
      float bitmap_x = (float)x - (float)bitmap.w / 2.0f;

      float red = 0.0f, green = 0.0f, blue = 0.0f;
      float max_depth = -inf;
      for (int sphere_idx = 0; sphere_idx < num_spheres; ++sphere_idx) {
        auto const &sphere = spheres[sphere_idx];

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
  }
}

void cpu_raytracer(thrust::host_vector<sphere> const &spheres,
                   cpu_bitmap &bitmap, hpc::experiment &xp) {
  auto timer = xp.measure<hpc::cpu_timer>("total");
  auto *spheres_ptr = thrust::raw_pointer_cast(spheres.data());
  cpu_raytracer_(spheres_ptr, spheres.size(), bitmap);
}