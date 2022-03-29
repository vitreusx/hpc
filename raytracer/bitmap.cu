#include "bitmap.h"

gpu_bitmap::gpu_bitmap(int w, int h) {
  pixels.resize(4 * w * h);
  this->w = w;
  this->h = h;
}

int gpu_bitmap::pix_off(int x, int y) const { return 4 * (x + w * y); }

gpu_bitmap::operator bitmap_view() {
  bitmap_view view;
  view.w = w;
  view.h = h;
  view.pixels = thrust::raw_pointer_cast(pixels.data());
  return view;
}
