#include "bitmap.h"

cpu_bitmap::cpu_bitmap(int w, int h) {
  pixels.resize(4 * w * h);
  this->w = w;
  this->h = h;
}

int cpu_bitmap::pix_off(int x, int y) const { return 4 * (x + y * w); }

std::ostream &operator<<(std::ostream &os, cpu_bitmap const &bitmap) {
  os << "P3\n";
  os << bitmap.w << " " << bitmap.h << " " << 0xff << '\n';
  for (int y = 0; y < bitmap.h; ++y) {
    for (int x = 0; x < bitmap.w; ++x) {
      auto off = bitmap.pix_off(x, y);
      os << (int)bitmap.pixels[off] << " " << (int)bitmap.pixels[off + 1] << " "
         << (int)bitmap.pixels[off + 2] << " ";
    }
    os << '\n';
  }
  return os;
}

cpu_bitmap::operator bitmap_view() {
  bitmap_view view;
  view.w = w;
  view.h = h;
  view.pixels = thrust::raw_pointer_cast(pixels.data());
  return view;
}