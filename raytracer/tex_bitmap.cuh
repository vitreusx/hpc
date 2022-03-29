#pragma once

struct tex_bitmap_view {
  cudaTextureObject_t tex;
  int w, h;
};

struct tex_bitmap {
  cudaArray *pixels;
  int w, h;
  cudaTextureObject_t tex;

  explicit tex_bitmap(int w, int h);
  ~tex_bitmap();

  operator tex_bitmap_view();
};