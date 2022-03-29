#include "tex_bitmap.cuh"

tex_bitmap::tex_bitmap(int w, int h) {
  this->w = w;
  this->h = h;

  auto channel_desc = cudaCreateChannelDesc<unsigned char>();
  cudaMallocArray(&pixels, &channel_desc, w, h);

  cudaResourceDesc res_desc = {};
  res_desc.resType = cudaResourceTypeArray;
  res_desc.res.array.array = pixels;

  cudaTextureDesc tex_desc = {};
  tex_desc.addressMode[0] = cudaAddressModeWrap;
  tex_desc.addressMode[1] = cudaAddressModeWrap;
  tex_desc.filterMode = cudaFilterModeLinear;
  tex_desc.readMode = cudaReadModeElementType;
  tex_desc.normalizedCoords = 1;

  tex = {};
  cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr);
}

tex_bitmap::~tex_bitmap() {
  cudaDestroyTextureObject(tex);
  cudaFree(pixels);
}

tex_bitmap::operator tex_bitmap_view() {
  tex_bitmap_view view;
  view.tex = tex;
  view.w = w;
  view.h = h;
  return view;
}