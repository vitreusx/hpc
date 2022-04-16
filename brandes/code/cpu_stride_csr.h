#pragma once
#include "graph.h"
#include <hpc/math.h>

struct cpu_stride_csr_repr {
  explicit cpu_stride_csr_repr(graph const &g, int mdeg) {
    n = g.n;
    m = g.m;

    auto deg = std::vector<int>(n, 0);
    for (auto const &e : g.edges)
      ++deg[e.first];

    vn = 0;
    nvir = std::vector<int>(n);
    for (int v = 0; v < n; ++v) {
      nvir[v] = hpc::div_round_up(deg[v], mdeg);
      vn += nvir[v];
    }

    vmap = std::vector<int>(vn);
    ptrs = std::vector<int>(n + 1);
    offset = std::vector<int>(vn);

    int cur = 0, vcur = 0;
    for (int v = 0; v < n; ++v) {
      ptrs[v] = cur;
      for (int voff = 0; voff < nvir[v]; ++voff) {
        offset[vcur] = voff;
        vmap[vcur] = v;
        ++vcur;
      }
      cur += deg[v];
    }
    ptrs[n] = cur;

    adjs = std::vector<int>(m);
    for (auto const &e : g.edges)
      adjs[ptrs[e.first]++] = e.second;
    
    for (int v = 0; v < n; ++v)
      ptrs[v] -= deg[v];
  }

  int n, vn, m;
  std::vector<int> offset, vmap, nvir, ptrs, adjs;
};