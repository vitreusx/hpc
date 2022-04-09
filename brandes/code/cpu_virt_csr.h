#pragma once
#include "graph.h"
#include <hpc/math.h>

struct cpu_virt_csr_repr {
  explicit cpu_virt_csr_repr(graph const &g, int mdeg) {
    n = g.n;
    m = g.m;

    auto deg = std::vector<int>(n, 0);
    for (auto const &e : g.edges)
      ++deg[e.first];

    vn = 0;
    for (int v = 0; v < n; ++v)
      vn += hpc::div_round_up(deg[v], mdeg);

    vmap = std::vector<int>(vn);
    auto ptrs = std::vector<int>(n + 1);
    vptrs = std::vector<int>(vn + 1);

    int cur = 0, vcur = 0;
    for (int v = 0; v < n; ++v) {
      ptrs[v] = cur;
      for (int voff = 0; voff < deg[v]; voff += mdeg) {
        vmap[vcur] = v;
        vptrs[vcur] = cur + voff;
        ++vcur;
      }
      cur += deg[v];
    }
    vptrs[vn] = cur;

    adjs = std::vector<int>(m);
    for (auto const &e : g.edges)
      adjs[ptrs[e.first]++] = e.second;
  }

  int n, vn, m;
  std::vector<int> vmap, vptrs, adjs;
};