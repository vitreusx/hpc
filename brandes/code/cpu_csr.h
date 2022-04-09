#pragma once
#include "graph.h"

struct cpu_csr_repr {
  explicit cpu_csr_repr(graph const &g) {
    n = g.n;
    m = g.m;

    ptrs = std::vector<int>(n + 1);

    auto deg = std::vector<int>(n, 0);
    for (auto const &e : g.edges)
      ++deg[e.first];

    auto cur = 0;
    for (int v = 0; v < n; ++v) {
      ptrs[v] = cur;
      cur += deg[v];
    }
    ptrs[n] = cur;

    adjs = std::vector<int>(m);
    for (auto const &e : g.edges)
      adjs[ptrs[e.first]++] = e.second;

    for (int v = 0; v < n; ++v)
      ptrs[v] -= deg[v];
  }

  int n, m;
  std::vector<int> ptrs, adjs;
};