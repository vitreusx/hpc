#pragma once
#include "graph.h"

struct cpu_csr_repr {
  explicit cpu_csr_repr(graph const &g) {
    n = g.n;
    m = g.m;

    offs = std::vector<int>(n + 1);
    adjs = std::vector<int>(m);

    std::vector<int> deg(n, 0);
    for (auto const &e : g.edges) {
      ++deg[e.first];
    }

    auto cur = 0;
    for (int v = 0; v <= n; ++v) {
      offs[v] = cur;
      if (v < n)
        cur += deg[v];
    }

    for (auto const &e : g.edges) {
      adjs[offs[e.first]++] = e.second;
    }

    for (int v = 0; v < n; ++v)
      offs[v] -= deg[v];
  }

  int n, m;
  std::vector<int> offs, adjs;
};