#pragma once
#include "graph.h"

struct cpu_coo_repr {
  explicit cpu_coo_repr(graph const &g) {
    n = g.n;
    m = g.m;

    is = std::vector<int>(m);
    adjs = std::vector<int>(m);

    for (int e = 0; e < m; ++e) {
      is[e] = g.edges[e].first;
      adjs[e] = g.edges[e].second;
    }
  }

  int n, m;
  std::vector<int> is, adjs;
};