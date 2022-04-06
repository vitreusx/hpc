#pragma once
#include "graph.h"

struct cpu_coo_repr {
  explicit cpu_coo_repr(graph const &g) {
    n = g.n;
    m = g.m;

    is = std::vector<int>(m);
    adjs = std::vector<int>(m);

    for (int e_idx = 0; e_idx < g.edges.size(); ++e_idx) {
      is[e_idx] = g.edges[e_idx].first;
      adjs[e_idx] = g.edges[e_idx].second;
    }
  }

  int n, m;
  std::vector<int> is, adjs;
};