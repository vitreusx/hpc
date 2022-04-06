#pragma once
#include <istream>
#include <vector>
#include <algorithm>

class graph {
public:
  int n = 0, m = 0;
  std::vector<std::pair<int, int>> edges = {};

  static graph from_stream(std::istream &is) {
    auto g = graph();

    int u, v;
    g.n = 0;
    while (is >> u >> v) {
      g.n = std::max(g.n, v + 1);
      g.edges.emplace_back(u, v);
      g.edges.emplace_back(v, u);
    }

    std::sort(g.edges.begin(), g.edges.end());

    g.m = g.edges.size();
    return g;
  }
};