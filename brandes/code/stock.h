#pragma once
#include "cpu_csr.h"
#include "graph.h"
#include <hpc/cpu_timer.h>
#include <hpc/scoped_timer.h>
#include <queue>
#include <stack>

struct result {
  std::vector<float> bc;
  int total_ms, kernel_ms;
};

result stock_impl(graph const &g) {
  result res;
  double dur;

  {
    auto timer = hpc::scoped_timer<hpc::cpu_timer>(dur);

    auto csr = cpu_csr_repr(g);

    res.bc = std::vector<float>(g.n, 0.0f);

    std::stack<int> S;
    std::queue<int> Q;
    std::vector<std::vector<int>> P(g.n);
    std::vector<float> sigma(g.n), delta(g.n);
    std::vector<int> d(g.n);

    for (int s = 0; s < g.n; ++s) {
      for (int v = 0; v < g.n; ++v)
        P[v].clear();
      std::fill(sigma.begin(), sigma.end(), 0.0f);
      std::fill(d.begin(), d.end(), -1);

      Q.push(s);
      sigma[s] = 1.0f;
      d[s] = 0;

      // Forward phase: BFS from $s$
      while (!Q.empty()) {
        auto v = Q.front();
        Q.pop();
        S.push(v);
        for (auto w_off = csr.offs[v]; w_off < csr.offs[v + 1]; ++w_off) {
          auto w = csr.adjs[w_off];
          if (d[w] < 0) {
            Q.push(w);
            d[w] = d[v] + 1;
          }
          if (d[w] == d[v] + 1) {
            sigma[w] += sigma[v];
            P[w].push_back(v);
          }
        }
      }

      // Backward phase: Back propagation
      for (int v = 0; v < g.n; ++v)
        if (d[v] >= 0)
          delta[v] = 1.0f / sigma[v];

      while (!S.empty()) {
        auto w = S.top();
        S.pop();
        for (auto const &v : P[w]) {
          delta[v] += delta[w];
        }
      }

      // Update bc values by using Equation (5)
      for (int v = 0; v < g.n; ++v)
        if (d[v] >= 0 && v != s)
          res.bc[v] += (delta[v] * sigma[v] - 1.0f);
    }
  }

  res.total_ms = res.kernel_ms = std::floor(1000.0 * dur);
  return res;
}