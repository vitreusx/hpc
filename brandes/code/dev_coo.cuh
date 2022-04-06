#pragma once
#include "graph.h"
#include "cpu_coo.h"
#include <hpc/dev_buffer.cuh>
#include <hpc/copy.cuh>

struct dev_coo_repr {
  explicit dev_coo_repr(graph const& g) {
    auto cpu_coo = cpu_coo_repr(g);

    int num_is = cpu_coo.is.size();
    is = hpc::dev_buffer<int>(num_is);
    hpc::copy<int>(hpc::host_iter<int>(cpu_coo.is.data()),
        cpu_coo.is.data() + num_is, is);

    int num_adjs = cpu_coo.adjs.size();
    adjs = hpc::dev_buffer<int>(num_adjs);
    hpc::copy<int>(hpc::host_iter<int>(cpu_coo.adjs.data()),
        cpu_coo.adjs.data() + num_adjs, adjs);
  }

  int n, m;
  hpc::dev_buffer<int> is, adjs;
};