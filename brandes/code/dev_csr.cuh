#pragma once
#include "cpu_csr.h"
#include <hpc/copy.cuh>
#include <hpc/dev_buffer.cuh>
#include <hpc/host_buffer.cuh>
#include <hpc/math.h>

struct dev_csr_repr {
  explicit dev_csr_repr(graph const &g) {
    auto cpu_csr = cpu_csr_repr(g);
    n = cpu_csr.n;
    m = cpu_csr.m;

    int num_ptrs = cpu_csr.ptrs.size();
    ptrs = hpc::dev_buffer<int>(num_ptrs);
    hpc::copy<int>(hpc::host_iter<int>(cpu_csr.ptrs.data()),
                   cpu_csr.ptrs.data() + num_ptrs, ptrs);

    int num_adjs = cpu_csr.adjs.size();
    adjs = hpc::dev_buffer<int>(num_adjs);
    hpc::copy<int>(hpc::host_iter<int>(cpu_csr.adjs.data()),
                   cpu_csr.adjs.data() + num_adjs, adjs);
  }

  int n, m;
  hpc::dev_buffer<int> ptrs, adjs;
};