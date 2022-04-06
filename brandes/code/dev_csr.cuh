#pragma once
#include "cpu_csr.h"
#include <hpc/copy.cuh>
#include <hpc/dev_buffer.cuh>
#include <hpc/host_buffer.cuh>
#include <hpc/math.cuh>

struct dev_csr_repr {
  explicit dev_csr_repr(graph const &g) {
    auto cpu_csr = cpu_csr_repr(g);
    n = cpu_csr.n;
    m = cpu_csr.m;

    int num_offs = cpu_csr.offs.size();
    offs = hpc::dev_buffer<int>(num_offs);
    hpc::copy<int>(hpc::host_iter<int>(cpu_csr.offs.data()),
                   cpu_csr.offs.data() + num_offs, offs);

    int num_adjs = cpu_csr.adjs.size();
    adjs = hpc::dev_buffer<int>(num_adjs);
    hpc::copy<int>(hpc::host_iter<int>(cpu_csr.adjs.data()),
                   cpu_csr.adjs.data() + num_adjs, adjs);
  }

  int n, m;
  hpc::dev_buffer<int> offs, adjs;
};