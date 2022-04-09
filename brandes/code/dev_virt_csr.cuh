#pragma once
#include "cpu_virt_csr.h"
#include <hpc/copy.cuh>
#include <hpc/dev_buffer.cuh>

struct dev_virt_csr_repr {
  explicit dev_virt_csr_repr(graph const &g, int mdeg) {
    auto cpu_virt_csr = cpu_virt_csr_repr(g, mdeg);

    n = cpu_virt_csr.n;
    vn = cpu_virt_csr.vn;
    m = cpu_virt_csr.m;

    int num_vmap = cpu_virt_csr.vmap.size();
    vmap = hpc::dev_buffer<int>(num_vmap);
    hpc::copy<int>(hpc::host_iter<int>(cpu_virt_csr.vmap.data()),
                   cpu_virt_csr.vmap.data() + num_vmap, vmap);

    int num_vptrs = cpu_virt_csr.vptrs.size();
    vptrs = hpc::dev_buffer<int>(num_vptrs);
    hpc::copy<int>(hpc::host_iter<int>(cpu_virt_csr.vptrs.data()),
                   cpu_virt_csr.vptrs.data() + num_vptrs, vptrs);

    int num_adjs = cpu_virt_csr.adjs.size();
    adjs = hpc::dev_buffer<int>(num_adjs);
    hpc::copy<int>(hpc::host_iter<int>(cpu_virt_csr.adjs.data()),
                   cpu_virt_csr.adjs.data() + num_adjs, adjs);
  }

  int n, vn, m;
  hpc::dev_buffer<int> vmap, vptrs, adjs;
};