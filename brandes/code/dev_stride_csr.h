#pragma once
#include "cpu_stride_csr.h"
#include <hpc/copy.cuh>
#include <hpc/dev_buffer.cuh>

struct dev_stride_csr_repr {
  explicit dev_stride_csr_repr(graph const &g, int mdeg) {
    auto cpu_stride_csr = cpu_stride_csr_repr(g, mdeg);

    n = cpu_stride_csr.n;
    vn = cpu_stride_csr.vn;
    m = cpu_stride_csr.m;

    int num_offset = cpu_stride_csr.offset.size();
    offset = hpc::dev_buffer<int>(num_offset);
    hpc::copy<int>(hpc::host_iter<int>(cpu_stride_csr.offset.data()),
                   cpu_stride_csr.offset.data() + num_offset, offset);

    int num_vmap = cpu_stride_csr.vmap.size();
    vmap = hpc::dev_buffer<int>(num_vmap);
    hpc::copy<int>(hpc::host_iter<int>(cpu_stride_csr.vmap.data()),
                   cpu_stride_csr.vmap.data() + num_vmap, vmap);

    int num_nvir = cpu_stride_csr.nvir.size();
    nvir = hpc::dev_buffer<int>(num_nvir);
    hpc::copy<int>(hpc::host_iter<int>(cpu_stride_csr.nvir.data()),
                   cpu_stride_csr.nvir.data() + num_nvir, nvir);

    int num_ptrs = cpu_stride_csr.ptrs.size();
    ptrs = hpc::dev_buffer<int>(num_ptrs);
    hpc::copy<int>(hpc::host_iter<int>(cpu_stride_csr.ptrs.data()),
                   cpu_stride_csr.ptrs.data() + num_ptrs, ptrs);

    int num_adjs = cpu_stride_csr.adjs.size();
    adjs = hpc::dev_buffer<int>(num_adjs);
    hpc::copy<int>(hpc::host_iter<int>(cpu_stride_csr.adjs.data()),
                   cpu_stride_csr.adjs.data() + num_adjs, adjs);
  }

  int n, vn, m;
  hpc::dev_buffer<int> offset, vmap, nvir, ptrs, adjs;
};