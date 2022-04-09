#pragma once
#include "dev_csr.cuh"
#include <hpc/cuda_timer.cuh>
#include <hpc/dev_var.cuh>
#include <hpc/scoped_timer.h>

namespace vertex {
__global__ void setup_bc(int n, float *bc) {
  int v = threadIdx.x + blockIdx.x * blockDim.x;
  if (v < n)
    bc[v] = 0.0f;
}

__global__ void setup_vert_props(int n, int s, float *sigma, int *d) {
  int v = threadIdx.x + blockIdx.x * blockDim.x;
  if (v < n) {
    if (v != s) {
      sigma[v] = 0.0f;
      d[v] = -1;
    } else {
      sigma[v] = 1.0f;
      d[v] = 0;
    }
  }
}

__global__ void setup_edge_props(int m, int *P) {
  int e = threadIdx.x + blockIdx.x * blockDim.x;
  if (e < m)
    P[e] = 0;
}

__global__ void forward_step(int ell, int n, int *d, int *ptrs, int *adjs,
                             int *P, float *sigma, bool *cont) {
  int u = threadIdx.x + blockIdx.x * blockDim.x;
  if (u < n && d[u] == ell) {
    for (auto vptr = ptrs[u]; vptr < ptrs[u + 1]; ++vptr) {
      auto v = adjs[vptr];

      if (d[v] < 0) {
        d[v] = ell + 1;
        *cont = true;
      }

      if (d[v] == ell + 1) {
        atomicAdd(&sigma[v], sigma[u]);
        P[vptr] = 1;
      }
    }
  }
}

__global__ void set_delta(int n, int *d, float *sigma, float *delta) {
  int v = threadIdx.x + blockIdx.x * blockDim.x;
  if (v < n && d[v] >= 0)
    delta[v] = 1.0f / sigma[v];
}

__global__ void backward_step(int ell, int n, int *d, int *ptrs, int *adjs,
                              int *P, float *delta) {
  int u = threadIdx.x + blockDim.x * blockIdx.x;
  if (u < n && d[u] == ell) {
    for (auto vptr = ptrs[u]; vptr < ptrs[u + 1]; ++vptr) {
      if (P[vptr]) {
        auto v = adjs[vptr];
        delta[u] += delta[v];
      }
    }
  }
}

__global__ void update_bc_values(int n, int s, float *bc, float *delta,
                                 float *sigma, int *d) {
  int v = threadIdx.x + blockDim.x * blockIdx.x;
  if (v < n && v != s && d[v] >= 0)
    bc[v] += delta[v] * sigma[v] - 1.0f;
}

struct result {
  hpc::host_buffer<float> bc;
  int total_ms, kernel_ms;
};

result impl(graph const &g) {
  double total_dur, kernel_dur;
  hpc::host_buffer<float> host_bc;

  {
    auto total = hpc::scoped_timer<hpc::cuda_timer>(total_dur);

    host_bc = hpc::host_buffer<float>(g.n);

    int ell;
    hpc::dev_var<bool> cont(true);
    hpc::dev_buffer<int> d(g.n), P(g.m);
    hpc::dev_buffer<float> sigma(g.n), delta(g.n), dev_bc(g.n);
    dev_csr_repr csr(g);

    dim3 block = 256;
    dim3 grid_n = hpc::div_round_up(g.n, block.x);
    dim3 grid_m = hpc::div_round_up(g.m, block.x);

    {
      auto kernel = hpc::scoped_timer<hpc::cuda_timer>(kernel_dur);

      setup_bc<<<grid_n, block>>>(g.n, dev_bc);

      for (int s = 0; s < g.n; ++s) {
        ell = 0;
        setup_vert_props<<<grid_n, block>>>(g.n, s, sigma, d);
        setup_edge_props<<<grid_m, block>>>(g.m, P);

        cont = true;
        while (cont) {
          cont = false;
          forward_step<<<grid_n, block>>>(ell, g.n, d, csr.ptrs, csr.adjs, P,
                                          sigma, cont.get());
          ell += 1;
        }

        set_delta<<<grid_n, block>>>(g.n, d, sigma, delta);

        while (ell > 1) {
          ell -= 1;
          backward_step<<<grid_n, block>>>(ell, g.n, d, csr.ptrs, csr.adjs, P,
                                           delta);
        }

        update_bc_values<<<grid_n, block>>>(g.n, s, dev_bc, delta, sigma, d);
      }
    }

    hpc::copy<float>(dev_bc.begin(), dev_bc.end(), host_bc);
  }

  result res;
  res.bc = std::move(host_bc);
  res.total_ms = std::floor(total_dur);
  res.kernel_ms = std::floor(kernel_dur);
  return res;
}
} // namespace vertex
