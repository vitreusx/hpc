#pragma once
#include "dev_virt_csr.cuh"

namespace virt {
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

__global__ void forward_step(int ell, int vn, int *d, int *vmap, int *vptrs,
                             int *adjs, float *sigma, bool *cont) {
  int virt_u = threadIdx.x + blockIdx.x * blockDim.x;
  if (virt_u < vn) {
    int u = vmap[virt_u];
    if (d[u] == ell) {
      for (auto v_ptr = vptrs[virt_u]; v_ptr < vptrs[virt_u + 1]; ++v_ptr) {
        auto v = adjs[v_ptr];

        if (d[v] < 0) {
          d[v] = ell + 1;
          *cont = true;
        }

        if (d[v] == ell + 1)
          atomicAdd(&sigma[v], sigma[u]);
      }
    }
  }
}

__global__ void set_delta(int n, int *d, float *sigma, float *delta) {
  int v = threadIdx.x + blockIdx.x * blockDim.x;
  if (v < n && d[v] >= 0)
    delta[v] = 1.0f / sigma[v];
}

__global__ void backward_step(int ell, int vn, int *d, int *vmap, int *vptrs,
                              int *adjs, float *delta) {
  int virt_u = threadIdx.x + blockDim.x * blockIdx.x;
  if (virt_u < vn) {
    int u = vmap[virt_u];
    if (d[u] == ell) {
      float sum = 0.0f;
      for (auto v_ptr = vptrs[virt_u]; v_ptr < vptrs[virt_u + 1]; ++v_ptr) {
        auto v = adjs[v_ptr];
        if (d[v] == ell + 1)
          sum += delta[v];
      }
      atomicAdd(&delta[u], sum);
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

result impl(graph const &g, int mdeg = 4) {
  double total_dur, kernel_dur;
  hpc::host_buffer<float> host_bc;

  {
    auto total = hpc::scoped_timer<hpc::cuda_timer>(total_dur);

    host_bc = hpc::host_buffer<float>(g.n);

    int ell;
    hpc::dev_var<bool> cont(true);
    hpc::dev_buffer<int> d(g.n);
    hpc::dev_buffer<float> sigma(g.n), delta(g.n), dev_bc(g.n);
    dev_virt_csr_repr virt_csr(g, mdeg);

    dim3 block = 256;
    dim3 grid_n = hpc::div_round_up(g.n, block.x);
    dim3 grid_vn = hpc::div_round_up(virt_csr.vn, block.x);

    {
      auto kernel = hpc::scoped_timer<hpc::cuda_timer>(kernel_dur);

      setup_bc<<<grid_n, block>>>(g.n, dev_bc);

      for (int s = 0; s < g.n; ++s) {
        ell = 0;
        setup_vert_props<<<grid_n, block>>>(g.n, s, sigma, d);

        cont = true;
        while (cont) {
          cont = false;
          forward_step<<<grid_vn, block>>>(ell, virt_csr.vn, d, virt_csr.vmap,
                                           virt_csr.vptrs, virt_csr.adjs, sigma,
                                           cont.get());
          ell += 1;
        }

        set_delta<<<grid_n, block>>>(g.n, d, sigma, delta);

        while (ell > 1) {
          ell -= 1;
          backward_step<<<grid_vn, block>>>(ell, virt_csr.vn, d, virt_csr.vmap,
                                            virt_csr.vptrs, virt_csr.adjs,
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
} // namespace virt