#include <future>
#include <hpc/config.cuh>
#include <hpc/cpu_timer.h>
#include <hpc/cuda_timer.cuh>
#include <hpc/experiment.h>
#include <iostream>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

template <typename T>
void cpu_stencil(T const *in, T *out, int size, int R, hpc::experiment &xp) {
  auto timer = xp.measure<hpc::cpu_timer>("");
  for (int out_idx = 0; out_idx < size; ++out_idx) {
    T res = (T)0;
    for (int in_idx = out_idx - R; in_idx <= out_idx + R; ++in_idx) {
      if (0 <= in_idx && in_idx < size)
        res += in[in_idx];
    }
    out[out_idx] = res;
  }
}

template <typename T>
__global__ void v1_kernel(T const *in, T *out, int size, int R) {
  int out_idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (out_idx >= size)
    return;

  T res = (T)0;
  for (int in_idx = out_idx - R; in_idx <= out_idx + R; ++in_idx) {
    if (0 <= in_idx && in_idx < size)
      res += in[in_idx];
  }
  out[out_idx] = res;
}

template <typename T>
__host__ __device__ T div_round_up(T const &p, T const &q) {
  return (p + q - 1) / q;
}

template <typename T, typename F>
void gpu_wrapper(T const *in, T *out, int size, hpc::experiment &xp, F func) {
  auto total = xp.measure<hpc::cuda_timer>("total");
  thrust::device_vector<T> dev_in(size), dev_out(size);
  thrust::fill(dev_out.begin(), dev_out.end(), (T)0);
  thrust::copy(in, in + size, dev_in.begin());
  {
    auto kernel = xp.measure<hpc::cuda_timer>("kernel");
    auto *dev_in_ptr = thrust::raw_pointer_cast(dev_in.data());
    auto *dev_out_ptr = thrust::raw_pointer_cast(dev_out.data());
    func(dev_in_ptr, dev_out_ptr);
  }
  thrust::copy(dev_out.begin(), dev_out.end(), out);
}

template <typename T>
void gpu_stencil_v1(T const *in, T *out, int size, int R, int block_size,
                    hpc::experiment &xp) {
  gpu_wrapper<T>(
      in, out, size, xp, [&](auto *dev_in_ptr, auto *dev_out_ptr) -> void {
        dim3 block(block_size);
        dim3 grid(div_round_up(size, block_size));
        v1_kernel<T><<<grid, block>>>(dev_in_ptr, dev_out_ptr, size, R);
      });
}

template <typename T>
__global__ void v2_kernel(T const *in, T *out, int size, int R) {
  extern __shared__ T sm_in[];

  T out_val = 0;
  int sec_off = (int)blockIdx.x * (int)blockDim.x,
      out_val_off = sec_off + (int)threadIdx.x;
  int laneIdx = (int)threadIdx.x % WARP_SIZE;

  int block_R = div_round_up(R, (int)blockDim.x);
  for (int in_sec_ord = -block_R; in_sec_ord <= block_R; ++in_sec_ord) {
    int in_sec_off = sec_off + in_sec_ord * (int)blockDim.x;

    int sm_in_val_off = in_sec_off + (int)threadIdx.x;
    __syncthreads();
    if (0 <= sm_in_val_off && sm_in_val_off < size)
      sm_in[threadIdx.x] = in[sm_in_val_off];
    else
      sm_in[threadIdx.x] = (T)0;
    __syncthreads();

    for (int tile_off = 0; tile_off < blockDim.x; tile_off += WARP_SIZE) {
      auto in_tile_off = in_sec_off + tile_off;

      if (__any_sync(FULL_MASK,
                     (in_tile_off <= out_val_off + R) ||
                         (out_val_off - R <= in_tile_off + WARP_SIZE - 1))) {
        T in_val = sm_in[tile_off + laneIdx], tmp;

        for (int lane = 0; lane < WARP_SIZE; ++lane) {
          tmp = __shfl_sync(FULL_MASK, in_val, lane);
          auto delta = (in_tile_off + lane) - out_val_off;
          if (-R <= delta && delta <= R)
            out_val += tmp;
        }
      }
    }
  }

  if (0 <= out_val_off && out_val_off < size)
    out[out_val_off] = out_val;
}

template <typename T>
void gpu_stencil_v2(T const *in, T *out, int size, int R, int block_size,
                    hpc::experiment &xp) {
  gpu_wrapper<T>(
      in, out, size, xp, [&](auto *dev_in_ptr, auto *dev_out_ptr) -> void {
        dim3 block(block_size);
        dim3 grid(div_round_up(size, block_size));
        auto sm = block_size * sizeof(T);
        v2_kernel<<<grid, block, sm>>>(dev_in_ptr, dev_out_ptr, size, R);
      });
}

template <typename T>
__global__ void v3_kernel(T const *in, T *out, int size, int R) {
  extern __shared__ T sm_in[];

  T out_val = 0;
  int sec_off = (int)blockDim.x * (int)blockIdx.x;
  int out_idx = sec_off + (int)threadIdx.x;

  int block_R = div_round_up(R, (int)blockDim.x);
  for (int block_delta = -block_R; block_delta <= block_R; ++block_delta) {
    int in_sec_off = sec_off + block_delta * (int)blockDim.x;

    int sm_in_idx = in_sec_off + (int)threadIdx.x;
    __syncthreads();
    if (0 <= sm_in_idx && sm_in_idx < size)
      sm_in[threadIdx.x] = in[sm_in_idx];
    else
      sm_in[threadIdx.x] = (T)0;
    __syncthreads();

    int start_idx = max(in_sec_off, out_idx - R) - in_sec_off;
    int end_idx =
        min(in_sec_off + (int)blockDim.x, out_idx + R + 1) - in_sec_off;
    for (int in_idx = start_idx; in_idx < end_idx; ++in_idx)
      out_val += sm_in[in_idx];
  }

  if (0 <= out_idx && out_idx < size)
    out[out_idx] = out_val;
}

template <typename T>
void gpu_stencil_v3(T const *in, T *out, int size, int R, int block_size,
                    hpc::experiment &xp) {
  gpu_wrapper<T>(
      in, out, size, xp, [&](auto *dev_in_ptr, auto *dev_out_ptr) -> void {
        dim3 block(block_size);
        dim3 grid(div_round_up(size, block_size));
        auto sm = block_size * sizeof(T);
        v3_kernel<<<grid, block, sm>>>(dev_in_ptr, dev_out_ptr, size, R);
      });
}

struct random_source {
  thrust::default_random_engine eng;
  thrust::uniform_real_distribution<float> dist;

  random_source() : eng(time(nullptr)) {}

  __host__ __device__ float operator()(int) { return dist(eng); }
};

bool allclose(float a, float b) { return abs(b - a) < 1e-4 * abs(a); }

void allclose(thrust::host_vector<float> const &A,
              thrust::host_vector<float> const &B) {
  if (A.size() != B.size())
    throw std::runtime_error("A and B do not have the same size");

  for (int idx = 0; idx < A.size(); ++idx) {
    if (!allclose(A[idx], B[idx])) {
      std::stringstream error_ss;
      error_ss << "A and B differ at index " << idx << "(" << A[idx]
               << " != " << B[idx] << ")";
      throw std::runtime_error(error_ss.str());
    }
  }
}

int main() {
  using namespace std::chrono_literals;
  int num_reps = 16;
  auto timeout = 5.0s;

  hpc::experiment::header({"rep", "algo", "size", "R", "block"});
  for (int rep = 0; rep < num_reps; ++rep) {
    for (auto size : {1'000, 1'000'000, 64'000'000}) {
      thrust::host_vector<float> host_in(size), host_out(size),
          host_dev_out(size);
      auto *host_in_ptr = thrust::raw_pointer_cast(host_in.data());
      auto *host_out_ptr = thrust::raw_pointer_cast(host_out.data());
      auto *host_dev_out_ptr = thrust::raw_pointer_cast(host_dev_out.data());

      thrust::counting_iterator<int> idx_seq;
      thrust::transform(idx_seq, idx_seq + size, host_in.begin(),
                        random_source());

      for (auto R : {3, 30, 300, 3'000}) {
        bool cpu_completed = false;

        {
          auto xp = hpc::experiment(rep, "cpu", size, R, "");
          auto fut = std::async([&]() -> void {
            cpu_stencil(host_in_ptr, host_out_ptr, size, R, xp);
            cpu_completed = true;
          });

          fut.wait_for(timeout);
          if (!cpu_completed)
            xp.clear();
        }

        for (auto block_size : {32, 64, 128, 192, 256, 512, 1024}) {
          {
            auto xp = hpc::experiment(rep, "gpu-v1", size, R, block_size);
            gpu_stencil_v1(host_in_ptr, host_dev_out_ptr, size, R, block_size,
                           xp);
          }
          if (cpu_completed)
            allclose(host_out, host_dev_out);

          {
            auto xp = hpc::experiment(rep, "gpu-v2", size, R, block_size);
            gpu_stencil_v2(host_in_ptr, host_dev_out_ptr, size, R, block_size,
                           xp);
          }
          if (cpu_completed)
            allclose(host_out, host_dev_out);

          {
            auto xp = hpc::experiment(rep, "gpu-v3", size, R, block_size);
            gpu_stencil_v3(host_in_ptr, host_dev_out_ptr, size, R, block_size,
                           xp);
          }
          if (cpu_completed)
            allclose(host_out, host_dev_out);
        }
      }
    }
  }

  return EXIT_SUCCESS;
}