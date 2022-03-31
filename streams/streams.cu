#include <hpc/cuda_timer.cuh>
#include <hpc/experiment.h>
#include <hpc/stream.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

__global__ void kernel(float *a, float *b, float *c, int num_items) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < num_items) {
    int tid1 = tid + 1;
    int tid2 = tid + 2;
    float aSum = (a[tid] + a[tid1] + a[tid2]) / 3.0f;
    float bSum = (b[tid] + b[tid1] + b[tid2]) / 3.0f;
    c[tid] = (aSum + bSum) / 2;
  }
}

template <typename T>
__host__ __device__ T div_round_up(T const &p, T const &q) {
  return (p + q - 1) / q;
}

void func(float const *a, float const *b, float *c, int size, int num_streams,
          int block_size, hpc::experiment &xp) {
  auto total = xp.measure<hpc::cuda_timer>("total");

  std::vector<hpc::stream> streams(num_streams);
  thrust::device_vector<float> dev_a(size + 2 * num_streams),
      dev_b(size + 2 * num_streams), dev_c(size);
  auto *dev_a_ptr = thrust::raw_pointer_cast(dev_a.data());
  auto *dev_b_ptr = thrust::raw_pointer_cast(dev_b.data());
  auto *dev_c_ptr = thrust::raw_pointer_cast(dev_c.data());

  auto sentinel = [&](int stream_idx) -> int {
    int value = ((size_t)stream_idx * (size_t)size) / (size_t)num_streams;
    return value;
  };

  {
    auto no_alloc = xp.measure<hpc::cuda_timer>("no-alloc");

    for (int stream_idx = 0; stream_idx < num_streams; ++stream_idx) {
      auto &stream = streams[stream_idx];

      {
        std::stringstream stream_ss{};
        stream_ss << "stream-" << stream_idx;
        auto stream_timer =
            xp.measure<hpc::cuda_timer>(stream_ss.str(), stream);

        auto host_from = sentinel(stream_idx),
             dev_from = host_from + 2 * stream_idx,
             host_to = sentinel(stream_idx + 1);
        int num_items = host_to - host_from;

        if (stream_idx != num_streams - 1) {
          cudaMemcpyAsync(dev_a_ptr + dev_from, a + host_from,
                          (num_items + 2) * sizeof(float),
                          cudaMemcpyHostToDevice, stream);
          cudaMemcpyAsync(dev_b_ptr + dev_from, b + host_from,
                          (num_items + 2) * sizeof(float),
                          cudaMemcpyHostToDevice, stream);
        } else {
          cudaMemcpyAsync(dev_a_ptr + dev_from, a + host_from,
                          num_items * sizeof(float), cudaMemcpyHostToDevice,
                          stream);
          cudaMemcpyAsync(dev_a_ptr + dev_from + num_items, a,
                          2 * sizeof(float), cudaMemcpyHostToDevice, stream);
          cudaMemcpyAsync(dev_b_ptr + dev_from, b + host_from,
                          num_items * sizeof(float), cudaMemcpyHostToDevice,
                          stream);
          cudaMemcpyAsync(dev_b_ptr + dev_from + num_items, b,
                          2 * sizeof(float), cudaMemcpyHostToDevice, stream);
        }

        {
          std::stringstream kernel_ss{};
          kernel_ss << "kernel-" << stream_idx;
          auto stream_kernel =
              xp.measure<hpc::cuda_timer>(kernel_ss.str(), stream);

          kernel<<<div_round_up(num_items, block_size), block_size, 0,
                   stream>>>(dev_a_ptr + dev_from, dev_b_ptr + dev_from,
                             dev_c_ptr + host_from, num_items);
        }

        cudaMemcpyAsync(c + host_from, dev_c_ptr + host_from,
                        num_items * sizeof(float), cudaMemcpyDeviceToHost,
                        stream);
      }
    }

    for (auto const &stream : streams) {
      cudaStreamSynchronize(stream);
    }
  }
}

struct random_source {
  thrust::default_random_engine eng;
  thrust::uniform_real_distribution<float> dist;

  random_source() : eng(time(nullptr)) {}

  __host__ __device__ float operator()(int) { return dist(eng); }
};

int main() {
  hpc::experiment::header({"rep", "size", "num_streams", "block_size"});
  for (auto const &size : {1 << 10, 1 << 14, 1 << 18, 1 << 22, 1 << 26}) {
    thrust::device_vector<float> a(size), b(size), c(size), ref_c(size);
    auto *a_ptr = thrust::raw_pointer_cast(a.data());
    auto *b_ptr = thrust::raw_pointer_cast(b.data());
    auto *c_ptr = thrust::raw_pointer_cast(c.data());

    for (int rep = 0; rep < 16; ++rep) {
      thrust::counting_iterator<int> idx_seq;
      auto source = random_source();
      thrust::transform(idx_seq, idx_seq + size, a.begin(), source);
      thrust::transform(idx_seq, idx_seq + size, b.begin(), source);

      for (auto const &num_streams : {1, 2, 4, 8, 16, 32, 64}) {
        for (auto const &block_size : {32, 64, 128, 256, 512, 1024}) {
          {
            hpc::experiment xp(rep, size, num_streams, block_size);
            func(a_ptr, b_ptr, c_ptr, size, num_streams, block_size, xp);
          }

          if (num_streams == 1) {
            thrust::copy(c.begin(), c.end(), ref_c.begin());
          } else {
            if (c != ref_c) {
              std::stringstream error_ss{};
              error_ss << "for size=" << size << ", num_streams=" << num_streams
                       << ", the result is wrong";
              throw std::runtime_error(error_ss.str());
            }
          }
        }
      }
    }
  }

  return EXIT_SUCCESS;
}