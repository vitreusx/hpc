#include <cuda_runtime.h>
#include <hpc/copy.cuh>
#include <hpc/cuda_timer.cuh>
#include <hpc/dev_buffer.cuh>
#include <hpc/experiment.h>
#include <hpc/host_buffer.cuh>
#include <hpc/stream.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

__global__ void kernel(float *a, float *b, float *c, int num_items) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < num_items) {
    auto va = a[tid], vb = b[tid], vc = 0.0f;
    for (int i = 0; i < 224; ++i) {
      vc = sin(vc + va) * cos(vc + vb);
    }
    c[tid] = vc;
  }
}

template <typename T>
__host__ __device__ T div_round_up(T const &p, T const &q) {
  return (p + q - 1) / q;
}

void ref(hpc::host_iter<float const> a, hpc::host_iter<float const> b,
         hpc::host_iter<float> c, int size, int block_size,
         hpc::experiment &xp) {
  auto total = xp.measure<hpc::cuda_timer>("total");

  hpc::dev_buffer<float> dev_a(size), dev_b(size), dev_c(size);
  {
    auto stream_timer = xp.measure<hpc::cuda_timer>("stream-0");

    hpc::copy<float>(a, a + size, dev_a);
    hpc::copy<float>(b, b + size, dev_b);

    {
      auto kernel_timer = xp.measure<hpc::cuda_timer>("kernel-0");

      kernel<<<div_round_up(size, block_size), block_size>>>(dev_a, dev_b,
                                                             dev_c, size);
    }

    hpc::copy<float>(dev_c, dev_c + size, c);
  }
}

void func(hpc::host_iter<float const> a, hpc::host_iter<float const> b,
          hpc::host_iter<float> c, int size, int num_streams, int block_size,
          hpc::experiment &xp) {
  auto total = xp.measure<hpc::cuda_timer>("total");

  std::vector<std::shared_ptr<hpc::stream>> streams(num_streams);
  hpc::dev_buffer<float> dev_a(size), dev_b(size), dev_c(size);

  auto sentinel = [&](int stream_idx) -> int {
    int value = ((size_t)stream_idx * (size_t)size) / (size_t)num_streams;
    return value;
  };

  for (int stream_idx = 0; stream_idx < num_streams; ++stream_idx) {
    auto &stream = streams[stream_idx];
    stream = std::make_shared<hpc::stream>();

    {
      std::stringstream stream_ss{};
      stream_ss << "stream-" << stream_idx;
      auto stream_timer = xp.measure<hpc::cuda_timer>(stream_ss.str(), stream);

      auto from = sentinel(stream_idx), to = sentinel(stream_idx + 1);
      int num_items = to - from;

      hpc::copy<float>(a + from, a + to, dev_a + from, *stream);
      hpc::copy<float>(b + from, b + to, dev_b + from, *stream);

      {
        std::stringstream kernel_ss{};
        kernel_ss << "kernel-" << stream_idx;
        auto stream_kernel =
            xp.measure<hpc::cuda_timer>(kernel_ss.str(), stream);

        kernel<<<div_round_up(num_items, block_size), block_size, 0, *stream>>>(
            dev_a + from, dev_b + from, dev_c + from, num_items);
      }

      hpc::copy<float>(dev_c + from, dev_c + to, c + from, *stream);
    }
  }

  for (auto const &stream : streams) {
    cudaCheck(cudaStreamSynchronize(*stream));
  }
}

struct random_source {
  thrust::default_random_engine eng;
  thrust::uniform_real_distribution<float> dist;

  random_source() : eng(time(nullptr)) {}

  __host__ __device__ float operator()(int) { return dist(eng); }
};

template <typename T>
bool operator!=(hpc::host_buffer<T> const &A, hpc::host_buffer<T> const &B) {
  if (A.size() != B.size())
    return true;

  for (int i = 0; i < A.size(); ++i)
    if (A[i] != B[i])
      return true;

  return false;
}

int main() {
  hpc::experiment::header({"rep", "size", "num_streams", "block_size"});
  for (auto const &size : {1 << 10, 1 << 14, 1 << 18, 1 << 22, 1 << 26}) {
    hpc::host_buffer<float> a(size), b(size), c(size), ref_c(size);

    for (int rep = 0; rep < 16; ++rep) {
      thrust::counting_iterator<int> idx_seq;
      auto source = random_source();
      thrust::transform(idx_seq, idx_seq + size, (float *)a, source);
      thrust::transform(idx_seq, idx_seq + size, (float *)b, source);

      for (auto const &block_size : {32, 64, 128, 256, 512, 1024}) {
        {
          hpc::experiment xp(rep, size, 0, block_size);
          ref(a, b, c, size, block_size, xp);
        }

        hpc::copy<float>(c.begin(), c.end(), ref_c);

        for (auto const &num_streams : {1, 2, 4, 8, 16, 32, 64}) {
          {
            hpc::experiment xp(rep, size, num_streams, block_size);
            func(a, b, c, size, num_streams, block_size, xp);
          }

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

  return EXIT_SUCCESS;
}