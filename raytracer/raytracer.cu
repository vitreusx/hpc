#include "cpu.h"
#include "stock.cuh"
#include "v1.cuh"

int main() {
  int w = 1024, h = 1024;
  cpu_bitmap cpu_bm(w, h), dev_res_bm(w, h);

  int num_spheres = 100;
  thrust::host_vector<sphere> cpu_spheres(num_spheres);

  std::vector<dim3> blocks = {dim3(8, 4),   dim3(8, 8),   dim3(8, 16),
                              dim3(16, 16), dim3(24, 24), dim3(16, 32),
                              dim3(32, 32)};

  hpc::experiment::header({"rep", "algo", "x", "y"});
  for (int rep = 0; rep < 64; ++rep) {
    thrust::counting_iterator<int> idx_seq;
    thrust::transform(idx_seq, idx_seq + cpu_spheres.size(),
                      cpu_spheres.begin(), random_sphere_source(cpu_bm));

    {
      auto xp = hpc::experiment(rep, "cpu", "", "");
      cpu_raytracer(cpu_spheres, cpu_bm, xp);
    }

    for (auto const &block : blocks) {
      std::stringstream params_ss;
      params_ss << block.x << "-" << block.y;

      {
        auto xp = hpc::experiment(rep, "stock", block.x, block.y);
        stock_raytracer(block, cpu_spheres, dev_res_bm, xp);
      }
      assert_equal("stock", cpu_bm, dev_res_bm);

      {
        auto xp = hpc::experiment(rep, "v1", block.x, block.y);
        v1_raytracer(block, cpu_spheres, dev_res_bm, xp);
      }
      assert_equal("v1", cpu_bm, dev_res_bm);
    }
  }

  return EXIT_SUCCESS;
}