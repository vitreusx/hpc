#include "benchmark.h"
#include <iomanip>
#include <iostream>
#include <omp.h>

double power(double x, long n) {
  if (n == 0) {
    return 1;
  }

  return x * power(x, n - 1);
}

double calcPi(long n) {
  if (n < 0) {
    return 0;
  }

  return 1.0 / power(16, n) *
             (4.0 / (8 * n + 1.0) - 2.0 / (8 * n + 4.0) - 1.0 / (8 * n + 5.0) -
              1.0 / (8 * n + 6.0)) +
         calcPi(n - 1);
}

double powerParallelReduction(double x, long n) {
  double res = 1.0;
#pragma omp parallel for default(none) shared(x, n) reduction(* : res)
  for (int idx = 0; idx < n; ++idx) {
    res *= x;
  }
  return res;
}

double powerParallelCritical(double x, long n) {
  double res = 1.0;

#pragma omp parallel default(none) shared(x, n, res)
  {
    double local_res = 1.0;
#pragma omp for nowait
    for (int idx = 0; idx < n; ++idx)
      local_res *= x;

#pragma omp critical
    res *= local_res;
  }

  return res;
}

double calcPiParallelReduction(long n) {
  if (n < 0) {
    return 0;
  } else {
    double res = 0.0;
#pragma omp parallel for default(none) shared(n) reduction(+ : res)
    for (int idx = 0; idx < n; ++idx) {
      res += 1.0 / power(16, idx) *
             (4.0 / (8.0 * idx + 1.0) - 2.0 / (8.0 * idx + 4.0) -
              1.0 / (8.0 * idx + 5.0) - 1.0 / (8.0 * idx + 6.0));
    }
    return res;
  }
}

double calcPiParallelCritical(long n) {
  if (n < 0) {
    return 0.0;
  } else {
    double res = 0.0;

#pragma omp parallel default(none) shared(n, res)
    {
      double local_res = 0.0;
#pragma omp for nowait
      for (int idx = 0; idx < n; ++idx) {
        res += 1.0 / powerParallelCritical(16, idx) *
               (4.0 / (8.0 * idx + 1.0) - 2.0 / (8.0 * idx + 4.0) -
                1.0 / (8.0 * idx + 5.0) - 1.0 / (8.0 * idx + 6.0));
      }

#pragma omp critical
      res += local_res;
    }

    return res;
  }
}

int main() {
  //  omp_set_nested(1);
  omp_set_max_active_levels(2);

  std::cout << "steps,threads,seq(mean),seq(sd),par_red(mean),par_red(sd)"
            << '\n';
  for (auto const &steps : {100, 1'000, 10'000, 100'000}) {
    for (auto const &threads : {2, 4, 8, 16, 32, 64}) {
      omp_set_num_threads(threads);

      auto [seq_mean, seq_sd] =
          benchmark([&]() -> auto { return calcPi(steps); });

      auto [red_mean, red_sd] =
          benchmark([&]() -> auto { return calcPiParallelReduction(steps); });

      std::cout << steps << "," << threads;
      std::cout << "," << seq_mean << "," << seq_sd;
      std::cout << "," << red_mean << "," << red_sd;
      std::cout << std::endl;
    }
  }

  return 0;
}
