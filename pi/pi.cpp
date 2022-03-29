#include <hpc/cpu_timer.h>
#include <hpc/experiment.h>
#include <hpc/omp_timer.h>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <vector>

double power(double x, long n) {
  if (n == 0) {
    return 1;
  }

  return x * power(x, n - 1);
}

double calcPi_(long n) {
  if (n < 0) {
    return 0;
  }

  return 1.0 / power(16, n) *
             (4.0 / (8 * n + 1.0) - 2.0 / (8 * n + 4.0) - 1.0 / (8 * n + 5.0) -
              1.0 / (8 * n + 6.0)) +
         calcPi_(n - 1);
}

double calcPi(long n, hpc::experiment &xp) {
  auto timer = xp.measure<hpc::cpu_timer>("");
  return calcPi_(n);
}

double powerParallelReduction(double x, long n) {
  double res = 1.0;
#pragma omp parallel for default(none) shared(x, n) reduction(* : res)
  for (int idx = 0; idx < n; ++idx)
    res *= x;

  return res;
}

double powerParallelCritical(double x, long n) {
  double res = 1.0;
#pragma omp parallel for default(none) shared(x, n, res)
  for (int idx = 0; idx < n; ++idx) {
#pragma omp critical
    res *= x;
  }

  return res;
}

double calcPiParallelReduction(long n, hpc::experiment &xp) {
  auto timer = xp.measure<hpc::omp_timer>("");
  double res = 0.0;

#pragma omp parallel for default(none) shared(n) reduction(+ : res)
  for (int i = 0; i < n; ++i) {
    res += 1.0 / power(16, i) *
           (4.0 / (8.0 * i + 1.0) - 2.0 / (8.0 * i + 4.0) -
            1.0 / (8.0 * i + 5.0) - 1.0 / (8.0 * i + 6.0));
  }

  return res;
}

double calcPiParallelCritical(long n, hpc::experiment &xp) {
  auto timer = xp.measure<hpc::omp_timer>("");
  double res = 0.0;

#pragma omp parallel for default(none) shared(n, res)
  for (int i = 0; i < n; ++i) {
    auto term = 1.0 / power(16, i) *
                (4.0 / (8.0 * i + 1.0) - 2.0 / (8.0 * i + 4.0) -
                 1.0 / (8.0 * i + 5.0) - 1.0 / (8.0 * i + 6.0));

#pragma omp critical
    res += term;
  }

  return res;
}

int main() {
  std::clog << "max threads = " << omp_get_max_threads() << '\n';
  std::vector<int> num_threads = {1, 2, 4, 8, 16, 32, 64};
  std::vector<int> sizes = {100, 1'000, 10'000, 100'000};

  hpc::experiment::header({"rep", "algo", "thr", "size"});
  for (int rep = 0; rep < 16; ++rep) {
    for (auto const &thr : num_threads) {
      omp_set_num_threads(thr);
      for (auto const &size : sizes) {
        double seq_res, par_red_res, par_crit_res;

        {
          auto xp = hpc::experiment(rep, "seq", thr, size);
          seq_res = calcPi(size, xp);
        }

        {
          auto xp = hpc::experiment(rep, "par-red", thr, size);
          par_red_res = calcPiParallelReduction(size, xp);
        }

        {
          auto xp = hpc::experiment(rep, "par-crit", thr, size);
          par_crit_res = calcPiParallelCritical(size, xp);
        }

        if (abs(par_red_res - seq_res) / seq_res > 1e-5)
          throw std::runtime_error(
              "\"parallel reduction\" version yield incorrect results!");

        if (abs(par_crit_res - seq_res) / seq_res > 1e-5)
          throw std::runtime_error(
              "\"parallel critical\" version yield incorrect results!");
      }
    }
  }

  return 0;
}
