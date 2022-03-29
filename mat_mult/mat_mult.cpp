#include <hpc/cpu_timer.h>
#include <hpc/experiment.h>
#include <hpc/omp_timer.h>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <thread>
#include <vector>

using Row = int *;
using Matrix = Row *;

Matrix generateMatrix(int size, bool empty = false) {
  Matrix result;
  result = new Row[size];

  for (int i = 0; i < size; i++) {
    result[i] = new int[size];
    for (int j = 0; j < size; j++) {
      result[i][j] = empty ? 0 : rand() % 100;
    }
  }

  return result;
}

Matrix matrixMult(Matrix first, Matrix second, int size, hpc::experiment &xp) {
  Matrix result = generateMatrix(size, true);

  {
    auto timer = xp.measure<hpc::cpu_timer>("");
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        auto res = 0;
        for (int k = 0; k < size; k++) {
          res += first[i][k] * second[k][j];
        }
        result[i][j] += res;
      }
    }
  }

  return result;
}

Matrix matrixMultParallel(Matrix first, Matrix second, int size,
                          hpc::experiment &xp) {
  Matrix result = generateMatrix(size, true);

  {
    auto timer = xp.measure<hpc::omp_timer>("");
#pragma omp parallel for default(none) shared(size, first, second, result)
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        int res = 0;
        for (int k = 0; k < size; k++) {
          res += first[i][k] * second[k][j];
        }
        result[i][j] += res;
      }
    }
  }

  return result;
}

void check(Matrix first, Matrix second, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (first[i][j] != second[i][j])
        throw std::runtime_error("matrices not equal!");
    }
  }
}

int main() {
  std::vector<int> sizes = {128, 192, 256, 340, 512};
  auto max_threads = (int)std::thread::hardware_concurrency();
  std::clog << "num threads = " << max_threads << '\n';

  omp_set_num_threads(max_threads);
  srand(time(nullptr));

  hpc::experiment::header({"rep", "algo", "size"});
  for (int rep = 0; rep < 16; ++rep) {
    for (auto const &size : sizes) {
      auto first = generateMatrix(size);
      auto second = generateMatrix(size);

      auto seq_xp = hpc::experiment(rep, "seq", size);
      auto seq_res = matrixMult(first, second, size, seq_xp);

      auto par_xp = hpc::experiment(rep, "par", size);
      auto par_res = matrixMultParallel(first, second, size, par_xp);

      check(seq_res, par_res, size);
    }
  }

  return 0;
}
