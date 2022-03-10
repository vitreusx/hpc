#include "benchmark.h"
#include <iostream>
#include <omp.h>
#include <thread>

#define SIZE 500

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

Matrix matrixMult(Matrix first, Matrix second, int size) {
  Matrix result = generateMatrix(size, true);

  auto bench = benchmark();
  for (int rep = 0; rep < 16; ++rep) {
    auto timer = bench.measure();
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

  std::cout << "[seq] " << bench << '\n';
  return result;
}

Matrix matrixMultParallel(Matrix first, Matrix second, int size) {
  Matrix result = generateMatrix(size, true);

  auto bench = benchmark();
  for (int rep = 0; rep < 16; ++rep) {
    auto timer = bench.measure();
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

  std::cout << "[par] " << bench << '\n';
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
  std::cout << "num threads = " << max_threads << '\n';
  
  omp_set_num_threads(max_threads);
  srand(time(nullptr));

  for (auto const &size : sizes) {
    std::cout << "size = " << size << '\n';
    auto first = generateMatrix(size);
    auto second = generateMatrix(size);
    auto sequentialResult = matrixMult(first, second, size);
    auto parallelResult = matrixMultParallel(first, second, size);
    check(sequentialResult, parallelResult, size);
  }
  return 0;
}
