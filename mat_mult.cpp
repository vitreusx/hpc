#include "benchmark.h"
#include <assert.h>
#include <iostream>
#include <math.h>
#include <omp.h>
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

Matrix matrixMult(Matrix first, Matrix second, int size) {
  Matrix result = generateMatrix(size, true);

  benchmark(
      [&]() -> void {
        for (int i = 0; i < size; i++) {
          for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
              result[i][j] += first[i][k] * second[k][j];
            }
          }
        }
      },
      "Sequential");

  return result;
}

Matrix matrixMultParallel(Matrix first, Matrix second, int size) {
  Matrix result = generateMatrix(size, true);

  benchmark(
      [&]() -> void {
#pragma omp parallel for default(none) shared(result, first, second, size)     \
    collapse(3)
        {
          for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
              for (int k = 0; k < size; k++) {
                result[i][j] += first[i][k] * second[k][j];
              }
            }
          }
        }
      },
      "Parallel");

  return result;
}

void check(Matrix first, Matrix second, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (first[i][j] != second[i][j])
        throw std::runtime_error("matrices not equal");
    }
  }
}

int main() {
  std::vector<int> sizes = {512, 640, 792, 1024, 1280, 1532, 2048};

  for (auto const &size : sizes) {
    std::cout << "checking for " << size << '\n';
    auto first = generateMatrix(size);
    auto second = generateMatrix(size);
    auto sequentialResult = matrixMult(first, second, size);
    auto parallelResult = matrixMultParallel(first, second, size);
    check(sequentialResult, parallelResult, size);
  }
  return 0;
}
