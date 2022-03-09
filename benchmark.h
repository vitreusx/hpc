#pragma once
#include <cmath>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

template <typename F> auto benchmark(F func, int num_repeats = 8) {
  std::vector<double> times;

  for (int rep_idx = 0; rep_idx < num_repeats; ++rep_idx) {
    auto start = omp_get_wtime();
    func();
    auto end = omp_get_wtime();
    times.push_back(end - start);
  }

  double mean = 0.0;
  for (auto const &t : times)
    mean += t;
  mean /= (double)times.size();

  double sd = 0.0;
  for (auto const &t : times)
    sd += (t - mean) * (t - mean);
  sd = std::sqrt(sd / (double)(times.size() - 1));

  return std::make_pair(mean, sd);
}