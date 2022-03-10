#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

class benchmark;
class timer;

class timer {
public:
  ~timer();

private:
  friend class benchmark;
  benchmark *super;
  explicit timer(benchmark *super);

  using time_point_t = decltype(omp_get_wtime());
  time_point_t start;
};

class benchmark {
public:
  timer measure();
  void reset();

public:
  std::vector<double> samples;
  double mean() const;
  double sd() const;
  double quantile(double q) const;

  friend std::ostream &operator<<(std::ostream &os, benchmark const &bench);
};
