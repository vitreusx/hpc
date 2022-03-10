#include "benchmark.h"

timer::timer(benchmark *super) : super{super} { start = omp_get_wtime(); }

timer::~timer() {
  auto end = omp_get_wtime();
  auto dur_s = end - start;
  super->samples.push_back(dur_s);
}

timer benchmark::measure() { return timer(this); }

double benchmark::mean() const {
  double sum = 0.0;
  for (auto const &t : samples)
    sum += t;
  return sum / (double)samples.size();
}

double benchmark::sd() const {
  auto m = mean();
  double var = 0.0;
  for (auto const &t : samples)
    var += (t - m) * (t - m);
  return sqrt(var / (double)(samples.size() - 1));
}

double benchmark::quantile(double q) const {
  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  return sorted[(size_t)(q * (double)sorted.size())];
}

std::ostream &operator<<(std::ostream &os, benchmark const &bench) {
  os << std::scientific << bench.mean() << " (sd = " << std::scientific
     << bench.sd() << ", med = " << std::scientific << bench.quantile(0.5)
     << ")";
  return os;
}

void benchmark::reset() { samples = {}; }