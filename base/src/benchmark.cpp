#include "benchmark.h"
#include <algorithm>
#include <cmath>
#include <iomanip>

namespace hpc {
int benchmark::n() const { return (int)samples.size(); }

double benchmark::mean() const {
  double total = 0.0;
  for (auto t : samples)
    total += t;
  return total / (double)n();
}

double benchmark::sd() const {
  double mu = mean(), var = 0.0;
  for (auto t : samples)
    var += (t - mu) * (t - mu);
  return sqrt(var / (double)(n() - 1));
}

double benchmark::median() const { return quantile(0.5); }

double benchmark::quantile(double q) const {
  if (!sorted) {
    std::sort(samples.begin(), samples.end());
    sorted = true;
  }

  auto idx = (int)(q * (double)n());
  if (idx < 0)
    idx = 0;
  if (idx >= n())
    idx = n() - 1;

  return samples[idx];
}

std::ostream &operator<<(std::ostream &os, benchmark const &bench) {
  auto fmt = os.rdstate();
  os << std::scientific << std::setprecision(3);
  os << bench.mean() << " [sd = " << bench.sd() << "; " << bench.quantile(0.25)
     << " - " << bench.quantile(0.5) << " - " << bench.quantile(0.75) << "]";
  os.setstate(fmt);
  return os;
}

void benchmark::add_sample(double sample) {
  samples.push_back(sample);
  sorted = false;
}

void benchmark::reset() { samples = {}; }

} // namespace hpc