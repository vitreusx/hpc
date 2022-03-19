#pragma once
#include "scoped_timer.h"
#include <memory>
#include <ostream>
#include <vector>

namespace hpc {
class benchmark;

std::ostream &operator<<(std::ostream &os, benchmark const &bench);

template <typename Timer> class bench_timer {
public:
  ~bench_timer();

private:
  friend class benchmark;
  template <typename... Args>
  bench_timer(benchmark &bench, Args &&...args)
      : bench{bench}, timer{dur, std::forward<Args>(args)...} {}

private:
  explicit bench_timer(benchmark &bench, std::shared_ptr<timer> timer);

  benchmark &bench;
  double dur;
  scoped_timer<Timer> timer;
};

class benchmark {
public:
  template <typename Timer, typename... Args>
  bench_timer<Timer> measure(Args &&...args) {
    return bench_timer<Timer>(*this, std::forward<Args>(args)...);
  }

  void reset();
  void add_sample(double sample);

public:
  int n() const;
  double mean() const;
  double sd() const;
  double median() const;
  double quantile(double q) const;

  friend std::ostream &operator<<(std::ostream &os, benchmark const &bench);

public:
  mutable bool sorted = true;
  mutable std::vector<double> samples;
};

template <typename Timer> bench_timer<Timer>::~bench_timer<Timer>() {
  timer.~scoped_timer<Timer>();
  bench.add_sample(dur);
}

} // namespace hpc