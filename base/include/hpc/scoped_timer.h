#pragma once
#include "timer.h"
#include <utility>

namespace hpc {
template <typename Timer> class scoped_timer {
public:
  template <typename... Args>
  scoped_timer(double &dur, Args &&...args)
      : dur{dur}, timer{std::forward<Args>(args)...} {
    timer.start();
  }

  ~scoped_timer() {
    timer.end();
    dur = timer.dur();
  }

private:
  double &dur;
  Timer timer;
};
} // namespace hpc