#pragma once
#include "timer.h"
#include <utility>

namespace hpc {
template <typename Timer> class scoped_timer {
public:
  template <typename... Args>
  explicit scoped_timer(double &dur, Args &&...args)
      : _timer{std::forward<Args>(args)...}, dur{dur} {
    _timer.start();
  }

  ~scoped_timer() {
    _timer.end();
    dur = _timer.dur();
  }

private:
  Timer _timer;
  double &dur;
};
} // namespace hpc