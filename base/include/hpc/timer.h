#pragma once

namespace hpc {
class timer {
public:
  virtual ~timer() = default;
  
  virtual void start() = 0;
  virtual void end() = 0;
  virtual double dur() const = 0;
};
} // namespace hpc