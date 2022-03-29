#pragma once
#include "timer.h"
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace hpc {
class experiment;

class section_timer {
public:
  explicit section_timer(experiment &parent, std::string name,
                         std::unique_ptr<timer> timer_ptr);
  ~section_timer();

  section_timer(section_timer const &) = delete;
  section_timer(section_timer &&) = default;

private:
  experiment &parent;
  std::string name;
  std::unique_ptr<timer> timer_ptr;
};

template <typename... Params> struct join;

template <> struct join<> {
  static void func(std::stringstream &) {}
};

template <typename Head, typename... Tail> struct join<Head, Tail...> {
  static void func(std::stringstream &ss, Head &&head, Tail &&...tail) {
    ss << head << ',';
    join<Tail...>::func(ss, std::forward<Tail>(tail)...);
  }
};

class experiment {
public:
  template <typename... Params> explicit experiment(Params &&...params) {
    std::stringstream prefix_ss{};
    join<Params...>::func(prefix_ss, std::forward<Params>(params)...);
    prefix = prefix_ss.str();
  }
  ~experiment();

  experiment(experiment const &) = delete;
  experiment(experiment &&) = default;

  static void header(std::vector<std::string> const &columns);

  template <typename Timer, typename... Args>
  section_timer measure(std::string const &section, Args &&...args) {
    auto timer_ptr = std::make_unique<Timer>(std::forward<Args>(args)...);
    return section_timer(*this, section, std::move(timer_ptr));
  }

  void clear();

private:
  friend class section_timer;
  std::string prefix;
  std::unordered_map<std::string, std::unique_ptr<timer>> sections;
};

} // namespace hpc