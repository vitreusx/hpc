#include "experiment.h"
#include <iomanip>
#include <iostream>
#include <sstream>

namespace hpc {

section_timer::section_timer(experiment &parent, std::string name,
                             std::unique_ptr<timer> timer_ptr)
    : parent(parent), name(std::move(name)), timer_ptr(std::move(timer_ptr)) {
  this->timer_ptr->start();
}

section_timer::~section_timer() {
  timer_ptr->end();
  auto sec_pair = std::make_pair(name, std::move(timer_ptr));
  parent.sections.insert(std::move(sec_pair));
}

void experiment::header(std::vector<std::string> const &columns) {
  std::stringstream header_ss{};
  for (auto const &column : columns)
    header_ss << column << ',';
  header_ss << "section,dur";
  std::cout << header_ss.str() << '\n';
}

experiment::~experiment() {
  auto fmt = std::cout.flags();
  std::cout << std::scientific;

  for (auto const &sec : sections) {
    std::cout << prefix << sec.first << ',' << sec.second->dur() << '\n';
  }

  std::cout.flags(fmt);
}

void experiment::clear() { sections.clear(); }

} // namespace hpc