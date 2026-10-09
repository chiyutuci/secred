#pragma once

#include "ginac/ginac.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>

template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &os, const std::pair<T1, T2> &p);

// std vector output
template <typename T>
inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  os << "[";
  for (auto it = v.begin(); it != v.end(); ++it) {
    if (it != v.begin())
      os << ", ";
    os << *it;
  }
  os << "]";
  return os;
}

// std pair output
template <typename T1, typename T2>
inline std::ostream &operator<<(std::ostream &os, const std::pair<T1, T2> &p) {
  os << "<" << p.first << ", " << p.second << ">";
  return os;
}

// error message
inline void process_finish(const std::string &result) {
  std::cout << "\n Finish: " << result << "\n" << std::endl;
  exit(EXIT_SUCCESS);
}

// time stamp
void time_stamp(std::ostream &, const std::string &);

// new symbols generation
std::vector<GiNaC::symbol> generate_symbols(const std::string &pre, unsigned n);

// System RAM, both fields in bytes. `available` is the kernel's estimate of
// how much memory a new process can get without swapping (MemAvailable, which
// already accounts for reclaimable page cache). Falls back to sysinfo() when
// /proc/meminfo is unavailable.
struct SystemMemory {
  std::uint64_t total = 0;
  std::uint64_t available = 0;
};
SystemMemory system_memory();

// Current resident memory (RSS) of this process, in bytes, read from
// /proc/self/status VmRSS.  This is the "used" signal against which the
// configurable max_memory budget is enforced (so a smaller max_memory lets a
// run simulate a machine with less RAM).
std::uint64_t process_used_memory();