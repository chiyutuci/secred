#pragma once

#include "ginac/ginac.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &os, const std::pair<T1, T2> &p);

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

template <typename T1, typename T2>
inline std::ostream &operator<<(std::ostream &os, const std::pair<T1, T2> &p) {
  os << "<" << p.first << ", " << p.second << ">";
  return os;
}

inline void process_finish(const std::string &result) {
  std::cout << "\n Finish: " << result << "\n" << std::endl;
  exit(EXIT_SUCCESS);
}

// Print msg prefixed with a local timestamp.
void time_stamp(std::ostream &, const std::string &);

// Create symbols pre1..preN.
std::vector<GiNaC::symbol> generate_symbols(const std::string &pre, unsigned n);

// Machine memory totals from /proc/meminfo.
struct SystemMemory {
  std::uint64_t total = 0;
  std::uint64_t available = 0;
};
// Total and available memory.
SystemMemory system_memory();

// Current resident set size.
std::uint64_t process_used_memory();

// Peak resident set size.
std::uint64_t process_peak_memory();
