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

// High-water mark of this process's resident memory, in bytes (/proc/self/status
// VmHWM).  The end-of-run Timing Statistics block reports it: it is the figure
// that says how close the run came to the max_memory budget.  Falls back to
// getrusage's ru_maxrss when /proc is unavailable.
std::uint64_t process_peak_memory();

// High-water mark of the reaped child processes' resident memory, in bytes
// (getrusage RUSAGE_CHILDREN).  SecRed launches exactly one child — the global
// Kira reduction — so this is the peak of that run, and the kernel attributes a
// child's own children to it through the process it waited on: the figure
// includes everything Kira ran, not just Kira itself.  The end-of-run peak that
// includes the external solver is max(process_peak_memory(), this).  0 before any
// child has been reaped.
std::uint64_t child_peak_memory();

// Equations the external solver selected out of the system it was handed, read
// from its log line "Number of selected equations to reduce: N equations".  When
// a log holds several selection passes the last one wins, since that is the pass
// whose reconstruction the run reports.  nullopt when the line is absent: an
// unreadable log must not be reported as "0 equations selected".
std::optional<std::uint64_t> parse_selected_equations(const std::string &text);

// The same count, read from the solver's log file (<cache_dir>/global/kira.log).
// nullopt when the file is missing or holds no selection line.
std::optional<std::uint64_t> selected_equations_from_log(const std::string &path);