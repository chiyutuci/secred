#pragma once

#include "ginac/ginac.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
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

// The cgroup's memory allowance.  `limited` is false when the cgroup is
// unlimited ("max" in v2, the near-2^63 sentinel in v1) or no cgroup is
// readable; `current` is the group's usage at the time of the read.
struct CgroupMemory {
  bool limited = false;
  std::uint64_t limit = 0;
  std::uint64_t current = 0;
};

// Parse a memory.max / memory.limit_in_bytes value.  nullopt means unlimited,
// either because the file says "max" or because it carries v1's
// "no limit" sentinel; malformed text is also nullopt.
std::optional<std::uint64_t> parse_cgroup_memory_limit(const std::string &text);

// Read the memory allowance and usage the process's cgroup imposes.  `root` is
// the cgroup mount (v2's /sys/fs/cgroup, or its parent for a v1 fixture) and
// `cgroup_file` is normally /proc/self/cgroup.  Both default to the live
// locations and are parameters so tests can use a fixture tree.
CgroupMemory cgroup_memory(
    const std::filesystem::path &root = "/sys/fs/cgroup",
    const std::filesystem::path &cgroup_file = "/proc/self/cgroup");

// The legacy cap on the default memory budget, now a ceiling rather than the
// default itself.
inline constexpr std::uint64_t kDefaultMemoryCeiling =
    512ULL * 1024 * 1024 * 1024;

// The default max_memory: 80% of min(available, the cgroup's remaining
// allowance), capped at `ceiling`.  A machine with no readable memory figures
// falls back to the ceiling; a cgroup that has used its whole allowance gives
// zero.  The 80% factor leaves room for the allocator's overhead and for
// concurrent processes.
std::uint64_t default_memory_budget(
    std::uint64_t available, std::uint64_t total, const CgroupMemory &cgroup,
    std::uint64_t ceiling = kDefaultMemoryCeiling);

// Current resident set size.
std::uint64_t process_used_memory();

// Peak resident set size.
std::uint64_t process_peak_memory();
