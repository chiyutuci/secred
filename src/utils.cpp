#include "utils.h"

#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <cctype>
#include <optional>
#include <string>

std::vector<GiNaC::symbol> generate_symbols(const std::string &pre,
                                            unsigned n) {
  std::vector<GiNaC::symbol> symbols(n);
  for (size_t i = 0; i < n; ++i)
    symbols[i] = GiNaC::symbol(pre + std::to_string(i + 1));
  return symbols;
}

void time_stamp(std::ostream &out, const std::string &msg) {
  auto now = std::chrono::system_clock::now();
  std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
  std::tm *localtime = std::localtime(&currentTime);
  out << msg << std::asctime(localtime) << std::endl;
}

SystemMemory system_memory() {
  SystemMemory sm;

  std::ifstream in("/proc/meminfo");
  if (in.is_open()) {
    std::string key;
    std::uint64_t kb;
    while (in >> key) {
      if (key == "MemTotal:") {
        in >> kb;
        sm.total = kb * 1024;
      } else if (key == "MemAvailable:") {
        in >> kb;
        sm.available = kb * 1024;
        if (sm.total)
          return sm;
      }
    }
  }

  struct sysinfo si;
  if (sysinfo(&si) == 0) {
    std::uint64_t unit = si.mem_unit ? si.mem_unit : 1;
    sm.total = (std::uint64_t)si.totalram * unit;
    std::uint64_t free =
        ((std::uint64_t)si.freeram + (std::uint64_t)si.bufferram) * unit;
    sm.available = free > sm.total ? sm.total : free;
  }
  return sm;
}

std::uint64_t process_used_memory() {

  std::ifstream in("/proc/self/status");
  if (in.is_open()) {
    std::string key;
    std::uint64_t kb;
    while (in >> key) {
      if (key == "VmRSS:") {
        in >> kb;
        return kb * 1024;
      }
    }
  }

  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) == 0)
    return (std::uint64_t)ru.ru_maxrss * 1024;
  return 0;
}

std::uint64_t process_peak_memory() {

  std::ifstream in("/proc/self/status");
  if (in.is_open()) {
    std::string key;
    std::uint64_t kb;
    while (in >> key) {
      if (key == "VmHWM:") {
        in >> kb;
        return kb * 1024;
      }
    }
  }

  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) == 0)
    return (std::uint64_t)ru.ru_maxrss * 1024;
  return 0;
}
