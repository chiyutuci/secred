#include "utils.h"

#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <algorithm>
#include <charconv>
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
  // localtime() and asctime() return pointers into process-wide static buffers,
  // and glibc's localtime() also rewrites the shared timezone state; every sector
  // worker calls this, so concurrent calls were a data race.  The _r forms write
  // into caller-owned storage, and asctime_r keeps the exact format asctime used.
  std::tm local {};
  ::localtime_r(&currentTime, &local);
  char stamp[32];
  ::asctime_r(&local, stamp);
  out << msg << stamp << std::endl;
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

namespace {

std::string trim_copy(const std::string &text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

std::optional<std::uint64_t> parse_byte_count(const std::string &text) {
  const std::string trimmed = trim_copy(text);
  if (trimmed.empty())
    return std::nullopt;
  std::uint64_t value = 0;
  const char *begin = trimmed.data();
  const char *end = begin + trimmed.size();
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc() || result.ptr != end)
    return std::nullopt;
  return value;
}

std::optional<std::string> read_text_file(const std::filesystem::path &path) {
  std::ifstream in(path);
  if (!in.is_open())
    return std::nullopt;
  std::string text;
  std::getline(in, text);
  return text;
}

bool has_memory_controller(const std::string &controllers) {
  std::size_t begin = 0;
  while (begin <= controllers.size()) {
    const std::size_t end = controllers.find(',', begin);
    const std::string token =
        controllers.substr(begin, end == std::string::npos ? std::string::npos
                                                           : end - begin);
    if (token == "memory")
      return true;
    if (end == std::string::npos)
      break;
    begin = end + 1;
  }
  return false;
}

// Read a limit/usage pair from `base`; the first file that yields a limit
// decides.  An unlimited limit leaves `limited` false.
void read_limit_pair(CgroupMemory &cg, const std::filesystem::path &limit_file,
                     const std::filesystem::path &usage_file) {
  const auto text = read_text_file(limit_file);
  if (!text)
    return;
  const auto limit = parse_cgroup_memory_limit(*text);
  if (!limit)
    return;
  cg.limited = true;
  cg.limit = *limit;
  if (const auto usage = read_text_file(usage_file))
    cg.current = parse_byte_count(*usage).value_or(0);
}

} // namespace

std::optional<std::uint64_t> parse_cgroup_memory_limit(
    const std::string &text) {
  const std::string trimmed = trim_copy(text);
  if (trimmed == "max")
    return std::nullopt;
  const auto value = parse_byte_count(trimmed);
  if (!value)
    return std::nullopt;
  // cgroup v1 reports PAGE_COUNTER_MAX * PAGE_SIZE, near 2^63, instead of a
  // separate "unlimited" word.  Nothing today consumes an exbibyte; treat
  // values that large as the sentinel they are.
  constexpr std::uint64_t kV1Unlimited = 1ULL << 60;
  if (*value >= kV1Unlimited)
    return std::nullopt;
  return value;
}

CgroupMemory cgroup_memory(const std::filesystem::path &root,
                           const std::filesystem::path &cgroup_file) {
  CgroupMemory cg;

  std::string v2_path;
  std::string v1_path;
  bool have_v2 = false;
  bool have_v1 = false;
  if (std::ifstream groups(cgroup_file); groups.is_open()) {
    std::string line;
    while (std::getline(groups, line)) {
      const auto first = line.find(':');
      if (first == std::string::npos)
        continue;
      const auto second = line.find(':', first + 1);
      if (second == std::string::npos)
        continue;
      const std::string hierarchy = line.substr(0, first);
      const std::string controllers = line.substr(first + 1, second - first - 1);
      const std::string path = line.substr(second + 1);
      if (!have_v2 && hierarchy == "0" && controllers.empty()) {
        v2_path = path;
        have_v2 = true;
      } else if (!have_v1 && has_memory_controller(controllers)) {
        v1_path = path;
        have_v1 = true;
      }
    }
  }

  const auto relative = [](const std::string &path) {
    std::filesystem::path p(path);
    return p.relative_path();
  };

  if (have_v2) {
    const std::filesystem::path base = root / relative(v2_path);
    read_limit_pair(cg, base / "memory.max", base / "memory.current");
    if (!cg.limited) {
      // A cgroup namespace can hide the leaf; the mount root's limit still
      // applies to us.
      read_limit_pair(cg, root / "memory.max", root / "memory.current");
    }
  } else if (have_v1) {
    const std::filesystem::path base =
        root / "memory" / relative(v1_path);
    read_limit_pair(cg, base / "memory.limit_in_bytes",
                    base / "memory.usage_in_bytes");
  } else {
    read_limit_pair(cg, root / "memory.max", root / "memory.current");
    if (!cg.limited)
      read_limit_pair(cg, root / "memory" / "memory.limit_in_bytes",
                      root / "memory" / "memory.usage_in_bytes");
  }
  return cg;
}

std::uint64_t default_memory_budget(std::uint64_t available,
                                    std::uint64_t total,
                                    const CgroupMemory &cgroup,
                                    std::uint64_t ceiling) {
  // No readable memory figures at all: keep the historical ceiling instead of
  // refusing every run on an unusual system.
  if (available == 0 && total == 0)
    return ceiling;

  std::uint64_t mem = available != 0 ? available : total;
  if (cgroup.limited) {
    const std::uint64_t left = cgroup.limit > cgroup.current
                                   ? cgroup.limit - cgroup.current
                                   : 0;
    mem = std::min(mem, left);
  }
  if (mem == 0)
    return 0;
  mem -= mem / 5; // 80%, rounded down
  return std::min(mem, ceiling);
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
