#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"

#include "sector.h"
#include "thread_count.h"
#include "utils.h"

// The typed, validated settings of one run.  Parsing and the machine-aware
// memory defaults live here, separate from the family model they configure.
// Field names keep the spelling they had inside Family so the split stays a
// move; RunContext passes the settings to the stages as a const reference.
struct RunSettings {
  unsigned _sthreads = secred::default_thread_count();
  std::string _parallel_mode = "fine";

  std::string _cache_dir = "cache";

  unsigned _subsector_filter_n = 1;

  unsigned _seed_level_offset = kSeedLevelOffset;
  bool _seed_level_offset_set = false;

  ProbeMode _probe_mode = ProbeMode::Targets;
  bool _probe_mode_set = false;

  // Seed truncation (Kira's truncate_sp): off by default, on when a probe option
  // is named.  An explicit `truncate: false` forces the pre-2026-10-17 walk --
  // unscoped probe and no mandatory rows -- overriding `probe_mode`.
  bool _truncate = false;
  bool _truncate_set = false;

  bool _resume = false;
  std::uint64_t _input_hash = 0;

  bool _mandatory_list = true;

  bool _sector_debug = false;

  bool _symmetry = true;

  OrderingMode _ordering_mode = OrderingMode::Balance;

  std::uint64_t _memory_reserve = 30ULL * 1024 * 1024 * 1024;

  // The default budget comes from the machine; an explicit setting can only
  // lower it.
  std::uint64_t _max_memory = 512ULL * 1024 * 1024 * 1024;

  std::vector<unsigned> _cut;
  unsigned _ncut = 0;

  unsigned _gthreads = secred::default_thread_count();

  unsigned _max_primes = 500;

  bool _memory_reserve_set = false;
  bool _max_memory_set = false;
  std::uint64_t _machine_budget = 0;
  std::uint64_t _machine_available = 0;
  std::uint64_t _machine_total = 0;
  CgroupMemory _cgroup;
  bool _dry_run = false;

  // Read the settings block.  Unknown keys are refused unless the caller
  // allowed them; keys removed with a replacement always fail.
  void parse(const YAML::Node &settings, bool allow_unknown_settings);

  // Validate the family's cut mask against the model's propagator count.
  void parse_cut(const YAML::Node &cut, unsigned nprops);

  // Size the default budget from the machine and its cgroup, and reconcile an
  // explicit max_memory / memory_reserve with it.
  void resolve_memory();
};

namespace secred {

std::string gib_text(std::uint64_t bytes);

inline std::uint64_t saturating_mul(std::uint64_t a, std::uint64_t b) {
  if (a == 0 || b == 0)
    return 0;
  const std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  return a > max / b ? max : a * b;
}

inline std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) {
  const std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  return b > max - a ? max : a + b;
}

} // namespace secred
