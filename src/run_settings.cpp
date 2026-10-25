#include "run_settings.h"

#include <cmath>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

// Every key the settings block understands, including the two classes of
// retired keys handled below.  Anything else is a typo or an option this build
// does not have, and running anyway would silently ignore what the input asked
// for.
bool known_settings_key(const std::string &key) {
  static const std::set<std::string> keys = {
      // live settings
      "cache_dir",         "sthreads",
      "gthreads",          "parallel_mode",
      "subsector_filter_n", "seed_level_offset",
      "probe_mode",        "truncate",
      "mandatory_list",    "sector_debug",
      "ordering_mode",     "max_memory",
      "memory_reserve",    "max_primes",
      "symmetry",          "resume",
      // removed with a replacement (the settings block refuses them)
      "global_tool",       "threads",
      "global_finiteflow_threads",
      // removed without a replacement (old inputs keep running)
      "boundary_mode",     "equation_selection",
      "sector_elimination", "global_otf",
      "subsector_depth_offset"};
  return keys.contains(key);
}

std::uint64_t gib_to_bytes(double gib, const std::string &key,
                           bool allow_zero) {
  constexpr double kMaxGiB = 1e9; // keeps the byte count far from overflowing
  if (!std::isfinite(gib) || gib < 0 || (!allow_zero && gib == 0) ||
      gib > kMaxGiB)
    throw std::runtime_error(
        "settings." + key + " must be a " +
        (allow_zero ? std::string("non-negative") : std::string("positive")) +
        " number of GiB, at most 1e9");
  const std::uint64_t bytes = (std::uint64_t)(gib * 1024.0 * 1024.0 * 1024.0);
  if (!allow_zero && bytes == 0)
    throw std::runtime_error("settings." + key +
                             " is smaller than one byte");
  return bytes;
}

} // namespace

namespace secred {

std::string gib_text(std::uint64_t bytes) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2)
      << (double)bytes / (1024.0 * 1024.0 * 1024.0) << " GiB";
  return out.str();
}

} // namespace secred

void RunSettings::parse(const YAML::Node &s, bool allow_unknown_settings) {
  if (!s)
    return;
  if (!s.IsMap())
    throw std::runtime_error("settings must be a mapping");

  for (auto it = s.begin(); it != s.end(); ++it) {
    const std::string key = it->first.as<std::string>();
    if (known_settings_key(key))
      continue;
    if (allow_unknown_settings) {
      std::cerr << "warning: ignoring unknown setting settings." << key
                << "\n";
      continue;
    }
    throw std::runtime_error("unknown setting settings." + key +
                             "; see the README for the supported settings");
  }

  for (const auto &[key, replacement] :
       std::initializer_list<std::pair<const char *, const char *>>{
           {"global_tool",
            "the global tool is fixed: a symbolic system (one or more symbols) "
            "selects FireFly, and a numeric system selects the FiniteFlow numeric "
            "path"},
           {"threads", "use settings.sthreads"},
           {"global_finiteflow_threads", "use settings.gthreads"}}) {
    if (s[key])
      throw std::runtime_error("settings." + std::string(key) +
                               " was removed; " + replacement);
  }
  if (s["cache_dir"])       _cache_dir       = s["cache_dir"].as<std::string>();
  if (s["sthreads"])        _sthreads        = s["sthreads"].as<unsigned>();

  _gthreads = s["gthreads"] ? s["gthreads"].as<unsigned>() : _sthreads;
  const auto check_threads = [](unsigned value, const char *key) {
    if (value < 1 || value > secred::kMaxConfiguredThreads)
      throw std::runtime_error(
          "settings." + std::string(key) + " must be between 1 and " +
          std::to_string(secred::kMaxConfiguredThreads) + ", got " +
          std::to_string(value));
  };
  check_threads(_sthreads, "sthreads");
  check_threads(_gthreads, "gthreads");
  if (s["parallel_mode"]) {
    _parallel_mode = s["parallel_mode"].as<std::string>();
    if (_parallel_mode != "layer" && _parallel_mode != "fine")
      throw std::runtime_error(
          "parallel_mode must be 'layer' or 'fine', got '" + _parallel_mode + "'");
  }

  if (s["subsector_filter_n"])
    _subsector_filter_n = s["subsector_filter_n"].as<unsigned>();

  if (s["seed_level_offset"]) {
    _seed_level_offset = s["seed_level_offset"].as<unsigned>();
    _seed_level_offset_set = true;
  }

  if (s["probe_mode"]) {
    const std::string pm = s["probe_mode"].as<std::string>();
    if      (pm == "targets") _probe_mode = ProbeMode::Targets;
    else if (pm == "all")     _probe_mode = ProbeMode::All;
    else
      throw std::runtime_error(
          "probe_mode must be 'targets' or 'all', got '" + pm + "'");
    _probe_mode_set = true;
  }

  if (s["truncate"]) {
    _truncate = s["truncate"].as<bool>();
    _truncate_set = true;
  }

  // With no `truncate` key, naming a probe option opts the run into the seed
  // truncation, whichever value it takes; with neither key the truncation is
  // off.  An explicit `truncate: false` force-overrides the probe: the walk is
  // unscoped (`all`) and the mandatory rows default off -- the
  // pre-2026-10-17 behaviour, whatever `probe_mode` says.  While the rule is
  // on, the probe scope defaults to `targets` and the mandatory default
  // follows it.
  if (!_truncate_set)
    _truncate = _probe_mode_set;

  if (!_truncate) {
    _probe_mode = ProbeMode::All;
    _mandatory_list = false;
  } else {
    if (!_probe_mode_set)
      _probe_mode = ProbeMode::Targets;
    _mandatory_list = _probe_mode == ProbeMode::Targets;
  }
  if (s["mandatory_list"]) _mandatory_list = s["mandatory_list"].as<bool>();

  if (s["sector_debug"]) _sector_debug = s["sector_debug"].as<bool>();

  if (s["ordering_mode"]) {
    std::string om = s["ordering_mode"].as<std::string>();
    if      (om == "normal")  _ordering_mode = OrderingMode::Normal;
    else if (om == "balance") _ordering_mode = OrderingMode::Balance;
    else
      throw std::runtime_error(
          "ordering_mode must be 'normal' or 'balance', got '" + om + "'");
  }

  if (s["max_memory"]) {
    _max_memory = gib_to_bytes(s["max_memory"].as<double>(), "max_memory",
                               false);
    _max_memory_set = true;
  }

  if (s["memory_reserve"]) {
    _memory_reserve = gib_to_bytes(s["memory_reserve"].as<double>(),
                                   "memory_reserve", true);
    _memory_reserve_set = true;
  }

  if (s["max_primes"]) {
    const unsigned long long mp = s["max_primes"].as<unsigned long long>();
    if (mp < 1 || mp > std::numeric_limits<unsigned>::max())
      throw std::runtime_error("settings.max_primes must be a positive integer");
    _max_primes = static_cast<unsigned>(mp);
  }

  if (s["symmetry"]) _symmetry = s["symmetry"].as<bool>();

  if (s["resume"]) {
    if (!s["resume"].IsScalar())
      throw std::runtime_error(
          "settings.resume must be false, true or auto");
    const std::string text = s["resume"].Scalar();
    if (text == "true" || text == "auto" || text == "yes" || text == "on")
      _resume = true;
    else if (text == "false" || text == "no" || text == "off" || text == "0")
      _resume = false;
    else
      throw std::runtime_error(
          "settings.resume must be false, true or auto, got '" + text + "'");
  }
}

void RunSettings::parse_cut(const YAML::Node &cut, unsigned nprops) {
  if (cut) {
    _cut = cut.as<std::vector<unsigned>>();
    if (_cut.size() != nprops)
      throw std::runtime_error(
          "family.cut must have one entry per propagator (" +
          std::to_string(nprops) + "), got " + std::to_string(_cut.size()));
    for (unsigned value : _cut)
      if (value > 1)
        throw std::runtime_error("family.cut entries must be 0 or 1");
  }
  _ncut = std::accumulate(_cut.begin(), _cut.end(), 0);
}

void RunSettings::resolve_memory() {
  // The default memory budget follows the machine.  An explicit max_memory can
  // lower it but not raise it past what the machine and its cgroup can give.
  const SystemMemory machine = system_memory();
  _machine_total = machine.total;
  _machine_available = machine.available;
  _cgroup = cgroup_memory();
  _machine_budget = default_memory_budget(
      machine.available, machine.total, _cgroup, kDefaultMemoryCeiling);

  if (!_max_memory_set) {
    if (_machine_budget == 0)
      throw std::runtime_error(
          "no memory is available: the cgroup has reached its memory limit");
    _max_memory = _machine_budget;
  } else if (_machine_budget != 0 && _max_memory > _machine_budget) {
    std::cout << "\n #0.0   settings.max_memory " << secred::gib_text(_max_memory)
              << " exceeds the machine budget " << secred::gib_text(_machine_budget)
              << "; using the budget.\n";
    _max_memory = _machine_budget;
  }

  if (_memory_reserve_set) {
    if (_memory_reserve >= _max_memory)
      throw std::runtime_error(
          "settings.memory_reserve (" + secred::gib_text(_memory_reserve) +
          ") must be smaller than settings.max_memory (" +
          secred::gib_text(_max_memory) + ")");
  } else if (_memory_reserve >= _max_memory) {
    const std::uint64_t adapted = _max_memory / 4;
    std::cout << "\n #0.0   settings.memory_reserve default "
              << secred::gib_text(_memory_reserve) << " does not fit the "
              << secred::gib_text(_max_memory) << " memory budget; using "
              << secred::gib_text(adapted) << ".\n";
    _memory_reserve = adapted;
  }
}
