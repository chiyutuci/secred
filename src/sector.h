#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "arith/ffint.h"
#include "arith/numeric.h"
#include "arith/symbolic.h"
#include "equation.h"
#include "utils.h"

#include "fflow/graph.hh"
#include "ginac/ginac.h"
#include "yaml-cpp/yaml.h"

enum class OrderingMode { Normal, Balance };

struct SectorConfig {
  std::string cache_dir      = "cache";
  unsigned subsector_depth_offset = 1;
  OrderingMode ordering_mode      = OrderingMode::Balance;
};

class Reduce;

class RawIntegral {
public:
  // Indices are the powers of each propagator (denominator > 0, numerator
  // < 0); in practice they stay within ±10, so one byte each suffices.
  // MAX_PROPS caps the number of propagators (the largest example has 14).
  static constexpr unsigned MAX_PROPS = 32;

  RawIntegral() : nprops_(0) {}

  explicit RawIntegral(unsigned n) : nprops_(n) {}

  explicit RawIntegral(unsigned n, int i) : nprops_(n) {
    for (unsigned k = 0; k < n; ++k)
      indices_[k] = (int8_t)i;
  }

  explicit RawIntegral(const std::vector<bool> &lines) : nprops_(lines.size()) {
    for (unsigned i = 0; i < lines.size(); ++i)
      indices_[i] = lines[i] ? 1 : 0;
  }

  explicit RawIntegral(const std::vector<int> &v) : nprops_(v.size()) {
    for (unsigned i = 0; i < v.size(); ++i)
      indices_[i] = (int8_t)v[i];
  }

  explicit RawIntegral(std::vector<int> &&v) : nprops_(v.size()) {
    for (unsigned i = 0; i < v.size(); ++i)
      indices_[i] = (int8_t)v[i];
  }

  int operator[](unsigned i) const { return indices_[i]; }

  int8_t &operator[](unsigned i) { return indices_[i]; }

  // length of propagators
  [[nodiscard]] unsigned size() const { return nprops_; }

  // sum of positive indices
  [[nodiscard]] unsigned depth() const {
    unsigned s = 0;
    for (unsigned i = 0; i < nprops_; ++i)
      if (indices_[i] > 0)
        s += indices_[i];
    return s;
  }

  // sum of negative indices
  [[nodiscard]] unsigned rank() const {
    unsigned s = 0;
    for (unsigned i = 0; i < nprops_; ++i)
      if (indices_[i] < 0)
        s -= indices_[i];
    return s;
  }

  // max negative indices
  [[nodiscard]] int maxneg() const {
    int res = -1;
    for (unsigned i = 0; i < nprops_; ++i)
      if (indices_[i] < res)
        res = indices_[i];
    return res;
  }

  // sector of the integral
  [[nodiscard]] unsigned sector() const {
    unsigned sector = 0;
    for (unsigned i = 0; i < nprops_; ++i)
      if (indices_[i] > 0)
        sector |= 1 << i;
    return sector;
  }

  bool operator==(const RawIntegral &other) const {
    if (nprops_ != other.nprops_)
      return false;
    for (unsigned i = 0; i < nprops_; ++i)
      if (indices_[i] != other.indices_[i])
        return false;
    return true;
  }

  bool operator!=(const RawIntegral &other) const {
    return !(*this == other);
  }

  friend bool operator<(const RawIntegral &lhs, const RawIntegral &rhs) {
    unsigned n = std::min(lhs.nprops_, rhs.nprops_);
    for (unsigned i = 0; i < n; ++i)
      if (lhs.indices_[i] != rhs.indices_[i])
        return lhs.indices_[i] < rhs.indices_[i];
    return lhs.nprops_ < rhs.nprops_;
  }

  friend RawIntegral operator+(const RawIntegral &lhs, const RawIntegral &rhs) {
    RawIntegral integral(lhs.size());
    for (unsigned i = 0; i < integral.size(); ++i)
      integral[i] = lhs[i] + rhs[i];
    return integral;
  }

  friend RawIntegral operator-(const RawIntegral &lhs, const RawIntegral &rhs) {
    RawIntegral integral(lhs.size());
    for (unsigned i = 0; i < integral.size(); ++i)
      integral[i] = lhs[i] - rhs[i];
    return integral;
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const RawIntegral &integral) {
    os << "[";
    for (unsigned i = 0; i < integral.nprops_; ++i) {
      if (i)
        os << ", ";
      os << (int)integral.indices_[i];
    }
    os << "]";
    return os;
  }

  // Parse the operator<< text form "[a, b, c]" back into an integral.
  static RawIntegral from_text(const std::string &s) {
    std::vector<int> v;
    size_t i = 0;
    while (i < s.size() && (s[i] == '[' || s[i] == ' ')) ++i;
    while (i < s.size() && s[i] != ']') {
      size_t start = i;
      while (i < s.size() && s[i] != ',' && s[i] != ']') ++i;
      std::string tok = s.substr(start, i - start);
      size_t b = tok.find_first_not_of(' ');
      size_t e = tok.find_last_not_of(' ');
      v.push_back(std::stoi(tok.substr(b, e - b + 1)));
      while (i < s.size() && (s[i] == ',' || s[i] == ' ')) ++i;
    }
    return RawIntegral(std::move(v));
  }

  friend class std::hash<RawIntegral>;

private:
  std::array<int8_t, MAX_PROPS> indices_{};
  uint8_t nprops_ = 0;
};

namespace std {
template <> struct hash<RawIntegral> {
  std::size_t operator()(const RawIntegral &integral) const {
    std::size_t hash = integral.size();
    for (unsigned i = 0; i < integral.size(); ++i)
      hash ^= integral[i] + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};
} // namespace std

namespace YAML {
template <> struct convert<RawIntegral> {
  static bool decode(const Node &node, RawIntegral &integral) {
    if (!node.IsSequence())
      return false;
    integral = RawIntegral(node.as<std::vector<int>>());
    return true;
  }
};
} // namespace YAML

// ibp relation
// first:  integral indices
// second: coefficient
typedef std::vector<std::pair<RawIntegral, GiNaC::ex>> IBPProto;
// ibp relation over finite field
// first:  integral indices
// second: coefficients of indices
typedef std::vector<std::pair<RawIntegral, std::vector<rational>>> IBPProtoFF;
// ibp relation over finite field, coefficients as finite-field residues
// (ffint) — the compact form fed to FiniteFlow's sparse solver.  _ibpFF keeps
// the full rational coefficients for the global-equation emission path.
typedef std::vector<std::pair<RawIntegral, std::vector<ffint>>> IBPProtoFFmod;
typedef std::vector<std::pair<RawIntegral, std::vector<GiNaC::ex>>> IBPProtoM;

// One term in a global equation: the integral and its coefficient as a string.
// Coefficient strings come from rational::str() (ff_n path) or GiNaC stream
// output (ff_m path) and are written verbatim into the kira system file.
using GlobalEquation = std::vector<std::pair<RawIntegral, std::string>>;

// Returns the index of the equation in the raw spill file whose leading term
// (largest integral ID) is smallest, i.e. the equation with the simplest
// leading term.  Used by _run_global_kira to pick the single equation into
// which the auxiliary 'dum' variable is injected so that kira accepts a
// symbol-free rational system.
unsigned
select_dum_eq_from_file(const std::string &path,
                        const std::unordered_map<RawIntegral, unsigned> &id_map);

// Wall-clock cost of one sector's adaptive two-pass seeding (ATPS).
// probeNanos is the sector's single probe pass, fullNanos the full pass that
// follows.  The memory gate (SectorRunGuard) is constructed inside each pass,
// so a headroom wait is counted in the pass that waited.  `expanded` records
// that the probe found the initial depth/rank too small, so they were raised
// once before the full pass.
struct SectorTiming {
  unsigned id = 0;
  bool expanded = false;
  std::uint64_t probeNanos = 0;
  std::uint64_t fullNanos = 0;
};

// One row of sector_timing.csv: "sector,expanded,probe_s,full_s".
std::string format_timing_row(const SectorTiming &t);

// Shared reduction state passed to each Sector during parallel execution.
// Owned by Reduce; Sector methods receive it by reference.
struct ReductionContext {
  std::vector<std::vector<RawIntegral>> allMasters; // [sector_index] → master integrals
  std::map<unsigned, std::set<RawIntegral>> targets;   // sector_id → required integrals
  std::map<unsigned, std::set<RawIntegral>> uptargets; // sector_id → integrals from higher layers
  std::mutex mastersMutex;
  std::mutex uptargetsMutex;
  unsigned totalSectors = 0;
  unsigned completedSectors = 0;
  std::mutex progressMutex;
  // Incremental disk spill of the selected equations from all sectors. They are
  // appended to a raw file (RawIntegral + coefficient text per line, blank line
  // between equations) instead of accumulating in RAM; only the distinct
  // integral set is kept resident, for global ID assignment in _run_global_kira.
  std::string rawEquationsPath;
  std::unique_ptr<std::ofstream> rawEqOut;        // append stream, guarded by globalEqMutex
  std::unordered_set<RawIntegral> distinctIntegrals; // resident distinct-integral set
  std::atomic<std::size_t> globalEquationsCount{0};  // equations spilled so far
  std::mutex globalEqMutex;

  // Append one equation to the spill file and record its integrals. Thread-safe
  // (guarded by globalEqMutex). `rawEqOut` must have been opened beforehand.
  void appendRawEquation(const GlobalEquation &eq) {
    std::lock_guard<std::mutex> lk(globalEqMutex);
    for (const auto &[integ, coeff] : eq) {
      distinctIntegrals.insert(integ);
      *rawEqOut << integ << '\t' << coeff << '\n';
    }
    *rawEqOut << '\n';
    globalEquationsCount.fetch_add(1, std::memory_order_relaxed);
  }

  // Per-sector seed-range narrowing statistics (updated atomically by worker threads).
  // narrowingSuccess: sectors where the initial depth/rank bounds (derived from targets)
  //   were sufficient — the probe pass returned 0.
  // narrowingFail: sectors where the probe pass reported the bounds too small and
  //   they were raised once before the full pass.
  std::atomic<unsigned> narrowingSuccess{0};
  std::atomic<unsigned> narrowingFail{0};

  // Total raw equations generated across all sectors in the full (second) pass.
  std::atomic<unsigned> rawEquationsTotal{0};

  // Sector-reduction (ATPS) wall-clock totals in nanoseconds.  probeNanos
  // accumulates every sector's probe pass, fullNanos the full passes; their
  // ratio is the probe pass's share of the reduction work.  The memory gate
  // lives inside each pass, so headroom waits are included in these figures.
  std::atomic<std::uint64_t> probeNanos{0};
  std::atomic<std::uint64_t> fullNanos{0};
  // Serializes appends to sector_timing.csv.
  std::mutex timingMutex;

  // Add one sector's ATPS timing to the two counters above.  Kept separate from
  // recordTiming's CSV append so it can be exercised without file I/O.
  void accumulateTiming(const SectorTiming &t) {
    probeNanos.fetch_add(t.probeNanos, std::memory_order_relaxed);
    fullNanos.fetch_add(t.fullNanos, std::memory_order_relaxed);
  }

  // Accumulate one sector's ATPS timing and append its row to
  // sector_timing.csv (whose header Family::run_reduce writes).
  void recordTiming(const SectorTiming &t);

  // Memory gate based on this process's actual memory usage (RSS) rather than a
  // per-sector estimate.  A sector is admitted only while the process stays
  // under `maxMemory - memoryReserve`: `maxMemory` is the allowed total (set it
  // smaller to simulate a machine with less RAM), `memoryReserve` is the
  // headroom kept free below that ceiling.
  std::uint64_t maxMemory = 512ULL * 1024 * 1024 * 1024; // allowed total (bytes)
  std::uint64_t memoryReserve = 60ULL * 1024 * 1024 * 1024; // headroom below ceiling
  std::mutex admitMutex;
  std::condition_variable admitCv;
  // Injectable probe returning this process's current memory usage (bytes), for
  // tests; defaults to reading /proc/self/status VmRSS.
  std::function<std::uint64_t()> memoryProbe = [] {
    return process_used_memory();
  };

  // Number of sectors currently executing the (memory-gated) reduction.
  std::atomic<unsigned> runningSectors{0};
  // Set when a sector is admitted while memory is tight (headroom in
  // (reserve, 2*reserve]).  It makes admission edge-triggered in that regime so
  // a just-freed chunk of memory is not claimed by a burst of waiters all
  // seeing the same snapshot.  `availAtAdmit` records the headroom at that
  // admission; the flag is re-armed once the admitted sector has consumed at
  // least `rearmDrop` bytes (avail falls that far below the snapshot) — not only
  // when the headroom is fully exhausted — so a small sector does not keep the
  // flag set and stall every later waiter.
  bool headroomTaken = false;
  std::uint64_t availAtAdmit = 0;
  // Minimum drop in avail below the admission snapshot (bytes) required to
  // re-arm the burst flag: a sector is considered to have "started consuming
  // memory" only once avail has fallen this far, so VmRSS noise does not re-arm
  // the flag prematurely.
  std::uint64_t rearmDrop = 10ULL * 1024 * 1024 * 1024; // 10 GB

  // Admit one sector, all decisions under admitMutex.  `avail` is the headroom
  // under the ceiling, avail = maxMemory - memoryProbe():
  //   - nothing running: the sole sector always proceeds (deadlock escape);
  //   - avail > 2*reserve: plenty of room — admit freely (full concurrency);
  //   - reserve < avail <= 2*reserve: tight — admit at most one per free event
  //     (headroomTaken), re-arming once the admitted sector's consumption shows
  //     up in avail, so the drip never stalls while the burst is still damped.
  void waitForMemoryHeadroom() {
    std::unique_lock<std::mutex> lk(admitMutex);
    for (;;) {
      std::uint64_t used = memoryProbe();
      std::uint64_t avail = used < maxMemory ? maxMemory - used : 0;
      unsigned running = runningSectors.load(std::memory_order_relaxed);

      if (running == 0) {
        runningSectors.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      if (avail > 2 * memoryReserve) {
        headroomTaken = false;  // burst limiter irrelevant with this much room
        runningSectors.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      // Re-arm once the just-admitted sector has consumed at least rearmDrop of
      // memory (avail fell that far below the admission snapshot), or once the
      // headroom is exhausted.
      if (avail <= memoryReserve ||
          (avail < availAtAdmit && availAtAdmit - avail >= rearmDrop))
        headroomTaken = false;
      if (avail > memoryReserve && !headroomTaken) {
        headroomTaken = true;
        availAtAdmit = avail;
        runningSectors.fetch_add(1, std::memory_order_relaxed);
        return;
      }

      admitCv.wait_for(lk, std::chrono::milliseconds(500));
    }
  }

  void releaseMemorySlot() {
    {
      std::lock_guard<std::mutex> lk(admitMutex);
      runningSectors.fetch_sub(1, std::memory_order_relaxed);
    }
    admitCv.notify_all();
  }

  ReductionContext() = default;
  ReductionContext(const ReductionContext &) = delete;
  ReductionContext &operator=(const ReductionContext &) = delete;
};

// Write a standard kira jobs YAML file for reduce_user_defined_system.
// otf = on-the-fly loading: when true, Kira reads the system incrementally
// instead of materialising it, which is useful when the system is very large
// at the cost of slower random access.
void createJobFile(const std::string &filename, bool otf = false);

class Sector {
public:
  friend class SectorTestHelper;

  static Sector make(unsigned index, unsigned id, unsigned nprops,
                     const std::vector<bool> &lines,
                     const std::set<unsigned> &trivials);

  unsigned id() const { return _id; }

  // Numeric FiniteFlow reduction. The structural elimination runs on the
  // finite-field `ibps_mod`; the selected independent equations are emitted
  // with the full rational coefficients from `ibps` (or the symbolic
  // `*ibps_sym` on the signed path).
  void run_reduce_ff_n(const std::vector<IBPProtoFF> &,
                       const std::vector<IBPProtoFFmod> &,
                       const std::vector<IBPProtoM> *ibps_sym, const SectorConfig &,
                       ReductionContext &);

  void clear();

private:
  unsigned sector_reduction_ff_n(const std::vector<IBPProtoFF> &,
                                 const std::vector<IBPProtoFFmod> &,
                                 const std::vector<IBPProtoM> *ibps_sym,
                                 const SectorConfig &, ReductionContext &,
                                 bool skip_subsectors);

  void _generate_seeds(OrderingMode mode = OrderingMode::Balance);
  void _generate_combinations(int number, int sum);
  static bool _compare_subintegral(const RawIntegral &, const RawIntegral &);
  static void _print_progress(ReductionContext &);

private:
  // sector index
  unsigned _index = 0;
  // sector number
  unsigned _id = 0;
  // number of propagators
  unsigned _nprops = 0;
  // maximum depth
  unsigned _depth = 0;
  // maximum rank
  unsigned _rank = 0;
  // lines
  std::vector<bool> _lines;

  // combinations[<number, sum>] -> [combinations]
  std::map<std::pair<int, int>, std::vector<std::vector<int>>> _combinations;
  // seeds: weight to integral
  std::vector<RawIntegral> _seeds;
  // self master integrls
  std::vector<RawIntegral> _masters;
  // subsector master intgerals
  std::vector<RawIntegral> _submasters;
  // seeds: integral to weight
  std::unordered_map<RawIntegral, unsigned> _weights;
  // trivial subsectors
  std::set<unsigned> _trivialsectors;
};
