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

// 分 sector 阶段的积分次序。seed 与子 sector 积分的列号都由它派生（列号越小越复杂）。
//   Normal     = 生成次序（depth 主、rank 次），不排序。
//   Balance    = 稳定排序：(超出本 sector 基础层的深度, rank, 活线最大幂)。
//   Complexity = 稳定排序：(depth+rank, depth, rank, 最大分母幂, 最大分子幂)。
//                Balance/Normal 只作用于 seed；Complexity 还作用于子 sector 积分
//                （本模式下的 5 键不是全序，故排序必须稳定，见 ARCHITECTURE.md §5.6）。
enum class OrderingMode { Normal, Balance, Complexity };

// 分 sector 阶段的消元引擎。
//   Fflow = 旧行为：FiniteFlow 的 SparseLinearSolver（learn + mark_and_sweep_eqs，
//           前向+反向整链闭包）。
//   Pyred = 与全局阶段同一套：src/pyred_solver 的引擎（先排序再消元）做 learn，
//           再用前向闭包 + 数值充分性修复做选取（src/global_select.h）。
// 只在分 sector 阶段生效，与全局阶段的 settings.equation_selection 正交。
enum class SectorElimination { Fflow, Pyred };

// 种子边界规则（settings.boundary_mode），语义见 SeedBound。
//   Box   = 旧行为：depth 与 rank 两个独立的边界。
//   Total = 两个边界仍作硬上限，另加一条总幂 depth+rank 的边界（初值取
//           depth+rank，即整只种子盒的总幂上端，所以硬上限放行的种子一个都
//           不会被它切掉；这条边界只决定探测的判据与子 sector 过滤）。
enum class BoundaryMode { Box, Total };

struct SectorConfig {
  std::string cache_dir      = "cache";
  unsigned subsector_filter_n = 1;
  OrderingMode ordering_mode      = OrderingMode::Balance;
  SectorElimination sector_elimination = SectorElimination::Fflow;
  BoundaryMode boundary_mode      = BoundaryMode::Box;
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

  // max positive index (largest denominator power); 1 if no index is positive,
  // mirroring the -1 sentinel of maxneg() so that -maxneg() (the largest
  // numerator power) also reads 1 for an integral with no negative index
  [[nodiscard]] int maxpos() const {
    int res = 1;
    for (unsigned i = 0; i < nprops_; ++i)
      if (indices_[i] > res)
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

// The "complexity" key (OrderingMode::Complexity), ascending in all five keys:
// total power depth+rank, then depth, then rank, then the largest denominator
// power, then the largest numerator power.  The rank key can never decide by
// itself (rank = total power - depth), but it is kept so the key reads exactly
// as specified.
//
// This is NOT a total order: distinct integrals tied on all five keys compare
// equal, so every caller must supply its own deterministic tie-break (a stable
// sort, or extra keys).  The sector stage (seeds, subsector columns) and the
// global ID numbering both use this single definition.
inline bool complexity_less(const RawIntegral &a, const RawIntegral &b) {
  unsigned ta = a.depth() + a.rank();
  unsigned tb = b.depth() + b.rank();
  if (ta != tb) return ta < tb;
  if (a.depth() != b.depth()) return a.depth() < b.depth();
  if (a.rank() != b.rank()) return a.rank() < b.rank();
  if (a.maxpos() != b.maxpos()) return a.maxpos() < b.maxpos();
  return -a.maxneg() < -b.maxneg();
}

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

// Wall-clock cost of one sector's adaptive seeding (ATPS).  Which pass is which
// depends on the engine.  Under `fflow` the sector runs a probe pass and then the
// full pass that follows: probeNanos is the probe, fullNanos that full pass.  Under
// `pyred` every round is a full pass (`skip_subsectors = false`): probeNanos
// accumulates the reporting rounds that were discarded — each charged the
// cache-directory removal that preceded it — and fullNanos is the one settling
// round that spilled, 0 for a sector dropped at the round cap.  The two engines'
// figures are therefore not comparable with each other, nor with this project's
// pre-stage-S runs.  The memory gate (SectorRunGuard) is constructed inside each
// pass, so a headroom wait is counted in the pass that waited.  `expanded` records
// that the initial depth/rank were found too small and the box was widened before
// the settling pass (`fflow`: at most once, before the single full pass; `pyred`:
// once per reporting round).
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
  //   were sufficient — the first pass settled without reporting (`fflow`: the probe
  //   pass returned 0; `pyred`: the settling round reported nothing).
  // narrowingFail: sectors that had to widen the bounds at least once before settling.
  std::atomic<unsigned> narrowingSuccess{0};
  std::atomic<unsigned> narrowingFail{0};

  // Total raw equations generated across all sectors in the passes that count: the
  // full (second) pass under `fflow`, every round — the discarded ones included —
  // under `pyred`.
  std::atomic<unsigned> rawEquationsTotal{0};

  // Sector-reduction (ATPS) wall-clock totals in nanoseconds: the sum of the
  // per-sector `SectorTiming` figures above.  probeNanos accumulates every sector's
  // probe pass (under `pyred`: the discarded reporting rounds), fullNanos the
  // settling passes; their ratio is the discarded rounds' share of the reduction
  // work — under `pyred` that is the share of the work the round loop throws away,
  // not what `fflow`'s cheaper seed-only probe costs, so the figure is not
  // comparable across engines.  The memory gate lives inside each pass, so headroom
  // waits are included in these figures.
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

// The seed boundary a sector is reduced with — the region of seeds that may
// generate equations, and the quantity the subsector filter and the probe pass
// act on.
//
//   Box   (default, the original scheme): two independent bounds.  A seed is
//         admitted iff depth() < depth && rank() < rank, and both the filter and
//         the probe test the depth alone against the depth bound.
//   Total: the same two bounds stay as hard caps — so a seed such as
//         (depth, rank) = (3, 6) can never enter — and a third cap on the total
//         power depth() + rank() is added on top.  The cap starts at the top of
//         the seed box's own total-power range: a generated seed has
//         depth() in [popcount(id), depth] and rank() in [0, rank], so the most
//         extreme one it can build has total power depth + rank, and that is the
//         initial value of `limit`.  The comparison is strict, so at that value
//         the cap cuts nothing at all: with depth() <= depth and
//         rank() <= rank the only seed reaching total == depth + rank is the box
//         corner (depth, rank) itself, which the hard caps already reject — the
//         Total-mode admission set is exactly the Box-mode one.  What the cap
//         still decides is the shape of the seed box's top corner (seed_top())
//         and the total-power half of the probe's narrowing-failure criterion,
//         which then measures depth + rank instead of the depth.  The filter reads
//         the seed box too — but from its top, one step inside the cap, so its
//         offset n means the same thing in both modes.
//
//         Admission is the cap's only remaining job, and even that is nominal at
//         the cap initial() builds: the probe does not read the cap, it reads the
//         seed box (see `exceeded()`), and the repair (`widen_depth()`,
//         `widen_rank()`) grows the box and carries the cap along at its top, so a
//         sector whose requests reach past the box is repaired the same way in
//         both modes — by one more layer of seeds.  The cap earns its keep on a
//         boundary built by hand with the cap biting into the box (the corner-cut
//         and +1 generations of bench/boundary), where it is the box's shape that
//         decides admission — and on the deep, low-rank seed the total-power half
//         of the criterion measures past its depth cap but not past the box top,
//         which `exceeded()` reports separately as `past_depth_cap()`.
//
// `quantity()` names the figure the mode counts in: depth in Box, depth + rank in
// Total.  `boundary()` is the boundary itself in that figure — the depth bound in
// Box, the cap in Total — and `admits()` is the only member that reads it;
// `limit` is meaningful only in Total mode.  The probe and the subsector filter
// instead measure against `seed_top()`, the top of the seed box, so that their
// criteria mean the same thing in both modes and do not shift when the cap moves.
struct SeedBound {
  BoundaryMode mode = BoundaryMode::Box;
  // Box: admitted iff depth() < depth && rank() < rank.  Both modes keep this.
  unsigned depth = 0;
  unsigned rank = 0;
  // Total mode only: admitted iff depth() + rank() < limit.  Unused in Box mode,
  // where the boundary is `depth` -- read it through boundary(), never directly.
  // The strict comparison is why initial() starts the cap at the top of the seed
  // box's total-power range rather than at the corner the hard caps allow.
  unsigned limit = 0;

  // The boundary a sector starts from, given the box initial_seed_bounds
  // derived.  In Total mode the cap starts at the largest total power a generated
  // seed can have (depth + rank), so it cuts no seed the hard caps admit (see
  // above).
  static SeedBound initial(BoundaryMode m, unsigned depth, unsigned rank) {
    SeedBound b;
    b.mode = m;
    b.depth = depth;
    b.rank = rank;
    if (m == BoundaryMode::Total)
      b.limit = depth + rank;
    return b;
  }

  // The quantity this mode counts in: depth, or depth + rank in Total mode.
  // The probe and the subsector filter compare it against seed_top(); admission
  // compares it against boundary() (Total only).
  unsigned quantity(const RawIntegral &i) const {
    return mode == BoundaryMode::Total ? i.depth() + i.rank() : i.depth();
  }

  // The boundary admission tests, in that quantity: the depth bound in Box mode,
  // the cap in Total.  A widening moves this -- in Total mode it moves the box
  // and carries the cap along at its top (see widen_depth).
  unsigned boundary() const {
    return mode == BoundaryMode::Total ? limit : depth;
  }

  // The largest quantity a seed the boundary admits can reach: the top of the
  // seed box, one step inside the boundary in each hard figure -- depth - 1 and
  // rank - 1 -- and, in Total mode, no higher than the cap allows (the box corner
  // depth + rank - 2, lowered to limit - 1 once the cap bites into the box).
  // Signed: a degenerate hand-built box (depth 0) reports -1 rather than
  // wrapping, and the comparison below is done in int anyway.  This is the
  // reference the subsector filter counts its layers from -- what the seeds
  // reach, not where the boundary sits -- and the triple that
  // format_seed_box_line prints.
  int seed_top() const {
    if (mode == BoundaryMode::Total)
      return std::min((int)depth + (int)rank - 2, (int)limit - 1);
    return (int)depth - 1;
  }

  // May this seed generate equations (a row of the sector's system)?
  bool admits(const RawIntegral &i) const {
    if (i.depth() >= depth || i.rank() >= rank)
      return false;
    return mode != BoundaryMode::Total || quantity(i) < boundary();
  }

  // The box-top half of the probe criterion: is this needed seed's quantity (its
  // depth in Box mode, its total power depth + rank in Total) past the top of the
  // seed box?  Measuring against the seed box rather than against the boundary is
  // what makes this half mean the same thing in both modes: in Box mode it is the
  // historical test (`depth() >= depth`, i.e. `depth() > seed_top()`, exactly this
  // expression), while the pre-fix Total-mode criterion compared against the cap,
  // which sits at or above the box top and so never fired on the refusals that
  // matter -- the ones the hard caps make.
  //
  // This is also the half that names the *box top* as what has to move: the repair
  // it asks for is a widening that carries the Total cap along at the top of the
  // new box (see widen_depth's carry_cap), so `exceeded()` reports it separately.
  bool past_top(const RawIntegral &i) const {
    return (int)quantity(i) > seed_top();
  }

  // The depth-cap half: does the depth hard cap refuse this seed on its own?  Only
  // Total mode can report this while past_top() stays silent, and that is the gap
  // this half closes.  The other half measures depth + rank, so a large rank figure
  // -- which a deep target with many numerator powers asks for -- lifts seed_top()
  // well above the depth cap, and a deep, low-rank seed slips under it: on F234's
  // sector 246 the box (7, 5) has seed_top() 10, and the seed [0,1,2,0,1,1,1,1,0]
  // (depth 7, rank 0, total 7) is refused by the depth cap while its total power
  // sits three layers below the top.  admits() refuses it, so it generates no
  // equation and survives as a master -- and the pre-fix criterion never reported
  // it, where Box mode (whose quantity *is* the depth) repaired it by widening.
  // In Box mode the two halves coincide -- quantity is the depth and seed_top() is
  // depth - 1 -- so this clause adds nothing there.
  bool past_depth_cap(const RawIntegral &i) const { return i.depth() >= depth; }

  // Did this needed seed land past the top of the seed box, or reach the depth hard
  // cap under it?  That disjunction is the probe pass's narrowing-failure criterion
  // (see ARCHITECTURE.md §5.6): such a seed is one the boundary refuses to admit,
  // so it generates no equation, the reduction cannot close on it, and it survives
  // as an unreduced master.  Either half is one widening away from admission, which
  // is why both are reported.
  //
  // The criterion is still one-sided on the rank axis: a seed the rank cap refuses
  // whose total power is inside the box (a wide, shallow integral) is not reported,
  // and widening the box would not admit it either -- those are the "outer" seeds
  // the sector legitimately hands on as masters.
  bool exceeded(const RawIntegral &i) const {
    return past_top(i) || past_depth_cap(i);
  }

  // Which figure of the seed box has to grow for a seed exceeded() reports to be
  // admitted again?  Bit 1: the depth figure -- the depth hard cap itself, or in
  // Total mode the cap holding the box's top corner down (widening the depth
  // moves both, see widen_depth).  Bit 2: the rank hard cap.  Box mode keeps the
  // historical repair, which grows the depth bound and leaves the rank bound
  // alone, even for a seed that also sits on the rank bound.
  //
  // This names a figure, not the repair's cap policy: the caller pairs it with
  // past_top() to decide whether a depth widening also carries the Total cap along
  // (see widen_depth's carry_cap, and the bit-4 note in sector.cpp).
  unsigned refusal(const RawIntegral &i) const {
    if (mode != BoundaryMode::Total)
      return 1;
    unsigned bits = 0;
    if (i.depth() >= depth)
      bits |= 1;
    if (i.rank() >= rank)
      bits |= 2;
    return bits == 0 ? 1u : bits;
  }

  // The subsector filter, applied at both sites in Sector::sector_reduction_ff_n
  // to integrals that are not seeds: keep the integral when its quantity lies
  // within n layers of the seed box's top, i.e. drop it when it is more than n
  // below seed_top().  The reference is the seed box, so n means the same thing
  // in both modes and does not shift when the Total cap moves: a subsector
  // integral comes from a seed plus one IBP shift, and none of the shifts the
  // generator produces lands past the top of the seed box on the benchmark
  // families (measured: n = 0 leaves every sector with no subsector integral at
  // all), so n = 0 drops all of them, n = 1 keeps the top layer alone, n = 2 the
  // two top layers, and so on.  The rule is one-sided (it has no upper end), so
  // a subsector integral past the top layer would be kept rather than dropped.
  // Box mode is the historical rule `depth < depthBound - n`; Total
  // mode reads the same rule in total power, measured from the box corner rather
  // than from the cap.  The subtraction is done in int: seed_top() + 1 < n must
  // not wrap around.
  bool filter_drops(const RawIntegral &i, unsigned n) const {
    return (int)quantity(i) < seed_top() + 1 - (int)n;
  }

  // The probe found the box too small for a seed the sector still needs: raise it
  // by one layer, in the figure SeedBound::refusal named -- widen_depth for bit 1,
  // widen_rank for bit 2.  Both grow a hard bound, so neither can admit a seed the
  // other hard bound rejects, and both admit the seed the probe reported: the
  // probe's next pass therefore either finds nothing more to repair or reports a
  // seed the new layer brought in.  In Box mode widen_rank never fires (refusal
  // returns bit 1 alone there, the historical repair) and raises no figure
  // seed_top() reads, so its loop always terminates on the first pass that reports
  // a depth violation.
  //
  // In Total mode the cap rides at the top of the box: initial() builds it there
  // (limit == depth + rank) and these keep it there, so an admission rule the cap
  // would otherwise cut is never cut -- and seed_top() rises by exactly one layer,
  // since the box's top corner moves by one and the cap with it.  A hand-built
  // boundary whose cap bit into the box (the corner-cut generation) is restored to
  // the box top instead, i.e. its top rises further in one step; std::max keeps a
  // cap already above the box top from being dragged down.
  //
  // carry_cap == false deepens the box *without* moving the cap, which is the
  // repair for a report the depth hard cap made on its own (past_depth_cap() with
  // past_top() silent -- the caller's bit 4).  Nothing about the total power was
  // wrong there, so there is nothing for the cap to follow: the box grows one layer
  // deeper and keeps its total-power ceiling, admitting exactly the deep, low-rank
  // seeds the cap let slip past the criterion, and the layer the cap would have
  // added on the corner stays cut.  seed_top() still rises by one layer -- the
  // largest admitted total power rises from depth + rank - 2 to depth + rank - 1 --
  // so the subsector filter's reference moves with it either way.
  void widen_depth(bool carry_cap = true) {
    ++depth;
    if (mode == BoundaryMode::Total && carry_cap)
      limit = std::max(limit, depth + rank);
  }

  void widen_rank() {
    ++rank;
    if (mode == BoundaryMode::Total)
      limit = std::max(limit, depth + rank);
  }

  // Apply one pass's report to the box: the widening policy both engines share, so
  // that the driver's loops read the bits in exactly one place.  Bit 1 deepens the
  // box -- carrying the Total cap along only when the box top was what was reported
  // (bit 4), because a report the depth hard cap made on its own has nothing wrong
  // with its total power (see widen_depth's carry_cap) -- and bit 2 widens the rank.
  // A report always pins at least one of the two hard figures, so a lone bit 4 (the
  // hint without the figure it belongs to) widens nothing.
  void widen_for(unsigned refusalBits) {
    if (refusalBits & 1)
      widen_depth(/*carry_cap=*/(refusalBits & 4) != 0);
    if (refusalBits & 2)
      widen_rank();
  }
};

// How many passes a sector may spend settling its seed box before the driver gives
// up on it with a warning; one cap for both engines (the fflow probe loop and the
// pyred round loop).
inline constexpr unsigned kMaxRounds = 8;

// What the driver does after one pass of a sector: Settle means the pass reported
// nothing, so it propagated its masters and uptargets and spilled its equations;
// Widen means it reported a seed outside the box, so its work is thrown away and
// the caller widens by the reported bits (SeedBound::widen_for) and runs again;
// GiveUp means the cap is reached with the pass still reporting, so the sector is
// dropped -- it contributes no masters, no uptargets and no equations, exactly as a
// dropped reporting pass does today.  Named here, next to the box rule, so the
// termination policy is testable without running a reduction.
enum class RoundAction { Settle, Widen, GiveUp };

// Round indices are 0-based, so `round + 1` is the number of passes run: the cap
// fires on the kMaxRounds-th reporting pass, matching the probe loop that widens
// only while rounds remain.
constexpr RoundAction round_action(unsigned round, unsigned refusalBits) {
  if (refusalBits == 0)
    return RoundAction::Settle;
  if (round + 1 == kMaxRounds)
    return RoundAction::GiveUp;
  return RoundAction::Widen;
}

// Initial seed-box bounds (max depth, max rank) for a sector: the smallest box
// that covers everything the sector is asked for.  A target needs one layer
// above itself (the equations then reduce it away), an uptarget is reduced in
// place, so it only needs its own depth.  There is no extra floor: a sector
// whose requests reach no further than its own base layer starts at zero excess
// depth.  Both sets only ever hold integrals of this very sector (their lines
// are the sector's lines), so the result always satisfies
// depth >= popcount(id) — the excess-depth arithmetic in _generate_seeds stays
// non-negative and the probe pass raises the bound once if the small box turns
// out to be insufficient.
std::pair<unsigned, unsigned>
initial_seed_bounds(unsigned id, const std::set<RawIntegral> &targets,
                    const std::set<RawIntegral> &uptargets);

// One sector log's seed-box line: the seeds the sector was actually reduced
// with and, after "probe start:", the seeds the probe pass ran with before its
// first widening.  Each triple is "depth rank total" and reports the extents
// of the seeds the boundary admits -- the seeds are what generates the sector's
// equations (SeedBound::admits), and they sit strictly inside the boundary in
// both hard figures, so each is one step below its bound.  depth is the largest
// depth, the sum of a seed's positive indices, i.e. depth() as RawIntegral
// counts it (an absolute figure: a seed of this sector has depth >= popcount(id),
// its base layer); rank is the largest rank, min(rank, limit - popcount(id)) - 1
// because the cap can bite the box's rank corner first; and in Total mode the
// line gains "total: T", the largest total power depth + rank, which is the box
// corner depth + rank - 2 at the initial cap (limit == depth + rank, the top of
// the seed box's total-power range, cutting no seed the hard caps admit) and is
// lowered by a cap that bites -- Box mode omits the field.  The two triples
// differ exactly where the probe widened the box: each widening raises the box
// top by one layer, and in Total mode the cap with it, since it rides at the top
// of the box.  A probe pass appends "probe rounds: N" (1-based) to its own line,
// so the log says how many widenings the sector needed; the full pass prints the
// line without it.  A box from
// initial_seed_bounds always has rank >= 1, so no field underflows there, but a
// base-layer box (depth == popcount(id)) admits none of its own seeds, and the
// line then still prints the box -- that this box admits no seed is what the
// probe reports and repairs.
std::string format_seed_box_line(unsigned id, const SeedBound &bound,
                                 const SeedBound &initial);

// Dump a sector's target list (one operator<< form per line) to `path`.  The
// stream is closed before returning, so the file on disk is a complete,
// newline-terminated snapshot from the moment this returns.  Holding the
// stream open for the sector's lifetime left it at the 8 KiB filebuf boundary,
// cut mid-number, until the sector finished.
void writeTargetList(const std::string &path,
                     const std::set<RawIntegral> &targets);

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
  static bool _compare_complexity(const RawIntegral &, const RawIntegral &);
  // Sort the discovered subsector integrals into column order (most complex
  // first) for the given ordering mode; see the definition for the tie rules.
  static void _order_subintegrals(std::vector<RawIntegral> &, OrderingMode);
  static void _print_progress(ReductionContext &);

private:
  // sector index
  unsigned _index = 0;
  // sector number
  unsigned _id = 0;
  // number of propagators
  unsigned _nprops = 0;
  // the seed boundary this sector is reduced with (see SeedBound); its depth and
  // rank members are the box _generate_seeds generates over
  SeedBound _bound;
  // the boundary as derived for the probe pass, before any widening; kept so the
  // log can report where the probe started (see format_seed_box_line)
  SeedBound _initialBound;
  // which probe pass is running (1-based), set by run_reduce_ff_n; the log reports
  // it so the number of widenings a sector needed is readable
  unsigned _probeRounds = 0;
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
