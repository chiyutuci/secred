#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
  std::string cache_dir = "cache";
  unsigned subsector_filter_n = 1;
  OrderingMode ordering_mode = OrderingMode::Balance;
};

class Reduce;
class SymmetryTable; // src/symmetry.h

class RawIntegral {
public:
  // Indices are the powers of each propagator (denominator > 0, numerator
  // < 0); in practice they stay within ±10, so one byte each suffices.
  // MAX_PROPS caps the number of propagators (the largest example has 14).
  static constexpr unsigned MAX_PROPS = 32;

  RawIntegral() : nprops_(0) {}

  explicit RawIntegral(unsigned n) : nprops_(cap(n)) {}

  explicit RawIntegral(unsigned n, int i) : nprops_(cap(n)) {
    for (unsigned k = 0; k < nprops_; ++k)
      indices_[k] = (int8_t)i;
  }

  explicit RawIntegral(const std::vector<bool> &lines) : nprops_(cap(lines.size())) {
    for (unsigned i = 0; i < nprops_; ++i)
      indices_[i] = lines[i] ? 1 : 0;
  }

  explicit RawIntegral(const std::vector<int> &v) : nprops_(cap(v.size())) {
    for (unsigned i = 0; i < nprops_; ++i)
      indices_[i] = (int8_t)v[i];
  }

  explicit RawIntegral(std::vector<int> &&v) : nprops_(cap(v.size())) {
    for (unsigned i = 0; i < nprops_; ++i)
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

  bool operator!=(const RawIntegral &other) const { return !(*this == other); }

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
    while (i < s.size() && (s[i] == '[' || s[i] == ' '))
      ++i;
    while (i < s.size() && s[i] != ']') {
      size_t start = i;
      while (i < s.size() && s[i] != ',' && s[i] != ']')
        ++i;
      std::string tok = s.substr(start, i - start);
      size_t b = tok.find_first_not_of(' ');
      size_t e = tok.find_last_not_of(' ');
      v.push_back(std::stoi(tok.substr(b, e - b + 1)));
      while (i < s.size() && (s[i] == ',' || s[i] == ' '))
        ++i;
    }
    return RawIntegral(std::move(v));
  }

  friend class std::hash<RawIntegral>;

private:
  // The storage holds MAX_PROPS indices; a longer input is truncated rather than
  // written past the end (the old loops wrote indices_[32], i.e. nprops_ itself).
  static unsigned cap(std::size_t n) {
    return n > MAX_PROPS ? MAX_PROPS : static_cast<unsigned>(n);
  }

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

// The balance key, ascending in all five keys: total power depth+rank, then
// depth, then rank, then the largest denominator power, then the largest
// numerator power.  The rank key can never decide by itself (rank = total power
// - depth), but it is kept so the key reads exactly as specified.
//
// This is NOT a total order: distinct integrals tied on all five keys compare
// equal, so every caller must supply its own deterministic tie-break (a stable
// sort, or extra keys).  The sector stage (seeds, subsector columns) and the
// global ID numbering both use this single definition.
inline bool balance_less(const RawIntegral &a, const RawIntegral &b) {
  unsigned ta = a.depth() + a.rank();
  unsigned tb = b.depth() + b.rank();
  if (ta != tb)
    return ta < tb;
  if (a.depth() != b.depth())
    return a.depth() < b.depth();
  if (a.rank() != b.rank())
    return a.rank() < b.rank();
  if (a.maxpos() != b.maxpos())
    return a.maxpos() < b.maxpos();
  return -a.maxneg() < -b.maxneg();
}

// One term in a global equation: the integral and its coefficient as a string.
// Coefficient strings come from rational::str() (ff_n path) or GiNaC stream
// output (ff_m path) and are written verbatim into the global system file.
using GlobalEquation = std::vector<std::pair<RawIntegral, std::string>>;

// Wall-clock cost of one sector's adaptive seeding (ATPS).  Every round is a
// full pass: probeNanos accumulates the reporting rounds that were discarded --
// each charged the cache-directory removal that preceded it -- and fullNanos is
// the one settling round that spilled, 0 for a sector dropped at the round cap.
// The memory gate (SectorRunGuard) is constructed inside each pass, so a
// headroom wait is counted in the pass that waited.  `expanded` records that
// the initial depth/rank were found too small and the box was widened at least
// once before the settling round.
struct SectorTiming {
  unsigned id = 0;
  bool expanded = false;
  std::uint64_t probeNanos = 0;
  std::uint64_t fullNanos = 0;
};

// One row of sector_timing.csv: "sector,expanded,probe_s,full_s".
std::string format_timing_row(const SectorTiming &t);

// Little-endian putters for the spill frame.  The format is read back by
// src/spill_reader.cpp, which is byte-order independent for the same reason.
inline void spill_put_u32(std::string &out, std::uint32_t v) {
  for (unsigned i = 0; i < 4; ++i)
    out.push_back(static_cast<char>((v >> (8 * i)) & 0xff));
}

inline void spill_put_u64(std::string &out, std::uint64_t v) {
  for (unsigned i = 0; i < 8; ++i)
    out.push_back(static_cast<char>((v >> (8 * i)) & 0xff));
}

// "SRSP" + u32 version 1, written by openSpill and recognised by the reader.
inline constexpr char kSpillMagic[4] = {'S', 'R', 'S', 'P'};
inline constexpr std::uint32_t kSpillVersion = 1;
inline constexpr std::uint64_t kSpillHeaderBytes = 8;

// Shared reduction state passed to each Sector during parallel execution.
// Owned by Reduce; Sector methods receive it by reference.
struct ReductionContext {
  std::vector<std::vector<RawIntegral>>
      allMasters; // [sector_index] → master integrals
  std::map<unsigned, std::set<RawIntegral>>
      targets; // sector_id → required integrals
  std::map<unsigned, std::set<RawIntegral>>
      uptargets; // sector_id → integrals from higher layers
  // The detected sector symmetries, owned by Reduce and read by every sector
  // (spec 4.3).  Set at the end of Reduce::prepare_sectors and read-only after,
  // so no lock is needed.  A table with no prescriptions when the feature is
  // off, so every consumer sees nothing to do.
  const SymmetryTable *symmetries = nullptr;
  std::mutex mastersMutex;
  std::mutex uptargetsMutex;
  unsigned totalSectors = 0;
  unsigned completedSectors = 0;
  std::mutex progressMutex;
  // Incremental disk spill of the selected equations from all sectors.
  //
  // The equations never accumulate in RAM: each sector appends its settling
  // round's equations to one file, which the global stage reads back through
  // src/spill_reader.h.  The file is the 8-byte header, one frame per equation
  // (u32 n_terms, then (u32 integral index, u32 coefficient index) per term, all
  // little-endian), the two interned tables -- the distinct integrals and the
  // distinct coefficient texts, each stored once -- and a footer with the table
  // offsets and the two counters.
  //
  // Interning is what makes a term 8 bytes: on F234, 4.8 M spilled terms carry
  // 7,276 distinct coefficients (57 KB of text) and 330 k distinct integrals, so
  // the text this replaced was 2.35 GB of mostly repetition.
  std::string rawEquationsPath;
  std::unique_ptr<std::ofstream>
      rawEqOut; // append stream, guarded by globalEqMutex
  std::vector<RawIntegral>
      spillIntegrals; // distinct integrals, in first-seen order
  std::vector<std::string>
      spillCoeffs; // distinct coefficient texts, in first-seen order
  std::atomic<std::size_t> globalEquationsCount{0}; // equations spilled so far
  std::mutex globalEqMutex;

  // Open the spill at `rawEquationsPath` for writing, creating its directory, and
  // write the header.  Called once per run, before the sector phase; a path that
  // cannot be opened is a hard failure (the global stage cannot run without it).
  void openSpill() {
    if (rawEquationsPath.empty())
      throw std::runtime_error("the raw-equation spill has no path");
    const std::size_t slash = rawEquationsPath.find_last_of('/');
    if (slash != std::string::npos)
      std::filesystem::create_directories(rawEquationsPath.substr(0, slash));
    rawEqOut = std::make_unique<std::ofstream>(
        rawEquationsPath, std::ios::binary | std::ios::trunc);
    if (!*rawEqOut)
      throw std::runtime_error("cannot open the raw-equation spill at " +
                               rawEquationsPath);
    spillIntegrals.clear();
    spillCoeffs.clear();
    spillIntegralIndex.clear();
    spillCoeffIndex.clear();
    spillBuf.clear();
    spillBytes = kSpillHeaderBytes;
    spillTerms = 0;
    globalEquationsCount.store(0, std::memory_order_relaxed);
    std::string header(kSpillMagic, sizeof kSpillMagic);
    spill_put_u32(header, kSpillVersion);
    rawEqOut->write(header.data(), static_cast<std::streamsize>(header.size()));
  }

  // Append one equation to the spill and record its integrals. Thread-safe
  // (guarded by globalEqMutex). `rawEqOut` must have been opened by openSpill.
  void appendRawEquation(const GlobalEquation &eq) {
    std::lock_guard<std::mutex> lk(globalEqMutex);
    spillBuf.clear();
    spill_frame(eq);
    spill_commit_locked(1);
  }

  // Append a whole burst of equations under one lock.  A sector's settling round
  // produces thousands of equations at once, and the lock used to be taken once per
  // equation -- 931,597 acquisitions on F234, ~11 us each, which was most of the
  // sector phase's wall time.  One acquisition per burst, and interning inside it,
  // is what removes that serialisation: the frames written are byte-identical to
  // the per-equation ones (`append_raw_equations_batch_matches_single_appends`).
  void appendRawEquations(const std::vector<GlobalEquation> &eqs) {
    if (eqs.empty())
      return;
    std::lock_guard<std::mutex> lk(globalEqMutex);
    spillBuf.clear();
    for (const GlobalEquation &eq : eqs)
      spill_frame(eq);
    spill_commit_locked(eqs.size());
  }

  // Close the spill: write the two interned tables and the footer, flush, release
  // the stream, and hand the caller the distinct integrals.  Their order here is
  // the first-seen one; the numbering order is imposed by
  // `order_global_integrals` (global_order.h), which is a total order on the
  // integrals themselves, so no property of this order can reach the result.
  void finishSpill(std::vector<RawIntegral> &integrals) {
    if (!rawEqOut)
      return;
    const std::uint64_t integral_offset = spillBytes;
    spillBuf.clear();
    spill_put_u64(spillBuf, spillIntegrals.size());
    for (const RawIntegral &integ : spillIntegrals) {
      spillBuf.push_back(
          static_cast<char>(static_cast<std::uint8_t>(integ.size())));
      for (unsigned i = 0; i < integ.size(); ++i)
        spillBuf.push_back(static_cast<char>(static_cast<std::int8_t>(integ[i])));
    }
    const std::uint64_t coeff_offset = integral_offset + spillBuf.size();
    spill_put_u64(spillBuf, spillCoeffs.size());
    for (const std::string &coeff : spillCoeffs) {
      spill_put_u32(spillBuf, static_cast<std::uint32_t>(coeff.size()));
      spillBuf += coeff;
    }
    spill_put_u64(spillBuf, integral_offset);
    spill_put_u64(spillBuf, coeff_offset);
    spill_put_u64(spillBuf, globalEquationsCount.load(std::memory_order_relaxed));
    spill_put_u64(spillBuf, spillTerms);
    spillBuf.append(kSpillMagic, sizeof kSpillMagic);
    rawEqOut->write(spillBuf.data(), static_cast<std::streamsize>(spillBuf.size()));
    rawEqOut->flush();
    rawEqOut.reset();
    spillBuf.clear();
    spillBuf.shrink_to_fit();
    integrals.swap(spillIntegrals);
  }

  // The run spilled nothing: close the stream and remove the header-only file.
  void discardSpill() {
    if (rawEqOut)
      rawEqOut->flush();
    rawEqOut.reset();
    if (!rawEquationsPath.empty())
      std::filesystem::remove(rawEquationsPath);
  }

  // The index of `integ` in the interned table, appending it on first use.  Called
  // with the lock held, so the table and the index cannot disagree.
  std::uint32_t spill_integral_index(const RawIntegral &integ) {
    const auto it = spillIntegralIndex.try_emplace(
        integ, static_cast<std::uint32_t>(spillIntegrals.size()));
    if (it.second)
      spillIntegrals.push_back(integ);
    return it.first->second;
  }

  std::uint32_t spill_coeff_index(const std::string &coeff) {
    const auto it = spillCoeffIndex.try_emplace(
        coeff, static_cast<std::uint32_t>(spillCoeffs.size()));
    if (it.second)
      spillCoeffs.push_back(coeff);
    return it.first->second;
  }

  // One equation's frame into spillBuf.  Called with the lock held.
  void spill_frame(const GlobalEquation &eq) {
    spill_put_u32(spillBuf, static_cast<std::uint32_t>(eq.size()));
    for (const auto &[integ, coeff] : eq) {
      spill_put_u32(spillBuf, spill_integral_index(integ));
      spill_put_u32(spillBuf, spill_coeff_index(coeff));
    }
    spillTerms += eq.size();
  }

  // Write the frames spillBuf holds and count them.  Called with the lock held.
  void spill_commit_locked(std::size_t n_equations) {
    rawEqOut->write(spillBuf.data(), static_cast<std::streamsize>(spillBuf.size()));
    spillBytes += spillBuf.size();
    globalEquationsCount.fetch_add(n_equations, std::memory_order_relaxed);
  }

  // Interner state.  Guarded by globalEqMutex like the stream itself.
  std::unordered_map<RawIntegral, std::uint32_t> spillIntegralIndex;
  std::unordered_map<std::string, std::uint32_t> spillCoeffIndex;
  std::string spillBuf;            // one burst's frames, reused across bursts
  std::uint64_t spillBytes = 0;    // header + frames written so far
  std::uint64_t spillTerms = 0;    // terms written so far

  // Per-sector seed-range narrowing statistics (updated atomically by worker
  // threads). narrowingSuccess: sectors where the initial depth/rank bounds
  // (derived from
  //   targets) were sufficient -- the settling round reported nothing.
  // narrowingFail: sectors that had to widen the bounds at least once before
  // settling.
  std::atomic<unsigned> narrowingSuccess{0};
  std::atomic<unsigned> narrowingFail{0};

  // Total raw equations generated across all sectors, counting every round --
  // the discarded ones included.
  std::atomic<unsigned> rawEquationsTotal{0};

  // Sector-reduction (ATPS) wall-clock totals in nanoseconds: the sum of the
  // per-sector `SectorTiming` figures above.  probeNanos accumulates every
  // discarded reporting round, fullNanos the settling passes; their ratio is
  // the share of the reduction work the round loop throws away.  The memory
  // gate lives inside each pass, so headroom waits are included in these
  // figures.
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
  std::uint64_t maxMemory =
      512ULL * 1024 * 1024 * 1024; // allowed total (bytes)
  std::uint64_t memoryReserve =
      60ULL * 1024 * 1024 * 1024; // headroom below ceiling
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
  // least `rearmDrop` bytes (avail falls that far below the snapshot) — not
  // only when the headroom is fully exhausted — so a small sector does not keep
  // the flag set and stall every later waiter.
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
        headroomTaken = false; // burst limiter irrelevant with this much room
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

// The seed boundary a sector is reduced with: the region of seeds that may
// generate equations, and the quantity the subsector filter measures in.
//
// `depth` and `rank` are hard caps (a seed is admitted only if depth() < depth
// and rank() < rank); `limit` caps the total power depth() + rank(), starting
// at the top of the seed box's own total-power range (limit == depth + rank),
// where the strict comparison cuts nothing -- the only seed reaching that total
// is the box corner, which the hard caps reject anyway.  `quantity()` is the
// figure everything here counts in and `boundary()` the cap; the criterion and
// the subsector filter measure against `seed_top()` instead, so the filter's
// offset n does not shift when the cap moves.
struct SeedBound {
  unsigned depth = 0;
  unsigned rank = 0;
  // Admitted iff depth() + rank() < limit; read it through boundary(), never
  // directly.  The strict comparison is why initial() starts the cap at the top
  // of the seed box's total-power range rather than at the corner the hard caps
  // allow.
  unsigned limit = 0;

  // The boundary a sector starts from, given the box initial_seed_bounds
  // derived.
  static SeedBound initial(unsigned depth, unsigned rank) {
    SeedBound b;
    b.depth = depth;
    b.rank = rank;
    b.limit = depth + rank;
    return b;
  }

  // The quantity the boundary, the criterion and the filter count in: the total
  // power depth() + rank().
  unsigned quantity(const RawIntegral &i) const { return i.depth() + i.rank(); }

  // The cap on the quantity; a widening moves it (see widen_depth).
  unsigned boundary() const { return limit; }

  // The largest quantity an admitted seed can reach: the box corner depth +
  // rank
  // - 2, lowered to limit - 1 once the cap bites into the box.  Signed, so a
  // degenerate hand-built box (depth 0) reports -1 rather than wrapping.  It is
  // the reference the subsector filter counts its layers from -- what the seeds
  // reach, not where the boundary sits -- and the triple format_seed_box_line
  // prints.
  int seed_top() const {
    return std::min((int)depth + (int)rank - 2, (int)limit - 1);
  }

  // May this seed generate equations (a row of the sector's system)?
  bool admits(const RawIntegral &i) const {
    if (i.depth() >= depth || i.rank() >= rank)
      return false;
    return quantity(i) < boundary();
  }

  // The box-top half of the criterion.  Measuring against the seed box rather
  // than the cap is what makes it fire -- the cap sits at or above the box top
  // -- and it names the box top as what has to move (see widen_depth's
  // carry_cap).
  bool past_top(const RawIntegral &i) const {
    return (int)quantity(i) > seed_top();
  }

  // The depth-cap half.  A large rank figure lifts seed_top() well above the
  // depth cap, so a deep, low-rank seed slips under the top yet is refused by
  // admits() -- on F234's sector 246 the box (7, 5) has seed_top() 10, and the
  // seed [0,1,2,0,1,1,1,1,0] (depth 7, rank 0, total 7) sits three layers below
  // the top while the depth cap refuses it.  This half is what reports it.
  bool past_depth_cap(const RawIntegral &i) const { return i.depth() >= depth; }

  // The narrowing-failure criterion: the seed is either past the box top or
  // under it at the depth hard cap.  Either way the boundary refuses it, it
  // generates no equation, and it survives as an unreduced master -- and either
  // way one widening is enough to admit it, which is why both halves are
  // reported.
  //
  // The criterion is one-sided on the rank axis: a wide, shallow seed the rank
  // cap refuses is not reported, since widening would not admit it either --
  // those are the "outer" seeds the sector hands on as masters.
  bool exceeded(const RawIntegral &i) const {
    return past_top(i) || past_depth_cap(i);
  }

  // Which figure has to grow for a seed exceeded() reports to be admitted
  // again: bit 1 the depth hard cap, bit 2 the rank hard cap.  It names a
  // figure, not the repair's cap policy -- the caller pairs it with past_top()
  // to decide whether a depth widening also carries the cap along
  // (widen_depth's carry_cap).
  unsigned refusal(const RawIntegral &i) const {
    unsigned bits = 0;
    if (i.depth() >= depth)
      bits |= 1;
    if (i.rank() >= rank)
      bits |= 2;
    return bits == 0 ? 1u : bits;
  }

  // The subsector filter, applied at both sites in
  // Sector::sector_reduction_ff_n to integrals that are not seeds: drop the
  // integral when its quantity is more than n layers below seed_top(), so n = 0
  // drops every subsector integral and n = 1 keeps the top layer alone.  The
  // reference is the seed box, so n does not shift when the cap moves.  The
  // rule has no upper end, and the subtraction is done in int: seed_top() + 1 <
  // n must not wrap.
  bool filter_drops(const RawIntegral &i, unsigned n) const {
    return (int)quantity(i) < seed_top() + 1 - (int)n;
  }

  // Raise the box by one layer in the figure refusal() named -- widen_depth for
  // bit 1, widen_rank for bit 2 -- which admits the reported seed, so the next
  // pass either settles or reports a seed the new layer brought in.
  //
  // The cap rides at the top of the box (limit == depth + rank, as initial()
  // built it), so it never cuts an admission the hard caps allow and seed_top()
  // rises by exactly one layer; std::max keeps a cap already above the box top
  // from being dragged down.
  //
  // carry_cap == false deepens the box without moving the cap, repairing a
  // report the depth hard cap made on its own (the caller's bit 4): the total
  // power was not wrong, so the box grows one layer deeper and keeps its
  // ceiling.
  void widen_depth(bool carry_cap = true) {
    ++depth;
    if (carry_cap)
      limit = std::max(limit, depth + rank);
  }

  void widen_rank() {
    ++rank;
    limit = std::max(limit, depth + rank);
  }

  // Apply one pass's report to the box: bit 1 deepens it (carrying the cap only
  // when bit 4 says the box top was what was reported), bit 2 widens the rank.
  // A report always pins a hard figure, so a lone bit 4 widens nothing.
  void widen_for(unsigned refusalBits) {
    if (refusalBits & 1)
      widen_depth(/*carry_cap=*/(refusalBits & 4) != 0);
    if (refusalBits & 2)
      widen_rank();
  }
};

// How many passes a sector may spend settling its seed box before the driver
// gives up on it with a warning.
inline constexpr unsigned kMaxRounds = 8;

// What the driver does after one pass of a sector: Settle means the pass
// reported nothing, so it propagated its masters and uptargets and spilled its
// equations; Widen means it reported a seed outside the box, so its work is
// thrown away and the caller widens by the reported bits (SeedBound::widen_for)
// and runs again; GiveUp means the cap is reached with the pass still
// reporting, so the sector is dropped -- it contributes no masters, no
// uptargets and no equations.  Named here, next to the box rule, so the
// termination policy is testable without running a reduction.
enum class RoundAction { Settle, Widen, GiveUp };

// Round indices are 0-based, so `round + 1` is the number of passes run: the
// cap fires on the kMaxRounds-th reporting pass.
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
// depth >= popcount(id) -- the excess-depth arithmetic in _generate_seeds stays
// non-negative, and the round loop widens the box if it turns out too small.
std::pair<unsigned, unsigned>
initial_seed_bounds(unsigned id, const std::set<RawIntegral> &targets,
                    const std::set<RawIntegral> &uptargets);

// One sector log's seed-box line: "depth rank total" for the seeds the boundary
// admits -- one step below each hard bound, since a seed sits strictly inside
// it -- then the same triple for the box initial_seed_bounds derived, after
// "start:". The settling round appends "rounds: N", so the log says how many
// widenings the sector needed.  The arithmetic is in format_seed_box_line
// (sector.cpp).
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
                       const std::vector<IBPProtoM> *ibps_sym,
                       const SectorConfig &, ReductionContext &);

  void clear();

private:
  unsigned sector_reduction_ff_n(const std::vector<IBPProtoFF> &,
                                 const std::vector<IBPProtoFFmod> &,
                                 const std::vector<IBPProtoM> *ibps_sym,
                                 const SectorConfig &, ReductionContext &);

  void _generate_seeds(OrderingMode mode = OrderingMode::Balance);
  void _generate_combinations(int number, int sum);
  static bool _compare_subintegral(const RawIntegral &, const RawIntegral &);
  static bool _compare_balance(const RawIntegral &, const RawIntegral &);
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
  // the seed boundary this sector is reduced with (see SeedBound); its depth
  // and rank members are the box _generate_seeds generates over
  SeedBound _bound;
  // the boundary as derived from the targets, before any widening; kept so the
  // log can report where the sector started (see format_seed_box_line)
  SeedBound _initialBound;
  // the round now running (1-based), set by run_reduce_ff_n; the log reports it
  // so the number of widenings a sector needed is readable
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
