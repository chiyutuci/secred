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
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
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

// Integral ordering used when seeding a sector.
enum class OrderingMode { Normal, Balance };

// Probe scope: only input-target masters may widen, or every out-of-box master.
enum class ProbeMode { Targets, All };

class RawIntegral;
struct SeedTruncation;

// Per-run options the sectors read.
struct SectorConfig {
  std::string cache_dir = "cache";
  unsigned subsector_filter_n = 1;
  OrderingMode ordering_mode = OrderingMode::Balance;

  bool sector_debug = false;

  const std::set<RawIntegral> *preferred = nullptr;

  const SeedTruncation *truncation = nullptr;

  ProbeMode probe_mode = ProbeMode::Targets;

  bool mandatory_list = true;
};

class Reduce;
class SymmetryTable;
namespace secred {
struct RowTable;
}

// A propagator-index vector; the core integral label.
class RawIntegral {
public:

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

  [[nodiscard]] unsigned size() const { return nprops_; }

  // Total positive power.
  [[nodiscard]] unsigned depth() const {
    unsigned s = 0;
    for (unsigned i = 0; i < nprops_; ++i)
      if (indices_[i] > 0)
        s += indices_[i];
    return s;
  }

  // Total negative power.
  [[nodiscard]] unsigned rank() const {
    unsigned s = 0;
    for (unsigned i = 0; i < nprops_; ++i)
      if (indices_[i] < 0)
        s -= indices_[i];
    return s;
  }

  [[nodiscard]] int maxneg() const {
    int res = -1;
    for (unsigned i = 0; i < nprops_; ++i)
      if (indices_[i] < res)
        res = indices_[i];
    return res;
  }

  [[nodiscard]] int maxpos() const {
    int res = 1;
    for (unsigned i = 0; i < nprops_; ++i)
      if (indices_[i] > res)
        res = indices_[i];
    return res;
  }

  // Bit mask of the lines with positive power.
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

  // Parse "[a, b, ...]".
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
}

namespace YAML {
template <> struct convert<RawIntegral> {
  static bool decode(const Node &node, RawIntegral &integral) {
    if (!node.IsSequence())
      return false;
    integral = RawIntegral(node.as<std::vector<int>>());
    return true;
  }
};
}

// Default relaxation of Kira's seed-truncation level.
inline constexpr unsigned kSeedLevelOffset = 1;

// Kira-style seed-truncation rule for one run.
struct SeedTruncation {

  bool active = false;

  int l = 0;

  unsigned level_offset = kSeedLevelOffset;

  unsigned dots = 0;

  std::unordered_map<unsigned, unsigned> s_exc;
};

// Largest dots among a set of integrals.
unsigned max_seed_dots(const std::vector<RawIntegral> &integrals);

// Dots derived from the targets (no floor).
unsigned seed_dots_from_targets(const std::vector<RawIntegral> &targets);

// Truncation offset derived from the targets.
unsigned automatic_seed_level_offset(const std::vector<RawIntegral> &targets);

// Build the truncation rule from the targets.
SeedTruncation derive_seed_truncation(
    const std::vector<RawIntegral> &targets, unsigned top,
    unsigned level_offset = kSeedLevelOffset);

// Dots a settled basis needs above the rule's limit, or 0.
unsigned seed_dots_refinement(const SeedTruncation &truncation,
                              const std::vector<RawIntegral> &masters);

// Masters that are requests the rule cut, in master order.
std::vector<RawIntegral> seed_request_refinement(
    const std::vector<RawIntegral> &masters,
    const std::map<unsigned, std::unordered_set<RawIntegral>> &cutRequests);

// Depth and rank caps the truncation rule applies to a sector.
std::pair<unsigned, unsigned>
truncated_seed_caps(unsigned id, unsigned depth, unsigned rank,
                    const SeedTruncation &truncation);

// Generated IBP rows in the supported coefficient representations.
typedef std::vector<std::pair<RawIntegral, GiNaC::ex>> IBPProto;

typedef std::vector<std::pair<RawIntegral, std::vector<rational>>> IBPProtoFF;

typedef std::vector<std::pair<RawIntegral, std::vector<ffint>>> IBPProtoFFmod;
typedef std::vector<std::pair<RawIntegral, std::vector<GiNaC::ex>>> IBPProtoM;

// Balance ordering: total power, depth, rank, max positive, max negative.
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

// One equation as integral/coefficient pairs.
using GlobalEquation = std::vector<std::pair<RawIntegral, std::string>>;

// Probe and full-pass timings for one sector.
struct SectorTiming {
  unsigned id = 0;
  bool expanded = false;
  std::uint64_t probeNanos = 0;
  std::uint64_t fullNanos = 0;
};

std::string format_timing_row(const SectorTiming &t);

inline void spill_put_u32(std::string &out, std::uint32_t v) {
  for (unsigned i = 0; i < 4; ++i)
    out.push_back(static_cast<char>((v >> (8 * i)) & 0xff));
}

inline void spill_put_u64(std::string &out, std::uint64_t v) {
  for (unsigned i = 0; i < 8; ++i)
    out.push_back(static_cast<char>((v >> (8 * i)) & 0xff));
}

inline constexpr char kSpillMagic[4] = {'S', 'R', 'S', 'P'};
inline constexpr std::uint32_t kSpillVersion = 1;
inline constexpr std::uint64_t kSpillHeaderBytes = 8;

// Shared state for one sector-phase pass.
struct ReductionContext {
  std::vector<std::vector<RawIntegral>>
      allMasters;
  std::map<unsigned, std::set<RawIntegral>>
      targets;
  std::map<unsigned, std::set<RawIntegral>>
      uptargets;

  std::map<unsigned, std::unordered_set<RawIntegral>> cutRequests;

  std::mutex cutRequestsMutex;

  bool exemptCutRequests = false;

  const SymmetryTable *symmetries = nullptr;

  const secred::RowTable *rowTable = nullptr;
  std::mutex mastersMutex;
  std::mutex uptargetsMutex;
  unsigned totalSectors = 0;
  unsigned completedSectors = 0;
  std::mutex progressMutex;

  std::string rawEquationsPath;
  std::unique_ptr<std::ofstream>
      rawEqOut;
  std::vector<RawIntegral>
      spillIntegrals;
  std::vector<std::string>
      spillCoeffs;
  std::atomic<std::size_t> globalEquationsCount{0};
  std::mutex globalEqMutex;

  // Start a new spilled equation system.
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

  void appendRawEquation(const GlobalEquation &eq) {
    std::lock_guard<std::mutex> lk(globalEqMutex);
    spillBuf.clear();
    spill_frame(eq);
    spill_commit_locked(1);
  }

  void appendRawEquations(const std::vector<GlobalEquation> &eqs) {
    if (eqs.empty())
      return;
    std::lock_guard<std::mutex> lk(globalEqMutex);
    spillBuf.clear();
    for (const GlobalEquation &eq : eqs)
      spill_frame(eq);
    spill_commit_locked(eqs.size());
  }

  // Write the spill trailer and hand back its integral list.
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

  // Drop the spill file.
  void discardSpill() {
    if (rawEqOut)
      rawEqOut->flush();
    rawEqOut.reset();
    if (!rawEquationsPath.empty())
      std::filesystem::remove(rawEquationsPath);
  }

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

  void spill_frame(const GlobalEquation &eq) {
    spill_put_u32(spillBuf, static_cast<std::uint32_t>(eq.size()));
    for (const auto &[integ, coeff] : eq) {
      spill_put_u32(spillBuf, spill_integral_index(integ));
      spill_put_u32(spillBuf, spill_coeff_index(coeff));
    }
    spillTerms += eq.size();
  }

  void spill_commit_locked(std::size_t n_equations) {
    rawEqOut->write(spillBuf.data(), static_cast<std::streamsize>(spillBuf.size()));
    spillBytes += spillBuf.size();
    globalEquationsCount.fetch_add(n_equations, std::memory_order_relaxed);
  }

  std::unordered_map<RawIntegral, std::uint32_t> spillIntegralIndex;
  std::unordered_map<std::string, std::uint32_t> spillCoeffIndex;
  std::string spillBuf;
  std::uint64_t spillBytes = 0;
  std::uint64_t spillTerms = 0;

  std::atomic<unsigned> narrowingSuccess{0};
  std::atomic<unsigned> narrowingFail{0};

  std::atomic<unsigned> truncatedSectors{0};
  std::atomic<unsigned> truncationLifted{0};

  std::atomic<unsigned> boxSpillRaised{0};
  std::atomic<unsigned> boxSpillExtraDepth{0};
  std::atomic<unsigned> boxSpillExtraRank{0};

  std::atomic<unsigned> probeScopeSkipped{0};

  std::atomic<unsigned> rawEquationsTotal{0};

  std::atomic<std::uint64_t> probeNanos{0};
  std::atomic<std::uint64_t> fullNanos{0};

  std::mutex timingMutex;

  void accumulateTiming(const SectorTiming &t) {
    probeNanos.fetch_add(t.probeNanos, std::memory_order_relaxed);
    fullNanos.fetch_add(t.fullNanos, std::memory_order_relaxed);
  }

  void recordTiming(const SectorTiming &t);

  void resetRunStatistics() {
    narrowingSuccess.store(0, std::memory_order_relaxed);
    narrowingFail.store(0, std::memory_order_relaxed);
    truncatedSectors.store(0, std::memory_order_relaxed);
    truncationLifted.store(0, std::memory_order_relaxed);
    boxSpillRaised.store(0, std::memory_order_relaxed);
    boxSpillExtraDepth.store(0, std::memory_order_relaxed);
    boxSpillExtraRank.store(0, std::memory_order_relaxed);
    probeScopeSkipped.store(0, std::memory_order_relaxed);
    rawEquationsTotal.store(0, std::memory_order_relaxed);
    probeNanos.store(0, std::memory_order_relaxed);
    fullNanos.store(0, std::memory_order_relaxed);
    runningSectors.store(0, std::memory_order_relaxed);
    headroomTaken = false;
    availAtAdmit = 0;
  }

  std::uint64_t maxMemory =
      512ULL * 1024 * 1024 * 1024;
  std::uint64_t memoryReserve =
      30ULL * 1024 * 1024 * 1024;
  std::mutex admitMutex;
  std::condition_variable admitCv;

  std::function<std::uint64_t()> memoryProbe = [] {
    return process_used_memory();
  };

  std::atomic<unsigned> runningSectors{0};

  bool headroomTaken = false;
  std::uint64_t availAtAdmit = 0;

  std::uint64_t rearmDrop = 10ULL * 1024 * 1024 * 1024;

  // Block until a sector may run within maxMemory.
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
        headroomTaken = false;
        runningSectors.fetch_add(1, std::memory_order_relaxed);
        return;
      }

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

  // Release the memory admission slot.
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

// Per-sector seed box and truncation caps.
struct SeedBound {
  unsigned depth = 0;
  unsigned rank = 0;

  unsigned limit = 0;

  unsigned trunc_depth = std::numeric_limits<unsigned>::max();
  unsigned trunc_rank = std::numeric_limits<unsigned>::max();

  // Boundary quantity cap for a box of this size.
  static unsigned cap_for(unsigned depth, unsigned rank) {
    return (unsigned)std::max(0, (int)depth + (int)rank - 2);
  }

  // The starting box for a sector.
  static SeedBound initial(unsigned depth, unsigned rank) {
    SeedBound b;
    b.depth = depth;
    b.rank = rank;
    b.limit = cap_for(depth, rank);

    b.trunc_depth = depth;
    b.trunc_rank = rank;
    return b;
  }

  unsigned quantity(const RawIntegral &i) const { return i.depth() + i.rank(); }

  unsigned boundary() const { return limit; }

  int seed_top() const {
    return std::min((int)depth + (int)rank - 2, (int)limit - 1);
  }

  const std::unordered_set<RawIntegral> *requestExempt = nullptr;

  bool is_request_exempt(const RawIntegral &i) const {
    return requestExempt != nullptr &&
           requestExempt->find(i) != requestExempt->end();
  }

  // Whether the rule generates an equation for this seed.
  bool admits(const RawIntegral &i) const {
    if (i.depth() >= depth || i.rank() >= rank)
      return false;
    if ((i.depth() >= trunc_depth || i.rank() >= trunc_rank) &&
        !is_request_exempt(i))
      return false;
    return quantity(i) < boundary();
  }

  // Whether the truncation rule refuses this seed.
  bool truncation_refused(const RawIntegral &i) const {
    if (i.depth() >= depth || i.rank() >= rank)
      return false;
    if (is_request_exempt(i))
      return false;
    return i.depth() >= trunc_depth || i.rank() >= trunc_rank;
  }

  // Whether the seed is beyond the box's top.
  bool past_top(const RawIntegral &i) const {
    return (int)quantity(i) > seed_top();
  }

  bool past_depth_cap(const RawIntegral &i) const { return i.depth() >= depth; }

  // Whether the probe must report this seed.
  bool exceeded(const RawIntegral &i) const {
    return past_top(i) || truncation_refused(i);
  }

  // Refusal bits: 1 depth, 2 rank, 8 top, 16/32 truncation.
  unsigned refusal(const RawIntegral &i) const {
    unsigned bits = 0;
    if (i.depth() >= depth)
      bits |= 1;
    if (i.rank() >= rank)
      bits |= 2;
    if (bits != 0)
      return bits;

    if (past_top(i))
      return 8u;
    if (i.depth() >= trunc_depth)
      bits |= 16;
    if (i.rank() >= trunc_rank)
      bits |= 32;
    return bits;
  }

  bool filter_drops(const RawIntegral &i, unsigned n) const {
    return (int)quantity(i) < seed_top() + 1 - (int)n;
  }

  void widen_depth(bool carry_cap = true) {
    const bool lifted = trunc_depth == depth;
    ++depth;
    if (lifted)
      trunc_depth = depth;
    if (carry_cap)
      limit = std::max(limit, cap_for(depth, rank));
  }

  void widen_rank() {
    const bool lifted = trunc_rank == rank;
    ++rank;
    if (lifted)
      trunc_rank = rank;
    limit = std::max(limit, cap_for(depth, rank));
  }

  void widen_cap() { ++limit; }

  // Apply the widening a refusal calls for.
  void widen_for(unsigned refusalBits) {
    if (refusalBits & 48u) {
      if (refusalBits & 16u)
        trunc_depth = depth;
      if (refusalBits & 32u)
        trunc_rank = rank;
      return;
    }
    if (refusalBits & 8) {
      widen_cap();
      return;
    }
    if (refusalBits & 1)
      widen_depth((refusalBits & 4) != 0);
    if (refusalBits & 2)
      widen_rank();
  }
};

// Probe rounds per sector.
inline constexpr unsigned kMaxRounds = 8;

enum class RoundAction { Settle, Widen, GiveUp };

// What the next probe round does.
constexpr RoundAction round_action(unsigned round, unsigned refusalBits) {
  if (refusalBits == 0)
    return RoundAction::Settle;
  if (round + 1 == kMaxRounds)
    return RoundAction::GiveUp;
  return RoundAction::Widen;
}

// Starting depth and rank for a sector from its targets.
std::pair<unsigned, unsigned>
initial_seed_bounds(unsigned id, const std::set<RawIntegral> &targets,
                    const std::set<RawIntegral> &uptargets);

std::string format_seed_box_line(const SeedBound &bound,
                                 const SeedBound &initial);

// Result of one probe walk.
struct ProbeCriterion {

  unsigned refusal_bits = 0;

  std::vector<RawIntegral> own_masters;

  unsigned scope_skipped = 0;
  std::vector<RawIntegral> accepted;
};

// Walk a reduction and collect refusals beyond the box.
ProbeCriterion probe_criterion(const SeedBound &bound,
                               const std::vector<std::size_t> &needed_indep,
                               const std::vector<RawIntegral> &seeds,
                               unsigned nvars, bool collect,
                               const std::unordered_set<std::size_t> *scope =
                                   nullptr);

std::string format_probe_report(unsigned round, const SeedBound &bound,
                                const ProbeCriterion &crit);

void writeTargetList(const std::string &path,
                     const std::set<RawIntegral> &targets);

// One subsector's seeding, reduction and probe loop.
class Sector {
public:
  friend class SectorTestHelper;

  // Build a sector from its positive-line mask.
  static Sector make(unsigned index, unsigned id, unsigned nprops,
                     const std::vector<bool> &lines,
                     const std::set<unsigned> &trivials);

  unsigned id() const { return _id; }

  // Reduce the sector and spill its equations.
  void run_reduce_ff_n(const std::vector<IBPProtoFF> &,
                       const std::vector<IBPProtoFFmod> &,
                       const std::vector<IBPProtoM> *ibps_sym,
                       const SectorConfig &, ReductionContext &);

  // Reset per-pass state.
  void clear();

private:
  unsigned sector_reduction_ff_n(const std::vector<IBPProtoFF> &,
                                 const std::vector<IBPProtoFFmod> &,
                                 const std::vector<IBPProtoM> *ibps_sym,
                                 const SectorConfig &, ReductionContext &);

  void _generate_seeds(OrderingMode mode = OrderingMode::Balance,
                       const std::set<RawIntegral> *preferred = nullptr);
  void _generate_combinations(int number, int sum);
  static bool _compare_subintegral(const RawIntegral &, const RawIntegral &);
  static bool _compare_balance(const RawIntegral &, const RawIntegral &);

  static void _order_subintegrals(std::vector<RawIntegral> &, OrderingMode,
                                  const std::set<RawIntegral> *preferred = nullptr);
  static void _print_progress(ReductionContext &);

private:

  unsigned _index = 0;

  unsigned _id = 0;

  unsigned _nprops = 0;

  SeedBound _bound;

  SeedBound _initialBound;

  unsigned _probeRounds = 0;

  std::string _probeReport;

  std::set<RawIntegral> _inputTargets;

  std::unordered_set<RawIntegral> _requestExempt;

  std::vector<bool> _lines;

  std::map<std::pair<int, int>, std::vector<std::vector<int>>> _combinations;

  std::vector<RawIntegral> _seeds;

  std::vector<RawIntegral> _masters;

  std::vector<RawIntegral> _submasters;

  std::unordered_map<RawIntegral, unsigned> _weights;

  std::set<unsigned> _trivialsectors;
};
