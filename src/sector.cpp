#include "sector.h"

#include <algorithm>
#include <chrono>
#include <fflow/alg_functions.hh>
#include <fflow/analytic_solver.hh>
#include <fflow/graph.hh>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <malloc.h>
#include <memory>
#include <mutex>
#include <sstream>
#include <tuple>

#include "global_select.h"
#include "numeric_solver_mod.h"
#include "sector_engine.h"

using namespace fflow;

// GiNaC is not thread-safe: concurrent `ex` operations (multiplication, expand,
// streaming) from worker threads corrupt GiNaC's internal state and crash.  The
// signed path runs its symbolic emission in parallel sector threads, so all its
// GiNaC work is serialized through this mutex.
static std::mutex ginac_mutex;

// RAII guard: waits until the system has memory headroom, then marks this
// sector "running" for the duration of one elimination (probe or full).  The
// slot is released on every return path, so probe iterations are gated too and
// runningSectors tracks the true (probe + full) concurrency.
struct SectorRunGuard {
  ReductionContext &ctx;
  explicit SectorRunGuard(ReductionContext &ctx) : ctx(ctx) {
    ctx.waitForMemoryHeadroom();
  }
  ~SectorRunGuard() { ctx.releaseMemorySlot(); }
};

Sector Sector::make(unsigned index, unsigned id, unsigned nprops,
                    const std::vector<bool> &lines,
                    const std::set<unsigned> &trivials) {
  Sector s;
  s._index = index;
  s._id = id;
  s._nprops = nprops;
  s._lines = lines;
  s._trivialsectors = trivials;
  return s;
}

bool Sector::_compare_subintegral(const RawIntegral &i1,
                                  const RawIntegral &i2) {
  int nprops = i1.size();
  if (i1.depth() < i2.depth())
    return true;
  else if (i1.depth() > i2.depth())
    return false;
  else {
    if (i1.rank() < i2.rank())
      return true;
    else if (i1.rank() > i2.rank())
      return false;
    else {
      if (i1.sector() < i2.sector())
        return true;
      else if (i1.sector() > i2.sector())
        return false;
      else {
        for (int i = 0; i < nprops; i++) {
          if (i1[i] > 0) {
            if (i1[i] < i2[i])
              return true;
            else if (i1[i] > i2[i])
              return false;
          }
        }
        for (int i = 0; i < nprops; i++) {
          if (i1[i] <= 0) {
            if (i1[i] < i2[i])
              return false;
            else if (i1[i] > i2[i])
              return true;
          }
        }
      }
    }
  }
  return false;
}

// OrderingMode::Complexity — the five-key order itself is complexity_less()
// (src/sector.h), shared with the global ID numbering.  Ascending = simplest
// first, the same direction as Balance and the same direction the column
// convention wants (a larger label is a smaller, more complex column).  The
// five keys do not form a total order — integrals that tie on all of them are
// ordered by whatever order the caller's input had, so both call sites must
// sort stably (the global numbering adds its own tie-break keys instead).
bool Sector::_compare_complexity(const RawIntegral &i1,
                                 const RawIntegral &i2) {
  return complexity_less(i1, i2);
}

// Order the discovered subsector integrals into column order: the most complex
// integral first, so it takes the smallest subsector column (nvars).  The mode
// picks the key: Complexity uses the same five-key order as the seeds, the other
// modes the canonical per-index _compare_subintegral.  Which subsector integral
// sits in the smallest column decides the elimination pivots exactly as it does
// for the seeds.  The Complexity key is not a total order, so that path sorts
// stably: ties keep the order the vector arrives in (the discovery builds it
// from a std::map, i.e. RawIntegral::operator<), which the final reverse then
// mirrors — tied integrals come back in the reverse of the arrival order.
void Sector::_order_subintegrals(std::vector<RawIntegral> &subInts,
                                 OrderingMode mode) {
  if (mode == OrderingMode::Complexity)
    std::stable_sort(subInts.begin(), subInts.end(), _compare_complexity);
  else
    std::sort(subInts.begin(), subInts.end(), _compare_subintegral);
  std::reverse(subInts.begin(), subInts.end());
}

void createJobFile(const std::string &filename, bool otf) {
  std::ofstream file(filename);

  file << "jobs:" << std::endl;
  file << "  - reduce_user_defined_system:" << std::endl;
  file << "      input_system: {files: [\"system\"], config: false, otf: "
       << (otf ? "true" : "false") << "}" << std::endl;
  file << "      select_integrals:" << std::endl;
  file << "        select_mandatory_list:" << std::endl;
  file << "          - [targets]" << std::endl;
  file << std::endl;
  file << "      conditional: true" << std::endl;
  file << std::endl;
  file << "      run_firefly: true" << std::endl;
  file << std::endl;
  file << "  - kira2file:" << std::endl;
  file << "      target:" << std::endl;
  file << "       - [targets]" << std::endl;

  file.close();
}

void writeTargetList(const std::string &path,
                     const std::set<RawIntegral> &targets) {
  std::ofstream file(path);
  for (const auto &integral : targets)
    file << integral << '\n';
  // The stream dies here: the buffered tail is flushed and the file is a
  // complete snapshot on disk even though the sector keeps running.
}

std::pair<unsigned, unsigned>
initial_seed_bounds(unsigned id, const std::set<RawIntegral> &targets,
                    const std::set<RawIntegral> &uptargets) {
  unsigned rank = 1;
  unsigned depth = std::popcount(id);
  for (const auto &integral : targets) {
    rank = std::max(rank, integral.rank() + 1);
    depth = std::max(depth, integral.depth() + 1);
  }
  for (const auto &integral : uptargets) {
    rank = std::max(rank, integral.rank() + 1);
    depth = std::max(depth, integral.depth());
  }
  return {depth, rank};
}

std::string format_seed_box_line(unsigned id, const SeedBound &bound,
                                 const SeedBound &initial) {
  // The three figures describe the seeds the boundary admits, not the boundary:
  // the seeds are what generates the sector's equations (SeedBound::admits), and
  // they sit strictly inside it in depth and in rank, so every figure here is one
  // step below the corresponding bound.  Reported are the extents of the admitted
  // set along each axis -- its largest depth (the sum of the positive indices, as
  // depth() counts it), its largest rank, and -- in Total mode, where the
  // total-power cap cuts the (depth, rank) box along a diagonal -- its largest
  // total power.  Each extent accounts for all three admission rules: a seed of
  // this sector has depth >= popcount(id), so the rank extent reads
  // min(rank, limit - popcount(id)) - 1 whenever the cap bites, and with the cap at
  // its initial value (the top of the seed box's total-power range, depth + rank)
  // the total-power figure is the box corner depth + rank - 2.  "probe start"
  // repeats the same three figures for the boundary the probe pass ran with,
  // before its single relaxation: the two are equal wherever the relaxation admits
  // no further seed (which is always so from the initial cap, one diagonal above
  // the box) and differ wherever it does, i.e. from a cap that was biting.
  auto append = [id](std::ostringstream &oss, const SeedBound &b) {
    const int depth = (int)b.depth;
    const int rank = (int)b.rank;
    if (b.mode == BoundaryMode::Total) {
      const int limit = (int)b.limit;
      oss << " depth: " << std::min(depth, limit) - 1
          << " rank: " << std::min(rank, limit - (int)std::popcount(id)) - 1
          << " total: " << std::min(depth + rank - 2, limit - 1);
    } else {
      oss << " depth: " << depth - 1 << " rank: " << rank - 1;
    }
  };
  std::ostringstream oss;
  append(oss, bound);
  oss << "  probe start:";
  append(oss, initial);
  return oss.str();
}

void Sector::_generate_combinations(int number, int sum) {
  for (int num = 0; num <= number; ++num)
    for (int s = 0; s <= sum; ++s) {
      if (_combinations.contains({num, s}))
        continue;
      else if (num == 0)
        _combinations[{num, s}] = {{}};
      else if (num == 1)
        _combinations[{num, s}] = {{s}};
      else if (s == 0)
        _combinations[{num, s}] = {std::vector<int>(num, 0)};
      else
        for (int i = 0; i <= s; ++i) {
          for (const auto &comb : _combinations[{num - 1, s - i}]) {
            std::vector<int> newComb = comb;
            newComb.emplace_back(i);
            _combinations[{num, s}].emplace_back(std::move(newComb));
          }
        }
    }
}

void Sector::_generate_seeds(OrderingMode mode) {
  int lines = std::popcount(_id);
  int zeros = (int)_nprops - lines;
  int number = std::max(lines, zeros);
  // The generation extent is the box itself, so it is the same in both boundary
  // modes: the Total-mode total-power cap only decides which of these seeds
  // generate equations (SeedBound::admits) and which of them the filter and the
  // probe keep, not which seeds exist, and every seed still gets a weight and
  // hence a column.  A relaxed boundary therefore never changes nvars.
  int sum = std::max((int)_bound.depth - (int)std::popcount(_id), (int)_bound.rank);
  _generate_combinations(number, sum);

  for (int depth = 0; depth <= _bound.depth - std::popcount(_id); ++depth) {
    if (zeros != 0)
      for (int rank = 0; rank <= _bound.rank; ++rank)
        for (const auto &depthComb : _combinations[{lines, depth}])
          for (const auto &rankComb : _combinations[{zeros, rank}]) {
            RawIntegral integral(_lines);
            unsigned posLines = 0;
            unsigned posZeros = 0;
            for (unsigned i = 0; i < _nprops; i++) {
              if (_lines[i])
                integral[i] += depthComb[posLines++];
              else
                integral[i] -= rankComb[posZeros++];
            }
            _seeds.emplace_back(std::move(integral));
          }
    else {
      for (const auto &depthComb : _combinations[{lines, depth}]) {
        RawIntegral integral(_lines);
        unsigned posLines = 0;
        for (unsigned i = 0; i < _nprops; i++)
          if (_lines[i])
            integral[i] += depthComb[posLines++];
        _seeds.emplace_back(std::move(integral));
      }
    }
  }

  int base_depth = (int)std::popcount(_id);

  if (mode == OrderingMode::Balance) {
    // Balance ordering: (excess_depth ASC, rank ASC, max_active_power ASC).
    // Seeds with the smallest excess depth and lowest rank are placed first,
    // making them preferred pivots in the Gaussian elimination and increasing
    // the chance that higher-depth, higher-rank seeds are found redundant.
    std::stable_sort(
        _seeds.begin(), _seeds.end(),
        [this, base_depth](const RawIntegral &a, const RawIntegral &b) {
          int ea = (int)a.depth() - base_depth;
          int eb = (int)b.depth() - base_depth;
          if (ea != eb)
            return ea < eb;
          if (a.rank() != b.rank())
            return a.rank() < b.rank();
          int max_a = 1, max_b = 1;
          for (unsigned i = 0; i < _nprops; i++) {
            if (_lines[i]) {
              max_a = std::max(max_a, a[i]);
              max_b = std::max(max_b, b[i]);
            }
          }
          return max_a < max_b;
        });
  } else if (mode == OrderingMode::Complexity) {
    // Complexity ordering: (depth+rank ASC, depth ASC, rank ASC, max
    // denominator power ASC, max numerator power ASC).  Stable, so seeds tying
    // on all five keys keep the generation order (depth-major, rank-minor).
    std::stable_sort(_seeds.begin(), _seeds.end(), _compare_complexity);
  }
  // OrderingMode::Normal: keep generation order, no sort.

  for (unsigned i = 0; i < _seeds.size(); ++i)
    _weights[_seeds[i]] = i;

  // _combinations is only a scratch table consumed to build _seeds/_weights; it
  // is never read afterwards, so release it before the FiniteFlow elimination.
  _combinations.clear();
}

std::string format_timing_row(const SectorTiming &t) {
  std::ostringstream os;
  os << t.id << "," << (t.expanded ? 1 : 0) << "," << std::fixed
     << std::setprecision(3) << (t.probeNanos / 1e9) << ","
     << (t.fullNanos / 1e9) << "\n";
  return os.str();
}

void ReductionContext::recordTiming(const SectorTiming &t) {
  accumulateTiming(t);
  std::lock_guard<std::mutex> lk(timingMutex);
  std::ofstream csv("sector_timing.csv", std::ios::app);
  if (csv.is_open())
    csv << format_timing_row(t);
}

void Sector::run_reduce_ff_n(const std::vector<IBPProtoFF> &ibps,
                             const std::vector<IBPProtoFFmod> &ibps_mod,
                             const std::vector<IBPProtoM> *ibps_sym,
                             const SectorConfig &cfg, ReductionContext &ctx) {
  if (ctx.targets[_id].empty() && ctx.uptargets[_id].empty()) {
    _print_progress(ctx);
    return;
  }

  // Smallest box covering the sector's targets; see initial_seed_bounds, and
  // SeedBound for the boundary rule built on top of it.  Keep it: the probe
  // below may relax the boundary once, and a sector's log reports both so the
  // probe's starting point stays visible.
  unsigned boxDepth, boxRank;
  std::tie(boxDepth, boxRank) = initial_seed_bounds(_id, ctx.targets[_id],
                                                    ctx.uptargets[_id]);
  _bound = SeedBound::initial(cfg.boundary_mode, boxDepth, boxRank);
  _initialBound = _bound;

  namespace fs = std::filesystem;
  std::string sectorpath =
      cfg.cache_dir + "/sectors/sector" + std::to_string(_id);

  // First run(s): skip all subsector integrals to check that the box covers every
  // integral the sector still needs.  A pass that reports an inadmissible seed
  // (SeedBound::exceeded -- past the top of the box, or through its depth cap under
  // that top) reports, per offending seed, which figure of the box refuses it
  // (SeedBound::refusal); widening that figure by one layer admits it, so the loop
  // repeats until a pass reports nothing.  A depth widening carries the Total cap
  // along only when the box top is what was reported (bit 4); a report the depth cap
  // made alone deepens the box and leaves the cap where it is.  The passes are cheap
  // next to the full one -- they carry no subsector integrals -- and the box they
  // settle on is what the full pass below reduces with.
  unsigned result = 0;
  unsigned probeRounds = 0;
  auto probeBegin = std::chrono::steady_clock::now();
  while (true) {
    _probeRounds = probeRounds + 1;
    clear();
    result = sector_reduction_ff_n(ibps, ibps_mod, ibps_sym, cfg, ctx, true);
    if (result == 0)
      break;
    ++probeRounds;
    if (probeRounds == kMaxRounds) {
      // Every widening so far has brought in a seed the reduction still could not
      // close on.  The bound is a heuristic input, not a guarantee, so stop here
      // and let the full pass -- and its own report -- speak for the sector.
      std::cerr << "warning: sector " << _id << ": probe still finds seeds"
                << " outside the box after " << probeRounds << " widenings"
                << " (depth " << _bound.depth << ", rank " << _bound.rank
                << (_bound.mode == BoundaryMode::Total
                        ? ", total " + std::to_string(_bound.limit)
                        : std::string())
                << ")\n";
      break;
    }
    _bound.widen_for(result);
  }
  bool probeExpanded = probeRounds > 0;
  auto probeEnd = std::chrono::steady_clock::now();
  if (probeExpanded)
    ctx.narrowingFail.fetch_add(1, std::memory_order_relaxed);
  else
    ctx.narrowingSuccess.fetch_add(1, std::memory_order_relaxed);

  // Second run: full computation with filtered subsector integrals.  The memory
  // gate + running-sector accounting happen inside sector_reduction_ff_n.  It
  // also discards the failed probe's cache, so the probe above leaves it in
  // place.
  auto fullBegin = std::chrono::steady_clock::now();
  fs::remove_all(sectorpath);
  clear();
  unsigned fullResult =
      sector_reduction_ff_n(ibps, ibps_mod, ibps_sym, cfg, ctx, false);
  auto fullEnd = std::chrono::steady_clock::now();
  if ((fullResult & 3) != 0) {
    // That a single expansion always suffices is what the two-pass scheme
    // assumes; if it ever fails to hold, the equations just collected are
    // incomplete, so report it rather than reduce silently with too narrow a
    // bound.  Bit 4 is the probe's repair hint, not a figure of the box, so it is
    // masked out here (and it cannot arrive alone -- a seed that reports it also
    // pins a hard cap).
    std::cerr << "warning: sector " << _id << ": full pass still needs a wider"
              << " bound (code " << (fullResult & 3) << ", depth " << _bound.depth
              << ", rank " << _bound.rank
              << (_bound.mode == BoundaryMode::Total
                      ? ", total " + std::to_string(_bound.limit - 1)
                      : std::string())
              << ")\n";
  }

  {
    SectorTiming t;
    t.id = _id;
    t.expanded = probeExpanded;
    t.probeNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       probeEnd - probeBegin)
                       .count();
    t.fullNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      fullEnd - fullBegin)
                      .count();
    ctx.recordTiming(t);
  }

  clear();
  _print_progress(ctx);
  // The FiniteFlow session/matrix were destroyed when sector_reduction_ff_n
  // returned and clear() released the seed buffers; ask glibc to return the
  // freed arena memory to the OS now so the process RSS (and hence the memory
  // gate) drops promptly instead of in batches.
  malloc_trim(0);
}

unsigned Sector::sector_reduction_ff_n(const std::vector<IBPProtoFF> &ibps,
                                       const std::vector<IBPProtoFFmod> &ibps_mod,
                                       const std::vector<IBPProtoM> *ibps_sym,
                                       const SectorConfig &cfg,
                                       ReductionContext &ctx,
                                       bool skip_subsectors) {
  namespace fs = std::filesystem;
  std::string sectorpath =
      cfg.cache_dir + "/sectors/sector" + std::to_string(_id);
  fs::create_directories(sectorpath);
  std::ofstream file(sectorpath + "/log");
  file << "sector " << _id << ":" << '\n';

  // Wait for system memory headroom BEFORE generating seeds.  A blocked sector
  // must not hold its seed/weight/combination tables: those are exactly the
  // "dead weight" that keeps MemAvailable below the reserve and starves the one
  // sector the escape hatch (runningSectors == 0) lets through.  The guard
  // releases on every return path, so probe iterations are gated too.
  SectorRunGuard guard(ctx);

  _generate_seeds(cfg.ordering_mode);

  std::vector<EquationFF> system;
  std::vector<std::pair<uint32, uint32>> eqrecord;

  // subsector integrals
  std::map<RawIntegral, unsigned int> subsector_map;
  unsigned int sub_int_id = _seeds.size();

  // create a new graph
  Session session;
  unsigned graphId = session.new_graph();
  Graph *graph = session.graph(graphId);

  // set input vars
  graph->set_input_vars(0);
  std::vector<unsigned> inputnodes;

  // prepare sparse solver data
  typedef NumericSparseSolverModData Data;
  std::unique_ptr<NumericSparseSolverMod> algptr(new NumericSparseSolverMod());
  std::unique_ptr<Data> data(new Data());
  auto &alg = *algptr;

  unsigned nvars = _seeds.size();
  unsigned nseeds = 0;
  unsigned nouter = 0;
  unsigned nitems = 0;
  unsigned nsubitems = 0;

  // first pass: discover subsector integrals
  time_stamp(file, " begin searching subsectors: ");
  for (unsigned itseed = 0; itseed < nvars; ++itseed) {
    if (_bound.admits(_seeds[itseed])) {
      ++nseeds;
      RawIntegral seed = _seeds[itseed];
      for (unsigned itibp = 0; itibp < ibps_mod.size(); ++itibp) {
        for (const auto &item : ibps_mod[itibp]) {
          RawIntegral integral = seed + item.first;
          ffint coeff(0);
          for (unsigned p = 0; p < _nprops; ++p)
            coeff += ffint(seed[p]) * item.second[p];
          coeff += item.second.back();
          if (coeff.is_zero())
            continue;
          if (!_weights.contains(integral)) {
            if (skip_subsectors ||
                _bound.filter_drops(integral, cfg.subsector_filter_n) ||
                _trivialsectors.contains(integral.sector()))
              continue;
            if (subsector_map.find(integral) == subsector_map.end()) {
              subsector_map[integral] = sub_int_id++;
            }
            nsubitems++;
          } else {
            nitems++;
          }
        }
      }
    } else
      ++nouter;
  }
  time_stamp(file, " finish searching subsectors: ");

  std::vector<RawIntegral> subInts;
  for (auto item : subsector_map) {
    subInts.emplace_back(item.first);
  }
  _order_subintegrals(subInts, cfg.ordering_mode);
  subsector_map.clear();
  for (unsigned i = 0; i < subInts.size(); ++i) {
    subsector_map[subInts[i]] = i;
  }
  unsigned nsubint = subInts.size();

  system.clear();
  eqrecord.clear();

  time_stamp(file, "\n begin generating system: ");
  for (unsigned itseed = 0; itseed < nvars; ++itseed) {
    if (_bound.admits(_seeds[itseed])) {
      RawIntegral seed = _seeds[itseed];
      for (unsigned itibp = 0; itibp < ibps_mod.size(); ++itibp) {
        EquationFF equation;
        // generate the ibp equation
        for (const auto &item : ibps_mod[itibp]) {
          RawIntegral integral = seed + item.first;
          // check if coefficient is zero
          ffint coeff(0);
          for (unsigned p = 0; p < _nprops; ++p)
            coeff += ffint(seed[p]) * item.second[p];
          coeff += item.second.back();
          if (coeff.is_zero())
            continue;

          unsigned weight = 0;
          if (!_weights.contains(integral)) {
            if (skip_subsectors ||
                _bound.filter_drops(integral, cfg.subsector_filter_n) ||
                _trivialsectors.contains(integral.sector()))
              continue;
            weight = subsector_map[integral] + nvars;
            equation.insert(weight, coeff);
          } else {
            weight = _weights[integral];
            equation.insert(nvars - 1 - _weights[integral], coeff);
          }
        }
        if (equation.empty())
          continue;
        equation.sort();

        system.emplace_back(std::move(equation));
        eqrecord.emplace_back(itseed, itibp);
      }
    }
  }
  time_stamp(file, " finish generating system: ");

  // Count raw equations for the full (second) pass only.
  if (!skip_subsectors)
    ctx.rawEquationsTotal.fetch_add(system.size(), std::memory_order_relaxed);

  // set target integrals
  for (const auto &integral : ctx.uptargets[_id])
    ctx.targets[_id].insert(integral);
  // Diagnostic snapshot of the list this sector reduces.  writeTargetList
  // closes its stream, so the file is complete on disk from here on, not only
  // once the sector finishes.
  writeTargetList(sectorpath + "/targets", ctx.targets[_id]);
  unsigned neededvars = ctx.targets[_id].size();
  file << '\n' << format_seed_box_line(_id, _bound, _initialBound)
       // How many probe passes it took to settle the bound: the driver widens the
       // box and probes again while a pass still reports a seed outside it, so
       // "probe rounds: 3" says the leading triple is two widenings past "probe
       // start".  The full pass reports the count of the probe passes before it.
       << "  probe rounds: " << _probeRounds << '\n';
  file << "\n integrals: " << nvars << " sub integrals: " << nsubint
       << '\n';
  file << " seeds: " << nseeds << "  outer: " << nouter << '\n';
  file << " original equations: " << system.size() << '\n';
  file << " items: " << nitems << "  sub items: " << nsubitems << '\n';
  auto neededVars = std::make_unique<unsigned[]>(neededvars);
  std::vector<RawIntegral> vtargets(ctx.targets[_id].begin(),
                                    ctx.targets[_id].end());
  for (unsigned i = 0; i < ctx.targets[_id].size(); ++i) {
    neededVars[i] = nvars - 1 - _weights[vtargets[i]];
  }

  // get system data
  time_stamp(file, "\n begin copying data: ");
  alg.rinfo.resize(system.size());
  alg.c.resize(system.size());
  for (unsigned i = 0; i < system.size(); ++i) {
    {
      unsigned csize = system[i].size();
      NumericSparseSolverMod::RowInfo &rinf = alg.rinfo[i];
      rinf.size = csize;
      rinf.cols.reset(new unsigned[csize]);
      alg.c[i].reset(new UInt[csize]);
      for (unsigned it = 0; it < csize; ++it) {
        rinf.cols.get()[it] = system[i][it];
        // ffint residue → UInt directly (already mod FiniteFlow's learn prime).
        alg.c[i][it] = system[i].coeff(it).value();
      }
    }
  }
  time_stamp(file, " finish copying data: ");

  // `system` has been copied into the sparse solver; release it now so the raw
  // equation system and the solver data do not overlap in peak memory.
  system.clear();
  system.shrink_to_fit();

  // -------------------------------------------------------------------------
  // 方程选取。给下游的是两样东西：必需主积分（needed_indep，列号）与选中的方程下标
  // （selected_eqs，指向 alg.rinfo/alg.c）。
  //
  // 两种引擎（settings.sector_elimination）：
  //   fflow（默认，旧行为）：FiniteFlow SparseLinearSolver 的 learn +
  //     mark_and_sweep_eqs（前向+反向整链闭包）。
  //   pyred：与全局阶段同一套 —— forward_eliminate/backward_eliminate 做 learn，
  //     前向闭包 + 数值充分性修复做选取（src/global_select.h），主积分用与全局阶段
  //     相同的公式（约化式右端的并集）。
  // needed_indep 一律是「按列升序」的列号（列越小越复杂），两种引擎同序。
  // -------------------------------------------------------------------------
  std::vector<std::size_t> needed_indep;
  std::vector<unsigned> selected_eqs;

  const unsigned ncols = nvars + nsubint;
  const uint64_t learn_prime = BIG_UINT_PRIMES[0];

  // pyred 分支的状态（fflow 分支不用）。sys_idx[j] = res_eqs 第 j 行来自
  // alg.rinfo/alg.c（= eqrecord）的哪一行 —— 选取结果要按它映射回去。
  std::vector<secred::ResEq> res_eqs;
  std::vector<uint32_t> sys_idx;
  secred::SubSolve pyred_learn;
  GlobalSelection pyred_sel;
  std::vector<unsigned> pyred_dep_targets;

  SparseLinearSolver *sls = nullptr;
  unsigned solverId = 0;

  if (cfg.sector_elimination == SectorElimination::Pyred) {
    // ---- pyred：与全局阶段同一套 learn（globalreduce.cpp 的 learn 三连）----
    time_stamp(file, "\n beging learning: ");
    // res_eqs 第 j 行 ← alg.rinfo/eqrecord 的第 sys_idx[j] 行。今天 res_eq_from_row
    // 丢不掉任何行（方程构造已滤掉空行，且 id = ncols-1-col 是单射，同 id 不会相消），
    // 所以 sys_idx 恒为 [0, 1, ...]；仍然显式映射，免得下游悄悄依赖这个不变量。
    std::vector<uint32_t> all_idx;
    res_eqs.reserve(alg.rinfo.size());
    sys_idx.reserve(alg.rinfo.size());
    for (unsigned i = 0; i < alg.rinfo.size(); ++i) {
      secred::ResEq re = secred::res_eq_from_row(
          alg.rinfo[i].cols.get(), alg.c[i].get(), alg.rinfo[i].size, ncols,
          learn_prime);
      if (!re.empty()) {
        res_eqs.push_back(std::move(re));
        sys_idx.push_back(i);
      }
    }
    all_idx.resize(res_eqs.size());
    for (uint32_t j = 0; j < all_idx.size(); ++j)
      all_idx[j] = j;
    secred::solve_res_subset(res_eqs, all_idx, learn_prime, pyred_learn,
                             /*want_fwd=*/true, /*want_bwd=*/true);
    time_stamp(file, " finish learning: ");

    // 主积分：与全局阶段同一公式（约化式右端 ∪ 本阶段约化不了的 target），
    // id 降序 → 翻成列号即升序，与 legacy 的 needed_indepvars() 同序。
    std::vector<unsigned> target_ids;
    target_ids.reserve(neededvars);
    for (unsigned i = 0; i < neededvars; ++i)
      target_ids.push_back(ncols - 1 - neededVars[i]);
    for (unsigned id : secred::sector_masters(pyred_learn, target_ids))
      needed_indep.push_back(ncols - 1 - id);
    for (unsigned id : target_ids)
      if (pyred_learn.solmap.find(id) != pyred_learn.solmap.end())
        pyred_dep_targets.push_back(id);
  } else {
    // ---- fflow（默认，旧行为，逐行未动）----
    alg.init(alg.c.size(), nvars + nsubint, neededVars.get(), neededvars, *data);

    // attach solver node
    solverId =
        graph->new_node(std::move(algptr), std::move(data), inputnodes.data());

    // set only homogeneous
    Algorithm *presol = session.algorithm(graphId, solverId);
    SparseLinearSolver &psls = *static_cast<SparseLinearSolver *>(presol);
    psls.only_homogeneous();
    session.invalidate_subctxt_alg_data(graphId, solverId);

    // set output node
    session.set_output_node(graphId, solverId);

    // solver learn
    time_stamp(file, "\n beging learning: ");
    Ret lret = session.learn(graphId);
    if (lret != SUCCESS) {
      std::cerr << "learn failed!" << std::endl;
      exit(1);
    }
    time_stamp(file, " finish learning: ");

    sls = static_cast<SparseLinearSolver *>(session.algorithm(graphId, solverId));
    needed_indep.assign(sls->needed_indepvars(),
                        sls->needed_indepvars() + sls->n_needed_indepvars());
  }

  // mark and sweep
  //
  // The narrowing-failure criterion: a needed independent variable that is a seed
  // the boundary does not admit means the box was too small (that seed generates no
  // equation, the reduction cannot close on it, and it stays a master).  Two things
  // make a seed the box generates inadmissible -- its quantity (depth in Box mode,
  // total power depth + rank in Total mode) landing past the top of the seed box,
  // or, in Total mode only, its depth reaching the depth hard cap while a large rank
  // figure keeps its total power under that top.  See SeedBound; the second one is
  // the deep, low-rank seed F234's sector 246 kept as a master.
  //
  // The return value names the figure of the box that has to grow: bit 1 the depth
  // (or, in Total mode, the cap that holds the box top down), bit 2 the rank, and
  // bit 4 the *repair* a depth widening has to use -- set when some reported seed
  // was past the box top, i.e. when the total power is what was wrong and the cap
  // therefore has to ride along at the top of the widened box (SeedBound::
  // widen_depth).  A depth report made by the depth cap alone leaves bit 4 clear
  // and deepens the box without moving the cap.  Seeds the rank cap refuses while
  // their total power is inside the box -- the wide, shallow "outer" integrals a
  // sector hands on as masters -- are still not reported: widening the box would
  // not admit them.
  unsigned refusalBits = 0;
  for (unsigned i = 0; i < needed_indep.size(); ++i) {
    if (needed_indep[i] >= nvars)
      continue;
    const RawIntegral &seed = _seeds[nvars - 1 - needed_indep[i]];
    if (!_bound.exceeded(seed))
      continue;
    refusalBits |= _bound.refusal(seed);
    if (_bound.past_top(seed))
      refusalBits |= 4;
  }
  if (refusalBits != 0)
    return refusalBits;

  // First run only checks depth/rank; skip master propagation and equation
  // collection.
  if (skip_subsectors)
    return 0;

  file << "\n masters: ";
  file << needed_indep.size() << '\n';
  for (unsigned i = 0; i < needed_indep.size(); ++i) {
    if (needed_indep[i] < nvars) {
      RawIntegral inte = _seeds[nvars - 1 - needed_indep[i]];
      {
        std::lock_guard<std::mutex> lock(ctx.mastersMutex);
        ctx.allMasters[_index].emplace_back(inte);
      }
      file << "\n  " << inte;
    } else {
      RawIntegral subint = subInts[needed_indep[i] - nvars];
      {
        std::lock_guard<std::mutex> lock(ctx.uptargetsMutex);
        ctx.uptargets[subint.sector()].insert(subint);
      }
      file << "\n  " << nsubint - 1 - needed_indep[i] + nvars << ": "
           << subint << " #" << subint.sector();
      subsector_map[subint] = -1;
    }
  }
  file << '\n';

  if (cfg.sector_elimination == SectorElimination::Pyred) {
    // ---- pyred 选取：前向闭包 + 数值充分性修复（与全局阶段同一个选择器）----
    std::unordered_map<unsigned, unsigned> master_index; // id -> 序号
    for (std::size_t j = 0; j < needed_indep.size(); ++j)
      master_index[ncols - 1 - needed_indep[j]] = j;

    pyred_sel = select_global_forward(
        pyred_learn.eq_index, pyred_learn.fwd_deps, pyred_learn.solmap,
        pyred_dep_targets, master_index,
        [&](const std::vector<uint32_t> &idx, secred::SubSolve &out) {
          secred::solve_res_subset(res_eqs, idx, learn_prime, out,
                                   /*want_fwd=*/true, /*want_bwd=*/false);
        });
    // 选取给出的是 res_eqs 下标，下游要的是 eqrecord/alg.rinfo 的行号。
    std::vector<unsigned> sel_idx;
    if (pyred_sel.fallback) {
      // 兜底：今天 mark_and_sweep_eqs 的语义（前向 ∪ 反向整链闭包），永不缺方程。
      std::cerr << "sector " << _id
                << ": forward selection fell back to the legacy closure\n";
      sel_idx = secred::legacy_closure(pyred_dep_targets, pyred_learn.fwd_deps,
                                       pyred_learn.bwd_deps,
                                       pyred_learn.eq_index);
    } else {
      sel_idx.assign(pyred_sel.needed_eq_idx.begin(),
                     pyred_sel.needed_eq_idx.end());
    }
    selected_eqs.reserve(sel_idx.size());
    for (unsigned j : sel_idx)
      selected_eqs.push_back(sys_idx[j]);
  } else {
    sls->mark_and_sweep_eqs(session.alg_data(graphId, solverId));
    session.invalidate_subctxt_alg_data(graphId, solverId);
    selected_eqs.assign(sls->indep_eqs(),
                        sls->indep_eqs() + sls->n_indep_eqs());
  }

  file << "\n remained equations: " << selected_eqs.size() << '\n';
  if (cfg.sector_elimination == SectorElimination::Pyred)
    file << " selection: forward (repair rounds " << pyred_sel.n_repair_rounds
         << ", +" << pyred_sel.n_repair_added << " added; fwd-only "
         << pyred_sel.n_fwd_only << "; fallback "
         << (pyred_sel.fallback ? "yes" : "no") << ")\n";

  // Regenerate each selected equation from its (seed, ibp) record and spill
  // it to the raw equation file (ctx.appendRawEquation) for the global
  // reduction. Subsector master integrals are forwarded by the
  // needed_indepvars() propagation above (the sweep result); the subsector-depth
  // filter (offset +1) keeps the full directly-generated subsector layer in the
  // columns, so nothing extra needs to be forwarded here.
  {
    std::vector<GlobalEquation> local_eqs;
    std::vector<RawIntegral> new_uptargets;
    local_eqs.reserve(selected_eqs.size());

    for (unsigned i = 0; i < selected_eqs.size(); ++i) {
      unsigned eqnum = selected_eqs[i];
      unsigned seed_idx = eqrecord[eqnum].first;
      unsigned ibp_idx = eqrecord[eqnum].second;
      const RawIntegral &seed = _seeds[seed_idx];

      GlobalEquation eq;
      if (ibps_sym) {
        // Signed/symbolic path: the elimination structure is decided on the
        // prime-substituted numeric system, but each surviving equation is
        // re-expressed with its full symbolic coefficient.  GiNaC is not
        // thread-safe, so the symbolic arithmetic is serialized.
        std::lock_guard<std::mutex> ginac_guard(ginac_mutex);
        for (const auto &item : (*ibps_sym)[ibp_idx]) {
          RawIntegral integral = seed + item.first;
          GiNaC::ex coeff;
          for (unsigned p = 0; p < _nprops; ++p)
            coeff += item.second[p] * seed[p];
          coeff += item.second.back();
          coeff = coeff.expand();
          if (coeff.is_zero())
            continue;

          std::ostringstream oss;
          oss << coeff;
          if (_weights.contains(integral)) {
            eq.emplace_back(integral, oss.str());
          } else if (!_trivialsectors.contains(integral.sector())) {
            eq.emplace_back(integral, oss.str());
            new_uptargets.push_back(integral);
          }
        }
      } else {
        for (const auto &item : ibps[ibp_idx]) {
          RawIntegral integral = seed + item.first;
          rational coeff;
          coeff.set_zero();
          for (unsigned p = 0; p < _nprops; ++p)
            coeff += rational(seed[p]) * item.second[p];
          coeff += item.second.back();
          if (coeff.is_zero())
            continue;

          if (_weights.contains(integral)) {
            eq.emplace_back(integral, coeff.str());
          } else if (!_trivialsectors.contains(integral.sector())) {
            eq.emplace_back(integral, coeff.str());
            new_uptargets.push_back(integral);
          }
        }
      }
      if (!eq.empty())
        local_eqs.push_back(std::move(eq));
    }

    if (!new_uptargets.empty()) {
      std::lock_guard lk(ctx.uptargetsMutex);
      for (const auto &integ : new_uptargets)
        ctx.uptargets[integ.sector()].insert(integ);
    }
    for (auto &eq : local_eqs)
      ctx.appendRawEquation(eq);
  }

  return 0;
}

void Sector::_print_progress(ReductionContext &ctx) {
  std::lock_guard<std::mutex> lock(ctx.progressMutex);
  ++ctx.completedSectors;
  std::cout << "  [ " << ctx.completedSectors << " / " << ctx.totalSectors
            << " ] sectors done, " << ctx.runningSectors.load()
            << " running" << std::endl;
}

void Sector::clear() {
  _combinations.clear();
  // `.clear()` only empties the containers; the underlying buffers stay at
  // their peak capacity (a large sector's _seeds can be hundreds of MB).  A
  // completed sector must actually release that memory, otherwise the peak
  // buffers of every finished sector stay resident for the whole run.
  _seeds.clear();
  _seeds.shrink_to_fit();
  _weights.clear();
  _weights.rehash(0);  // also drop the bucket array (nodes are freed by clear())
  _submasters.clear();
  _submasters.shrink_to_fit();
  _masters.clear();
  _masters.shrink_to_fit();
}

unsigned
select_dum_eq_from_file(const std::string &path,
                        const std::unordered_map<RawIntegral, unsigned> &id_map) {
  // Scan the raw spill file (one term per line "<integral>\t<coeff>", equations
  // separated by blank lines) and return the equation index whose largest
  // integral ID is smallest. Same selection rule as the former in-memory
  // select_dum_eq.
  std::ifstream in(path);
  unsigned best = 0, best_max = UINT_MAX;
  unsigned eq_idx = 0, cur_max = 0;
  bool have_term = false;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) {
      if (have_term && cur_max < best_max) {
        best_max = cur_max;
        best = eq_idx;
      }
      ++eq_idx;
      cur_max = 0;
      have_term = false;
      continue;
    }
    auto tab = line.find('\t');
    if (tab == std::string::npos)
      continue;
    RawIntegral integ = RawIntegral::from_text(line.substr(0, tab));
    auto it = id_map.find(integ);
    if (it != id_map.end()) {
      cur_max = std::max(cur_max, it->second);
      have_term = true;
    }
  }
  return best;
}