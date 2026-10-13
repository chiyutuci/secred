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
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <tuple>

#include "global_select.h"
#include "numeric_solver_mod.h"
#include "row_coeff.h"
#include "sector_engine.h"
#include "symmetry.h"

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

// The balance order is balance_less() (src/sector.h), shared with the global ID
// numbering.  Ascending = simplest first, the direction the column convention
// wants (a larger label is a smaller, more complex column).  The five keys do
// not form a total order — integrals that tie on all of them are ordered by
// whatever order the caller's input had, so both call sites must sort stably
// (the global numbering adds its own tie-break keys instead).
bool Sector::_compare_balance(const RawIntegral &i1, const RawIntegral &i2) {
  return balance_less(i1, i2);
}

// Order the discovered subsector integrals into column order: the most complex
// integral first, so it takes the smallest subsector column (nvars).  The mode
// picks the key: Balance uses the same five-key order as the seeds, Normal the
// canonical per-index _compare_subintegral.  Which subsector integral sits in
// the smallest column decides the elimination pivots exactly as it does for the
// seeds.  The balance key is not a total order, so that path sorts stably: ties
// keep the order the vector arrives in (the discovery builds it from a
// std::map, i.e. RawIntegral::operator<), which the final reverse then mirrors —
// tied integrals come back in the reverse of the arrival order.
void Sector::_order_subintegrals(std::vector<RawIntegral> &subInts,
                                 OrderingMode mode) {
  if (mode == OrderingMode::Balance)
    std::stable_sort(subInts.begin(), subInts.end(), _compare_balance);
  else
    std::sort(subInts.begin(), subInts.end(), _compare_subintegral);
  std::reverse(subInts.begin(), subInts.end());
}

void createJobFile(const std::string &filename) {
  std::ofstream file(filename);

  file << "jobs:" << std::endl;
  file << "  - reduce_user_defined_system:" << std::endl;
  file << "      input_system: {files: [\"system\"], config: false, otf: false}"
       << std::endl;
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
  // The three figures are the extents of the admitted seeds, not of the boundary:
  // a seed sits strictly inside both hard bounds, and a seed of this sector has
  // depth >= popcount(id), so the rank extent reads min(rank, limit - popcount(id))
  // - 1 whenever the cap bites and the total-power figure is the box corner
  // depth + rank - 2 at the initial cap.  "start:" repeats them for the box
  // initial_seed_bounds derived.
  auto append = [id](std::ostringstream &oss, const SeedBound &b) {
    const int depth = (int)b.depth;
    const int rank = (int)b.rank;
    const int limit = (int)b.limit;
    oss << " depth: " << std::min(depth, limit) - 1
        << " rank: " << std::min(rank, limit - (int)std::popcount(id)) - 1
        << " total: " << std::min(depth + rank - 2, limit - 1);
  };
  std::ostringstream oss;
  append(oss, bound);
  oss << "  start:";
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

  if (mode == OrderingMode::Balance) {
    // Balance ordering: (depth+rank ASC, depth ASC, rank ASC, max denominator
    // power ASC, max numerator power ASC).  Stable, so seeds tying on all five
    // keys keep the generation order (depth-major, rank-minor).
    std::stable_sort(_seeds.begin(), _seeds.end(), _compare_balance);
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
  // Sectors no request reaches never start: they get neither a round nor a cache
  // directory.  Keep this above everything else.
  if (ctx.targets[_id].empty() && ctx.uptargets[_id].empty()) {
    _print_progress(ctx);
    return;
  }

  // Smallest box covering the sector's targets; see initial_seed_bounds, and
  // SeedBound for the boundary rule built on top of it.  Keep it: a widening below
  // may relax the boundary once or more, and a sector's log reports both so the
  // starting point stays visible.
  unsigned boxDepth, boxRank;
  std::tie(boxDepth, boxRank) = initial_seed_bounds(_id, ctx.targets[_id],
                                                    ctx.uptargets[_id]);
  _bound = SeedBound::initial(boxDepth, boxRank);
  _initialBound = _bound;

  namespace fs = std::filesystem;
  std::string sectorpath =
      cfg.cache_dir + "/sectors/sector" + std::to_string(_id);

  // One round = one pass over the current box, with the per-round tables released
  // first (the matrix and solver data were destroyed inside the previous pass).
  auto run_round = [&] {
    clear();
    return sector_reduction_ff_n(ibps, ibps_mod, ibps_sym, cfg, ctx);
  };

  // Whether the box was widened at all, i.e. whether the sector settled on a box
  // past the one initial_seed_bounds returned.
  bool expanded = false;
  std::uint64_t probeNanos = 0;  // time in rounds whose work was thrown away
  std::uint64_t fullNanos = 0;   // time in the round that settled and spilled

  // Every round is a full pass, so the criterion that decides the box sees the
  // subsector columns too.  A pass that reports an inadmissible seed
  // (SeedBound::exceeded -- past the top of the seed box, or through its depth cap
  // under that top) spills nothing: it returns at the criterion, before the
  // masters/uptargets propagation and before the equations are appended, so the
  // next round starts from a clean directory and an unchanged ctx and only the box
  // differs.  Widening the reported figure by one layer admits that seed
  // (_bound.widen_for; a depth widening carries the cap along only when the box top
  // was what was reported, bit 4), and the loop repeats until a round reports
  // nothing -- that round is the one that settles and spills.
  for (unsigned round = 0;; ++round) {
    // A re-run can find a directory an earlier run left behind, and a reporting
    // round leaves its own partial files (targets, the seed-box line; see below).
    // The deletion is charged to the round that follows it.
    const auto roundBegin = std::chrono::steady_clock::now();
    fs::remove_all(sectorpath);
    _probeRounds = round + 1;
    const unsigned result = run_round();
    const std::uint64_t nanos =
        (std::uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - roundBegin)
            .count();
    const RoundAction action = round_action(round, result);
    if (action == RoundAction::Settle) {
      fullNanos = nanos;
      break;
    }
    probeNanos += nanos;
    if (action == RoundAction::GiveUp) {
      // Every widening so far has brought in a seed the reduction still could not
      // close on.  The bound is a heuristic input, not a guarantee, so stop here:
      // this round reported, so it settles nothing and the sector contributes no
      // masters, no uptargets and no equations.
      std::cerr << "warning: sector " << _id << ": still finds seeds outside the"
                << " box after " << kMaxRounds << " passes (depth " << _bound.depth
                << ", rank " << _bound.rank << ", total " << _bound.limit << ")\n";
      break;
    }
    expanded = true;
    _bound.widen_for(result);
  }

  if (expanded)
    ctx.narrowingFail.fetch_add(1, std::memory_order_relaxed);
  else
    ctx.narrowingSuccess.fetch_add(1, std::memory_order_relaxed);

  {
    SectorTiming t;
    t.id = _id;
    t.expanded = expanded;
    t.probeNanos = probeNanos;
    t.fullNanos = fullNanos;
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
                                       ReductionContext &ctx) {
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

  // the sector's matrix: rows are the generated equations, columns the seeds
  // (ascending column number = simpler) followed by the subsector integrals
  NumericSparseSolverMod alg;

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
            if (_bound.filter_drops(integral, cfg.subsector_filter_n) ||
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
            if (_bound.filter_drops(integral, cfg.subsector_filter_n) ||
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

  // Count raw equations of this round.
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
       // How many passes it took to settle the bound: the driver widens the box and
       // runs another pass while one still reports a seed outside it, so "rounds: 3"
       // says the leading triple is two widenings past "start".
       << "  rounds: " << _probeRounds << '\n';
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
  // Equation selection.  Two things go downstream: the needed masters
  // (needed_indep, as column numbers, ascending = most complex first) and the
  // selected equations (selected_eqs, indices into alg.rinfo/alg.c).
  // -------------------------------------------------------------------------
  std::vector<std::size_t> needed_indep;
  std::vector<unsigned> selected_eqs;

  const unsigned ncols = nvars + nsubint;
  const uint64_t learn_prime = BIG_UINT_PRIMES[0];

  // sys_idx[j] = the row of alg.rinfo/alg.c (= eqrecord) that row j of res_eqs
  // came from; the selection result is mapped back through it.
  std::vector<secred::ResEq> res_eqs;
  std::vector<uint32_t> sys_idx;
  secred::SubSolve ffred_learn;
  GlobalSelection ffred_sel;
  std::vector<unsigned> ffred_dep_targets;

  // Learn the system (globalreduce.cpp's learn): row j of res_eqs comes from row
  // sys_idx[j] of alg.rinfo/eqrecord, which the selection result is mapped back
  // through.  res_eq_from_row drops no row today -- the equation builder already
  // filters empty rows and id = ncols - 1 - col is injective -- so sys_idx is
  // always [0, 1, ...], but it is kept explicit so no downstream code comes to
  // depend on that.
  time_stamp(file, "\n beging learning: ");
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
  secred::solve_res_subset(res_eqs, all_idx, learn_prime, ffred_learn,
                           /*want_fwd=*/true, /*want_bwd=*/true);
  time_stamp(file, " finish learning: ");

  // Masters, by the global stage's formula: the RHS of every reduction union the
  // targets this stage cannot reduce.  id descending -> column ascending.
  std::vector<unsigned> target_ids;
  target_ids.reserve(neededvars);
  for (unsigned i = 0; i < neededvars; ++i)
    target_ids.push_back(ncols - 1 - neededVars[i]);
  for (unsigned id : secred::sector_masters(ffred_learn, target_ids))
    needed_indep.push_back(ncols - 1 - id);
  for (unsigned id : target_ids)
    if (ffred_learn.solmap.find(id) != ffred_learn.solmap.end())
      ffred_dep_targets.push_back(id);

  // mark and sweep
  //
  // A needed independent variable whose seed the boundary does not admit means the
  // box was too small: that seed generates no equation, the reduction cannot close
  // on it, and it stays a master (SeedBound names the two ways that happens).
  //
  // The return value names the figure that has to grow: bit 1 the depth, bit 2 the
  // rank, bit 4 the *repair* -- set when the total power was what was wrong, so the
  // cap must ride along at the top of the widened box (SeedBound::widen_depth).  A
  // depth report from the depth cap alone leaves bit 4 clear and deepens the box
  // without moving the cap.
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

  // Selection: forward closure + numeric sufficiency repair, the global stage's
  // own selector (global_select.h).
  std::unordered_map<unsigned, unsigned> master_index; // id -> position
  for (std::size_t j = 0; j < needed_indep.size(); ++j)
    master_index[ncols - 1 - needed_indep[j]] = j;

  ffred_sel = select_global_forward(
      ffred_learn.eq_index, ffred_learn.fwd_deps, ffred_learn.solmap,
      ffred_dep_targets, master_index,
      [&](const std::vector<uint32_t> &idx, secred::SubSolve &out) {
        secred::solve_res_subset(res_eqs, idx, learn_prime, out,
                                 /*want_fwd=*/true, /*want_bwd=*/false);
      });
  // The selection names res_eqs rows; downstream wants eqrecord/alg.rinfo rows.
  std::vector<unsigned> sel_idx;
  if (ffred_sel.fallback) {
    // Fallback: the forward union the full backward closure, which can never
    // miss an equation.
    std::cerr << "sector " << _id
              << ": forward selection fell back to the legacy closure\n";
    sel_idx = secred::legacy_closure(ffred_dep_targets, ffred_learn.fwd_deps,
                                     ffred_learn.bwd_deps,
                                     ffred_learn.eq_index);
  } else {
    sel_idx.assign(ffred_sel.needed_eq_idx.begin(),
                   ffred_sel.needed_eq_idx.end());
  }
  selected_eqs.reserve(sel_idx.size());
  for (unsigned j : sel_idx)
    selected_eqs.push_back(sys_idx[j]);

  file << "\n remained equations: " << selected_eqs.size() << '\n';
  file << " selection: forward (repair rounds " << ffred_sel.n_repair_rounds
       << ", +" << ffred_sel.n_repair_added << " added; fwd-only "
       << ffred_sel.n_fwd_only << "; fallback "
       << (ffred_sel.fallback ? "yes" : "no") << ")\n";

  // Regenerate each selected equation from its (seed, ibp) record and spill
  // it to the raw equation file (ctx.appendRawEquation) for the global
  // reduction. Subsector master integrals are forwarded by the master
  // propagation above (the sweep result); the subsector filter keeps the top
  // layer in the columns, so nothing extra needs to be forwarded here.
  {
    std::vector<GlobalEquation> local_eqs;
    std::vector<RawIntegral> new_uptargets;
    local_eqs.reserve(selected_eqs.size());

    // Symbolic coefficients (spec 4.3): a row is precompiled once, under the
    // lock, and every later seed is a lock-free integer evaluation.  A row the
    // precompiler cannot represent keeps the GiNaC path and its lock.
    // (row_coeff.h's names live in namespace secred; this file spells them the
    // way it already spells secred::ResEq.)
    secred::RowTable row_table;
    unsigned slow_eqs = 0;

    for (unsigned i = 0; i < selected_eqs.size(); ++i) {
      unsigned eqnum = selected_eqs[i];
      unsigned seed_idx = eqrecord[eqnum].first;
      unsigned ibp_idx = eqrecord[eqnum].second;
      const RawIntegral &seed = _seeds[seed_idx];

      GlobalEquation eq;
      if (ibps_sym) {
        // Signed/symbolic path: the elimination structure is decided on the
        // prime-substituted numeric system, but each surviving equation is
        // re-expressed with its full symbolic coefficient.
        if (!row_table.contains(ibp_idx)) {
          std::lock_guard<std::mutex> ginac_guard(ginac_mutex);
          row_table.add(ibp_idx, (*ibps_sym)[ibp_idx]);
        }
        bool via_ginac = false;
        const IBPProtoM &items = (*ibps_sym)[ibp_idx];
        for (unsigned j = 0; j < items.size(); ++j) {
          const RawIntegral integral = seed + items[j].first;
          std::string coeff_text;
          const secred::RowEval st = row_table.eval(ibp_idx, j, seed, coeff_text);
          if (st == secred::RowEval::Zero)
            continue;
          if (st == secred::RowEval::NotPrecompiled) {
            // The fallback keeps the previous behaviour exactly: GiNaC is not
            // thread-safe, so its arithmetic is serialized.
            if (!via_ginac) {
              via_ginac = true;
              ++slow_eqs;
            }
            std::lock_guard<std::mutex> ginac_guard(ginac_mutex);
            GiNaC::ex coeff;
            for (unsigned p = 0; p < _nprops; ++p)
              coeff += items[j].second[p] * seed[p];
            coeff += items[j].second.back();
            coeff = coeff.expand();
            if (coeff.is_zero())
              continue;
            std::ostringstream oss;
            oss << coeff;
            coeff_text = oss.str();
          }
          if (_weights.contains(integral)) {
            eq.emplace_back(integral, coeff_text);
          } else if (!_trivialsectors.contains(integral.sector())) {
            eq.emplace_back(integral, coeff_text);
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

    if (ibps_sym)
      file << "  symbolic coefficients: " << row_table.fast_rows
           << " rows precompiled, " << row_table.slow_rows
           << " via GiNaC, " << slow_eqs << " equations via GiNaC\n";

    // Sector symmetry relations (spec 5.1-5.3).  One equation per prescription
    // out of this sector, for every seed this round admitted -- the IBP seed set
    // (_bound.admits), which is what Kira generates its relations on (spec 5.2).
    // An image that is not a variable of this sector's own system is requested
    // from *its own* sector through the same uptargets push the IBP terms use;
    // the symmetry edge in _run_reduce_fine is what makes that sector run after
    // this one, and -- the edge putting `to` after this sector, every image
    // sector being a subset of `to` -- running after this sector is the only
    // thing needed.
    if (ctx.symmetries) {
      const std::vector<PropPrescription> &pres = ctx.symmetries->bySource(_id);
      if (!pres.empty()) {
        const auto trivial = [&](unsigned sector) {
          return _trivialsectors.contains(sector);
        };
        std::vector<GlobalEquation> sym_eqs;
        for (const PropPrescription &p : pres) {
          if (ibps_sym) {
            // The prescription's rows in the ring this sector works in: GiNaC
            // for a symbolic family, the pre-converted rationals otherwise
            // (spec 5.1).  GiNaC is not thread-safe, so its arithmetic takes the
            // same guard the IBP spill does.
            std::vector<std::vector<GiNaC::ex>> f_ex(p.f.rows(),
                                                     std::vector<GiNaC::ex>(p.f.cols()));
            std::vector<GiNaC::ex> c_ex(p.c.rows());
            for (unsigned i = 0; i < p.f.rows(); ++i)
              for (unsigned j = 0; j < p.f.cols(); ++j)
                f_ex[i][j] = p.f(i, j);
            for (unsigned i = 0; i < p.c.rows(); ++i)
              c_ex[i] = p.c(i, 0);
            std::lock_guard<std::mutex> ginac_guard(ginac_mutex);
            for (const RawIntegral &seed : _seeds) {
              if (!_bound.admits(seed))
                continue;
              std::vector<SymTerm<GiNaC::ex>> terms;
              if (!expand_prescription<GiNaC::ex>(p, f_ex, c_ex, seed, trivial, terms))
                continue;
              // Merge the image terms by integral and add the seed itself with
              // coefficient -1 (Kira's convention, spec 5.1): a seed the
              // correspondence fixes -- the all-ones seed of a sector, whose
              // in-sector lines all carry power one -- cancels to nothing here
              // and carries no equation.
              std::map<RawIntegral, GiNaC::ex> agg;
              for (const SymTerm<GiNaC::ex> &t : terms)
                agg[RawIntegral(t.powers)] += t.coeff;
              agg[seed] += GiNaC::ex(-1);
              GlobalEquation eq;
              for (const auto &[integ, coeff] : agg) {
                if (coeff.is_zero())
                  continue;
                std::ostringstream oss;
                oss << coeff;
                if (_weights.contains(integ)) {
                  eq.emplace_back(integ, oss.str());
                } else if (!_trivialsectors.contains(integ.sector())) {
                  eq.emplace_back(integ, oss.str());
                  new_uptargets.push_back(integ);
                }
              }
              if (!eq.empty())
                sym_eqs.push_back(std::move(eq));
            }
          } else {
            for (const RawIntegral &seed : _seeds) {
              if (!_bound.admits(seed))
                continue;
              std::vector<SymTerm<rational>> terms;
              if (!expand_prescription<rational>(p, p.f_num, p.c_num, seed, trivial, terms))
                continue;
              std::map<RawIntegral, rational> agg;
              for (const SymTerm<rational> &t : terms)
                agg[RawIntegral(t.powers)] += t.coeff;
              agg[seed] += rational(-1);
              GlobalEquation eq;
              for (const auto &[integ, coeff] : agg) {
                if (coeff.is_zero())
                  continue;
                if (_weights.contains(integ)) {
                  eq.emplace_back(integ, coeff.str());
                } else if (!_trivialsectors.contains(integ.sector())) {
                  eq.emplace_back(integ, coeff.str());
                  new_uptargets.push_back(integ);
                }
              }
              if (!eq.empty())
                sym_eqs.push_back(std::move(eq));
            }
          }
        }
        for (auto &eq : sym_eqs)
          ctx.appendRawEquation(eq);
      }
    }

    if (!new_uptargets.empty()) {
      std::lock_guard lk(ctx.uptargetsMutex);
      for (const auto &integ : new_uptargets)
        ctx.uptargets[integ.sector()].insert(integ);
      // A self relation's image is a permutation of an admitted seed, so every
      // SeedBound rule (all permutation-invariant) admits it too and it is a
      // variable of this sector's own system already (spec 5.3).  A self image
      // landing here would be a column with no equation anywhere, i.e. a
      // spurious master; it cannot happen, so say so loudly if it does.
      if (ctx.symmetries && ctx.symmetries->representative(_id) == _id)
        for (const auto &integ : new_uptargets)
          if (integ.sector() == _id)
            std::cerr << "sector " << _id
                      << ": symmetry image missing from the local system\n";
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