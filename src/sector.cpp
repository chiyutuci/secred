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

#include "numeric_solver_mod.h"

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

std::string format_seed_box_line(unsigned id, unsigned depth, unsigned rank,
                                 unsigned initialDepth, unsigned initialRank) {
  // dot = excess depth of the box above the sector's own base layer, i.e.
  // depth - popcount(id): 0 means the box is exactly that base layer (the
  // smallest box initial_seed_bounds can return, where no seed generates
  // equations and the probe raises the bound).  "probe start" repeats the same
  // two figures for the box the probe pass ran with, before any raise; it equals
  // the leading pair when the probe did not raise.
  std::ostringstream oss;
  oss << " dot: " << (int)depth - (int)std::popcount(id) << " rank: " << rank - 1
      << "  probe start: dot: " << (int)initialDepth - (int)std::popcount(id)
      << " rank: " << initialRank - 1;
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
  int sum = std::max((int)_depth - std::popcount(_id), (int)_rank);
  _generate_combinations(number, sum);

  for (int depth = 0; depth <= _depth - std::popcount(_id); ++depth) {
    if (zeros != 0)
      for (int rank = 0; rank <= _rank; ++rank)
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

  // Smallest box covering the sector's targets; see initial_seed_bounds.  Keep
  // it: the probe below may raise the bounds once, and a sector's log reports
  // both so the probe's starting point stays visible.
  std::tie(_depth, _rank) = initial_seed_bounds(_id, ctx.targets[_id],
                                                ctx.uptargets[_id]);
  _initialDepth = _depth;
  _initialRank = _rank;

  namespace fs = std::filesystem;
  std::string sectorpath =
      cfg.cache_dir + "/sectors/sector" + std::to_string(_id);

  // First run: skip all subsector integrals to determine whether the initial
  // depth/rank (derived from the targets) are sufficient.  It runs exactly
  // once: where it reports the bounds too small, raising them once closes the
  // gap, so the full pass below runs directly instead of probing again.
  unsigned result;
  bool probeExpanded = false;
  auto probeBegin = std::chrono::steady_clock::now();
  clear();
  result = sector_reduction_ff_n(ibps, ibps_mod, ibps_sym, cfg, ctx, true);
  if (result != 0) {
    probeExpanded = true;
    if (result & 1)
      _depth++;
    if (result & 2)
      _rank++;
  }
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
  if (fullResult != 0) {
    // That a single expansion always suffices is what the two-pass scheme
    // assumes; if it ever fails to hold, the equations just collected are
    // incomplete, so report it rather than reduce silently with too narrow a
    // bound.
    std::cerr << "warning: sector " << _id << ": full pass still needs a wider"
              << " bound (code " << fullResult << ", depth " << _depth
              << ", rank " << _rank << ")\n";
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
    if (_seeds[itseed].depth() < _depth && _seeds[itseed].rank() < _rank) {
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
                (int)integral.depth() <
                    (int)_depth - (int)cfg.subsector_depth_offset ||
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
  std::sort(subInts.begin(), subInts.end(), _compare_subintegral);
  std::reverse(subInts.begin(), subInts.end());
  subsector_map.clear();
  for (unsigned i = 0; i < subInts.size(); ++i) {
    subsector_map[subInts[i]] = i;
  }
  unsigned nsubint = subInts.size();

  system.clear();
  eqrecord.clear();

  time_stamp(file, "\n begin generating system: ");
  for (unsigned itseed = 0; itseed < nvars; ++itseed) {
    if (_seeds[itseed].depth() < _depth && _seeds[itseed].rank() < _rank) {
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
                (int)integral.depth() <
                    (int)_depth - (int)cfg.subsector_depth_offset ||
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
  file << '\n' << format_seed_box_line(_id, _depth, _rank, _initialDepth,
                                       _initialRank)
       << '\n';
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

  alg.init(alg.c.size(), nvars + nsubint, neededVars.get(), neededvars, *data);

  // attach solver node
  unsigned solverId =
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

  // mark and sweep
  Algorithm *solver = session.algorithm(graphId, solverId);
  SparseLinearSolver &sls = *static_cast<SparseLinearSolver *>(solver);

  unsigned mdepth = 0, mrank = 0;
  for (unsigned i = 0; i < sls.n_needed_indepvars(); ++i) {
    if (sls.needed_indepvars()[i] < nvars) {
      unsigned weight = nvars - 1 - sls.needed_indepvars()[i];
      mdepth = std::max(mdepth, _seeds[weight].depth());
      mrank = std::max(mrank, _seeds[weight].rank());
    }
    if (mdepth >= _depth) {
      return 1;
    }
  }

  // First run only checks depth/rank; skip master propagation and equation
  // collection.
  if (skip_subsectors)
    return 0;

  file << "\n masters: ";
  file << sls.n_needed_indepvars() << '\n';
  for (unsigned i = 0; i < sls.n_needed_indepvars(); ++i) {
    if (sls.needed_indepvars()[i] < nvars) {
      RawIntegral inte = _seeds[nvars - 1 - sls.needed_indepvars()[i]];
      {
        std::lock_guard<std::mutex> lock(ctx.mastersMutex);
        ctx.allMasters[_index].emplace_back(inte);
      }
      file << "\n  " << inte;
    } else {
      RawIntegral subint = subInts[sls.needed_indepvars()[i] - nvars];
      {
        std::lock_guard<std::mutex> lock(ctx.uptargetsMutex);
        ctx.uptargets[subint.sector()].insert(subint);
      }
      file << "\n  " << nsubint - 1 - sls.needed_indepvars()[i] + nvars << ": "
           << subint << " #" << subint.sector();
      subsector_map[subint] = -1;
    }
  }
  file << '\n';

  sls.mark_and_sweep_eqs(session.alg_data(graphId, solverId));
  session.invalidate_subctxt_alg_data(graphId, solverId);

  file << "\n remained equations: " << sls.n_indep_eqs() << '\n';

  // Regenerate each selected equation from its (seed, ibp) record and spill
  // it to the raw equation file (ctx.appendRawEquation) for the global
  // reduction. Subsector master integrals are forwarded by the
  // needed_indepvars() propagation above (the sweep result); the subsector-depth
  // filter (offset +1) keeps the full directly-generated subsector layer in the
  // columns, so nothing extra needs to be forwarded here.
  {
    std::vector<GlobalEquation> local_eqs;
    std::vector<RawIntegral> new_uptargets;
    local_eqs.reserve(sls.n_indep_eqs());

    for (unsigned i = 0; i < sls.n_indep_eqs(); ++i) {
      unsigned eqnum = sls.indep_eqs()[i];
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