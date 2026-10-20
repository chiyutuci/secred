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
#include "spill_reader.h"
#include "symmetry.h"

using namespace fflow;

static std::mutex ginac_mutex;

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

bool Sector::_compare_balance(const RawIntegral &i1, const RawIntegral &i2) {
  return balance_less(i1, i2);
}

// Order sub-integrals within a sector.
void Sector::_order_subintegrals(std::vector<RawIntegral> &subInts,
                                 OrderingMode mode,
                                 const std::set<RawIntegral> *preferred) {
  if (mode == OrderingMode::Balance)
    std::stable_sort(subInts.begin(), subInts.end(), _compare_balance);
  else
    std::sort(subInts.begin(), subInts.end(), _compare_subintegral);
  std::reverse(subInts.begin(), subInts.end());

  if (preferred)
    std::stable_partition(subInts.begin(), subInts.end(),
                          [preferred](const RawIntegral &x) {
                            return !preferred->contains(x);
                          });
}

// Write the per-sector target list.
void writeTargetList(const std::string &path,
                     const std::set<RawIntegral> &targets) {
  std::ofstream file(path);
  for (const auto &integral : targets)
    file << integral << '\n';

}

std::pair<unsigned, unsigned>
initial_seed_bounds(unsigned id, const std::set<RawIntegral> &targets,
                    const std::set<RawIntegral> &uptargets) {
  unsigned rank = 1;
  unsigned depth = std::popcount(id);

  auto raise = [&](const std::set<RawIntegral> &requests) {
    for (const auto &integral : requests) {
      rank = std::max(rank, integral.rank() + 1);
      depth = std::max(depth, integral.depth() + 1);
    }
  };
  raise(targets);
  raise(uptargets);
  return {depth, rank};
}

unsigned max_seed_dots(const std::vector<RawIntegral> &integrals) {
  unsigned dots = 0;
  for (const RawIntegral &integral : integrals)
    dots = std::max(dots, integral.depth() - std::popcount(integral.sector()));
  return dots;
}

unsigned seed_dots_from_targets(const std::vector<RawIntegral> &targets) {
  return max_seed_dots(targets);
}

unsigned seed_dots_refinement(const SeedTruncation &truncation,
                              const std::vector<RawIntegral> &masters) {
  if (!truncation.active)
    return 0;
  const unsigned needed = max_seed_dots(masters);
  return needed > truncation.dots ? needed : 0;
}

std::vector<RawIntegral> seed_request_refinement(
    const std::vector<RawIntegral> &masters,
    const std::map<unsigned, std::unordered_set<RawIntegral>> &cutRequests) {
  std::vector<RawIntegral> offending;
  for (const RawIntegral &master : masters) {
    const auto sector = cutRequests.find(master.sector());
    if (sector != cutRequests.end() &&
        sector->second.find(master) != sector->second.end())
      offending.push_back(master);
  }
  return offending;
}

unsigned automatic_seed_level_offset(const std::vector<RawIntegral> &targets) {
  if (targets.empty())
    return kSeedLevelOffset;
  return seed_dots_from_targets(targets) == 0 ? 0u : kSeedLevelOffset;
}

SeedTruncation derive_seed_truncation(const std::vector<RawIntegral> &targets,
                                      unsigned top, unsigned level_offset) {
  SeedTruncation truncation;
  truncation.level_offset = level_offset;
  if (targets.empty())
    return truncation;
  truncation.active = true;
  unsigned s_max = 0;
  for (const RawIntegral &target : targets) {
    s_max = std::max(s_max, target.rank());
    truncation.dots =
        std::max(truncation.dots, target.depth() - std::popcount(target.sector()));
    unsigned &s_exc = truncation.s_exc[target.sector()];
    s_exc = std::max(s_exc, target.rank());
  }

  truncation.l =
      std::max(0, (int)std::popcount(top) - (int)s_max - (int)level_offset);
  return truncation;
}

std::pair<unsigned, unsigned>
truncated_seed_caps(unsigned id, unsigned depth, unsigned rank,
                    const SeedTruncation &truncation) {
  if (!truncation.active)
    return {depth, rank};
  const unsigned lines = std::popcount(id);
  const unsigned s_exc = truncation.s_exc.contains(id) ? truncation.s_exc.at(id) : 0u;

  const int keep_rank = std::max({1, (int)lines - truncation.l + 1, (int)s_exc});
  const unsigned cap_rank = (unsigned)std::max(0, keep_rank + 1);
  const unsigned cap_depth = lines + truncation.dots + 1;
  return {std::min(depth, cap_depth), std::min(rank, cap_rank)};
}

std::string format_seed_box_line(const SeedBound &bound,
                                 const SeedBound &initial) {

  auto append = [](std::ostringstream &oss, const SeedBound &b) {
    const int depth = (int)b.depth;
    const int rank = (int)b.rank;
    const int limit = (int)b.limit;
    oss << " depth: " << depth - 1
        << " rank: " << rank - 1
        << " total: " << std::min(depth + rank - 2, limit - 1);
  };
  std::ostringstream oss;
  append(oss, bound);

  if (bound.trunc_depth < bound.depth || bound.trunc_rank < bound.rank)
    oss << " trunc: depth " << (int)bound.trunc_depth - 1 << " rank "
        << (int)bound.trunc_rank - 1;
  oss << "  start:";
  append(oss, initial);
  return oss.str();
}

ProbeCriterion probe_criterion(const SeedBound &bound,
                               const std::vector<std::size_t> &needed_indep,
                               const std::vector<RawIntegral> &seeds,
                               unsigned nvars, bool collect,
                               const std::unordered_set<std::size_t> *scope) {
  ProbeCriterion out;
  for (const std::size_t col : needed_indep) {

    if (col >= nvars)
      continue;
    const RawIntegral &seed = seeds[nvars - 1 - col];
    if (collect)
      out.own_masters.push_back(seed);
    if (!bound.exceeded(seed))
      continue;

    if (scope != nullptr && scope->find(col) == scope->end()) {
      ++out.scope_skipped;
      if (collect)
        out.accepted.push_back(seed);
      continue;
    }
    out.refusal_bits |= bound.refusal(seed);

    out.refusal_bits |= 4;
  }
  return out;
}

std::string format_probe_report(unsigned round, const SeedBound &bound,
                                const ProbeCriterion &crit) {

  std::ostringstream oss;
  oss << "[probe] round " << round << " failed: depth: " << bound.depth
      << " rank: " << bound.rank << " limit: " << bound.limit
      << ", seed top: " << bound.seed_top() << '\n';

  unsigned refused = 0;
  for (const RawIntegral &m : crit.own_masters)
    if (bound.exceeded(m) &&
        std::find(crit.accepted.begin(), crit.accepted.end(), m) ==
            crit.accepted.end())
      ++refused;
  oss << "[probe] own masters: " << crit.own_masters.size()
      << ", refused: " << refused << ", accepted: " << crit.scope_skipped
      << ", refusal bits: " << crit.refusal_bits << '\n';

  for (const RawIntegral &m : crit.own_masters) {
    std::string verdict;
    if (bound.past_top(m))
      verdict += "PAST-TOP ";
    if (bound.past_depth_cap(m))
      verdict += "PAST-DEPTH-CAP ";
    if (bound.truncation_refused(m)) {
      if (m.depth() >= bound.trunc_depth)
        verdict += "PAST-TRUNC-DEPTH ";
      if (m.rank() >= bound.trunc_rank)
        verdict += "PAST-TRUNC-RANK ";
    }
    if (verdict.empty())
      verdict = "ok";
    else
      verdict.pop_back();

    if (std::find(crit.accepted.begin(), crit.accepted.end(), m) !=
        crit.accepted.end())
      verdict += " ACCEPTED";
    oss << "[probe]   " << m << "  depth: " << m.depth() << " rank: " << m.rank()
        << " total: " << m.depth() + m.rank() << "  " << verdict << '\n';
  }
  return oss.str();
}

// Enumerate the allowed depth/rank combinations.
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

// Generate the seeds of one sector.
void Sector::_generate_seeds(OrderingMode mode,
                             const std::set<RawIntegral> *preferred) {
  int lines = std::popcount(_id);
  int zeros = (int)_nprops - lines;
  int number = std::max(lines, zeros);

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

    std::stable_sort(_seeds.begin(), _seeds.end(), _compare_balance);
  }

  if (preferred)
    std::stable_partition(_seeds.begin(), _seeds.end(),
                          [preferred](const RawIntegral &x) {
                            return preferred->contains(x);
                          });

  for (unsigned i = 0; i < _seeds.size(); ++i)
    _weights[_seeds[i]] = i;

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

  _inputTargets = ctx.targets[_id];

  const auto targetBox = initial_seed_bounds(_id, ctx.targets[_id], {});
  const auto requestBox =
      initial_seed_bounds(_id, ctx.targets[_id], ctx.uptargets[_id]);
  if (requestBox != targetBox) {
    ctx.boxSpillRaised.fetch_add(1, std::memory_order_relaxed);
    ctx.boxSpillExtraDepth.fetch_add(requestBox.first - targetBox.first,
                                     std::memory_order_relaxed);
    ctx.boxSpillExtraRank.fetch_add(requestBox.second - targetBox.second,
                                    std::memory_order_relaxed);
  }

  const auto [boxDepth, boxRank] = requestBox;
  _bound = SeedBound::initial(boxDepth, boxRank);

  bool truncated = false;
  if (cfg.truncation && cfg.truncation->active) {
    std::tie(_bound.trunc_depth, _bound.trunc_rank) = truncated_seed_caps(
        _id, _bound.depth, _bound.rank, *cfg.truncation);
    truncated =
        _bound.trunc_depth < _bound.depth || _bound.trunc_rank < _bound.rank;
  }
  if (truncated)
    ctx.truncatedSectors.fetch_add(1, std::memory_order_relaxed);
  _requestExempt.clear();
  _bound.requestExempt = nullptr;
  if (truncated) {

    auto classify = [&](const std::set<RawIntegral> &requests) {
      for (const RawIntegral &request : requests) {
        if (!_bound.truncation_refused(request))
          continue;
        if (ctx.exemptCutRequests) {
          _requestExempt.insert(request);
        } else {
          std::lock_guard<std::mutex> lock(ctx.cutRequestsMutex);
          ctx.cutRequests[_id].insert(request);
        }
      }
    };
    classify(ctx.targets[_id]);
    classify(ctx.uptargets[_id]);
    if (!_requestExempt.empty())
      _bound.requestExempt = &_requestExempt;
  }
  bool truncationLifted = false;
  _initialBound = _bound;

  _probeReport.clear();

  namespace fs = std::filesystem;
  std::string sectorpath =
      cfg.cache_dir + "/sectors/sector" + std::to_string(_id);

  auto run_round = [&] {
    clear();
    return sector_reduction_ff_n(ibps, ibps_mod, ibps_sym, cfg, ctx);
  };

  bool expanded = false;
  bool widened = false;
  std::uint64_t probeNanos = 0;
  std::uint64_t fullNanos = 0;

  for (unsigned round = 0;; ++round) {

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

      std::cerr << "warning: sector " << _id << ": still finds seeds outside the"
                << " box after " << kMaxRounds << " passes (depth " << _bound.depth
                << ", rank " << _bound.rank << ", total " << _bound.limit << ")\n";
      break;
    }
    expanded = true;
    if ((result & 48u) && !truncationLifted) {
      truncationLifted = true;
      ctx.truncationLifted.fetch_add(1, std::memory_order_relaxed);

      std::lock_guard<std::mutex> lock(ctx.cutRequestsMutex);
      ctx.cutRequests[_id].clear();
    }
    const unsigned depthBefore = _bound.depth;
    const unsigned rankBefore = _bound.rank;
    const unsigned limitBefore = _bound.limit;
    _bound.widen_for(result);
    if (_bound.depth != depthBefore || _bound.rank != rankBefore ||
        _bound.limit != limitBefore)
      widened = true;
  }

  if (!_probeReport.empty()) {
    std::ofstream report(sectorpath + "/log", std::ios::app);
    if (report.is_open())
      report << _probeReport;
  }

  if (widened)
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

  malloc_trim(0);
}

// Reduce one sector on the numeric path.
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

  SectorRunGuard guard(ctx);

  _generate_seeds(cfg.ordering_mode, cfg.preferred);

  std::vector<EquationFF> system;
  std::vector<std::pair<uint32, uint32>> eqrecord;

  std::map<RawIntegral, unsigned int> subsector_map;
  unsigned int sub_int_id = _seeds.size();

  NumericSparseSolverMod alg;

  unsigned nvars = _seeds.size();
  unsigned nseeds = 0;
  unsigned nouter = 0;
  unsigned nitems = 0;
  unsigned nsubitems = 0;

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
  _order_subintegrals(subInts, cfg.ordering_mode, cfg.preferred);
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

        for (const auto &item : ibps_mod[itibp]) {
          RawIntegral integral = seed + item.first;

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

  ctx.rawEquationsTotal.fetch_add(system.size(), std::memory_order_relaxed);

  for (const auto &integral : ctx.uptargets[_id])
    ctx.targets[_id].insert(integral);

  writeTargetList(sectorpath + "/targets", ctx.targets[_id]);
  unsigned neededvars = ctx.targets[_id].size();
  file << '\n' << format_seed_box_line(_bound, _initialBound)

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

        alg.c[i][it] = system[i].coeff(it).value();
      }
    }
  }
  time_stamp(file, " finish copying data: ");

  system.clear();
  system.shrink_to_fit();

  std::vector<std::size_t> needed_indep;
  std::vector<unsigned> selected_eqs;

  const unsigned ncols = nvars + nsubint;
  const uint64_t learn_prime = BIG_UINT_PRIMES[0];

  std::vector<secred::ResEq> res_eqs;
  std::vector<uint32_t> sys_idx;
  secred::SubSolve ffred_learn;
  GlobalSelection ffred_sel;
  std::vector<unsigned> ffred_dep_targets;

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
                           true, true);
  time_stamp(file, " finish learning: ");

  std::vector<unsigned> target_ids;
  target_ids.reserve(neededvars);
  for (unsigned i = 0; i < neededvars; ++i)
    target_ids.push_back(ncols - 1 - neededVars[i]);
  for (unsigned id : secred::sector_masters(ffred_learn, target_ids))
    needed_indep.push_back(ncols - 1 - id);
  for (unsigned id : target_ids)
    if (ffred_learn.solmap.find(id) != ffred_learn.solmap.end())
      ffred_dep_targets.push_back(id);

  std::unordered_set<std::size_t> probe_scope;
  {
    std::vector<unsigned> input_ids;
    input_ids.reserve(_inputTargets.size());
    for (const RawIntegral &t : _inputTargets) {
      const auto w = _weights.find(t);
      if (w != _weights.end())
        input_ids.push_back(nsubint + w->second);
    }
    for (unsigned id : secred::sector_masters(ffred_learn, input_ids))
      probe_scope.insert(ncols - 1 - id);
  }

  const ProbeCriterion crit = probe_criterion(
      _bound, needed_indep, _seeds, nvars, cfg.sector_debug,
      cfg.probe_mode == ProbeMode::All ? nullptr : &probe_scope);
  if (crit.scope_skipped != 0)
    ctx.probeScopeSkipped.fetch_add(crit.scope_skipped,
                                    std::memory_order_relaxed);
  if (crit.refusal_bits != 0) {

    if (cfg.sector_debug)
      _probeReport += format_probe_report(_probeRounds, _bound, crit);
    return crit.refusal_bits;
  }

  file << "\n masters: ";
  file << needed_indep.size() << '\n';

  const std::vector<RawIntegral> sym_seeds =
      secred::symmetry_seed_masters(needed_indep, _seeds, nvars);
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

  std::unordered_map<unsigned, unsigned> master_index;
  for (std::size_t j = 0; j < needed_indep.size(); ++j)
    master_index[ncols - 1 - needed_indep[j]] = j;

  ffred_sel = select_global_forward(
      ffred_learn.eq_index, ffred_learn.fwd_deps, ffred_learn.bwd_deps,
      ffred_learn.solmap, ffred_dep_targets, master_index,
      [&](const std::vector<uint32_t> &idx, secred::SubSolve &out) {
        secred::solve_res_subset(res_eqs, idx, learn_prime, out,
                                 true, false);
      },
      64u, kRepairAlways);

  std::vector<unsigned> sel_idx;
  if (ffred_sel.fallback) {

    std::cerr << "sector " << _id
              << ": forward selection fell back to the legacy closure\n";
    sel_idx = secred::legacy_closure(ffred_dep_targets, ffred_learn.fwd_deps,
                                     ffred_learn.bwd_deps,
                                     ffred_learn.eq_index);
  } else {
    sel_idx.assign(ffred_sel.needed_eq_idx.begin(),
                   ffred_sel.needed_eq_idx.end());
  }

  const std::size_t n_selected_before = sel_idx.size();
  if (cfg.mandatory_list) {
    for (uint32_t r : secred::unreducible_subsector_rows(
             res_eqs, nsubint, master_index, ffred_learn.solmap))
      sel_idx.push_back(r);
    std::sort(sel_idx.begin(), sel_idx.end());
    sel_idx.erase(std::unique(sel_idx.begin(), sel_idx.end()), sel_idx.end());
  }
  const std::size_t n_mandatory_rows = sel_idx.size() - n_selected_before;

  selected_eqs.reserve(sel_idx.size());
  for (unsigned j : sel_idx)
    selected_eqs.push_back(sys_idx[j]);

  file << "\n remained equations: " << selected_eqs.size() << '\n';
  file << " selection: forward (repair rounds " << ffred_sel.n_repair_rounds
       << ", +" << ffred_sel.n_repair_added << " added; fwd closure "
       << ffred_sel.n_fwd_rows << " equations; mandatory " << n_mandatory_rows
       << "; fallback " << (ffred_sel.fallback ? "yes" : "no") << ")\n";

  {
    std::vector<GlobalEquation> local_eqs;
    std::vector<RawIntegral> new_uptargets;
    local_eqs.reserve(selected_eqs.size());

    const bool family_table = ctx.rowTable != nullptr;

    secred::RowTable local_table;
    const secred::RowTable *row_table = family_table ? ctx.rowTable : &local_table;
    unsigned slow_eqs = 0;

    for (unsigned i = 0; i < selected_eqs.size(); ++i) {
      unsigned eqnum = selected_eqs[i];
      unsigned seed_idx = eqrecord[eqnum].first;
      unsigned ibp_idx = eqrecord[eqnum].second;
      const RawIntegral &seed = _seeds[seed_idx];

      GlobalEquation eq;
      if (ibps_sym) {

        if (!family_table && !local_table.contains(ibp_idx)) {
          std::lock_guard<std::mutex> ginac_guard(ginac_mutex);
          local_table.add(ibp_idx, (*ibps_sym)[ibp_idx]);
        }
        bool via_ginac = false;
        const IBPProtoM &items = (*ibps_sym)[ibp_idx];
        for (unsigned j = 0; j < items.size(); ++j) {
          const RawIntegral integral = seed + items[j].first;
          std::string coeff_text;
          const secred::RowEval st = row_table->eval(ibp_idx, j, seed, coeff_text);
          if (st == secred::RowEval::Zero)
            continue;
          if (st == secred::RowEval::NotPrecompiled) {

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
      file << "  symbolic coefficients: " << row_table->fast_rows
           << " rows precompiled, " << row_table->slow_rows
           << " via GiNaC, " << slow_eqs << " equations via GiNaC\n";

    if (ctx.symmetries) {
      const std::vector<PropPrescription> &pres = ctx.symmetries->bySource(_id);
      if (!pres.empty()) {
        const auto trivial = [&](unsigned sector) {
          return _trivialsectors.contains(sector);
        };
        std::vector<GlobalEquation> sym_eqs;
        for (const PropPrescription &p : pres) {
          if (!p.has_numeric_rows) {

            std::vector<std::vector<GiNaC::ex>> f_ex(p.f.rows(),
                                                     std::vector<GiNaC::ex>(p.f.cols()));
            std::vector<GiNaC::ex> c_ex(p.c.rows());
            for (unsigned i = 0; i < p.f.rows(); ++i)
              for (unsigned j = 0; j < p.f.cols(); ++j)
                f_ex[i][j] = p.f(i, j);
            for (unsigned i = 0; i < p.c.rows(); ++i)
              c_ex[i] = p.c(i, 0);
            std::lock_guard<std::mutex> ginac_guard(ginac_mutex);

            for (const RawIntegral &seed : sym_seeds) {
              std::vector<SymTerm<GiNaC::ex>> terms;
              if (!expand_prescription<GiNaC::ex>(p, f_ex, c_ex, seed, trivial, terms))
                continue;

              std::vector<std::pair<RawIntegral, GiNaC::ex>> agg;
              agg.reserve(terms.size() + 1);
              for (const SymTerm<GiNaC::ex> &t : terms)
                agg.emplace_back(RawIntegral(t.powers), t.coeff);
              agg.emplace_back(seed, GiNaC::ex(-1));
              flat_merge(agg);
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
            for (const RawIntegral &seed : sym_seeds) {
              std::vector<SymTerm<rational>> terms;
              if (!expand_prescription<rational>(p, p.f_num, p.c_num, seed, trivial, terms))
                continue;
              std::vector<std::pair<RawIntegral, rational>> agg;
              agg.reserve(terms.size() + 1);
              for (const SymTerm<rational> &t : terms)
                agg.emplace_back(RawIntegral(t.powers), t.coeff);
              agg.emplace_back(seed, rational(-1));
              flat_merge(agg);
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
        ctx.appendRawEquations(sym_eqs);
      }
    }

    if (!new_uptargets.empty()) {
      std::lock_guard lk(ctx.uptargetsMutex);
      for (const auto &integ : new_uptargets)
        ctx.uptargets[integ.sector()].insert(integ);

      if (ctx.symmetries && ctx.symmetries->representative(_id) == _id)
        for (const auto &integ : new_uptargets)
          if (integ.sector() == _id)
            std::cerr << "sector " << _id
                      << ": symmetry image missing from the local system\n";
    }
    ctx.appendRawEquations(local_eqs);
  }

  return 0;
}

// Print the sector progress line.
void Sector::_print_progress(ReductionContext &ctx) {
  std::lock_guard<std::mutex> lock(ctx.progressMutex);
  ++ctx.completedSectors;
  std::cout << "  [ " << ctx.completedSectors << " / " << ctx.totalSectors
            << " ] sectors done, " << ctx.runningSectors.load()
            << " running" << std::endl;
}

void Sector::clear() {
  _combinations.clear();

  _seeds.clear();
  _seeds.shrink_to_fit();
  _weights.clear();
  _weights.rehash(0);
  _submasters.clear();
  _submasters.shrink_to_fit();
  _masters.clear();
  _masters.shrink_to_fit();
}
