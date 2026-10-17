#include "family.h"
#include "BS_thread_pool.hpp"
#include "global_order.h"
#include "globalreduce.h"
#include "log_files.h"
#include "masters_file.h"
#include "ginac/factor.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <unordered_map>

// Distinct primes substituted for the free symbols (d, kinematic invariants)
// so that a symbolic family can still be reduced structurally with the fast
// numeric FiniteFlow path.  The substitution is generic: with overwhelming
// probability the linear dependencies among equations are unchanged, so the
// (seed, ibp) structure recorded by FiniteFlow carries over to the full
// symbolic equations fed to the reconstruction.
//
// The primes are kept around 2^31 rather than 2^63: primes near 2^63 make the
// coefficients ~19 decimal digits, which slows FiniteFlow's finite-field
// elimination by orders of magnitude (measured ~300x on F234).  ~2^31 primes
// are still large enough to be generic while keeping the coefficients small
// enough for the elimination to stay fast.
static constexpr const char *kPrimeSubs[] = {
    "2147483647", "2147483629", "2147483587", "2147483579",
    "2147483563", "2147483549", "2147483543", "2147483497",
    "2147483489", "2147483477", "2147483423", "2147483399",
    "2147483353", "2147483323", "2147483269", "2147483249",
    "2147483237", "2147483179", "2147483171", "2147483137",
    "2147483123", "2147483077", "2147483069", "2147483059",
};
constexpr unsigned kNumPrimeSubs = sizeof(kPrimeSubs) / sizeof(kPrimeSubs[0]);

GiNaC::symtab Family::symtab;

void Family::run_reduce(Reduce &reduce) {
  // Open the incremental equation spill. Each sector appends its selected
  // equations here (instead of accumulating them in RAM), so the global system
  // never stays resident through the sector phase.
  reduce._ctx.rawEquationsPath = _cache_dir + "/raw_equations";
  reduce._ctx.openSpill();

  // Fresh per-sector ATPS timing table: a header line here, then one row per
  // sector appended by the worker threads once its full pass has finished.
  {
    std::ofstream csv("sector_timing.csv", std::ios::trunc);
    if (csv.is_open())
      csv << "sector,expanded,probe_s,full_s\n";
  }

  // Free the raw GiNaC IBP prototypes (_ibp): they are consumed by the
  // _ibpFF/_ibpFFmod/_ibpM generators at init time and never touched by the
  // sector phase. Releasing them before the parallel reduction removes the
  // largest prototype representation from the sector-phase memory peak.
  _ibp.clear();
  _ibp.shrink_to_fit();

  const auto sector_t0 = std::chrono::steady_clock::now();
  if (_parallel_mode == "fine")
    _run_reduce_fine(reduce);
  else
    _run_reduce_layer(reduce);
  const auto sector_t1 = std::chrono::steady_clock::now();
  const double sectorSeconds =
      std::chrono::duration<double>(sector_t1 - sector_t0).count();
  // Echoed here as well as in the end-of-run Timing Statistics block: the global
  // phase that follows can run for hours, and this is the number that says how
  // much of the run the sector phase took.
  std::cout << "\n  [timing] sector reduction: "
            << secred::format_duration(sectorSeconds) << "\n";

  // Release the remaining IBP prototypes and the intra-sector propagation
  // state.  They are only consumed while generating sector equations; without
  // this they stay resident through the global phase and keep the process
  // at its sector-reduction peak.  (_ibp itself was already released before the
  // parallel passes.)
  _ibpFF.clear();
  _ibpFF.shrink_to_fit();
  _ibpFFmod.clear();
  _ibpFFmod.shrink_to_fit();
  _ibpM.clear();
  _ibpM.shrink_to_fit();
  reduce._ctx.uptargets.clear();

  // In-process global reduction (FiniteFlow for a numeric system, FireFly for a
  // symbolic one): the probe-narrowing and final reduction statistics are
  // consolidated into the Reduction Statistics block printed inside the
  // in-process reducer (after learn, before reconstruction), so the legacy
  // blocks below print only for a run that has no global system at all.
  const bool have_global_system = reduce._ctx.globalEquationsCount.load() > 0;

  // Probe narrowing statistics are final once the sector phase completes. Write
  // them to cache/sectors/sectors and echo them now — before the (possibly very
  // long) global reduction — so this feedback is not delayed.
  {
    unsigned ok = reduce._ctx.narrowingSuccess.load();
    unsigned fail = reduce._ctx.narrowingFail.load();
    unsigned total = ok + fail;
    double ratio = total ? 100.0 * static_cast<double>(ok) / total : 0.0;

    // ATPS pass totals are final at the same moment; report them alongside the
    // narrowing counts.  Formatted with snprintf so the stream state set for the
    // file is not relied on, and the numbers are echoed for both global tools
    // (the FiniteFlow path's consolidated block does not carry them).
    const double probeSeconds = reduce._ctx.probeNanos.load() / 1e9;
    const double fullSeconds = reduce._ctx.fullNanos.load() / 1e9;
    const double atpsSeconds = probeSeconds + fullSeconds;
    const double probeShare =
        atpsSeconds > 0 ? 100.0 * probeSeconds / atpsSeconds : 0.0;

    // Seed truncation (SeedTruncation): how many sectors the derived rule cut, and
    // how many of those the probe then made lift the exception -- the measure of
    // over-truncation, which costs one discarded pass per lifted sector.
    const unsigned trunc = reduce._ctx.truncatedSectors.load();
    const unsigned lifted = reduce._ctx.truncationLifted.load();

    // Seed-box diagnostics (2026-10-17): how far the recursive spill lifts the
    // box over the sector's own input targets.
    const unsigned spillRaised = reduce._ctx.boxSpillRaised.load();
    const unsigned spillDepth = reduce._ctx.boxSpillExtraDepth.load();
    const unsigned spillRank = reduce._ctx.boxSpillExtraRank.load();
    // Probe scope (fixed to the masters the input targets need): how many
    // out-of-box master reports the target-only walk accepted instead of
    // widening the box.
    const char *probeMode = "targets";
    const unsigned accepted = reduce._ctx.probeScopeSkipped.load();

    std::ofstream sfile(_cache_dir + "/sectors/sectors", std::ios::trunc);
    if (sfile.is_open())
      sfile << "narrowing succeeded: " << ok << "\n"
            << "narrowing failed:    " << fail << "\n"
            << "total sectors:       " << total << "\n"
            << "success ratio:       " << std::fixed << std::setprecision(1)
            << ratio << " %\n"
            << "probe seconds:       " << probeSeconds << "\n"
            << "full seconds:        " << fullSeconds << "\n"
            << "probe share:         " << probeShare << " %\n"
            << "truncated sectors:   " << trunc << "\n"
            << "truncation lifted:   " << lifted << "\n"
            << "truncation l:        " << reduce._truncation.l << "\n"
            << "truncation offset:   " << reduce._truncation.level_offset << "\n"
            << "box spill raised:    " << spillRaised << "\n"
            << "box spill extra:     " << spillDepth << " depth, " << spillRank
            << " rank\n"
            << "probe scope:         " << probeMode << "\n"
            << "probe accepted:      " << accepted << "\n";

    char atpsLine[128];
    std::snprintf(atpsLine, sizeof(atpsLine),
                  "\n  ATPS: probe %.1f s / full %.1f s / probe share %.1f %%",
                  probeSeconds, fullSeconds, probeShare);
    std::cout << atpsLine << std::endl;  // flush: precedes the global phase

    std::cout << "\n  Seed truncation: " << trunc << " sectors cut, " << lifted
              << " lifted by the probe (l = " << reduce._truncation.l
              << ", offset " << reduce._truncation.level_offset << ")"
              << std::endl;  // flush: same reason

    std::cout << "\n  Seed box: spill raised " << spillRaised
              << " sectors (+" << spillDepth << " depth, +" << spillRank
              << " rank over targets-only)" << std::endl;  // flush: same reason

    std::cout << "\n  Probe scope (" << probeMode << "): " << accepted
              << " out-of-box master reports accepted, not needed by the"
              << " input targets" << std::endl;  // flush: same reason

    // With the pure-numeric FiniteFlow global reduction the probe narrowing is
    // folded into the Reduction Statistics summary printed after learn.
    if (!have_global_system)
      std::cout << "\n Probe narrowing: succeeded " << ok << " / failed " << fail
                << " / total " << total << " ("
                << std::fixed << std::setprecision(1) << ratio << " % success)"
                << std::endl;  // flush so it precedes the global stage's output
  }

  // Send the spilled equations to the global stage (numeric and
  // signed paths alike).
  const auto global_t0 = std::chrono::steady_clock::now();
  GlobalReductionResult globalResult;
  if (reduce._ctx.globalEquationsCount.load() > 0)
    globalResult = _run_global_reduction(reduce);
  else
    reduce._ctx.discardSpill(); // nothing spilled: no file to read back
  const double globalSeconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - global_t0)
          .count();

  // Write the sector-local masters — the union of the per-sector masters, the
  // candidates the global step then selects from — to cache/masters_sector; the
  // basis the global reduction settles on is cache/masters_global.
  {
    std::vector<RawIntegral> sector_masters;
    for (const auto &sector : reduce._ctx.allMasters)
      sector_masters.insert(sector_masters.end(), sector.begin(), sector.end());
    secred::write_masters_file(_cache_dir + "/" + secred::kMastersSectorFile,
                               sector_masters);
  }

  // Count final master integrals: the basis the global phase settled on, which is
  // what it wrote to cache/masters_global.  With no global system there is no
  // global phase, and the sector-local union stands.
  unsigned masterCount = 0;
  if (have_global_system) {
    masterCount = globalResult.masters.size();
  } else {
    for (const auto &sector : reduce._ctx.allMasters)
      masterCount += sector.size();
  }

  // Print reduction statistics summary.  (Probe narrowing stats were already
  // printed before the global reduction; the in-process paths print their
  // consolidated Reduction Statistics inside the reducer itself instead.)
  if (!have_global_system) {
    const auto &ctx = reduce._ctx;
    std::vector<std::string> labels{"Master integrals",
                                    "Raw equations generated (full pass)",
                                    "Equations in global system"};
    const std::size_t w = secred::block_label_width(labels);

    std::vector<std::string> rows{
        secred::block_row("Master integrals", std::to_string(masterCount), w),
        secred::block_row("Raw equations generated (full pass)",
                          std::to_string(ctx.rawEquationsTotal.load()), w),
        secred::block_row("Equations in global system",
                          std::to_string(ctx.globalEquationsCount.load()), w)};

    std::cout << "\n"
              << secred::block_header("Reduction Statistics")
              << secred::block_rows(rows) << "\n"
              << secred::block_rule();
  }

  // Closing summary of where the run's wall-clock time went, with the global
  // stage split into its learn and reconstruction halves.
  {
    std::vector<secred::PhaseTiming> phases{
        {"Sector reduction", sectorSeconds},
        {"Global reduction", globalSeconds},
        {"Global learn", globalResult.learn_seconds},
        {"Reconstruction", globalResult.reconstruction_seconds}};
    std::cout << "\n"
              << secred::format_timing_statistics(
                     phases, secred::run_elapsed_seconds(),
                     process_peak_memory());
  }
}

void Family::_run_reduce_layer(Reduce &reduce) const {
  BS::thread_pool pool(_sthreads);
  SectorConfig scfg{_cache_dir,     _subsector_filter_n, _ordering_mode,
                    _sector_debug,  &reduce._preferred,  &reduce._truncation};

  // Hierarchical parallel execution: sectors are pre-sorted by descending
  // popcount(id), so iterating in order visits layers from highest to lowest.
  // All sectors within the same layer are submitted to the pool together;
  // pool.wait_for_tasks() is called at each layer boundary so that every
  // higher-layer sector has finished (and written its uptargets) before any
  // lower-layer sector reads them.
  unsigned current_layer = UINT_MAX;
  for (auto &sector : reduce._reduceSectors) {
    unsigned layer = std::popcount(sector.id());
    if (layer != current_layer) {
      if (current_layer != UINT_MAX)
        pool.wait_for_tasks();  // complete current layer before next layer starts
      current_layer = layer;
    }

    if (_symbols.size() == 0)
      pool.push_task([&sector, ibps = _ibpFF, ibps_mod = _ibpFFmod, scfg,
                      &ctx = reduce._ctx]() mutable {
        sector.run_reduce_ff_n(ibps, ibps_mod, nullptr, scfg, ctx);
      });
    else
      pool.push_task([&sector, ibps = _ibpFF, ibps_mod = _ibpFFmod,
                      ibps_sym = &_ibpM, scfg, &ctx = reduce._ctx]() mutable {
        sector.run_reduce_ff_n(ibps, ibps_mod, ibps_sym, scfg, ctx);
      });
  }
  pool.wait_for_tasks();  // complete the last (lowest) layer
}

void Family::_run_reduce_fine(Reduce &reduce) const {
  unsigned nsec = reduce._reduceSectors.size();

  // Map sector ID → index in _reduceSectors for O(1) parent lookup.
  std::unordered_map<unsigned, unsigned> id_to_idx;
  id_to_idx.reserve(nsec);
  for (unsigned i = 0; i < nsec; ++i)
    id_to_idx[reduce._reduceSectors[i].id()] = i;

  // For each sector j, its "parents" are the active sectors whose ID equals
  // j's ID with exactly one additional bit set.  When all parents of j have
  // finished, j can start.  We track this with an atomic counter per sector
  // and a children list on each parent.
  std::vector<std::vector<unsigned>> children(nsec);
  std::vector<unsigned> parent_count(nsec, 0);
  for (unsigned j = 0; j < nsec; ++j) {
    unsigned jid = reduce._reduceSectors[j].id();
    for (unsigned k = 0; k < _nprops; ++k) {
      if (jid & (1u << k)) continue;  // bit k already set → not a new parent
      auto it = id_to_idx.find(jid | (1u << k));
      if (it != id_to_idx.end()) {
        children[it->second].push_back(j);
        ++parent_count[j];
      }
    }
  }

  // One edge per cross prescription (spec 5.3): the image sector S' = rep(S) is
  // reduced *after* the sector S that emits the relation, because S requests the
  // image integrals from S' when it spills, and that request must be in place
  // before S' derives its box.  A representative has no cross prescription --
  // representative(s) == s is exactly "s has no out-edge" -- and restrictTo has
  // already removed every prescription with an endpoint outside the scheduled
  // set, so the lookup below always succeeds.  The edges are stars of depth one
  // (S -> rep(S), rep(rep(S)) == rep(S)), so they add no cycle to the DAG.
  for (unsigned j = 0; j < nsec; ++j) {
    unsigned sid = reduce._reduceSectors[j].id();
    unsigned rep = reduce._symmetries.representative(sid);
    if (rep == sid)
      continue;
    auto it = id_to_idx.find(rep);
    if (it == id_to_idx.end())
      continue;
    children[j].push_back(it->second);
    ++parent_count[it->second];
  }

  std::vector<std::atomic<unsigned>> pending(nsec);
  for (unsigned i = 0; i < nsec; ++i)
    pending[i].store(parent_count[i], std::memory_order_relaxed);

  SectorConfig scfg{_cache_dir,     _subsector_filter_n, _ordering_mode,
                    _sector_debug,  &reduce._preferred,  &reduce._truncation};

  std::atomic<unsigned> done_count{0};
  std::mutex done_mutex;
  std::condition_variable done_cv;

  // pool is declared before run_sector so that pool outlives run_sector:
  // destructor order is run_sector first (safe — all tasks done by then),
  // then pool.
  BS::thread_pool pool(_sthreads);

  std::function<void(unsigned)> run_sector = [&](unsigned i) {
    auto &sector = reduce._reduceSectors[i];

    if (_symbols.empty())
      sector.run_reduce_ff_n(_ibpFF, _ibpFFmod, nullptr, scfg, reduce._ctx);
    else
      sector.run_reduce_ff_n(_ibpFF, _ibpFFmod, &_ibpM, scfg, reduce._ctx);

    // Release memory ordering: our writes to ctx are visible to children.
    for (unsigned child : children[i]) {
      if (pending[child].fetch_sub(1, std::memory_order_acq_rel) == 1)
        pool.push_task([child, &run_sector] { run_sector(child); });
    }
    if (done_count.fetch_add(1, std::memory_order_acq_rel) + 1 == nsec)
      done_cv.notify_one();
  };

  for (unsigned i = 0; i < nsec; ++i)
    if (parent_count[i] == 0)
      pool.push_task([i, &run_sector] { run_sector(i); });

  {
    std::unique_lock<std::mutex> lock(done_mutex);
    done_cv.wait(lock, [&] { return done_count.load() == nsec; });
  }
  // Ensure the last notifying task has fully returned before run_sector
  // (captured by reference in pool tasks) goes out of scope.
  pool.wait_for_tasks();
}

GlobalReductionResult Family::_run_global_reduction(Reduce &reduce) const {
  auto &ctx = reduce._ctx;

  // Close the spill: all sectors have appended their equations, so its interned
  // tables go to disk and it hands back the distinct integrals, which the sort
  // below turns into global IDs (simpler first → lower ID, the masters-first
  // convention).
  std::vector<RawIntegral> all_integrals;
  ctx.finishSpill(all_integrals);

  // The order is fixed by src/global_order.h: sector popcount first, then either
  // the (depth, rank) key (Normal) or the balance key, with the sector mask and
  // the lexicographic order as the deterministic tie-breaks that make it a total
  // order.
  secred::order_global_integrals(all_integrals, _ordering_mode,
                                 &reduce._preferred);

  std::unordered_map<RawIntegral, unsigned> integral_id;
  integral_id.reserve(all_integrals.size());
  for (unsigned i = 0; i < all_integrals.size(); ++i)
    integral_id[all_integrals[i]] = i;

  // The global reduction, end to end in-process.  Which engine runs is fixed by
  // the system, not by a setting: a symbolic system (free symbols in the
  // coefficients) is reconstructed with FireFly's black box, which expresses the
  // coefficients as functions of those symbols; a numeric one goes through the
  // modular elimination + CRT path.  Neither is a fallback for the other -- a
  // system that reaches the wrong engine fails there rather than quietly being
  // reduced by the other.
  const bool symbolic = !_symbols.empty();
  const char *tool = symbolic ? "firefly" : "finiteflow";
  const std::string mastersGlobalPath =
      _cache_dir + "/" + secred::kMastersGlobalFile;

  GlobalReductionResult result;
  {
    // FireFly writes its log to a path it hardcodes in the working directory, so
    // the reconstruction is bracketed by the two calls that keep that file out of
    // the run's way and inside the cache directory (src/log_files.h).
    secred::reset_firefly_log();
    struct CollectFireflyLog {
      const std::string &cache_dir;
      ~CollectFireflyLog() { secred::collect_firefly_log(cache_dir); }
    } collect_log{_cache_dir};

    if (symbolic)
      result = secred::reduce_global_firefly_symbolic(
          ctx.rawEquationsPath, all_integrals, integral_id, reduce._rawTargets,
          _symbols, ctx.narrowingSuccess.load(), ctx.narrowingFail.load(),
          ctx.rawEquationsTotal.load(), ctx.globalEquationsCount.load(),
          mastersGlobalPath, _gthreads);
    else
      result = reduce_global_finiteflow(
          ctx.rawEquationsPath, all_integrals, integral_id, reduce._rawTargets,
          ctx.narrowingSuccess.load(), ctx.narrowingFail.load(),
          ctx.rawEquationsTotal.load(), ctx.globalEquationsCount.load(),
          mastersGlobalPath, _gthreads, _max_memory, _memory_reserve);
    std::filesystem::remove(ctx.rawEquationsPath);

    // A phase that did not complete has no table to write: stop the run before the
    // masters file and cache/reductions below turn a failure into an empty reduction.
    secred::require_global_reduction(result, tool);

    // Final reduction table under cache/reductions; blank line between targets.
    {
      std::ofstream of(_cache_dir + "/reductions");
      bool first = true;
      for (const auto &[integ, coeff] : result.reductions) {
        if (coeff == "-1") {
          if (!first)
            of << "\n";
          of << integ << "*(-1)\n";
          first = false;
        } else {
          of << integ << "*(" << coeff << ")\n";
        }
      }
    }
  }
  return result;
}


Family::Family(const YAML::Node &config) {
  if (!config["family"])
    throw std::runtime_error("family not found");
  YAML::Node familyConfig = config["family"];

  // check config file keys
  if (!familyConfig["name"])
    throw std::runtime_error("family name not found");
  if (!familyConfig["internals"])
    throw std::runtime_error("family loop momenta not found");
  if (!familyConfig["externals"])
    throw std::runtime_error("family external momenta not found");
  if (!familyConfig["propagators"])
    throw std::runtime_error("family propagators not found");
  if (!familyConfig["sps_rules"])
    throw std::runtime_error("family scalar product rules not found");

  // family name
  _name = familyConfig["name"].as<std::string>();
  // dimension of the family
  if (familyConfig["dimension"]) {
    GiNaC::parser parser;
    _dimension = parser(familyConfig["dimension"].as<std::string>());
  } else
    _dimension = GiNaC::possymbol("d");
  if (!is_a<GiNaC::numeric>(_dimension)) {
    _symbols.emplace_back("d");
    _dimension = _symbols.back();
    symtab["d"] = _dimension;
  }
  // loop momenta
  for (const auto &sym :
       familyConfig["internals"].as<std::vector<std::string>>()) {
    if (symtab.contains(sym))
      throw std::runtime_error("symbol " + sym + " defined more than once");
    _internals.emplace_back(sym);
    symtab[sym] = _internals.back();
  }
  // external momenta
  for (const auto &sym :
       familyConfig["externals"].as<std::vector<std::string>>()) {
    if (symtab.contains(sym))
      throw std::runtime_error("symbol " + sym + " defined more than once");
    _externals.emplace_back(sym);
    symtab[sym] = _externals.back();
  }
  // invariants
  if (familyConfig["invariants"] && !familyConfig["invariants"].IsNull()) {
    std::string one;
    if (familyConfig["invar_one"] && !familyConfig["invar_one"].IsNull())
      one = familyConfig["invar_one"].as<std::string>();
    for (const auto &inv :
         familyConfig["invariants"]
             .as<std::vector<std::pair<std::string, unsigned>>>()) {
      if (symtab.contains(inv.first))
        throw std::runtime_error("symbol " + inv.first +
                                 " defined more than once");
      _invariants.emplace_back(inv.first, inv.second);
      symtab[inv.first] = _invariants.back().first;
      if (inv.first != one)
        _symbols.emplace_back(_invariants.back().first);
    }
  }
  // the invariant set to one
  if (familyConfig["invar_one"] && !familyConfig["invar_one"].IsNull()) {
    auto one = familyConfig["invar_one"].as<std::string>();
    if (!symtab.contains(one))
      throw std::runtime_error("the invariant set to one is not valid");
    _one.append(symtab[one] == 1);
  }

  GiNaC::parser reader(symtab);

  _nints = _internals.size();
  _nexts = _externals.size();
  _nprops = _nints * _nexts + _nints * (_nints + 1) / 2;
  unsigned nsps = _nexts * (_nexts + 1) / 2;

  // invariant scalar products
  if (familyConfig["sps_rules"].size() != nsps)
    throw std::runtime_error(
        "number of scalar product rules does not match the topology");
  for (const auto &sp :
       familyConfig["sps_rules"].as<std::vector<std::vector<std::string>>>()) {
    if (sp.size() != 3 || !symtab.contains(sp[0]) || !symtab.contains(sp[1]))
      throw std::runtime_error(
          "scalar product rules contains invalid expressions");
    _spsRules.append(reader(sp[0]) * reader(sp[1]) == reader(sp[2]).subs(_one));
  }

  // propagators
  if (familyConfig["propagators"].size() != _nprops)
    throw std::runtime_error(
        "number of propagators does not match the topology");
  for (const auto &prop :
       familyConfig["propagators"]
           .as<std::vector<std::pair<std::string, std::string>>>()) {
    // Keep the flow as well: the reduction needs only D_k, but the symmetry
    // detector needs q_k, and q_k is not recoverable from D_k.
    _propMomenta.emplace_back(reader(prop.first));
    _propagators.emplace_back(
        (pow(reader(prop.first), 2) - pow(reader(prop.second), 2))
            .expand()
            .subs(_spsRules, GiNaC::subs_options::algebraic)
            .subs(_one)
            .expand());
  }

  // cuts
  if (familyConfig["cut"])
    _cut = familyConfig["cut"].as<std::vector<unsigned>>();
  _ncut = std::accumulate(_cut.begin(), _cut.end(), 0);

  // settings (top-level, optional)
  if (config["settings"]) {
    const YAML::Node &s = config["settings"];
    // Settings that used to exist and no longer do.  Each would otherwise be
    // ignored in silence, leaving a config asking for one thing and a run doing
    // another (a stale global_tool would pick the other global engine, a stale
    // threads would leave the thread count at its default), so they are refused
    // with the name of what replaced them.
    for (const auto &[key, replacement] :
         std::initializer_list<std::pair<const char *, const char *>>{
             {"global_tool",
              "the global tool is fixed: a symbolic system is reconstructed with "
              "FireFly, a numeric one with FiniteFlow"},
             {"threads", "use settings.sthreads"},
             {"global_finiteflow_threads", "use settings.gthreads"},
             {"probe_mode",
              "the probe scope is fixed to the input targets: only a master the "
              "sector's input targets need may widen its box"}}) {
      if (s[key])
        throw std::runtime_error("settings." + std::string(key) +
                                 " was removed; " + replacement);
    }
    if (s["cache_dir"])       _cache_dir       = s["cache_dir"].as<std::string>();
    if (s["sthreads"])        _sthreads        = s["sthreads"].as<unsigned>();
    // gthreads unset: the global phase uses the sector phase's thread count.
    _gthreads = s["gthreads"] ? s["gthreads"].as<unsigned>() : _sthreads;
    if (s["parallel_mode"]) {
      _parallel_mode = s["parallel_mode"].as<std::string>();
      if (_parallel_mode != "layer" && _parallel_mode != "fine")
        throw std::runtime_error(
            "parallel_mode must be 'layer' or 'fine', got '" + _parallel_mode + "'");
    }
    // subsector_filter_n: offset n of the subsector filter -- a subsector
    // integral is kept when its total power lies within n layers of the seed
    // box's top (SeedBound::seed_top); n = 0 therefore drops every subsector
    // integral.
    if (s["subsector_filter_n"])
      _subsector_filter_n = s["subsector_filter_n"].as<unsigned>();
    // seed_level_offset: how many levels the derived Kira truncation level l is
    // relaxed by (derive_seed_truncation).  Larger relaxes more, generating more
    // equations and cutting fewer seeds.  Unset, the rule chooses between the
    // default and 0 by reading the targets (automatic_seed_level_offset), so the
    // explicit/absent distinction is recorded.
    if (s["seed_level_offset"]) {
      _seed_level_offset = s["seed_level_offset"].as<unsigned>();
      _seed_level_offset_set = true;
    }
    // sector_debug: a diagnostic -- a pass that reports its seed box as too small
    // leaves its own masters, the figures the criterion read and the verdicts it
    // reached in the sector log, so a widening (or a sector given up on) can be
    // checked by hand.  It changes nothing about the reduction.
    if (s["sector_debug"]) _sector_debug = s["sector_debug"].as<bool>();
    // ordering_mode: integral order.  "balance" (default) orders the per-sector
    // seeds and subsector columns, and the global IDs (the global numbering, see
    // order_global_integrals below); "normal" keeps the generation order.
    if (s["ordering_mode"]) {
      std::string om = s["ordering_mode"].as<std::string>();
      if      (om == "normal")  _ordering_mode = OrderingMode::Normal;
      else if (om == "balance") _ordering_mode = OrderingMode::Balance;
      else
        throw std::runtime_error(
            "ordering_mode must be 'normal' or 'balance', got '" + om + "'");
    }
    // max_memory is specified in GB and stored internally in bytes. It is the
    // allowed total: sectors are admitted only while the process stays under
    // max_memory - memory_reserve (set it smaller to simulate less RAM).
    if (s["max_memory"])
      _max_memory = (std::uint64_t)(s["max_memory"].as<double>() * 1024 * 1024 * 1024);
    // memory_reserve: system memory (in GB) kept free during parallel sectors.
    if (s["memory_reserve"])
      _memory_reserve = (std::uint64_t)(s["memory_reserve"].as<double>() * 1024 * 1024 * 1024);
    // symmetry: detect the sector symmetry relations and use them in the
    // per-sector reduction (default true; spec 5.6/5.10).  With false no table
    // is built, nothing is dumped and the reduction is the symmetry-free one.
    if (s["symmetry"]) _symmetry = s["symmetry"].as<bool>();
  }

  std::cout << "\n #0.0   Parsing config file finished." << std::endl;

  _symIndices = generate_symbols("a", _nprops);
  _symProps = generate_symbols("D", _nprops);
}

void Family::init() {
  std::cout << "\n #0.1   Initializing integral family...\n";

  namespace fs = std::filesystem;
  if (fs::exists(_cache_dir))
    fs::remove_all(_cache_dir);
  fs::create_directories(_cache_dir);

  _compute_sps();
  _compute_symanzik();

  std::cout << "\n #0.1   Initializing integral family "
               "finished.\n";

  std::cout << "\n #0.2   Generating IBP relations...\n";
  _generate_ibp();
  std::cout << "\n #0.2   Generating IBP relations finished.\n";
}

void Family::init_reduce(const YAML::Node &config, Reduce &reduce) const {
  reduce._symbols = _symbols;
  reduce._symIndices = _symIndices;
  reduce._cut = _cut;
  reduce._ncut = _ncut;
  reduce._cache_dir = _cache_dir;
  reduce._ctx.maxMemory = _max_memory;
  reduce._ctx.memoryReserve = _memory_reserve;

  std::cout << "\n #0.3   Collecting target integrals...\n";
  if (!config["targets"])
    throw std::runtime_error("reduce targets not found");
  reduce._rawTargets = config["targets"].as<std::vector<RawIntegral>>();

  // find the top sector
  reduce._nprops = _nprops;
  reduce._seed_level_offset = _seed_level_offset;
  reduce._seed_level_offset_set = _seed_level_offset_set;
  std::vector<bool> hasLine(_nprops, false);
  for (const auto &integral : reduce._rawTargets) {
    if (integral.size() != _nprops) {
      std::stringstream ss;
      ss << integral;
      throw std::runtime_error("target integral " + ss.str() + " is invalid");
    }
    for (unsigned i = 0; i < _nprops; ++i) {
      if (integral[i] > 0 && !hasLine[i])
        hasLine[i] = true;
    }
    unsigned tarSector = integral.sector();
    reduce._targetSectors.insert(tarSector);
  }
  for (unsigned i = 0; i < _nprops; ++i)
    if (hasLine[i])
      reduce._top |= 1 << i;

  // The optional `preferred` list, at the same level as `targets` and with the
  // same syntax.  It only biases the master choice: an entry that takes part in
  // this reduction is ordered as its sector's simplest integral (see
  // SectorConfig::preferred), while an entry that never appears -- dropped by the
  // subsector filter, its sector not reached, or the entry reducible itself --
  // is ignored.  Nothing here extends the reduction, so the scheduled sectors
  // stay a function of `targets` alone.  A wrong index count is a malformed
  // input and throws, exactly as a wrong-sized target does.  `preferred:` with an
  // empty value asks for nothing, i.e. it is the absent case; any other value
  // that is not a list of index vectors -- a scalar, a map, a list of scalars --
  // makes the conversion below throw.
  if (config["preferred"] && !config["preferred"].IsNull()) {
    for (const auto &integral :
         config["preferred"].as<std::vector<RawIntegral>>()) {
      if (integral.size() != _nprops) {
        std::stringstream ss;
        ss << integral;
        throw std::runtime_error("preferred integral " + ss.str() +
                                 " is invalid");
      }
      reduce._preferred.insert(integral);
    }
  }
  std::cout << "\n #0.3   Collecting target integrals "
               "finished.\n";

  std::cout << "\n #0.4   Searching trivial sectors...\n";
  _search_trivial_sectors(reduce);
  std::cout << "\n #0.4   Searching trivial sectors "
               "finished.\n";

  // Sector symmetries (spec 4.2): the only slot where both the non-trivial flags
  // are known and no thread has started.  With the feature off the table stays
  // empty and every consumer sees nothing to do.
  if (_symmetry) {
    std::cout << "\n #0.4   Searching sector symmetries...\n";
    reduce._symmetries =
        SymmetryTable::find(symmetry_inputs(), reduce._sectors, reduce._top);
    if (_parallel_mode == "layer") {
      // A layer is submitted whole, so a same-layer edge cannot be honoured
      // (spec 5.5).
      reduce._symmetries.dropCross();
      std::cout << " #0.4   parallel_mode layer: cross-sector symmetry relations "
                   "are dropped, self relations are kept.\n";
    }
    // Spec 5.1: the rational rows exist for every prescription whose entries are
    // numbers -- all of them on a numeric family, most of them on a symbolic
    // one.  Filled here, before prepare_sectors starts any thread: the sector
    // stage reads them without a lock.
    reduce._symmetries.try_to_numeric_all();
    std::cout << "\n #0.4   Searching sector symmetries finished.\n";
  }

  // The settling round's precompiled rows (spec 4.3).  A row's coefficients do
  // not depend on the seed, so they do not depend on the sector either: the
  // table is the family's, built here -- single-threaded, before
  // prepare_sectors starts any sector thread -- and read-only from then on.
  // Built per sector it was the same GiNaC work done once per sector, each
  // under ginac_mutex, which serialised the settling rounds against each other.
  if (!_symbols.empty()) {
    reduce._rowTable.build_all(_ibpM);
    reduce._ctx.rowTable = &reduce._rowTable;
  }

  reduce.prepare_sectors();
}

void Family::print() const {

  std::cout << "\n----------------- Family Info "
               "------------------\n"
            << "\n  Topology: " << _name << "   Dimension: " << _dimension
            << "\n\n  Internals:";

  for (const auto &sym : _internals)
    std::cout << " " << sym;

  std::cout << "\n\n  Externals:";
  for (const auto &sym : _externals)
    std::cout << " " << sym;

  std::cout << "\n\n  Invariants:";
  for (const auto &inv : _invariants)
    std::cout << " " << inv.first;

  std::cout << "\n\n  Propagators: " << _nprops
            << "   IBP relations: " << _ibp.size();

  std::cout << std::endl;
}

void Family::_generate_ibp() {
  GiNaC::ex coeff, coeffD;
  std::map<RawIntegral, GiNaC::ex> equation;

  // i: l_i in derivatives
  // j: l_j or p_j in nominators
  // s: prop_s in denominators
  for (size_t i = 0; i < _nints; ++i) {
    for (size_t j = 0; j < _nints + _nexts; ++j) {
      equation.clear();
      // cases of i == j generate g^u_u = D
      if (i == j)
        equation[RawIntegral(_nprops, 0)] += _dimension;
      for (size_t s = 0; s < _nprops; ++s) {
        if (j < _nints)
          coeff = (-_symIndices[s]) * _internals[j] *
                  GiNaC::diff(_propagators[s], _internals[i]);
        else
          coeff = (-_symIndices[s]) * _externals[j - _nints] *
                  GiNaC::diff(_propagators[s], _internals[i]);
        coeff = coeff.expand()
                    .subs(_spsRules, GiNaC::subs_options::algebraic)
                    .expand();
        if (coeff != 0) {
          RawIntegral integral(_nprops, 0);
          integral[s] = 1;
          // substitute scalar products by propagators
          coeff = coeff.subs(_spsFromProps, GiNaC::subs_options::algebraic);
          // t: D_t in coeff
          for (size_t t = 0; t < _nprops; ++t) {
            coeffD = coeff.diff(_symProps[t]);
            if (coeffD != 0) {
              integral[t] -= 1;
              equation[integral] += coeffD;
              integral[t] += 1;
            }
            coeff = coeff.subs(_symProps[t] == 0);
            if (coeff == 0)
              break;
          }
          coeff = coeff.expand();
          if (coeff != 0)
            equation[integral] += coeff;
        }
      }
      // add to _ibp
      IBPProto ibp;
      for (const auto &item : equation | std::ranges::views::reverse) {
        coeff = item.second.expand();
        if (coeff != 0)
          ibp.emplace_back(item.first, coeff);
      }
      if (!ibp.empty())
        _ibp.push_back(std::move(ibp));
    }
  }
  _generate_li();

  if (_symbols.size() == 0) {
    _generate_ibp_ff_n();
  } else {
    _generate_ibp_ff_m();  // symbolic coefficients (for the global emission)
    _generate_ibp_ff_n();  // prime-substituted numeric (for FiniteFlow structure)
  }
}

void Family::_generate_li() {
  GiNaC::ex coeff1, coeff2, coeff, coeffD;
  std::map<RawIntegral, GiNaC::ex> equation;

  // (u, v): E(E-1)/2 equations
  for (unsigned r = 0; r < _nexts; ++r) {
    for (unsigned s = r + 1; s < _nexts; ++s) {
      equation.clear();
      // i: p_i
      // p: propators
      for (unsigned i = 0; i < _nexts; ++i) {
        for (unsigned p = 0; p < _nprops; ++p) {
          coeff1 = (2 * _propagators[p].diff(_externals[i]) *
                    (-_symIndices[p]) * _externals[r])
                       .expand()
                       .subs(_spsRules, GiNaC::subs_options::algebraic)
                       .expand();
          coeff2 = (_externals[i] * _externals[s])
                       .subs(_spsRules, GiNaC::subs_options::algebraic)
                       .expand();
          if (coeff1 != 0 && coeff2 != 0) {
            RawIntegral integral(_nprops, 0);
            integral[p] = 1;
            // substitute scalar products by propagators
            coeff = (coeff1 * coeff2)
                        .expand()
                        .subs(_spsFromProps, GiNaC::subs_options::algebraic);
            // t: D_t in coeff
            for (size_t t = 0; t < _nprops; ++t) {
              coeffD = coeff.diff(_symProps[t]);
              if (coeffD != 0) {
                integral[t] -= 1;
                equation[integral] += coeffD;
                integral[t] += 1;
              }
              coeff = coeff.subs(_symProps[t] == 0);
              if (coeff == 0)
                break;
            }
            coeff = coeff.expand();
            if (coeff != 0)
              equation[integral] += coeff;
          }

          coeff1 = (-2 * _propagators[p].diff(_externals[i]) *
                    (-_symIndices[p]) * _externals[s])
                       .expand()
                       .subs(_spsRules, GiNaC::subs_options::algebraic)
                       .expand();
          coeff2 = (_externals[i] * _externals[r])
                       .subs(_spsRules, GiNaC::subs_options::algebraic)
                       .expand();
          if (coeff1 != 0 && coeff2 != 0) {
            RawIntegral integral(_nprops, 0);
            integral[p] = 1;
            // substitute scalar products by propagators
            coeff = (coeff1 * coeff2)
                        .expand()
                        .subs(_spsFromProps, GiNaC::subs_options::algebraic);
            // t: D_t in coeff
            for (size_t t = 0; t < _nprops; ++t) {
              coeffD = coeff.diff(_symProps[t]);
              if (coeffD != 0) {
                integral[t] -= 1;
                equation[integral] += coeffD;
                integral[t] += 1;
              }
              coeff = coeff.subs(_symProps[t] == 0);
              if (coeff == 0)
                break;
            }
            coeff = coeff.expand();
            if (coeff != 0)
              equation[integral] += coeff;
          }
        }
      }
      // add to _ibp
      IBPProto ibp;
      for (const auto &item : equation | std::ranges::views::reverse) {
        coeff1 = item.second.expand();
        if (coeff1 != 0)
          ibp.emplace_back(item.first, coeff1);
      }
      if (!ibp.empty())
        _ibp.push_back(std::move(ibp));
    }
  }
}

void Family::_generate_ibp_ff_n() {
  // If the family carries free symbols (symbolic d or kinematic invariants),
  // substitute distinct large primes so the coefficients become pure numbers
  // and the numeric FiniteFlow path can be reused for structural reduction.
  // The symbolic coefficients are kept separately in _ibpM and used for the
  // final emission to the global system (see sector_reduction_ff_n).
  GiNaC::lst primeSubs;
  if (!_symbols.empty()) {
    for (unsigned i = 0; i < _symbols.size() && i < kNumPrimeSubs; ++i)
      primeSubs.append(_symbols[i] == GiNaC::numeric(kPrimeSubs[i]));
  }

  // convert to numeric equations: full rationals (for global-equation emission)
  // alongside finite-field residues (for the compact FiniteFlow solver input).
  // On the signed path the symbolic _ibpM supplies the coefficients for the
  // emission, so the full rational _ibpFF is dead weight there: build and store
  // only the finite-field residues.
  const bool keep_rational = _symbols.empty();
  for (const auto &ibp : _ibp) {
    IBPProtoFF ibpFF;
    IBPProtoFFmod ibpFFmod;
    for (const auto &term : ibp) {
      GiNaC::ex item = primeSubs.nops() ? term.second.subs(primeSubs).expand()
                                        : term.second;
      std::vector<rational> coeffs;
      std::vector<ffint> coeffs_mod;
      for (unsigned i = 0; i < _nprops; ++i) {
        std::stringstream ss;
        ss << item.coeff(_symIndices[i]);
        if (keep_rational)
          coeffs.emplace_back(ss.str());
        coeffs_mod.emplace_back(ss.str());
        item -= item.coeff(_symIndices[i]) * _symIndices[i];
        item = item.expand();
      }
      // constant term
      std::stringstream ss;
      ss << item;
      if (keep_rational)
        coeffs.emplace_back(ss.str());
      coeffs_mod.emplace_back(ss.str());
      if (keep_rational)
        ibpFF.emplace_back(term.first, std::move(coeffs));
      ibpFFmod.emplace_back(term.first, std::move(coeffs_mod));
    }
    if (keep_rational)
      _ibpFF.emplace_back(std::move(ibpFF));
    _ibpFFmod.emplace_back(std::move(ibpFFmod));
  }
}

void Family::_generate_ibp_ff_m() {
  // convert to mpoly equations
  for (const auto &ibp : _ibp) {
    IBPProtoM ibpM;
    for (const auto &term : ibp) {
      GiNaC::ex item = term.second;
      std::vector<GiNaC::ex> coeffs;
      for (unsigned i = 0; i < _nprops; ++i) {
        GiNaC::ex coe = item.coeff(_symIndices[i]);
        coeffs.emplace_back(coe);
        item -= item.coeff(_symIndices[i]) * _symIndices[i];
        item = item.expand();
      }
      // constant term
      coeffs.emplace_back(item);
      ibpM.emplace_back(term.first, std::move(coeffs));
    }
    _ibpM.emplace_back(std::move(ibpM));
  }
}

std::vector<bool> Family::nontrivial_sectors(unsigned top) const {
  GiNaC::ex gPoly = _uPoly + _fPoly;
  GiNaC::ex gDiff;

  std::vector<GiNaC::symbol> kvec = generate_symbols("k", _nprops);
  GiNaC::lst klst;
  for (const auto &sym : kvec)
    klst.append(sym);

  for (unsigned i = 0; i < _nprops; ++i)
    gDiff += kvec[i] * _symIndices[i] * gPoly.diff(_symIndices[i]);
  gDiff = (gDiff - gPoly).expand();

  std::vector<bool> nontrivial(top + 1, false);
  std::vector<unsigned> trivials;
  for (unsigned sector = top; sector >= (1 << _nints) - 1; --sector) {
    // check if this sector belongs to top sector
    if ((sector & top) == sector) {
      // check if this sector belongs to a trivial sector
      if (std::find_if(trivials.begin(), trivials.end(),
                       [sector](unsigned trivial) {
                         return (sector & trivial) == sector;
                       }) != trivials.end())
        continue;

      GiNaC::lst zeros;
      for (unsigned i = 0; i < _nprops; ++i)
        if ((sector & (1 << i)) == 0)
          zeros.append(_symIndices[i] == 0);
      GiNaC::ex gSector = gDiff.subs(zeros);

      // collect the k equations
      std::map<std::vector<unsigned>, GiNaC::ex> coeffs;
      for (unsigned item = 0; item < gSector.nops(); ++item) {
        GiNaC::ex term = gSector.op(item);
        std::vector<unsigned> indices(_nprops, 0);
        for (unsigned i = 0; i < _nprops; ++i) {
          if (sector & (1 << i)) {
            if (term.coeff(_symIndices[i], 2) != 0) {
              term = term.coeff(_symIndices[i], 2);
              indices[i] = 2;
            } else if (term.coeff(_symIndices[i], 1) != 0) {
              term = term.coeff(_symIndices[i], 1);
              indices[i] = 1;
            }
          }
        }
        coeffs[indices] += term;
      }
      GiNaC::lst eqs;
      for (const auto &item : coeffs) {
        GiNaC::ex coeff = item.second.expand();
        if (coeff != 0)
          eqs.append(coeff == 0);
      }
      // solve the k equations
      if (nops(GiNaC::lsolve(eqs, klst)) == 0)
        nontrivial[sector] = true;
      else
        trivials.push_back(sector);
    }
  }
  return nontrivial;
}

void Family::_search_trivial_sectors(Reduce &reduce) const {
  reduce._sectors = nontrivial_sectors(reduce._top);
}

SymmetryTable::Inputs Family::symmetry_inputs() const {
  return SymmetryTable::Inputs{_uPoly + _fPoly,  // by value: the sum has no owner
                               _propagators,
                               _propMomenta,
                               _internals,
                               _externals,
                               _spsRules,
                               _spsFromProps,
                               _symIndices,
                               _symProps,
                               _nprops,
                               _symbols.empty()};
}

void Family::_compute_sps() {
  GiNaC::matrix propSpMat(_nprops, _nprops);
  GiNaC::matrix propSpConst(_nprops, 1);

  // set all variables to zero to get constant items
  GiNaC::lst allZero;
  for (unsigned i = 0; i < _nints; ++i) {
    for (unsigned j = i; j < _nints; ++j)
      allZero.append(_internals[i] * _internals[j] == 0);
    for (unsigned j = 0; j < _nexts; ++j)
      allZero.append(_internals[i] * _externals[j] == 0);
  }
  // compute the matrix from sp to prop
  for (unsigned s = 0; s < _nprops; ++s) {
    propSpConst(s, 0) =
        _propagators[s].subs(allZero, GiNaC::subs_options::algebraic);

    unsigned index = 0;
    for (unsigned i = 0; i < _nints; ++i) {
      GiNaC::ex coeff = _propagators[s].diff(_internals[i]);
      propSpMat(s, index) = coeff.diff(_internals[i]) / 2;
      ++index;
      for (unsigned j = i + 1; j < _nints; ++j) {
        propSpMat(s, index) = coeff.diff(_internals[j]);
        ++index;
      }
      for (unsigned j = 0; j < _nexts; ++j) {
        propSpMat(s, index) = coeff.diff(_externals[j]);
        ++index;
      }
    }
  }
  // check if the matrix is invertible
  if (propSpMat.rank() < _nprops)
    throw std::runtime_error("the set of propagators is incomplete");
  // compute the expression for sps
  GiNaC::matrix spPropVec(_nprops, 1);
  for (unsigned i = 0; i < _nprops; ++i)
    spPropVec(i, 0) = _symProps[i] - propSpConst(i, 0);
  GiNaC::matrix propSpInv = propSpMat.inverse();
  spPropVec = propSpInv.mul(spPropVec);
  // fill the rule list
  unsigned index = 0;
  for (unsigned i = 0; i < _nints; ++i) {
    for (unsigned j = i; j < _nints; ++j) {
      _spsFromProps.append(_internals[i] * _internals[j] ==
                           spPropVec(index, 0));
      ++index;
    }
    for (unsigned j = 0; j < _nexts; ++j) {
      _spsFromProps.append(_internals[i] * _externals[j] ==
                           spPropVec(index, 0));
      ++index;
    }
  }
}

void Family::_compute_symanzik() {
  GiNaC::ex schwinger;
  for (unsigned i = 0; i < _nprops; ++i)
    schwinger += -_symIndices[i] * _propagators[i];

  // the constant item
  GiNaC::lst allZero;
  for (const auto &ex : _internals)
    allZero.append(ex == 0);
  GiNaC::ex J = -schwinger.subs(allZero, GiNaC::subs_options::algebraic);

  // matrix l.M.l
  GiNaC::matrix M(_nints, _nints);
  // vector l.v
  GiNaC::matrix V(_nints, 1);

  for (unsigned i = 0; i < _nints; ++i) {
    GiNaC::ex di = schwinger.diff(_internals[i]);
    for (unsigned j = i; j < _nints; ++j) {
      M(i, j) = di.diff(_internals[j]) / 2;
      if (i != j)
        M(j, i) = M(i, j);
    }
    V(i, 0) = -di.subs(allZero, GiNaC::subs_options::algebraic) / 2;
  }

  _uPoly = M.determinant().expand();
  _fPoly = (_uPoly * J +
            V.transpose().mul(M.inverse().mul_scalar(_uPoly)).mul(V)(0, 0))
               .expand()
               .subs(_spsRules, GiNaC::subs_options::algebraic)
               .expand();
}

void Reduce::print() const {
  std::cout << "\n----------------- Reduce Info ------------------\n"
            << "\n  Targets: " << _rawTargets.size()
            << "   Top sector: " << _top;

  std::cout << "\n\n  Non-trivial sectors: "
            << std::count_if(_sectors.begin(), _sectors.end(),
                             [](bool value) { return value; });

  std::cout << "\n\n------------------------------------------------\n"
            << std::endl;
}

void Reduce::prepare_sectors() {
  // Kira-style seed truncation, derived once from the input's targets and the
  // configured relaxation (settings.seed_level_offset; derive_seed_truncation).
  // When the config names no offset, the rule reads the targets and picks 0 for
  // a pure-denominator set and the shipped default otherwise
  // (automatic_seed_level_offset).  Built here, before any worker starts, so
  // every sector reads the same read-only rule.
  const unsigned level_offset =
      _seed_level_offset_set ? _seed_level_offset
                             : automatic_seed_level_offset(_rawTargets);
  _truncation = derive_seed_truncation(_rawTargets, _top, level_offset);

  _lines = std::vector<bool>(_nprops, false);
  for (unsigned i = 0; i < _nprops; ++i)
    if (_top & (1 << i))
      _lines[i] = true;

  if (_ncut != 0) {
    unsigned cut = 0;
    for (unsigned i = 0; i < _cut.size(); ++i) {
      if (_cut[i] == 1)
        cut += 1 << i;
    }
    for (unsigned i = 0; i < _sectors.size(); ++i) {
      if ((cut & i) != cut)
        _sectors[i] = false;
    }
  }

  unsigned topSect = 0;
  for (auto sector : _targetSectors)
    topSect = std::max(topSect, sector);
  for (unsigned i = 0; i < _sectors.size(); ++i) {
    if (i > topSect)
      _sectors[i] = false;
  }

  unsigned nsec = std::count_if(_sectors.begin(), _sectors.end(),
                                [](bool value) { return value; });
  if (nsec == 0)
    process_finish("no non-trivial sectors");
  // sort non-trivial sectors
  std::vector<unsigned> sectors(nsec);
  std::set<unsigned> trivialsectors;
  for (unsigned i = 0, j = nsec - 1; i < _sectors.size(); ++i)
    if (_sectors[i])
      sectors[j--] = i;
    else
      trivialsectors.insert(i);

  std::sort(sectors.begin(), sectors.end(), [](unsigned a, unsigned b) {
    unsigned alines = std::popcount(a);
    unsigned blines = std::popcount(b);
    if (alines > blines)
      return true;
    else if (alines == blines)
      return a > b;
    else
      return false;
  });

  // Pre-populate all sector keys in targets and uptargets before threads start.
  // This prevents operator[] from structurally modifying the maps during
  // parallel execution (concurrent structural modification is a data race).
  _ctx.targets.clear();
  _ctx.uptargets.clear();
  for (unsigned id : sectors) {
    _ctx.targets[id];   // default-inserts empty set for this key
    _ctx.uptargets[id]; // same
  }
  for (const auto &integral : _rawTargets) {
    _ctx.targets[integral.sector()].insert(integral);
  }

  namespace fs = std::filesystem;
  fs::create_directories(_cache_dir + "/sectors");
  std::ofstream sectorfile(_cache_dir + "/sectors/sectors");

  _ctx.allMasters = std::vector<std::vector<RawIntegral>>(nsec);
  _ctx.totalSectors = nsec;
  _ctx.completedSectors = 0;
  _reduceSectors.clear();
  _reduceSectors.reserve(nsec);
  for (unsigned i = 0; i < sectors.size(); ++i) {
    sectorfile << sectors[i] << ": " << std::popcount(sectors[i]) << std::endl;
    std::vector<bool> slines(_nprops, false);
    for (unsigned j = 0; j < _nprops; ++j)
      if (sectors[i] & (1 << j))
        slines[j] = true;
    _reduceSectors.push_back(Sector::make(i, sectors[i], _nprops, slines, trivialsectors));
  }

  // Spec 4.7: the table was built over non-trivial sectors, but this function's
  // own filters (the cut mask and the top filter) may have dropped an endpoint.
  // An edge to a sector that is never scheduled is a deadlock, so prune the
  // table to exactly the sectors that will run.  A source left without
  // prescriptions reports itself.
  {
    std::set<unsigned> reduced;
    for (const Sector &s : _reduceSectors)
      reduced.insert(s.id());
    _symmetries.restrictTo(reduced);
    _ctx.symmetries = &_symmetries;
  }
}