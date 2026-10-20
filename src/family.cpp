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
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>

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

// Run repeated sector passes until the seed basis settles, then the global stage.
void Family::run_reduce(Reduce &reduce) {

  constexpr unsigned kMaxSeedPasses = 4;

  reduce._ctx.exemptCutRequests = false;
  double sectorSeconds = 0.0;
  double globalSeconds = 0.0;
  double learnSeconds = 0.0;
  double reconstructionSeconds = 0.0;
  GlobalReductionResult globalResult;
  bool have_global_system = false;

  for (unsigned pass = 0; pass < kMaxSeedPasses; ++pass) {

    reduce._ctx.rawEquationsPath = _cache_dir + "/raw_equations";
    reduce._ctx.openSpill();

    {
      std::ofstream csv("sector_timing.csv", std::ios::trunc);
      if (csv.is_open())
        csv << "sector,expanded,probe_s,full_s\n";
    }

    _ibp.clear();
    _ibp.shrink_to_fit();

    const auto sector_t0 = std::chrono::steady_clock::now();
    if (_parallel_mode == "fine")
      _run_reduce_fine(reduce);
    else
      _run_reduce_layer(reduce);
    const auto sector_t1 = std::chrono::steady_clock::now();
    const double passSectorSeconds =
        std::chrono::duration<double>(sector_t1 - sector_t0).count();
    sectorSeconds += passSectorSeconds;

    std::cout << "\n  [timing] sector reduction: "
              << secred::format_duration(passSectorSeconds) << "\n";

    _ibpFF.clear();
    _ibpFF.shrink_to_fit();
    _ibpFFmod.clear();
    _ibpFFmod.shrink_to_fit();
    _ibpM.clear();
    _ibpM.shrink_to_fit();
    reduce._ctx.uptargets.clear();

    have_global_system = reduce._ctx.globalEquationsCount.load() > 0;

    {
      unsigned ok = reduce._ctx.narrowingSuccess.load();
      unsigned fail = reduce._ctx.narrowingFail.load();
      unsigned total = ok + fail;
      double ratio = total ? 100.0 * static_cast<double>(ok) / total : 0.0;

      const double probeSeconds = reduce._ctx.probeNanos.load() / 1e9;
      const double fullSeconds = reduce._ctx.fullNanos.load() / 1e9;
      const double atpsSeconds = probeSeconds + fullSeconds;
      const double probeShare =
          atpsSeconds > 0 ? 100.0 * probeSeconds / atpsSeconds : 0.0;

      const unsigned trunc = reduce._ctx.truncatedSectors.load();
      const unsigned lifted = reduce._ctx.truncationLifted.load();

      const unsigned spillRaised = reduce._ctx.boxSpillRaised.load();
      const unsigned spillDepth = reduce._ctx.boxSpillExtraDepth.load();
      const unsigned spillRank = reduce._ctx.boxSpillExtraRank.load();

      const bool probeAll = reduce._probe_mode == ProbeMode::All;
      const char *probeMode = probeAll ? "all" : "targets";
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
              << "truncation dots:     " << reduce._truncation.dots << "\n"
              << "box spill raised:    " << spillRaised << "\n"
              << "box spill extra:     " << spillDepth << " depth, " << spillRank
              << " rank\n"
              << "probe scope:         " << probeMode << "\n"
              << "probe accepted:      " << accepted << "\n";

      char atpsLine[128];
      std::snprintf(atpsLine, sizeof(atpsLine),
                    "\n  ATPS: probe %.1f s / full %.1f s / probe share %.1f %%",
                    probeSeconds, fullSeconds, probeShare);
      std::cout << atpsLine << std::endl;

      std::cout << "\n  Seed truncation: " << trunc << " sectors cut, " << lifted
                << " lifted by the probe (l = " << reduce._truncation.l
                << ", offset " << reduce._truncation.level_offset << ")"
                << std::endl;

      std::cout << "\n  Seed box: spill raised " << spillRaised
                << " sectors (+" << spillDepth << " depth, +" << spillRank
                << " rank over targets-only)" << std::endl;

      if (probeAll)
        std::cout << "\n  Probe scope (all): every out-of-box master report"
                  << " widens the box; seed_level_offset is void (offset 0)"
                  << std::endl;
      else
        std::cout << "\n  Probe scope (targets): " << accepted
                  << " out-of-box master reports accepted, not needed by the"
                  << " input targets" << std::endl;

      if (!have_global_system)
        std::cout << "\n Probe narrowing: succeeded " << ok << " / failed " << fail
                  << " / total " << total << " ("
                  << std::fixed << std::setprecision(1) << ratio << " % success)"
                  << std::endl;
    }

    const auto global_t0 = std::chrono::steady_clock::now();
    if (have_global_system)
      globalResult = _run_global_reduction(reduce, pass + 1 < kMaxSeedPasses);
    else
      reduce._ctx.discardSpill();
    globalSeconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - global_t0)
            .count();
    if (have_global_system) {
      learnSeconds += globalResult.learn_seconds;
      reconstructionSeconds += globalResult.reconstruction_seconds;
    }

    const unsigned seedDots =
        have_global_system
            ? seed_dots_refinement(reduce._truncation, globalResult.masters)
            : 0;

    const std::vector<RawIntegral> requestOffenders =
        have_global_system
            ? seed_request_refinement(globalResult.masters,
                                      reduce._ctx.cutRequests)
            : std::vector<RawIntegral>{};
    if (seedDots == 0 && requestOffenders.empty())
      break;
    const bool raiseDots = seedDots != 0;
    if (pass + 1 == kMaxSeedPasses) {
      if (raiseDots)
        std::cout << "\n  Seed truncation: the settled basis carries masters at"
                  << " dots " << seedDots << ", above the limit "
                  << reduce._truncation.dots << "; pass cap reached, keeping"
                  << " this basis" << std::endl;
      else
        std::cout << "\n  Seed truncation: the settled basis carries "
                  << requestOffenders.size()
                  << " masters the rule cut (requests no sector reduced);"
                  << " pass cap reached, keeping this basis" << std::endl;
      break;
    }
    if (raiseDots) {
      std::cout << "\n  Seed truncation: the settled basis carries masters at"
                << " dots " << seedDots << ", above the limit "
                << reduce._truncation.dots
                << "; raising the limit and reducing again" << std::endl;
    } else {
      std::cout << "\n  Seed truncation: the settled basis carries "
                << requestOffenders.size()
                << " masters the rule cut (requests no sector reduced);"
                << " honoring the request cut and reducing again" << std::endl;
      reduce._ctx.exemptCutRequests = true;
    }

    _generate_ibp();
    if (!_symbols.empty())
      reduce._rowTable.build_all(_ibpM);
    reduce.prepare_sectors();
    if (raiseDots)
      reduce._truncation.dots = seedDots;
    reduce._ctx.resetRunStatistics();
  }

  {
    std::vector<RawIntegral> sector_masters;
    for (const auto &sector : reduce._ctx.allMasters)
      sector_masters.insert(sector_masters.end(), sector.begin(), sector.end());
    secred::write_masters_file(_cache_dir + "/" + secred::kMastersSectorFile,
                               sector_masters);
  }

  unsigned masterCount = 0;
  if (have_global_system) {
    masterCount = globalResult.masters.size();
  } else {
    for (const auto &sector : reduce._ctx.allMasters)
      masterCount += sector.size();
  }

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

  {
    std::vector<secred::PhaseTiming> phases{
        {"Sector reduction", sectorSeconds},
        {"Global reduction", globalSeconds},
        {"Global learn", learnSeconds},
        {"Reconstruction", reconstructionSeconds}};
    std::cout << "\n"
              << secred::format_timing_statistics(
                     phases, secred::run_elapsed_seconds(),
                     process_peak_memory());
  }
}

// One pass, sectors scheduled layer by layer by sector width.
void Family::_run_reduce_layer(Reduce &reduce) const {
  BS::thread_pool pool(_sthreads);
  SectorConfig scfg{_cache_dir,     _subsector_filter_n, _ordering_mode,
                    _sector_debug,  &reduce._preferred,  &reduce._truncation,
                    reduce._probe_mode, reduce._mandatory_list};

  unsigned current_layer = UINT_MAX;
  for (auto &sector : reduce._reduceSectors) {
    unsigned layer = std::popcount(sector.id());
    if (layer != current_layer) {
      if (current_layer != UINT_MAX)
        pool.wait_for_tasks();
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
  pool.wait_for_tasks();
}

// One pass, all sectors submitted independently.
void Family::_run_reduce_fine(Reduce &reduce) const {
  unsigned nsec = reduce._reduceSectors.size();

  std::unordered_map<unsigned, unsigned> id_to_idx;
  id_to_idx.reserve(nsec);
  for (unsigned i = 0; i < nsec; ++i)
    id_to_idx[reduce._reduceSectors[i].id()] = i;

  std::vector<std::vector<unsigned>> children(nsec);
  std::vector<unsigned> parent_count(nsec, 0);
  for (unsigned j = 0; j < nsec; ++j) {
    unsigned jid = reduce._reduceSectors[j].id();
    for (unsigned k = 0; k < _nprops; ++k) {
      if (jid & (1u << k)) continue;
      auto it = id_to_idx.find(jid | (1u << k));
      if (it != id_to_idx.end()) {
        children[it->second].push_back(j);
        ++parent_count[j];
      }
    }
  }

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
                    _sector_debug,  &reduce._preferred,  &reduce._truncation,
                    reduce._probe_mode, reduce._mandatory_list};

  std::atomic<unsigned> done_count{0};
  std::mutex done_mutex;
  std::condition_variable done_cv;

  BS::thread_pool pool(_sthreads);

  std::function<void(unsigned)> run_sector = [&](unsigned i) {
    auto &sector = reduce._reduceSectors[i];

    if (_symbols.empty())
      sector.run_reduce_ff_n(_ibpFF, _ibpFFmod, nullptr, scfg, reduce._ctx);
    else
      sector.run_reduce_ff_n(_ibpFF, _ibpFFmod, &_ibpM, scfg, reduce._ctx);

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

  pool.wait_for_tasks();
}

// Build and solve the global system, then vote on the settled basis.
GlobalReductionResult Family::_run_global_reduction(Reduce &reduce,
                                                    bool allow_basis_vote) const {
  auto &ctx = reduce._ctx;

  BasisVote basis_vote;
  if (allow_basis_vote)
    basis_vote = [&reduce](const std::vector<RawIntegral> &masters) {
      if (seed_dots_refinement(reduce._truncation, masters) > 0)
        return true;
      return !seed_request_refinement(masters, reduce._ctx.cutRequests).empty();
    };

  std::vector<RawIntegral> all_integrals;
  ctx.finishSpill(all_integrals);

  secred::order_global_integrals(all_integrals, _ordering_mode,
                                 &reduce._preferred);

  std::unordered_map<RawIntegral, unsigned> integral_id;
  integral_id.reserve(all_integrals.size());
  for (unsigned i = 0; i < all_integrals.size(); ++i)
    integral_id[all_integrals[i]] = i;

  const GlobalEngine engine = global_engine_for(_symbols.size());
  const char *tool = global_engine_name(engine);
  const std::string mastersGlobalPath =
      _cache_dir + "/" + secred::kMastersGlobalFile;

  GlobalReductionResult result;
  {

    secred::reset_firefly_log();
    struct CollectFireflyLog {
      const std::string &cache_dir;
      ~CollectFireflyLog() { secred::collect_firefly_log(cache_dir); }
    } collect_log{_cache_dir};

    if (engine == GlobalEngine::FireFlySymbolic)
      result = secred::reduce_global_firefly_symbolic(
          ctx.rawEquationsPath, all_integrals, integral_id, reduce._rawTargets,
          _symbols, ctx.narrowingSuccess.load(), ctx.narrowingFail.load(),
          ctx.rawEquationsTotal.load(), ctx.globalEquationsCount.load(),
          mastersGlobalPath, _gthreads, _max_primes, basis_vote);
    else if (engine == GlobalEngine::FiniteFlowSymbolic)
      result = reduce_global_finiteflow_symbolic(
          ctx.rawEquationsPath, all_integrals, integral_id, reduce._rawTargets,
          _symbols, ctx.narrowingSuccess.load(), ctx.narrowingFail.load(),
          ctx.rawEquationsTotal.load(), ctx.globalEquationsCount.load(),
          mastersGlobalPath, _gthreads, _max_primes, basis_vote);
    else
      result = reduce_global_finiteflow(
          ctx.rawEquationsPath, all_integrals, integral_id, reduce._rawTargets,
          ctx.narrowingSuccess.load(), ctx.narrowingFail.load(),
          ctx.rawEquationsTotal.load(), ctx.globalEquationsCount.load(),
          mastersGlobalPath, _gthreads, _max_memory, _memory_reserve,
          _max_primes, basis_vote);
    std::filesystem::remove(ctx.rawEquationsPath);

    secred::require_global_reduction(result, tool);

    if (!result.reconstruction_skipped) {
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

// Read the input node into the family settings.
Family::Family(const YAML::Node &config) {
  if (!config["family"])
    throw std::runtime_error("family not found");
  YAML::Node familyConfig = config["family"];

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

  _name = familyConfig["name"].as<std::string>();

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

  for (const auto &sym :
       familyConfig["internals"].as<std::vector<std::string>>()) {
    if (symtab.contains(sym))
      throw std::runtime_error("symbol " + sym + " defined more than once");
    _internals.emplace_back(sym);
    symtab[sym] = _internals.back();
  }

  for (const auto &sym :
       familyConfig["externals"].as<std::vector<std::string>>()) {
    if (symtab.contains(sym))
      throw std::runtime_error("symbol " + sym + " defined more than once");
    _externals.emplace_back(sym);
    symtab[sym] = _externals.back();
  }

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

  if (familyConfig["propagators"].size() != _nprops)
    throw std::runtime_error(
        "number of propagators does not match the topology");
  for (const auto &prop :
       familyConfig["propagators"]
           .as<std::vector<std::pair<std::string, std::string>>>()) {

    _propMomenta.emplace_back(reader(prop.first));
    _propagators.emplace_back(
        (pow(reader(prop.first), 2) - pow(reader(prop.second), 2))
            .expand()
            .subs(_spsRules, GiNaC::subs_options::algebraic)
            .subs(_one)
            .expand());
  }

  if (familyConfig["cut"])
    _cut = familyConfig["cut"].as<std::vector<unsigned>>();
  _ncut = std::accumulate(_cut.begin(), _cut.end(), 0);

  if (config["settings"]) {
    const YAML::Node &s = config["settings"];

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

    if (s["max_memory"])
      _max_memory = (std::uint64_t)(s["max_memory"].as<double>() * 1024 * 1024 * 1024);

    if (s["memory_reserve"])
      _memory_reserve = (std::uint64_t)(s["memory_reserve"].as<double>() * 1024 * 1024 * 1024);

    if (s["max_primes"]) {
      const unsigned long long mp = s["max_primes"].as<unsigned long long>();
      if (mp < 1 || mp > std::numeric_limits<unsigned>::max())
        throw std::runtime_error("settings.max_primes must be a positive integer");
      _max_primes = static_cast<unsigned>(mp);
    }

    if (s["symmetry"]) _symmetry = s["symmetry"].as<bool>();
  }

  std::cout << "\n #0.0   Parsing config file finished." << std::endl;

  _symIndices = generate_symbols("a", _nprops);
  _symProps = generate_symbols("D", _nprops);
}

// Set up the family: symmetries, Symanzik polynomials and IBP rows.
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

// Collect the targets and the per-run options for a reduction.
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

  reduce._nprops = _nprops;
  reduce._seed_level_offset = _seed_level_offset;
  reduce._seed_level_offset_set = _seed_level_offset_set;
  reduce._probe_mode = _probe_mode;
  reduce._mandatory_list = _mandatory_list;
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

  if (_symmetry) {
    std::cout << "\n #0.4   Searching sector symmetries...\n";
    reduce._symmetries =
        SymmetryTable::find(symmetry_inputs(), reduce._sectors, reduce._top);
    if (_parallel_mode == "layer") {

      reduce._symmetries.dropCross();
      std::cout << " #0.4   parallel_mode layer: cross-sector symmetry relations "
                   "are dropped, self relations are kept.\n";
    }

    reduce._symmetries.try_to_numeric_all();
    std::cout << "\n #0.4   Searching sector symmetries finished.\n";
  }

  if (!_symbols.empty()) {
    reduce._rowTable.build_all(_ibpM);
    reduce._ctx.rowTable = &reduce._rowTable;
  }

  reduce.prepare_sectors();
}

// Print the family definition.
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

// Generate the IBP rows for the current run.
void Family::_generate_ibp() {
  GiNaC::ex coeff, coeffD;
  std::map<RawIntegral, GiNaC::ex> equation;

  for (size_t i = 0; i < _nints; ++i) {
    for (size_t j = 0; j < _nints + _nexts; ++j) {
      equation.clear();

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

          coeff = coeff.subs(_spsFromProps, GiNaC::subs_options::algebraic);

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
    _generate_ibp_ff_m();
    _generate_ibp_ff_n();
  }
}

// Generate the Lorentz-invariance rows.
void Family::_generate_li() {
  GiNaC::ex coeff1, coeff2, coeff, coeffD;
  std::map<RawIntegral, GiNaC::ex> equation;

  for (unsigned r = 0; r < _nexts; ++r) {
    for (unsigned s = r + 1; s < _nexts; ++s) {
      equation.clear();

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

            coeff = (coeff1 * coeff2)
                        .expand()
                        .subs(_spsFromProps, GiNaC::subs_options::algebraic);

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

            coeff = (coeff1 * coeff2)
                        .expand()
                        .subs(_spsFromProps, GiNaC::subs_options::algebraic);

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

// Generate the numeric-path IBP rows.
void Family::_generate_ibp_ff_n() {

  GiNaC::lst primeSubs;
  if (!_symbols.empty()) {
    for (unsigned i = 0; i < _symbols.size() && i < kNumPrimeSubs; ++i)
      primeSubs.append(_symbols[i] == GiNaC::numeric(kPrimeSubs[i]));
  }

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

// Generate the symbolic-path IBP rows.
void Family::_generate_ibp_ff_m() {

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

      coeffs.emplace_back(item);
      ibpM.emplace_back(term.first, std::move(coeffs));
    }
    _ibpM.emplace_back(std::move(ibpM));
  }
}

// Sectors worth reducing under the top-sector classification.
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

    if ((sector & top) == sector) {

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

      if (nops(GiNaC::lsolve(eqs, klst)) == 0)
        nontrivial[sector] = true;
      else
        trivials.push_back(sector);
    }
  }
  return nontrivial;
}

// Store the nontrivial sector list for the run.
void Family::_search_trivial_sectors(Reduce &reduce) const {
  reduce._sectors = nontrivial_sectors(reduce._top);
}

// Assemble the symmetry search inputs.
SymmetryTable::Inputs Family::symmetry_inputs() const {
  return SymmetryTable::Inputs{_uPoly + _fPoly,
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

// Solve for the scalar-product substitution rules.
void Family::_compute_sps() {
  GiNaC::matrix propSpMat(_nprops, _nprops);
  GiNaC::matrix propSpConst(_nprops, 1);

  GiNaC::lst allZero;
  for (unsigned i = 0; i < _nints; ++i) {
    for (unsigned j = i; j < _nints; ++j)
      allZero.append(_internals[i] * _internals[j] == 0);
    for (unsigned j = 0; j < _nexts; ++j)
      allZero.append(_internals[i] * _externals[j] == 0);
  }

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

  if (propSpMat.rank() < _nprops)
    throw std::runtime_error("the set of propagators is incomplete");

  GiNaC::matrix spPropVec(_nprops, 1);
  for (unsigned i = 0; i < _nprops; ++i)
    spPropVec(i, 0) = _symProps[i] - propSpConst(i, 0);
  GiNaC::matrix propSpInv = propSpMat.inverse();
  spPropVec = propSpInv.mul(spPropVec);

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

// Compute the Symanzik polynomials.
void Family::_compute_symanzik() {
  GiNaC::ex schwinger;
  for (unsigned i = 0; i < _nprops; ++i)
    schwinger += -_symIndices[i] * _propagators[i];

  GiNaC::lst allZero;
  for (const auto &ex : _internals)
    allZero.append(ex == 0);
  GiNaC::ex J = -schwinger.subs(allZero, GiNaC::subs_options::algebraic);

  GiNaC::matrix M(_nints, _nints);

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

// Print the settled run configuration.
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

// Derive the truncation and build the sector list.
void Reduce::prepare_sectors() {

  const unsigned level_offset =
      _probe_mode == ProbeMode::All
          ? 0
          : (_seed_level_offset_set ? _seed_level_offset
                                    : automatic_seed_level_offset(_rawTargets));
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

  _ctx.targets.clear();
  _ctx.uptargets.clear();
  _ctx.cutRequests.clear();
  for (unsigned id : sectors) {
    _ctx.targets[id];
    _ctx.uptargets[id];
    _ctx.cutRequests[id];
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

  {
    std::set<unsigned> reduced;
    for (const Sector &s : _reduceSectors)
      reduced.insert(s.id());
    _symmetries.restrictTo(reduced);
    _ctx.symmetries = &_symmetries;
  }
}
