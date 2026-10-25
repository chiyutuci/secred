#include "family.h"
#include "family_stages.h"
#include "globalreduce.h"
#include "masters_file.h"
#include "resume.h"
#include "sector_resume.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <unordered_set>

GiNaC::symtab Family::symtab;

namespace secred {

namespace {

bool path_has_prefix(const std::filesystem::path &prefix,
                     const std::filesystem::path &path) {
  auto p = prefix.begin();
  auto q = path.begin();
  for (; p != prefix.end(); ++p, ++q) {
    if (q == path.end() || *p != *q)
      return false;
  }
  return true;
}

// A symlink on the way to the cache would redirect the recursive delete, so the
// guard refuses such a path instead of resolving it.
bool has_symlink_component(const std::filesystem::path &path) {
  namespace fs = std::filesystem;
  fs::path prefix = path.root_path();
  const fs::path relative = path.relative_path();
  for (const fs::path &part : relative) {
    prefix /= part;
    std::error_code ec;
    const fs::file_status status = fs::symlink_status(prefix, ec);
    if (ec) {
      if (ec == std::errc::no_such_file_or_directory ||
          ec == std::errc::not_a_directory)
        return false;
      throw std::runtime_error("cannot inspect the cache path " +
                               prefix.string() + ": " + ec.message());
    }
    if (fs::is_symlink(status))
      return true;
  }
  return false;
}

// Names SecRed itself writes at the top of a cache.  An unmarked directory made
// only of these is a 1.0 cache and is adopted; anything else is refused.
bool is_known_cache_entry(const std::string &name) {
  static const std::unordered_set<std::string> known{
      kCacheMarkerFile, kManifestFile, "firefly.log", "masters_global",
      "masters_sector", "raw_equations", "reductions", "resume", "secred.log",
      "sectors"};
  return known.contains(name);
}

struct ResolvedCache {
  std::filesystem::path resolved;
  bool exists = false;
  bool marked = false;
};

ResolvedCache resolve_cache_directory(const std::string &cache_dir) {
  namespace fs = std::filesystem;

  if (cache_dir.empty())
    throw std::runtime_error("refusing an empty cache directory");

  std::error_code ec;
  fs::path path = fs::absolute(fs::path(cache_dir), ec);
  if (ec)
    throw std::runtime_error("cannot resolve the cache directory \"" + cache_dir +
                             "\": " + ec.message());
  path = path.lexically_normal();
  if (path.empty() || path == path.root_path())
    throw std::runtime_error(
        "refusing the filesystem root as a cache directory");

  if (has_symlink_component(path))
    throw std::runtime_error(
        "refusing a cache directory reached through a symlink: " +
        path.string());

  const fs::path resolved = fs::weakly_canonical(path, ec);
  if (ec)
    throw std::runtime_error("cannot resolve the cache directory \"" + cache_dir +
                             "\": " + ec.message());

  const fs::path cwd = fs::weakly_canonical(fs::current_path(), ec);
  if (ec)
    throw std::runtime_error("cannot resolve the current directory: " +
                             ec.message());
  if (resolved == cwd || path_has_prefix(resolved, cwd))
    throw std::runtime_error(
        "refusing a cache directory that is the current directory or one of "
        "its ancestors: " +
        resolved.string());

  if (const char *home = std::getenv("HOME"); home && *home) {
    std::error_code home_ec;
    const fs::path home_path = fs::weakly_canonical(fs::path(home), home_ec);
    if (!home_ec && home_path != home_path.root_path() &&
        (resolved == home_path || path_has_prefix(resolved, home_path)))
      throw std::runtime_error(
          "refusing a cache directory that is the home directory or one of its "
          "ancestors: " +
          resolved.string());
  }

  ec.clear();
  const fs::file_status status = fs::symlink_status(resolved, ec);
  if (ec && ec != std::errc::no_such_file_or_directory)
    throw std::runtime_error("cannot inspect the cache directory \"" +
                             resolved.string() + "\": " + ec.message());
  ResolvedCache info;
  info.resolved = resolved;
  info.exists = !ec && status.type() != fs::file_type::not_found;
  if (info.exists) {
    if (!fs::is_directory(status))
      throw std::runtime_error(
          "refusing a cache path that is not a directory: " +
          resolved.string());
    std::error_code marker_ec;
    info.marked =
        fs::is_regular_file(fs::symlink_status(resolved / kCacheMarkerFile,
                                               marker_ec));
  }
  return info;
}

void refuse_unknown_cache_entries(const std::filesystem::path &resolved) {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::directory_iterator it(resolved, ec);
  if (ec)
    throw std::runtime_error("cannot list the cache directory \"" +
                             resolved.string() + "\": " + ec.message());
  for (const fs::directory_entry &entry : it) {
    const std::string name = entry.path().filename().string();
    if (!is_known_cache_entry(name))
      throw std::runtime_error(
          "refusing to clear the cache directory \"" + resolved.string() +
          "\": unexpected entry \"" + name + "\"");
  }
}

void clear_cache_directory(const std::filesystem::path &resolved) {
  std::error_code ec;
  std::filesystem::remove_all(resolved, ec);
  if (ec)
    throw std::runtime_error("cannot clear the cache directory \"" +
                             resolved.string() + "\": " + ec.message());
}

void create_cache_directory(const std::filesystem::path &resolved) {
  std::error_code ec;
  std::filesystem::create_directories(resolved, ec);
  if (ec && !std::filesystem::is_directory(resolved))
    throw std::runtime_error("cannot create the cache directory \"" +
                             resolved.string() + "\": " + ec.message());
}

void write_cache_marker(const std::filesystem::path &resolved) {
  std::ofstream marker(resolved / kCacheMarkerFile, std::ios::trunc);
  marker << "secred cache directory\n";
  marker.flush();
  if (!marker)
    throw std::runtime_error("cannot write the cache marker in \"" +
                             resolved.string() + "\"");
}

} // namespace

std::filesystem::path prepare_cache_directory(const std::string &cache_dir) {
  namespace fs = std::filesystem;

  const ResolvedCache info = resolve_cache_directory(cache_dir);
  if (info.exists) {
    if (!info.marked)
      refuse_unknown_cache_entries(info.resolved);
    clear_cache_directory(info.resolved);
  }
  create_cache_directory(info.resolved);
  write_cache_marker(info.resolved);
  return info.resolved;
}

CachePreparation prepare_run_cache_directory(const std::string &cache_dir,
                                             bool resume,
                                             std::uint64_t input_hash) {
  namespace fs = std::filesystem;

  const ResolvedCache info = resolve_cache_directory(cache_dir);
  bool keep = false;
  if (info.exists && resume) {
    ResumeManifest manifest;
    std::string error;
    if (read_manifest(info.resolved / kManifestFile, manifest, &error) &&
        manifest.format == kResumeFormatVersion &&
        manifest.build == resume_build_stamp() &&
        manifest.input_hash == input_hash) {
      keep = true;
    } else {
      std::cout << "\n  Resume: the cache directory does not belong to this run"
                   " ("
                << (error.empty() ? "a different input or build" : error)
                << "); starting fresh" << std::endl;
    }
  }

  if (info.exists && !keep) {
    if (!info.marked)
      refuse_unknown_cache_entries(info.resolved);
    clear_cache_directory(info.resolved);
  }
  if (!keep) {
    create_cache_directory(info.resolved);
    write_cache_marker(info.resolved);
  } else if (!info.marked) {
    write_cache_marker(info.resolved);
  }

  if (!keep) {
    ResumeManifest manifest;
    manifest.build = resume_build_stamp();
    manifest.input_hash = input_hash;
    std::string error;
    if (!write_manifest(info.resolved / kManifestFile, manifest, &error))
      throw std::runtime_error("cannot write the resume manifest in \"" +
                               info.resolved.string() + "\": " + error);
  }
  return CachePreparation{info.resolved, keep};
}

} // namespace secred

// Read the input node into the family's model and settings.
Family::Family(const YAML::Node &config, const FamilyOptions &options) {
  _settings._dry_run = options.dry_run;
  if (!config["family"])
    throw std::runtime_error("family not found");
  const YAML::Node familyConfig = config["family"];

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

  _model.parse(familyConfig, symtab);
  _settings.parse_cut(familyConfig["cut"], _model._nprops);
  _settings.parse(config["settings"], options.allow_unknown_settings);
  _settings.resolve_memory();
  _settings._input_hash = secred::resume_config_hash(config);

  std::cout << "\n #0.0   Parsing config file finished." << std::endl;
}

// Set up the family: symmetries, Symanzik polynomials and IBP rows.
void Family::init() {
  std::cout << "\n #0.1   Initializing integral family...\n";

  if (_settings._dry_run) {
    std::cout << "\n #0.1   Dry run: leaving cache directory "
              << _settings._cache_dir << " untouched.\n";
  } else {
    const secred::CachePreparation cache =
        secred::prepare_run_cache_directory(_settings._cache_dir,
                                            _settings._resume,
                                            _settings._input_hash);
    _settings._cache_dir = cache.path.string();
    _resumed = cache.resumed;
    if (_resumed)
      std::cout << "\n #0.1   Resuming in cache directory "
                << _settings._cache_dir << std::endl;
  }

  _model._compute_sps();
  _model._compute_symanzik();

  std::cout << "\n #0.1   Initializing integral family "
               "finished.\n";

  std::cout << "\n #0.2   Generating IBP relations...\n";
  _generator.generate(_model);
  std::cout << "\n #0.2   Generating IBP relations finished.\n";
}

// Collect the targets and the per-run options for a reduction.
void Family::init_reduce(const YAML::Node &config, Reduce &reduce) const {
  const secred::RunContext ctx{_model, _settings, _generator, reduce};
  secred::SectorStage::prepare(ctx, config);
}

// Print the family definition.
void Family::print() const {

  std::cout << "\n----------------- Family Info "
               "------------------\n"
            << "\n  Topology: " << _model._name
            << "   Dimension: " << _model._dimension
            << "\n\n  Internals:";

  for (const auto &sym : _model._internals)
    std::cout << " " << sym;

  std::cout << "\n\n  Externals:";
  for (const auto &sym : _model._externals)
    std::cout << " " << sym;

  std::cout << "\n\n  Invariants:";
  for (const auto &inv : _model._invariants)
    std::cout << " " << inv.first;

  std::cout << "\n\n  Propagators: " << _model._nprops
            << "   IBP relations: " << _generator._ibp.size();

  std::cout << std::endl;
}

// Run repeated sector passes until the seed basis settles, then the global stage.
void Family::run_reduce(Reduce &reduce) {

  constexpr unsigned kMaxSeedPasses = 4;

  reduce._ctx.exemptCutRequests = false;

  // Checkpointing is opt-in: with resume off, sectors write no records and the
  // scheduler does not look for any.
  secred::ResumeControl resume_control;
  if (_settings._resume) {
    resume_control.cache_dir = _settings._cache_dir;
    resume_control.input_hash = _settings._input_hash;
    reduce._ctx.resume = &resume_control;
  } else {
    reduce._ctx.resume = nullptr;
    reduce._ctx.sectorPassDir.clear();
  }

  const secred::RunContext stage_ctx{_model, _settings, _generator, reduce};

  double sectorSeconds = 0.0;
  double globalSeconds = 0.0;
  double learnSeconds = 0.0;
  double reconstructionSeconds = 0.0;
  GlobalReductionResult globalResult;
  bool have_global_system = false;

  for (unsigned pass = 0; pass < kMaxSeedPasses; ++pass) {

    reduce._ctx.rawEquationsPath = _settings._cache_dir + "/raw_equations";
    reduce._ctx.openSpill();

    {
      std::ofstream csv("sector_timing.csv", std::ios::trunc);
      if (csv.is_open())
        csv << "sector,expanded,probe_s,full_s\n";
    }

    // Each pass owns a directory of sector records.  The pass key pins the
    // whole seed state, so records from a different truncation or options can
    // never be mistaken for this pass's.
    std::vector<bool> replayed(reduce._reduceSectors.size(), false);
    if (reduce._ctx.resume != nullptr) {
      const std::string pass_key = secred::sector_pass_key(
          _settings._input_hash, pass, reduce._truncation.active,
          reduce._truncation.l, reduce._truncation.level_offset,
          reduce._truncation.dots, reduce._ctx.exemptCutRequests);
      reduce._ctx.sectorPassDir =
          secred::sector_resume_dir(_settings._cache_dir, pass_key).string();
      std::error_code ec;
      std::filesystem::create_directories(reduce._ctx.sectorPassDir, ec);

      const std::vector<unsigned> ids = reduce.sector_ids();
      const secred::SectorReplayPlan plan = secred::plan_sector_replay(
          reduce._ctx.sectorPassDir, ids, reduce._ctx.targets);
      if (plan.replayed != 0) {
        const unsigned batches = secred::apply_sector_replay(plan, reduce._ctx);
        reduce._ctx.completedSectors += plan.replayed;
        replayed = plan.replay;
        std::cout << "\n  [resume] replayed " << plan.replayed << " of "
                  << ids.size() << " sectors (" << batches
                  << " equation batch(es))\n";
      }
    } else {
      reduce._ctx.sectorPassDir.clear();
    }

    _generator.clear_ibp();

    const auto sector_t0 = std::chrono::steady_clock::now();
    secred::SectorStage::run_pass(stage_ctx, replayed);
    // A sector that exhausted its probe rounds left its part of the spill
    // short.  The workers only recorded it; fail here, after every worker has
    // joined and before any global stage can read the incomplete system.
    reduce._ctx.throwIfFailed();
    const auto sector_t1 = std::chrono::steady_clock::now();
    const double passSectorSeconds =
        std::chrono::duration<double>(sector_t1 - sector_t0).count();
    sectorSeconds += passSectorSeconds;

    std::cout << "\n  [timing] sector reduction: "
              << secred::format_duration(passSectorSeconds) << "\n";

    _generator.release_sector_rows();
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

      const bool probeTargets = _settings._probe_mode == ProbeMode::Targets;
      const char *probeMode = probeTargets ? "targets" : "all";
      const unsigned accepted = reduce._ctx.probeScopeSkipped.load();

      std::ofstream sfile(_settings._cache_dir + "/sectors/sectors",
                          std::ios::trunc);
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

      if (reduce._truncation.active)
        std::cout << "\n  Seed truncation: " << trunc << " sectors cut, "
                  << lifted << " lifted by the probe (l = "
                  << reduce._truncation.l << ", offset "
                  << reduce._truncation.level_offset << ")" << std::endl;
      else
        std::cout << "\n  Seed truncation: off (the whole seed box is admitted)"
                  << std::endl;

      std::cout << "\n  Seed box: spill raised " << spillRaised
                << " sectors (+" << spillDepth << " depth, +" << spillRank
                << " rank over targets-only)" << std::endl;

      if (probeTargets)
        std::cout << "\n  Probe scope (targets): " << accepted
                  << " out-of-box master reports accepted, not needed by the"
                  << " input targets" << std::endl;
      else
        std::cout << "\n  Probe scope (all): every out-of-box master report"
                  << " widens the box; seed_level_offset is void (offset 0)"
                  << std::endl;

      if (!have_global_system)
        std::cout << "\n Probe narrowing: succeeded " << ok << " / failed " << fail
                  << " / total " << total << " ("
                  << std::fixed << std::setprecision(1) << ratio << " % success)"
                  << std::endl;
    }

    const auto global_t0 = std::chrono::steady_clock::now();
    if (have_global_system)
      globalResult =
          secred::GlobalStage::run(stage_ctx, pass + 1 < kMaxSeedPasses);
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

    _generator.generate(_model);
    if (!_model._symbols.empty())
      reduce._rowTable.build_all(_generator._ibpM);
    reduce.prepare_sectors();
    if (raiseDots)
      reduce._truncation.dots = seedDots;
    reduce._ctx.resetRunStatistics();
  }

  {
    std::vector<RawIntegral> sector_masters;
    for (const auto &sector : reduce._ctx.allMasters)
      sector_masters.insert(sector_masters.end(), sector.begin(), sector.end());
    secred::write_masters_file(_settings._cache_dir + "/" +
                                   secred::kMastersSectorFile,
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

DryRunReport Family::dry_run_report(const Reduce &reduce) const {
  DryRunReport report;
  report.name = _model._name;
  report.nprops = _model._nprops;
  report.targets = reduce._rawTargets.size();
  report.ibp_relations = _generator._ibpFFmod.size();
  report.threads_sector = _settings._sthreads;
  report.threads_global = _settings._gthreads;
  report.max_primes = _settings._max_primes;
  report.truncation = reduce._truncation.active;
  report.machine_total = _settings._machine_total;
  report.machine_available = _settings._machine_available;
  report.memory_budget = _settings._max_memory;
  report.memory_reserve = _settings._memory_reserve;
  report.cgroup_limited = _settings._cgroup.limited;
  report.cgroup_limit = _settings._cgroup.limit;
  report.cgroup_current = _settings._cgroup.current;

  static const std::set<RawIntegral> no_integrals;
  for (unsigned id : reduce.sector_ids()) {
    const auto targetIt = reduce._ctx.targets.find(id);
    const auto upIt = reduce._ctx.uptargets.find(id);
    const std::set<RawIntegral> &targets =
        targetIt == reduce._ctx.targets.end() ? no_integrals : targetIt->second;
    const std::set<RawIntegral> &uptargets =
        upIt == reduce._ctx.uptargets.end() ? no_integrals : upIt->second;

    DryRunSectorEstimate estimate;
    estimate.id = id;
    estimate.lines = std::popcount(id);
    estimate.targets = targets.size();
    const auto [depth, rank] = initial_seed_bounds(id, targets, uptargets);
    SeedBound bound = SeedBound::initial(depth, rank);
    if (reduce._truncation.active)
      std::tie(bound.trunc_depth, bound.trunc_rank) = truncated_seed_caps(
          id, bound.depth, bound.rank, reduce._truncation);
    estimate.depth = bound.depth;
    estimate.rank = bound.rank;

    const SeedEstimate seeds = estimate_seeds(id, _model._nprops, bound);
    estimate.seeds = seeds.generated;
    estimate.admitted = seeds.admitted;
    estimate.equations =
        secred::saturating_mul(seeds.admitted, _generator._ibpFFmod.size());

    report.generated_seeds =
        secred::saturating_add(report.generated_seeds, estimate.seeds);
    report.estimated_seeds =
        secred::saturating_add(report.estimated_seeds, estimate.admitted);
    report.estimated_equations =
        secred::saturating_add(report.estimated_equations, estimate.equations);
    report.per_sector.push_back(estimate);
  }
  report.sectors = report.per_sector.size();
  return report;
}

void Family::print_dry_run(const Reduce &reduce) const {
  const DryRunReport report = dry_run_report(reduce);
  std::cout << "\n----------------- Dry Run ------------------\n";
  std::cout << "\n  Family: " << report.name
            << "   Propagators: " << report.nprops
            << "   IBP relations: " << report.ibp_relations;
  std::cout << "\n  Targets: " << report.targets
            << "   Sectors: " << report.sectors
            << "   Seed truncation: " << (report.truncation ? "on" : "off");
  std::cout << "\n\n  Threads: sectors " << report.threads_sector
            << ", global " << report.threads_global
            << "   Max primes: " << report.max_primes;
  std::cout << "\n\n  Memory: machine "
            << secred::gib_text(report.machine_available) << " available of "
            << secred::gib_text(report.machine_total)
            << "\n          budget " << secred::gib_text(report.memory_budget)
            << ", reserve " << secred::gib_text(report.memory_reserve);
  if (report.cgroup_limited)
    std::cout << "\n          cgroup limit "
              << secred::gib_text(report.cgroup_limit) << ", current "
              << secred::gib_text(report.cgroup_current);
  std::cout << "\n\n  Seeds (initial boxes): " << report.generated_seeds
            << " generated, " << report.estimated_seeds << " admitted";
  std::cout << "\n  Raw equations: ~" << report.estimated_equations
            << " from the initial target boxes (subsector requests and probe"
               " widening add rows)";
  std::cout << "\n\n  Per-sector estimates:\n"
            << "    sector  lines  targets  depth  rank      seeds"
               "   admitted  equations";
  for (const DryRunSectorEstimate &s : report.per_sector)
    std::cout << "\n    " << std::setw(6) << s.id << std::setw(7) << s.lines
              << std::setw(9) << s.targets << std::setw(7) << s.depth
              << std::setw(6) << s.rank << std::setw(11) << s.seeds
              << std::setw(11) << s.admitted << std::setw(11) << s.equations;
  std::cout << "\n\n  Dry run: no reduction, cache write or log was started.\n"
            << std::endl;
}
