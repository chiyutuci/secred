#pragma once

#include <filesystem>
#include <iostream>
#include <ranges>
#include <string>

#include "ginac/ginac.h"
#include "yaml-cpp/yaml.h"

#include "family_generator.h"
#include "family_model.h"
#include "globalreduce.h"
#include "row_coeff.h"
#include "run_settings.h"
#include "sector.h"
#include "symmetry.h"
#include "thread_count.h"
#include "utils.h"

namespace secred {

// The marker a directory acquires once SecRed has created it as a cache.  It
// lets a later run tell its own directory from an arbitrary path it must not
// recursively delete.
inline constexpr const char *kCacheMarkerFile = ".secred_cache";

// Validate, clear and create the run's cache directory.  Refuses the root, the
// current working directory, symlinks, ancestors of the working directory or
// the home directory, and directories whose entries are not SecRed's own.  A
// directory without the marker is adopted only when every entry is a known
// cache artifact, so existing 1.0 caches keep working.
std::filesystem::path prepare_cache_directory(const std::string &cache_dir);

struct CachePreparation {
  std::filesystem::path path;
  bool resumed = false;
};

// The cache-directory policy for a run.  With `resume` on and a manifest whose
// input hash, format version and build stamp all match, the directory is kept;
// otherwise it is cleared exactly as prepare_cache_directory() would.  A
// directory that is not SecRed's is still refused before anything is deleted.
CachePreparation prepare_run_cache_directory(const std::string &cache_dir,
                                             bool resume,
                                             std::uint64_t input_hash);

} // namespace secred

// How the input is checked and whether the run may touch the disk.
struct FamilyOptions {
  // Accept settings keys this build does not know, with a warning.  Keys that
  // were removed with a replacement still fail: they are known to be
  // ineffective, not unknown.
  bool allow_unknown_settings = false;
  // Validate and estimate only: no cache directory, run log, sector file or
  // reduction output is written.
  bool dry_run = false;
};

// One sector's place in a dry-run estimate.  The box is the initial one from
// the target requests; the run itself may widen it after a probe.
struct DryRunSectorEstimate {
  unsigned id = 0;
  unsigned lines = 0;
  unsigned targets = 0;
  unsigned depth = 0;
  unsigned rank = 0;
  std::uint64_t seeds = 0;      // generated seed entries
  std::uint64_t admitted = 0;   // entries the box admits
  std::uint64_t equations = 0;  // admitted entries x IBP rows
};

// The summary --dry-run prints.  It is built without allocating the seed
// lists or the reduction system.  `estimated_equations` counts the initial
// target boxes only: probe widening and subsector requests add rows, while
// zero coefficients and the subsector filter remove some.
struct DryRunReport {
  std::string name;
  unsigned nprops = 0;
  std::size_t targets = 0;
  std::size_t sectors = 0;
  std::size_t ibp_relations = 0;
  unsigned threads_sector = 0;
  unsigned threads_global = 0;
  unsigned max_primes = 0;
  bool truncation = false;
  std::uint64_t machine_total = 0;
  std::uint64_t machine_available = 0;
  std::uint64_t memory_budget = 0;
  std::uint64_t memory_reserve = 0;
  bool cgroup_limited = false;
  std::uint64_t cgroup_limit = 0;
  std::uint64_t cgroup_current = 0;
  std::uint64_t generated_seeds = 0;   // initial-box seed entries
  std::uint64_t estimated_seeds = 0;   // entries the boxes admit
  std::uint64_t estimated_equations = 0; // upper bound on raw equations
  std::vector<DryRunSectorEstimate> per_sector;
};

// The input family: a model, its settings and its generated rows.  The stages
// in family_stages.h consume them through a RunContext; Family itself is the
// facade the driver and the tests use.
class Family {
public:
  explicit Family(const YAML::Node &, const FamilyOptions & = {});

  // Parse the family and build its IBP generators.
  void init();

  // Fill a Reduce from the targets and options.
  void init_reduce(const YAML::Node &, Reduce &) const;

  void print() const;

  // The dry-run summary for a Reduce prepared by init_reduce().
  DryRunReport dry_run_report(const Reduce &) const;
  void print_dry_run(const Reduce &) const;

  // Run the sector phase and the global reduction.
  void run_reduce(Reduce &);

  unsigned nprops() const { return _model._nprops; }

  const std::string &cache_dir() const { return _settings._cache_dir; }

  // Sectors with a non-trivial line mask below top.
  std::vector<bool> nontrivial_sectors(unsigned top) const {
    return _model.nontrivial_sectors(top);
  }

  SymmetryTable::Inputs symmetry_inputs() const {
    return _model.symmetry_inputs();
  }

  const FamilyModel &model() const { return _model; }
  const RunSettings &settings() const { return _settings; }
  const FamilyGenerator &generator() const { return _generator; }

  static GiNaC::symtab symtab;

private:
  FamilyModel _model;
  RunSettings _settings;
  FamilyGenerator _generator;
  bool _resumed = false;
};

// One reduction run: targets, sectors and the global context.
class Reduce {
public:

  // Build the sector list and the per-sector target maps.
  void prepare_sectors();

  void print() const;

  std::vector<unsigned> sector_ids() const {
    std::vector<unsigned> ids;
    ids.reserve(_reduceSectors.size());
    for (const auto &s : _reduceSectors)
      ids.push_back(s.id());
    return ids;
  }

  const SymmetryTable &symmetries() const { return _symmetries; }

  const secred::RowTable *row_table() const { return _ctx.rowTable; }

  const std::set<RawIntegral> &preferred() const { return _preferred; }

  const SeedTruncation &truncation() const { return _truncation; }

  ProbeMode probe_mode() const { return _probe_mode; }

  bool truncate() const { return _truncate; }

  bool mandatory_list() const { return _mandatory_list; }

  // The run's state.  It is public because the stages own pieces of it (the
  // targets, the sector plan, the context's statistics) and RunContext is the
  // interface that hands it to them; a friend declaration would make the
  // stages reach through Family instead.
  std::vector<GiNaC::possymbol> _symbols;
  std::vector<GiNaC::symbol> _symIndices;

  std::vector<RawIntegral> _rawTargets;

  std::set<RawIntegral> _preferred;
  std::set<unsigned> _targetSectors;

  SeedTruncation _truncation;

  unsigned _seed_level_offset = kSeedLevelOffset;
  bool _seed_level_offset_set = false;

  ProbeMode _probe_mode = ProbeMode::Targets;

  bool _truncate = false;

  bool _mandatory_list = true;

  unsigned _top = 0;

  unsigned _nprops = 0;

  std::vector<bool> _lines;

  std::string _cache_dir;

  std::vector<unsigned> _cut;
  unsigned _ncut = 0;

  std::vector<bool> _sectors;

  std::vector<Sector> _reduceSectors;

  SymmetryTable _symmetries;

  secred::RowTable _rowTable;
  ReductionContext _ctx;

  // Appended after 1.0 so _rowTable and _ctx keep their offsets.
  bool _dry_run = false;
};
