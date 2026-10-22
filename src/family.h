#pragma once

#include <iostream>
#include <ranges>
#include <string>

#include "ginac/ginac.h"
#include "yaml-cpp/yaml.h"

#include "globalreduce.h"
#include "row_coeff.h"
#include "sector.h"
#include "symmetry.h"
#include "utils.h"

// The global engine a system's symbol count selects: none -> the numeric
// FiniteFlow path, one or more -> FireFly.  The FiniteFlow rational-function
// engine (FiniteFlowSymbolic) is no longer selected by any symbol count; its
// branch and unit tests stay, and reduce_global_finiteflow_symbolic is still
// callable directly.
enum class GlobalEngine { FiniteFlowNumeric, FiniteFlowSymbolic, FireFlySymbolic };

inline GlobalEngine global_engine_for(std::size_t nsymbols) {
  if (nsymbols == 0)
    return GlobalEngine::FiniteFlowNumeric;
  return GlobalEngine::FireFlySymbolic;
}

// The name the engine's failures carry.
inline const char *global_engine_name(GlobalEngine engine) {
  return engine == GlobalEngine::FireFlySymbolic ? "firefly" : "finiteflow";
}

// The input family: kinematics, propagators, targets and run options.
class Family {
public:
  friend class FamilyTestHelper;

  explicit Family(const YAML::Node &);

  // Parse the family and build its IBP generators.
  void init();

  // Fill a Reduce from the targets and options.
  void init_reduce(const YAML::Node &, Reduce &) const;

  void print() const;

  // Run the sector phase and the global reduction.
  void run_reduce(Reduce &);

  unsigned nprops() const { return _nprops; }

  const std::string &cache_dir() const { return _cache_dir; }

  // Sectors with a non-trivial line mask below top.
  std::vector<bool> nontrivial_sectors(unsigned top) const;

  SymmetryTable::Inputs symmetry_inputs() const;

private:

  // Derive scalar products from the propagators.
  void _compute_sps();

  void _compute_symanzik();

  void _generate_ibp();

  void _generate_li();

  void _generate_ibp_ff_n();
  void _generate_ibp_ff_m();

  void _search_trivial_sectors(Reduce &) const;

  void _run_reduce_layer(Reduce &) const;
  void _run_reduce_fine(Reduce &) const;

  // Reduce the assembled system to the final basis.
  GlobalReductionResult _run_global_reduction(Reduce &, bool allow_basis_vote) const;

public:
  static GiNaC::symtab symtab;

private:

  std::string _name;

  GiNaC::ex _dimension;

  std::vector<GiNaC::possymbol> _internals;

  std::vector<GiNaC::possymbol> _externals;

  unsigned _nints = 0;

  unsigned _nexts = 0;

  std::vector<std::pair<GiNaC::possymbol, unsigned>> _invariants;

  std::vector<GiNaC::possymbol> _symbols;

  std::vector<GiNaC::ex> _propagators;

  std::vector<GiNaC::ex> _propMomenta;

  GiNaC::lst _spsRules;

  GiNaC::lst _spsFromProps;

  GiNaC::lst _one;

  unsigned _nprops = 0;

  unsigned _sthreads = 64;
  std::string _parallel_mode = "fine";

  std::string _cache_dir = "cache";

  unsigned _subsector_filter_n = 1;

  unsigned _seed_level_offset = kSeedLevelOffset;
  bool _seed_level_offset_set = false;

  ProbeMode _probe_mode = ProbeMode::Targets;
  bool _probe_mode_set = false;

  // Seed truncation (Kira's truncate_sp): off by default, on when a probe option
  // is named.  An explicit `truncate: false` forces the pre-2026-10-17 walk --
  // unscoped probe and no mandatory rows -- overriding `probe_mode`.
  bool _truncate = false;
  bool _truncate_set = false;

  bool _mandatory_list = true;

  bool _sector_debug = false;

  bool _symmetry = true;

  OrderingMode _ordering_mode      = OrderingMode::Balance;

  std::uint64_t _memory_reserve = 30ULL * 1024 * 1024 * 1024;

  std::uint64_t _max_memory = 512ULL * 1024 * 1024 * 1024;

  std::vector<unsigned> _cut;
  unsigned _ncut = 0;

  unsigned _gthreads = 64;

  unsigned _max_primes = 500;

  std::vector<GiNaC::symbol> _symIndices;

  std::vector<GiNaC::symbol> _symProps;

  GiNaC::ex _uPoly;

  GiNaC::ex _fPoly;

  std::vector<IBPProto> _ibp;

  std::vector<IBPProtoFF> _ibpFF;

  std::vector<IBPProtoFFmod> _ibpFFmod;
  std::vector<IBPProtoM> _ibpM;
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

  friend class Family;

private:

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
};
