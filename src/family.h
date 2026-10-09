#pragma once

#include <iostream>
#include <ranges>
#include <string>

#include "ginac/ginac.h"
#include "yaml-cpp/yaml.h"

#include "sector.h"
#include "utils.h"

class Family {
public:
  friend class FamilyTestHelper; // exposes private state to tests

  explicit Family(const YAML::Node &);

  // initialize the family
  void init();
  // prepare reduce information
  void init_reduce(const YAML::Node &, Reduce &) const;
  // print family information
  void print() const;

  // run the reduction
  void run_reduce(Reduce &);


private:
  // compute expressions of scalar products over propagators
  void _compute_sps();
  // compute symanzik polynomials
  void _compute_symanzik();

  // generate ibp relations
  void _generate_ibp();
  // generate li relations
  void _generate_li();
  // generate ibp over finite filed
  void _generate_ibp_ff_n();
  void _generate_ibp_ff_m();
  // search trivial sectors
  void _search_trivial_sectors(Reduce &) const;
  // parallel reduction strategies
  void _run_reduce_layer(Reduce &) const;
  void _run_reduce_fine(Reduce &) const;
  // post-reduction global kira solve (ff_n path only)
  void _run_global_kira(Reduce &) const;

public:
  static GiNaC::symtab symtab;

private:
  // family name
  std::string _name;
  // dimension of the family
  GiNaC::ex _dimension;
  // loop momenta
  std::vector<GiNaC::possymbol> _internals;
  // external momenta
  std::vector<GiNaC::possymbol> _externals;
  // number of internals
  unsigned _nints = 0;
  // number of externals
  unsigned _nexts = 0;
  // kinematics invariants
  //  first: name
  //  second: dimension
  std::vector<std::pair<GiNaC::possymbol, unsigned>> _invariants;
  // symbols
  std::vector<GiNaC::possymbol> _symbols;
  // propagators
  std::vector<GiNaC::ex> _propagators;
  // scalar products rules
  GiNaC::lst _spsRules;
  // scalar products over propagators
  GiNaC::lst _spsFromProps;
  // the invariant set to one
  GiNaC::lst _one;
  // number of propagators
  unsigned _nprops = 0;

  unsigned _threads = 64;
  std::string _parallel_mode = "fine";

  std::string _cache_dir      = "cache";
  std::string _job_dir        = "";
  std::string _kira_cmd       = "kira";
  std::string _kira_jobs_file = "jobs.yaml";
  unsigned _kira_bunch_size   = 64;
  unsigned _kira_parallel     = 64;
  unsigned _subsector_depth_offset = 1;
  OrderingMode _ordering_mode      = OrderingMode::Balance;
  // System-memory headroom kept free during parallel sector execution (default 60 GB).
  std::uint64_t _memory_reserve = 60ULL * 1024 * 1024 * 1024;
  // Allowed total memory for the reduction, in bytes (default 512 GB). Sectors
  // are admitted only while the process stays under _max_memory - _memory_reserve;
  // set smaller to simulate a machine with less RAM.
  std::uint64_t _max_memory = 512ULL * 1024 * 1024 * 1024;

  std::vector<unsigned> _cut;
  unsigned _ncut = 0;

  // on-the-fly loading in the global Kira reduction (parsed from settings:
  // global_otf: true/false); when false the whole system is materialised.
  bool _global_otf = false;
  // which tool performs the global reduction: "kira" (external Kira subprocess,
  // default) or "finiteflow" (in-process FiniteFlow; pyred + CRT for pure-numeric
  // systems, pyred + FiniteFlow rational-function reconstruction for symbolic
  // systems). Parsed from settings: global_tool.
  std::string _global_tool = "kira";
  // number of threads for the FiniteFlow global reconstruction
  // (parsed from settings: global_finiteflow_threads; default 64). The Gaussian
  // elimination itself is single-threaded (FiniteFlow), but the per-coefficient
  // rational reconstruction (CRT + rat_rec) is parallelised across threads.
  unsigned _global_finiteflow_threads = 64;

  // symbols for integral indices
  std::vector<GiNaC::symbol> _symIndices;
  // symbols for integral propagators
  std::vector<GiNaC::symbol> _symProps;
  // symanzik U
  GiNaC::ex _uPoly;
  // symanzik F
  GiNaC::ex _fPoly;

  // ibp relations prototype
  std::vector<IBPProto> _ibp;
  // ibp relations prototype over finite field
  std::vector<IBPProtoFF> _ibpFF;
  // ibp relations over finite field with ffint (residue) coefficients, for the
  // compact FiniteFlow sparse-solver input.
  std::vector<IBPProtoFFmod> _ibpFFmod;
  std::vector<IBPProtoM> _ibpM;
};

class Reduce {
public:
  // prepare sectors relations
  void prepare_sectors();

  // print reduciton job info
  void print() const;

  // ordered sector IDs after prepare_sectors() (descending popcount)
  std::vector<unsigned> sector_ids() const {
    std::vector<unsigned> ids;
    ids.reserve(_reduceSectors.size());
    for (const auto &s : _reduceSectors)
      ids.push_back(s.id());
    return ids;
  }

  friend class Family;

private:
  // symbols
  std::vector<GiNaC::possymbol> _symbols;
  std::vector<GiNaC::symbol> _symIndices;
  // raw target integrals
  std::vector<RawIntegral> _rawTargets;
  std::set<unsigned> _targetSectors;
  // top sector
  unsigned _top = 0;
  // number of propagators
  unsigned _nprops = 0;
  // top sector lines
  std::vector<bool> _lines;

  std::string _cache_dir;

  std::vector<unsigned> _cut;
  unsigned _ncut = 0;
  // non-trivial sectors
  // true: non-trivial
  // false: trivial or irrelevant
  std::vector<bool> _sectors;
  // the reduction jobs
  std::vector<Sector> _reduceSectors;
  ReductionContext _ctx;
};