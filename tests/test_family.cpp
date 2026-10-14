#include "test_runner.h"
#include "family.h"
#include "row_coeff.h"
#include <filesystem>
#include <sstream>

// ---------------------------------------------------------------------------
// Helper: clear the global GiNaC symbol table between tests so duplicate-
//         symbol checks don't interfere with each other.
// ---------------------------------------------------------------------------

static void clear_symtab() { Family::symtab.clear(); }

// Helper: builds a minimal one-loop box YAML node.
static YAML::Node box_config(const std::string &cache_dir = "test_cache_box") {
  std::string yaml = R"(
settings:
  cache_dir: )" + cache_dir + R"(
  threads: 4

family:
  name: box
  dimension: d
  internals: [l1]
  externals: [p1, p2, p3]
  invariants:
    - [s, 2]
    - [t, 2]
    - [m, 1]
  invar_one: m
  sps_rules:
    - [p1, p1, 0]
    - [p2, p2, 0]
    - [p3, p3, 0]
    - [p1, p2, "s/2"]
    - [p1, p3, "t/2"]
    - [p2, p3, "-s/2-t/2"]
  propagators:
    - [l1, 0]
    - [l1+p1, 0]
    - [l1+p1+p2, 0]
    - [l1+p1+p2+p3, 0]

targets:
  - [2, 1, 1, 1]
)";
  return YAML::Load(yaml);
}

// Helper: bubble topology (1 loop, 2 external, 2 propagators)
static YAML::Node bubble_config(const std::string &cache_dir = "test_cache_bubble") {
  std::string yaml = R"(
settings:
  cache_dir: )" + cache_dir + R"(

family:
  name: bubble
  dimension: d
  internals: [l]
  externals: [p]
  invariants:
    - [p2, 2]
  sps_rules:
    - [p, p, p2]
  propagators:
    - [l, 0]
    - [l+p, 0]

targets:
  - [1, 1]
)";
  return YAML::Load(yaml);
}

// ---------------------------------------------------------------------------
// Test accessor: exposes Family private internals via the friend declaration.
// ---------------------------------------------------------------------------

class FamilyTestHelper {
public:
  static const std::vector<IBPProto> &ibp(const Family &f) { return f._ibp; }
  static const std::vector<IBPProtoFF> &ibpFF(const Family &f) {
    return f._ibpFF;
  }
  static const std::vector<IBPProtoFFmod> &ibpFFmod(const Family &f) {
    return f._ibpFFmod;
  }
  static const std::vector<IBPProtoM> &ibpM(const Family &f) { return f._ibpM; }
  static const std::vector<GiNaC::possymbol> &symbols(const Family &f) {
    return f._symbols;
  }
  static std::uint64_t memory_reserve(const Family &f) {
    return f._memory_reserve;
  }
  static const std::string &global_tool(const Family &f) {
    return f._global_tool;
  }
  static unsigned global_finiteflow_threads(const Family &f) {
    return f._global_finiteflow_threads;
  }
  static unsigned threads(const Family &f) { return f._threads; }
  static const std::string &parallel_mode(const Family &f) {
    return f._parallel_mode;
  }
  static OrderingMode ordering_mode(const Family &f) {
    return f._ordering_mode;
  }
  static unsigned subsector_filter_n(const Family &f) {
    return f._subsector_filter_n;
  }
  static const std::string &cache_dir(const Family &f) {
    return f._cache_dir;
  }
};

// ---------------------------------------------------------------------------
// Family YAML parsing — valid configurations
// ---------------------------------------------------------------------------

TEST(family_parse_box_name) {
  clear_symtab();
  YAML::Node cfg = box_config("tc_name");
  Family f(cfg);
  // If construction succeeds, parsing was correct.
  // We verify observable state via print() not throwing.
  // (name is private; observable effect is successful construction)
}

TEST(family_parse_propagator_count_box) {
  // box: nints=1, nexts=3
  // nprops = nints*nexts + nints*(nints+1)/2 = 1*3 + 1 = 4
  clear_symtab();
  YAML::Node cfg = box_config("tc_nprops_box");
  // If the propagator count were wrong, yaml-cpp validation would throw
  // "number of propagators does not match the topology"
  Family f(cfg);
}

TEST(family_parse_propagator_count_bubble) {
  // bubble: nints=1, nexts=1
  // nprops = 1*1 + 1*2/2 = 1 + 1 = 2
  clear_symtab();
  YAML::Node cfg = bubble_config("tc_nprops_bubble");
  Family f(cfg);
}

TEST(family_parse_threads) {
  // The box config sets threads: 4 — verify the explicit value is stored.
  clear_symtab();
  YAML::Node cfg = box_config("tc_threads");
  Family f(cfg);
  REQUIRE(FamilyTestHelper::threads(f) == 4);
}

TEST(family_parse_threads_default) {
  // Without a threads entry the default (64) applies.
  clear_symtab();
  YAML::Node cfg = box_config("tc_threads_default");
  cfg["settings"].remove("threads");
  Family f(cfg);
  REQUIRE(FamilyTestHelper::threads(f) == 64);
}

TEST(family_parse_settings_cache_dir) {
  clear_symtab();
  YAML::Node cfg = box_config("my_custom_cache");
  Family f(cfg);
  // If the settings section is parsed, init() will use "my_custom_cache".
  // We verify by calling init() and checking the directory exists.
  f.init();
  REQUIRE(std::filesystem::exists("my_custom_cache"));
  std::filesystem::remove_all("my_custom_cache");
}

TEST(family_parse_memory_reserve_default) {
  clear_symtab();
  YAML::Node cfg = box_config("tc_memreserve_default");
  Family f(cfg);
  REQUIRE(FamilyTestHelper::memory_reserve(f) == 60ULL * 1024 * 1024 * 1024);
}

TEST(family_parse_memory_reserve_custom) {
  clear_symtab();
  YAML::Node cfg = box_config("tc_memreserve_custom");
  cfg["settings"]["memory_reserve"] = 40;
  Family f(cfg);
  REQUIRE(FamilyTestHelper::memory_reserve(f) == 40ULL * 1024 * 1024 * 1024);
}

TEST(family_parse_global_tool_default) {
  clear_symtab();
  YAML::Node cfg = box_config("tc_global_tool_default");
  Family f(cfg);
  REQUIRE(FamilyTestHelper::global_tool(f) == "finiteflow");
}

TEST(family_parse_global_tool_finiteflow) {
  clear_symtab();
  YAML::Node cfg = box_config("tc_global_tool_finiteflow");
  cfg["settings"]["global_tool"] = "finiteflow";
  Family f(cfg);
  REQUIRE(FamilyTestHelper::global_tool(f) == "finiteflow");
}

TEST(family_parse_global_tool_rejects_an_unknown_value) {
  clear_symtab();
  YAML::Node cfg = box_config("tc_global_tool_unknown");
  cfg["settings"]["global_tool"] = "fireflyy";
  REQUIRE_THROWS_AS(Family(cfg), std::runtime_error);
}

TEST(family_parse_global_tool_firefly) {
  clear_symtab();
  YAML::Node cfg = box_config("tc_global_tool_firefly");
  cfg["settings"]["global_tool"] = "firefly";
  Family f(cfg);
  REQUIRE(FamilyTestHelper::global_tool(f) == "firefly");
}

TEST(family_parse_global_tool_accepts_both_backends) {
  for (const char *tool : {"finiteflow", "firefly"}) {
    clear_symtab();
    YAML::Node cfg = box_config(std::string("tc_global_tool_ok_") + tool);
    cfg["settings"]["global_tool"] = tool;
    REQUIRE_NOTHROW(Family(cfg));
  }
}

TEST(family_parse_global_tool_rejects_an_external_solver) {
  // The global stage runs in this process; there is no external backend to name.
  clear_symtab();
  YAML::Node cfg = box_config("tc_global_tool_kira");
  cfg["settings"]["global_tool"] = "kira";
  REQUIRE_THROWS_AS(Family(cfg), std::runtime_error);
}

TEST(family_parse_global_finiteflow_threads_default) {
  clear_symtab();
  YAML::Node cfg = box_config("tc_global_finiteflow_threads_default");
  Family f(cfg);
  REQUIRE(FamilyTestHelper::global_finiteflow_threads(f) == 64);
}

TEST(family_parse_global_finiteflow_threads_custom) {
  clear_symtab();
  YAML::Node cfg = box_config("tc_global_finiteflow_threads_custom");
  cfg["settings"]["global_finiteflow_threads"] = 32;
  Family f(cfg);
  REQUIRE(FamilyTestHelper::global_finiteflow_threads(f) == 32);
}

TEST(family_parse_settings_custom_cache_dir) {
  clear_symtab();
  std::string yaml = R"(
settings:
  cache_dir: tc_custom_cache

family:
  name: box2
  dimension: d
  internals: [ll]
  externals: [q1, q2, q3]
  invariants:
    - [ss, 2]
    - [tt, 2]
    - [mm, 1]
  invar_one: mm
  sps_rules:
    - [q1, q1, 0]
    - [q2, q2, 0]
    - [q3, q3, 0]
    - [q1, q2, "ss/2"]
    - [q1, q3, "tt/2"]
    - [q2, q3, "-ss/2-tt/2"]
  propagators:
    - [ll, 0]
    - [ll+q1, 0]
    - [ll+q1+q2, 0]
    - [ll+q1+q2+q3, 0]
targets:
  - [1, 1, 1, 1]
)";
  YAML::Node cfg = YAML::Load(yaml);
  Family f(cfg);
  // The cache directory is what the run writes into; reading it back is the
  // verifiable signal here.
  REQUIRE(FamilyTestHelper::cache_dir(f) == "tc_custom_cache");
}

// ---------------------------------------------------------------------------
// Family YAML parsing — error cases
// ---------------------------------------------------------------------------

TEST(family_missing_family_key) {
  clear_symtab();
  YAML::Node cfg = YAML::Load("targets:\n  - [1,1,1,1]\n");
  REQUIRE_THROWS_AS(Family(cfg), std::runtime_error);
}

TEST(family_missing_name) {
  clear_symtab();
  YAML::Node cfg = YAML::Load(R"(
family:
  internals: [l]
  externals: [p]
  sps_rules:
    - [p, p, 0]
  propagators:
    - [l, 0]
    - [l+p, 0]
targets: []
)");
  REQUIRE_THROWS_AS(Family(cfg), std::runtime_error);
}

TEST(family_missing_internals) {
  clear_symtab();
  YAML::Node cfg = YAML::Load(R"(
family:
  name: bad
  externals: [p]
  sps_rules:
    - [p, p, 0]
  propagators:
    - [x, 0]
targets: []
)");
  REQUIRE_THROWS_AS(Family(cfg), std::runtime_error);
}

TEST(family_missing_externals) {
  clear_symtab();
  YAML::Node cfg = YAML::Load(R"(
family:
  name: bad2
  internals: [l]
  sps_rules:
    - [l, l, 0]
  propagators:
    - [l, 0]
targets: []
)");
  REQUIRE_THROWS_AS(Family(cfg), std::runtime_error);
}

TEST(family_missing_sps_rules) {
  clear_symtab();
  YAML::Node cfg = YAML::Load(R"(
family:
  name: bad3
  internals: [la]
  externals: [pa]
  propagators:
    - [la, 0]
    - [la+pa, 0]
targets: []
)");
  REQUIRE_THROWS_AS(Family(cfg), std::runtime_error);
}

TEST(family_missing_propagators) {
  clear_symtab();
  YAML::Node cfg = YAML::Load(R"(
family:
  name: bad4
  internals: [lb]
  externals: [pb]
  sps_rules:
    - [pb, pb, 0]
targets: []
)");
  REQUIRE_THROWS_AS(Family(cfg), std::runtime_error);
}

TEST(family_wrong_propagator_count) {
  // nprops should be 2 for 1 loop + 1 external, but we give 3
  clear_symtab();
  YAML::Node cfg = YAML::Load(R"(
family:
  name: bad5
  internals: [lc]
  externals: [pc]
  sps_rules:
    - [pc, pc, 0]
  propagators:
    - [lc, 0]
    - [lc+pc, 0]
    - [lc+pc+pc, 0]
targets: []
)");
  REQUIRE_THROWS_AS(Family(cfg), std::runtime_error);
}

TEST(family_wrong_sps_rules_count) {
  // box needs 6 sps_rules (nexts*(nexts+1)/2 = 3*4/2 = 6), but we give 3
  clear_symtab();
  YAML::Node cfg = YAML::Load(R"(
family:
  name: bad6
  internals: [ld]
  externals: [pd1, pd2, pd3]
  sps_rules:
    - [pd1, pd1, 0]
    - [pd2, pd2, 0]
    - [pd3, pd3, 0]
  propagators:
    - [ld, 0]
    - [ld+pd1, 0]
    - [ld+pd1+pd2, 0]
    - [ld+pd1+pd2+pd3, 0]
targets: []
)");
  REQUIRE_THROWS_AS(Family(cfg), std::runtime_error);
}

TEST(family_duplicate_symbol_rejected) {
  clear_symtab();
  // Using "l1" as both internal and external
  YAML::Node cfg = YAML::Load(R"(
family:
  name: dup
  internals: [l1]
  externals: [l1]
  sps_rules:
    - [l1, l1, 0]
  propagators:
    - [l1, 0]
    - [l1, 0]
targets: []
)");
  REQUIRE_THROWS_AS(Family(cfg), std::runtime_error);
}

TEST(family_invalid_invar_one) {
  clear_symtab();
  YAML::Node cfg = YAML::Load(R"(
family:
  name: badinvar
  internals: [le]
  externals: [pe]
  invariants:
    - [qe, 2]
  invar_one: nonexistent
  sps_rules:
    - [pe, pe, qe]
  propagators:
    - [le, 0]
    - [le+pe, 0]
targets: []
)");
  REQUIRE_THROWS_AS(Family(cfg), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Family::init() — calls _compute_sps + _compute_symanzik + _generate_ibp
// ---------------------------------------------------------------------------

TEST(family_init_creates_cache_dir) {
  clear_symtab();
  YAML::Node cfg = box_config("tc_init_box");
  Family f(cfg);
  f.init();
  REQUIRE(std::filesystem::exists("tc_init_box"));
  std::filesystem::remove_all("tc_init_box");
}

TEST(family_init_bubble_sps_non_singular) {
  // If the propagator matrix is singular, _compute_sps throws.
  // A valid bubble should not throw.
  clear_symtab();
  YAML::Node cfg = bubble_config("tc_bubble_sps");
  Family f(cfg);
  // Should not throw
  f.init();
  std::filesystem::remove_all("tc_bubble_sps");
}

TEST(family_init_generates_ibp_relations) {
  // Box with 1 loop + 3 externals generates IBP relations.
  // nints * (nints + nexts) = 1 * 4 = 4 IBP identities.
  // LI identities: nexts*(nexts-1)/2 = 3 identities.
  // Total: up to 7 IBPProto entries (some may be zero after substitution).
  // Observable: init() should complete without throwing.
  clear_symtab();
  YAML::Node cfg = box_config("tc_box_ibp");
  Family f(cfg);
  f.init();
  // Also test print() doesn't throw
  f.print();
  std::filesystem::remove_all("tc_box_ibp");
}

TEST(family_signed_path_generates_numeric_and_symbolic_ibp) {
  // A symbolic family (symbolic d, s, t) must generate the finite-field IBP
  // proto (used for the FiniteFlow structure) and the symbolic proto (used for
  // the symbolic reduction).  The two must share the same term structure so
  // the (seed, ibp) index recorded by the numeric elimination maps 1:1 onto the
  // symbolic equations.  The full rational _ibpFF is not needed on the signed
  // path (the symbolic _ibpM supplies the coefficients), so it stays empty.
  clear_symtab();
  YAML::Node cfg = box_config("tc_box_signed");
  Family f(cfg);
  f.init();

  const auto &ibp = FamilyTestHelper::ibp(f);
  const auto &ibpFF = FamilyTestHelper::ibpFF(f);
  const auto &ibpFFmod = FamilyTestHelper::ibpFFmod(f);
  const auto &ibpM = FamilyTestHelper::ibpM(f);

  REQUIRE(!ibp.empty());
  REQUIRE(ibpFF.empty());  // rational proto is dead weight on the signed path
  REQUIRE(ibpFFmod.size() == ibp.size());
  REQUIRE(ibpM.size() == ibp.size());

  for (size_t i = 0; i < ibp.size(); ++i) {
    REQUIRE(ibpFFmod[i].size() == ibpM[i].size());
    for (size_t j = 0; j < ibpFFmod[i].size(); ++j) {
      REQUIRE(ibpFFmod[i][j].first == ibpM[i][j].first);
      REQUIRE(ibpFFmod[i][j].second.size() == ibpM[i][j].second.size());
    }
  }

  std::filesystem::remove_all("tc_box_signed");
}

// ---------------------------------------------------------------------------
// init_reduce — target parsing and trivial sector detection
// ---------------------------------------------------------------------------

TEST(init_reduce_requires_targets) {
  clear_symtab();
  YAML::Node cfg = YAML::Load(R"(
settings:
  cache_dir: tc_notargets
family:
  name: nt
  internals: [lnt]
  externals: [pnt]
  sps_rules:
    - [pnt, pnt, 0]
  propagators:
    - [lnt, 0]
    - [lnt+pnt, 0]
)");  // no "targets" key
  Family f(cfg);
  f.init();
  Reduce r;
  REQUIRE_THROWS_AS(f.init_reduce(cfg, r), std::runtime_error);
  std::filesystem::remove_all("tc_notargets");
}

TEST(init_reduce_rejects_wrong_size_target) {
  clear_symtab();
  YAML::Node cfg = YAML::Load(R"(
settings:
  cache_dir: tc_wrongtarget
family:
  name: wt
  internals: [lwt]
  externals: [pwt]
  sps_rules:
    - [pwt, pwt, 0]
  propagators:
    - [lwt, 0]
    - [lwt+pwt, 0]
targets:
  - [1, 1, 1]
)");  // target has 3 indices but nprops=2
  Family f(cfg);
  f.init();
  Reduce r;
  REQUIRE_THROWS_AS(f.init_reduce(cfg, r), std::runtime_error);
  std::filesystem::remove_all("tc_wrongtarget");
}

TEST(init_reduce_bubble_succeeds) {
  clear_symtab();
  YAML::Node cfg = bubble_config("tc_bubble_reduce");
  Family f(cfg);
  f.init();
  Reduce r;
  // Should not throw; all sectors processed
  f.init_reduce(cfg, r);
  std::filesystem::remove_all("tc_bubble_reduce");
}

TEST(init_reduce_prints_correctly) {
  clear_symtab();
  YAML::Node cfg = bubble_config("tc_bubble_print");
  Family f(cfg);
  f.init();
  Reduce r;
  f.init_reduce(cfg, r);
  // Should not throw
  r.print();
  std::filesystem::remove_all("tc_bubble_print");
}

// The settling round's row table belongs to the family, not to the sector.
// Family::init_reduce builds it once -- before prepare_sectors starts any
// thread -- so a family's IBP generators are expanded once instead of once per
// sector under one process-wide mutex, and every later seed is a lock-free
// integer evaluation.  Read-only from that point on (ctx.rowTable).
TEST(init_reduce_builds_the_family_row_table_once) {
  clear_symtab();
  YAML::Node cfg = box_config("tc_rowtable_box");
  Family f(cfg);
  f.init();
  Reduce r;
  f.init_reduce(cfg, r);
  const secred::RowTable *table = r.row_table();
  REQUIRE(table != nullptr);
  const auto &ibpM = FamilyTestHelper::ibpM(f);
  REQUIRE(!ibpM.empty());
  unsigned total = 0;
  for (unsigned i = 0; i < ibpM.size(); ++i) {
    total += ibpM[i].size();
    if (!ibpM[i].empty())
      REQUIRE_MSG(table->contains(i), "generator " + std::to_string(i) +
                                          " is missing from the family row table");
  }
  // Every item row of every generator, not just the ones a first sector used.
  REQUIRE_EQ(table->fast_rows + table->slow_rows, total);
  std::filesystem::remove_all("tc_rowtable_box");
}

// A numeric family -- no symbols, so `dimension` is a number and the invariant
// is the one set to unity -- has no symbolic coefficients to precompile: its
// settling round evaluates integer coefficient vectors instead, and the table
// stays null so the sector never looks for one.
TEST(init_reduce_leaves_a_numeric_family_without_a_row_table) {
  clear_symtab();
  YAML::Node cfg = YAML::Load(R"(
settings:
  cache_dir: tc_rowtable_num
family:
  name: bubnum
  dimension: 1999/500
  internals: [l]
  externals: [p]
  invariants:
    - [p2, 2]
  invar_one: p2
  sps_rules:
    - [p, p, p2]
  propagators:
    - [l, 0]
    - [l+p, 0]

targets:
  - [1, 1]
)");
  Family f(cfg);
  f.init();
  Reduce r;
  f.init_reduce(cfg, r);
  REQUIRE(r.row_table() == nullptr);
  std::filesystem::remove_all("tc_rowtable_num");
}

// ---------------------------------------------------------------------------
// Hierarchical parallel execution — sector ordering invariant
//
// run_reduce processes layers in descending popcount order, relying on
// _reduceSectors being sorted that way by prepare_sectors().  These tests
// verify that invariant so the layer-barrier logic in run_reduce is correct.
// ---------------------------------------------------------------------------

TEST(sectors_ordered_by_descending_popcount_bubble) {
  // Bubble (sector 0b11 only after trivial-sector pruning):
  // single non-trivial sector → ordering trivially satisfied.
  clear_symtab();
  YAML::Node cfg = bubble_config("tc_order_bubble");
  Family f(cfg);
  f.init();
  Reduce r;
  f.init_reduce(cfg, r);
  const auto ids = r.sector_ids();
  for (unsigned i = 1; i < ids.size(); ++i) {
    REQUIRE(std::popcount(ids[i - 1]) >= std::popcount(ids[i]));
  }
  std::filesystem::remove_all("tc_order_bubble");
}

TEST(sectors_ordered_by_descending_popcount_box) {
  // Box topology has multiple layers; verify descending-popcount ordering.
  clear_symtab();
  YAML::Node cfg = box_config("tc_order_box");
  Family f(cfg);
  f.init();
  Reduce r;
  f.init_reduce(cfg, r);
  const auto ids = r.sector_ids();
  for (unsigned i = 1; i < ids.size(); ++i) {
    REQUIRE(std::popcount(ids[i - 1]) >= std::popcount(ids[i]));
  }
  std::filesystem::remove_all("tc_order_box");
}

TEST(sectors_same_layer_grouped_contiguously) {
  // Within a layer (same popcount), all those sectors must appear
  // consecutively — the layer-barrier logic in run_reduce depends on this.
  clear_symtab();
  YAML::Node cfg = box_config("tc_grouped_box");
  Family f(cfg);
  f.init();
  Reduce r;
  f.init_reduce(cfg, r);
  const auto ids = r.sector_ids();
  // scan: once we see a popcount N, we must not see a higher popcount later
  unsigned max_seen = UINT_MAX;
  for (unsigned id : ids) {
    unsigned pc = std::popcount(id);
    if (pc < max_seen)
      max_seen = pc;
    REQUIRE(pc <= max_seen);
  }
  std::filesystem::remove_all("tc_grouped_box");
}

// ---------------------------------------------------------------------------
// Propagator count formula:  nprops = nints*nexts + nints*(nints+1)/2
// ---------------------------------------------------------------------------

TEST(nprops_formula_1loop_1ext) {
  // 1*1 + 1*2/2 = 1 + 1 = 2
  // Bubble topology already tested above; just verify count via propagator
  // mismatch rejection (passing 3 propagators for a 2-prop topology throws)
  clear_symtab();
  YAML::Node cfg = YAML::Load(R"(
family:
  name: np1
  internals: [lf]
  externals: [pf]
  sps_rules:
    - [pf, pf, 0]
  propagators:
    - [lf, 0]
    - [lf+pf, 0]
    - [lf+pf+pf, 0]
targets: []
)");
  REQUIRE_THROWS_AS(Family(cfg), std::runtime_error);
}

TEST(nprops_formula_1loop_2ext) {
  // 1*2 + 1*2/2 = 2 + 1 = 3 propagators needed
  // Triangle topology: 3 propagators → should parse fine
  clear_symtab();
  YAML::Node cfg = YAML::Load(R"(
family:
  name: triangle
  internals: [lt]
  externals: [p1t, p2t]
  invariants:
    - [s12, 2]
  sps_rules:
    - [p1t, p1t, 0]
    - [p2t, p2t, 0]
    - [p1t, p2t, "s12/2"]
  propagators:
    - [lt, 0]
    - [lt+p1t, 0]
    - [lt+p1t+p2t, 0]
targets:
  - [1, 1, 1]
)");
  Family f(cfg);
}

TEST(nprops_formula_2loop_1ext) {
  // 2*1 + 2*3/2 = 2 + 3 = 5 propagators needed
  // Verify via exact count acceptance
  clear_symtab();
  YAML::Node cfg = YAML::Load(R"(
family:
  name: twoloop1ext
  internals: [l1a, l2a]
  externals: [pa]
  invariants:
    - [p2a, 2]
  sps_rules:
    - [pa, pa, p2a]
  propagators:
    - [l1a, 0]
    - [l2a, 0]
    - [l1a+l2a, 0]
    - [l1a+pa, 0]
    - [l2a+pa, 0]
targets:
  - [1, 1, 1, 1, 1]
)");
  Family f(cfg);
}

// ---------------------------------------------------------------------------
// parallel_mode — parse and validate the scheduling mode field
// ---------------------------------------------------------------------------

TEST(family_parse_parallel_mode_default) {
  // Without a parallel_mode entry the default ("fine") applies.
  clear_symtab();
  YAML::Node cfg = box_config("tc_pm_default");
  Family f(cfg);
  REQUIRE(FamilyTestHelper::parallel_mode(f) == "fine");
}

TEST(family_parse_parallel_mode_layer_explicit) {
  // Explicit parallel_mode: layer is still accepted (the default is "fine").
  clear_symtab();
  std::string yaml = R"(
settings:
  cache_dir: tc_pm_layer
  parallel_mode: layer
family:
  name: pm_layer
  dimension: d
  internals: [lpm]
  externals: [p1pm, p2pm, p3pm]
  invariants:
    - [spm, 2]
    - [tpm, 2]
    - [mpm, 1]
  invar_one: mpm
  sps_rules:
    - [p1pm, p1pm, 0]
    - [p2pm, p2pm, 0]
    - [p3pm, p3pm, 0]
    - [p1pm, p2pm, "spm/2"]
    - [p1pm, p3pm, "tpm/2"]
    - [p2pm, p3pm, "-spm/2-tpm/2"]
  propagators:
    - [lpm, 0]
    - [lpm+p1pm, 0]
    - [lpm+p1pm+p2pm, 0]
    - [lpm+p1pm+p2pm+p3pm, 0]
targets:
  - [1, 1, 1, 1]
)";
  YAML::Node cfg = YAML::Load(yaml);
  Family f(cfg);  // must not throw
}

TEST(family_parse_parallel_mode_fine) {
  // parallel_mode: fine selects the fine-grained dependency scheduler.
  clear_symtab();
  std::string yaml = R"(
settings:
  cache_dir: tc_pm_fine
  parallel_mode: fine
family:
  name: pm_fine
  dimension: d
  internals: [lfn]
  externals: [p1fn, p2fn, p3fn]
  invariants:
    - [sfn, 2]
    - [tfn, 2]
    - [mfn, 1]
  invar_one: mfn
  sps_rules:
    - [p1fn, p1fn, 0]
    - [p2fn, p2fn, 0]
    - [p3fn, p3fn, 0]
    - [p1fn, p2fn, "sfn/2"]
    - [p1fn, p3fn, "tfn/2"]
    - [p2fn, p3fn, "-sfn/2-tfn/2"]
  propagators:
    - [lfn, 0]
    - [lfn+p1fn, 0]
    - [lfn+p1fn+p2fn, 0]
    - [lfn+p1fn+p2fn+p3fn, 0]
targets:
  - [1, 1, 1, 1]
)";
  YAML::Node cfg = YAML::Load(yaml);
  Family f(cfg);  // must not throw
}

TEST(family_parse_parallel_mode_invalid) {
  // Unknown parallel_mode values must be rejected at construction time.
  clear_symtab();
  YAML::Node cfg = YAML::Load(R"(
settings:
  parallel_mode: concurrent
family:
  name: pm_bad
  internals: [lbad]
  externals: [pbad]
  sps_rules:
    - [pbad, pbad, 0]
  propagators:
    - [lbad, 0]
    - [lbad+pbad, 0]
targets: []
)");
  REQUIRE_THROWS_AS(Family(cfg), std::runtime_error);
}


// ---------------------------------------------------------------------------
// Removed options — the keys must be silently ignored
// (boundary_mode, equation_selection, sector_elimination, global_otf and
// subsector_depth_offset are hardcoded or gone; old input files must keep
// running while the removed keys take no effect)
// ---------------------------------------------------------------------------

TEST(family_parse_removed_options_are_ignored) {
  clear_symtab();
  YAML::Node cfg = box_config("tc_removed_keys");
  cfg["settings"]["boundary_mode"] = "totall";
  cfg["settings"]["equation_selection"] = "automatic";
  cfg["settings"]["sector_elimination"] = "ffredx";
  cfg["settings"]["global_otf"] = true;
  cfg["settings"]["subsector_depth_offset"] = 3;
  cfg["settings"]["subsector_filter_n"] = 2;
  Family f(cfg);
  // Only the surviving key has an effect.
  REQUIRE(FamilyTestHelper::subsector_filter_n(f) == 2u);
  REQUIRE(FamilyTestHelper::ordering_mode(f) == OrderingMode::Balance);
}

TEST(family_parse_removed_options_do_not_change_the_defaults) {
  clear_symtab();
  YAML::Node cfg = box_config("tc_removed_keys_only");
  cfg["settings"]["boundary_mode"] = "total";
  cfg["settings"]["equation_selection"] = "legacy";
  cfg["settings"]["sector_elimination"] = "fflow";
  cfg["settings"]["global_otf"] = true;
  cfg["settings"]["subsector_depth_offset"] = 3;
  Family f(cfg);
  REQUIRE(FamilyTestHelper::subsector_filter_n(f) == 1u);
}

// ---------------------------------------------------------------------------
// ordering_mode — integral order: per-sector seeds/columns, and under
// balance the global IDs too (see test_global_order.cpp)
// ---------------------------------------------------------------------------

TEST(family_parse_ordering_mode_default) {
  // Absent key → Balance.
  clear_symtab();
  YAML::Node cfg = box_config("tc_om_default");
  Family f(cfg);
  REQUIRE(FamilyTestHelper::ordering_mode(f) == OrderingMode::Balance);
}

TEST(family_parse_ordering_mode_existing_values) {
  clear_symtab();
  YAML::Node cfg_normal = box_config("tc_om_normal");
  cfg_normal["settings"]["ordering_mode"] = "normal";
  REQUIRE(FamilyTestHelper::ordering_mode(Family(cfg_normal)) ==
          OrderingMode::Normal);
  clear_symtab();
  YAML::Node cfg_balance = box_config("tc_om_balance");
  cfg_balance["settings"]["ordering_mode"] = "balance";
  REQUIRE(FamilyTestHelper::ordering_mode(Family(cfg_balance)) ==
          OrderingMode::Balance);
}

TEST(family_parse_ordering_mode_invalid) {
  // Typos throw instead of silently running Balance, matching parallel_mode.
  // "complexity" is the pre-rename spelling and is no longer accepted.
  clear_symtab();
  YAML::Node cfg = box_config("tc_om_bad");
  cfg["settings"]["ordering_mode"] = "complex";
  REQUIRE_THROWS_AS(Family(cfg), std::runtime_error);
  clear_symtab();
  YAML::Node cfg_old = box_config("tc_om_old_name");
  cfg_old["settings"]["ordering_mode"] = "complexity";
  REQUIRE_THROWS_AS(Family(cfg_old), std::runtime_error);
}

// ---------------------------------------------------------------------------
// subsector_filter_n — offset of the subsector filter
// ---------------------------------------------------------------------------

TEST(family_parse_subsector_filter_n_default) {
  clear_symtab();
  YAML::Node cfg = box_config("tc_sfn_default");
  Family f(cfg);
  REQUIRE(FamilyTestHelper::subsector_filter_n(f) == 1u);
}

TEST(family_parse_subsector_filter_n_value) {
  clear_symtab();
  YAML::Node cfg = box_config("tc_sfn_2");
  cfg["settings"]["subsector_filter_n"] = 2;
  Family f(cfg);
  REQUIRE(FamilyTestHelper::subsector_filter_n(f) == 2u);
}

TEST(family_parse_subsector_filter_n_zero) {
  // 0 is meaningful (no offset), not "unset".
  clear_symtab();
  YAML::Node cfg = box_config("tc_sfn_0");
  cfg["settings"]["subsector_filter_n"] = 0;
  Family f(cfg);
  REQUIRE(FamilyTestHelper::subsector_filter_n(f) == 0u);
}

TEST(family_parse_subsector_depth_offset_is_gone) {
  // The key is removed: an old input file still loads, but the value has no
  // effect on subsector_filter_n.
  clear_symtab();
  YAML::Node cfg = box_config("tc_sdo_removed");
  cfg["settings"]["subsector_depth_offset"] = 3;
  Family f(cfg);
  REQUIRE(FamilyTestHelper::subsector_filter_n(f) == 1u);
}

// The precompiler against the rows a real symbolic family generates (spec 4.4):
// the unit corpus is synthetic, this is the shape the benchmark families meet.
// Every box IBP row must precompile, and evaluating it must reproduce GiNaC's
// own sum, parsed back with the family's symbol table.
TEST(row_coeff_precompiles_real_box_ibp_rows) {
  clear_symtab();
  YAML::Node cfg = box_config("tc_row_coeff_box");
  Family f(cfg);
  f.init();
  const auto &ibpM = FamilyTestHelper::ibpM(f);
  REQUIRE(!ibpM.empty());
  const unsigned nprops = f.nprops();

  // Seeds of the shapes the settling round emits: a sector support with its
  // lowest line doubled, one negative power on a line inside the support (the
  // numerator case) and one above it (the constant case).
  std::vector<std::vector<int>> seeds;
  seeds.push_back(std::vector<int>(nprops, 0));
  for (unsigned neg = 0; neg < nprops; ++neg) {
    std::vector<int> sv(nprops, 0);
    sv[0] = 2;
    sv[neg] -= 1;
    seeds.push_back(sv);
  }

  secred::RowTable table; // single-threaded here; the sector holds ginac_mutex
  GiNaC::parser parser(Family::symtab);
  unsigned checked = 0;
  for (unsigned i = 0; i < ibpM.size(); ++i) {
    table.add(i, ibpM[i]);
    for (unsigned k = 0; k < ibpM[i].size(); ++k) {
      REQUIRE(table.rows[i][k].precompiled);
      ++checked;
      for (const std::vector<int> &sv : seeds) {
        const RawIntegral seed(sv);
        std::string got;
        const secred::RowEval st = table.eval(i, k, seed, got);
        GiNaC::ex want;
        for (unsigned p = 0; p < nprops; ++p)
          want += ibpM[i][k].second[p] * seed[p];
        want += ibpM[i][k].second.back();
        want = want.expand();
        if (st == secred::RowEval::Zero) {
          REQUIRE(want.is_zero());
          continue;
        }
        REQUIRE(st == secred::RowEval::Ok);
        REQUIRE_MSG((parser(got) - want).expand().is_zero(),
                    "box row " + std::to_string(i) + "/" + std::to_string(k) +
                        " on seed " + std::to_string(sv[0]) + " evaluated to '" + got +
                        "'");
      }
    }
  }
  REQUIRE(checked > 0);
  REQUIRE_EQ(table.fast_rows, checked);
  REQUIRE_EQ(table.slow_rows, 0u);
  std::filesystem::remove_all("tc_row_coeff_box");
}
