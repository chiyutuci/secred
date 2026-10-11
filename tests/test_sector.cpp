#include "test_runner.h"
#include "sector.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>

// Helper: unambiguously construct RawIntegral from an int initializer list.
static RawIntegral RI(std::initializer_list<int> il) {
  return RawIntegral(std::vector<int>(il));
}

// ---------------------------------------------------------------------------
// Test accessor: exposes Sector private internals via the friend declaration.
// ---------------------------------------------------------------------------

class SectorTestHelper {
public:
  static Sector make_sector(unsigned id, unsigned nprops,
                             const std::vector<bool> &lines, unsigned depth,
                             unsigned rank,
                             const std::set<unsigned> &trivials = {}) {
    Sector s;
    s._id     = id;
    s._nprops = nprops;
    s._lines  = lines;
    s._bound  = SeedBound::initial(BoundaryMode::Box, depth, rank);
    s._trivialsectors = trivials;
    return s;
  }

  // Proxy for the private static comparator.
  static bool compare_subintegral(const RawIntegral &a, const RawIntegral &b) {
    return Sector::_compare_subintegral(a, b);
  }

  // Proxy for the private static comparator.
  static bool compare_complexity(const RawIntegral &a, const RawIntegral &b) {
    return Sector::_compare_complexity(a, b);
  }

  // Proxy for the private subsector column ordering.
  static std::vector<RawIntegral> order_subintegrals(
      const std::vector<RawIntegral> &in, OrderingMode mode) {
    std::vector<RawIntegral> v = in;
    Sector::_order_subintegrals(v, mode);
    return v;
  }

  static void generate_combinations(Sector &s, int n, int sum) {
    s._generate_combinations(n, sum);
  }

  static void generate_seeds(Sector &s,
                              OrderingMode mode = OrderingMode::Balance) {
    s._generate_seeds(mode);
  }

  static const std::vector<RawIntegral> &seeds(const Sector &s) {
    return s._seeds;
  }

  static const std::unordered_map<RawIntegral, unsigned> &
  weights(const Sector &s) {
    return s._weights;
  }

  static const std::map<std::pair<int, int>, std::vector<std::vector<int>>> &
  combinations(const Sector &s) {
    return s._combinations;
  }

  static unsigned depth(const Sector &s) { return s._bound.depth; }
  static unsigned rank(const Sector &s)  { return s._bound.rank; }

  static void set_boundary(Sector &s, const SeedBound &b) { s._bound = b; }
  static const SeedBound &boundary(const Sector &s) { return s._bound; }
};

// ---------------------------------------------------------------------------
// SectorConfig — default values
// ---------------------------------------------------------------------------

TEST(sectorconfig_defaults) {
  SectorConfig cfg;
  REQUIRE(cfg.cache_dir      == "cache");
  REQUIRE(cfg.subsector_filter_n == 1u);
  REQUIRE(cfg.ordering_mode == OrderingMode::Balance);
  // 默认必须保持历史边界规则：depth / rank 两个独立的盒子边界。
  REQUIRE(cfg.boundary_mode == BoundaryMode::Box);
  // 默认必须保持历史行为：分 sector 阶段走 FiniteFlow 的 SparseLinearSolver。
  REQUIRE(cfg.sector_elimination == SectorElimination::Fflow);
}

TEST(sectorconfig_sector_elimination_configurable) {
  SectorConfig cfg;
  cfg.sector_elimination = SectorElimination::Pyred;
  REQUIRE(cfg.sector_elimination == SectorElimination::Pyred);
  REQUIRE(cfg.sector_elimination != SectorElimination::Fflow);
}

TEST(sectorconfig_subsector_filter_n_configurable) {
  SectorConfig cfg;
  cfg.subsector_filter_n = 2;
  REQUIRE(cfg.subsector_filter_n == 2u);
}

TEST(sectorconfig_boundary_mode_configurable) {
  SectorConfig cfg;
  cfg.boundary_mode = BoundaryMode::Total;
  REQUIRE(cfg.boundary_mode == BoundaryMode::Total);
  // 必须是新键才改变行为：默认值仍是 Box。
  REQUIRE(SectorConfig{}.boundary_mode == BoundaryMode::Box);
}

TEST(sectorconfig_custom_values) {
  SectorConfig cfg;
  cfg.cache_dir      = "/tmp/myrun";
  cfg.ordering_mode  = OrderingMode::Normal;
  REQUIRE(cfg.cache_dir     == "/tmp/myrun");
  REQUIRE(cfg.ordering_mode == OrderingMode::Normal);
}

TEST(sectorconfig_ordering_mode_complexity_configurable) {
  SectorConfig cfg;
  cfg.ordering_mode = OrderingMode::Complexity;
  REQUIRE(cfg.ordering_mode == OrderingMode::Complexity);
  // 必须是新键才改变行为：默认值仍是 Balance。
  REQUIRE(SectorConfig{}.ordering_mode == OrderingMode::Balance);
}

// ---------------------------------------------------------------------------
// Sector::_compare_subintegral — canonical ordering
// Ordering: depth ASC → rank ASC → sector ASC → positive indices ASC →
//           negative indices DESC
// ---------------------------------------------------------------------------

TEST(compare_depth_ordering) {
  RawIntegral shallow = RI({1, 1, 0, 0});  // depth = 2
  RawIntegral deep    = RI({1, 1, 1, 0});  // depth = 3
  REQUIRE( SectorTestHelper::compare_subintegral(shallow, deep));
  REQUIRE(!SectorTestHelper::compare_subintegral(deep, shallow));
}

TEST(compare_rank_ordering_when_depths_equal) {
  RawIntegral low_rank  = RI({1, 1, -1, 0});  // depth=2, rank=1
  RawIntegral high_rank = RI({1, 1, -2, 0});  // depth=2, rank=2
  REQUIRE( SectorTestHelper::compare_subintegral(low_rank,  high_rank));
  REQUIRE(!SectorTestHelper::compare_subintegral(high_rank, low_rank));
}

TEST(compare_sector_ordering_when_depth_rank_equal) {
  // depth=1, rank=0 for both; sector bitmask differs
  // pos 0 active → sector = 0b0001 = 1
  RawIntegral small_sector = RI({1, 0, 0, 0});
  // pos 1 active → sector = 0b0010 = 2
  RawIntegral large_sector = RI({0, 1, 0, 0});
  REQUIRE(small_sector.depth() == large_sector.depth());
  REQUIRE(small_sector.rank()  == large_sector.rank());
  REQUIRE( SectorTestHelper::compare_subintegral(small_sector, large_sector));
  REQUIRE(!SectorTestHelper::compare_subintegral(large_sector, small_sector));
}

TEST(compare_irreflexive) {
  RawIntegral ri = RI({1, -1, 2});
  REQUIRE(!SectorTestHelper::compare_subintegral(ri, ri));
}

TEST(compare_consistent_with_sort) {
  std::vector<RawIntegral> v = {
    RI({1, 1, 1}),  // depth=3
    RI({1, 0, 0}),  // depth=1
    RI({1, 1, 0}),  // depth=2
  };
  std::sort(v.begin(), v.end(), SectorTestHelper::compare_subintegral);
  REQUIRE(v[0].depth() <= v[1].depth());
  REQUIRE(v[1].depth() <= v[2].depth());
}

TEST(compare_positive_indices_ascending) {
  // Same depth (2), same rank (0).
  // a: pos 0 and 1 active → sector=3; b: only pos 0 → sector=1
  // smaller sector (1) comes first
  RawIntegral a = RI({1, 1, 0, 0});
  RawIntegral b = RI({2, 0, 0, 0});
  REQUIRE(SectorTestHelper::compare_subintegral(b, a));
}

TEST(compare_equal_integrals_return_false) {
  RawIntegral a = RI({2, -1, 3});
  RawIntegral b = RI({2, -1, 3});
  REQUIRE(!SectorTestHelper::compare_subintegral(a, b));
  REQUIRE(!SectorTestHelper::compare_subintegral(b, a));
}

// Negative indices: larger (less negative) wins within same depth/rank/sector.
TEST(compare_negative_indices_descending) {
  // Both have sector 0b01 (pos 0 active), depth=1, rank differs → rank ASC
  // low_neg: rank=1 (one -1); hi_neg: rank=2 (one -2)
  RawIntegral low_neg = RI({1, -1});
  RawIntegral hi_neg  = RI({1, -2});
  REQUIRE( SectorTestHelper::compare_subintegral(low_neg, hi_neg));
  REQUIRE(!SectorTestHelper::compare_subintegral(hi_neg,  low_neg));
}

// ---------------------------------------------------------------------------
// Sector::_generate_combinations — memoized partition generator
// ---------------------------------------------------------------------------

TEST(combinations_zero_number) {
  Sector s = SectorTestHelper::make_sector(1, 1, {true}, 2, 2);
  SectorTestHelper::generate_combinations(s, 0, 3);
  const auto &combs = SectorTestHelper::combinations(s);
  // combinations(0, s) = { {} } for any s
  REQUIRE(combs.at({0, 3}).size() == 1);
  REQUIRE(combs.at({0, 3})[0].empty());
}

TEST(combinations_number_1) {
  Sector s = SectorTestHelper::make_sector(1, 1, {true}, 2, 2);
  SectorTestHelper::generate_combinations(s, 1, 4);
  const auto &combs = SectorTestHelper::combinations(s);
  // combinations(1, 4) = { {4} }
  REQUIRE(combs.at({1, 4}).size() == 1);
  REQUIRE(combs.at({1, 4})[0][0] == 4);
}

TEST(combinations_sum_0) {
  Sector s = SectorTestHelper::make_sector(1, 3, {true, true, true}, 2, 2);
  SectorTestHelper::generate_combinations(s, 3, 0);
  const auto &combs = SectorTestHelper::combinations(s);
  // combinations(3, 0) = { {0,0,0} }
  REQUIRE(combs.at({3, 0}).size() == 1);
  for (int v : combs.at({3, 0})[0])
    REQUIRE(v == 0);
}

TEST(combinations_2_2_has_correct_count) {
  // partitions of 2 into 2 non-negative parts: {0,2},{1,1},{2,0} → 3
  Sector s = SectorTestHelper::make_sector(1, 2, {true, true}, 3, 3);
  SectorTestHelper::generate_combinations(s, 2, 2);
  const auto &combs = SectorTestHelper::combinations(s);
  REQUIRE(combs.at({2, 2}).size() == 3);
}

TEST(combinations_all_parts_sum_to_target) {
  Sector s = SectorTestHelper::make_sector(1, 3, {true, true, true}, 4, 4);
  SectorTestHelper::generate_combinations(s, 3, 4);
  const auto &combs = SectorTestHelper::combinations(s);
  for (const auto &comb : combs.at({3, 4})) {
    int sum = 0;
    for (int v : comb)
      sum += v;
    REQUIRE(sum == 4);
    REQUIRE(comb.size() == 3u);
  }
}

TEST(combinations_memoization) {
  // Calling twice with the same args should be a no-op (cache hit).
  Sector s = SectorTestHelper::make_sector(1, 2, {true, true}, 3, 3);
  SectorTestHelper::generate_combinations(s, 2, 3);
  auto size_before = SectorTestHelper::combinations(s).size();
  SectorTestHelper::generate_combinations(s, 2, 3);
  REQUIRE(SectorTestHelper::combinations(s).size() == size_before);
  REQUIRE(SectorTestHelper::combinations(s).contains({2, 3}));
}

TEST(combinations_3_3_nonnegative_parts) {
  // Every partition of 3 into 3 non-negative parts: C(3+3-1,3-1)=C(5,2)=10
  Sector s = SectorTestHelper::make_sector(1, 3, {true, true, true}, 4, 4);
  SectorTestHelper::generate_combinations(s, 3, 3);
  const auto &combs = SectorTestHelper::combinations(s);
  REQUIRE(combs.at({3, 3}).size() == 10);
  // Every part must be non-negative and sum to 3
  for (const auto &c : combs.at({3, 3})) {
    int sum = 0;
    for (int v : c) {
      REQUIRE(v >= 0);
      sum += v;
    }
    REQUIRE(sum == 3);
  }
}

// ---------------------------------------------------------------------------
// Sector::_generate_seeds — seed integral enumeration
// ---------------------------------------------------------------------------

TEST(seeds_single_propagator_sector) {
  // Sector 0b1 (only propagator 0 active): depth=2, rank=1, nprops=1
  std::vector<bool> lines = {true};
  Sector s = SectorTestHelper::make_sector(1, 1, lines, 2, 1);
  SectorTestHelper::generate_seeds(s);
  const auto &seeds = SectorTestHelper::seeds(s);

  REQUIRE(!seeds.empty());
  for (const auto &seed : seeds)
    REQUIRE(seed.size() == 1);
}

TEST(seeds_all_within_depth_rank_bounds) {
  // Sector 0b11 (both propagators active): depth=3, rank=2, nprops=2.
  // _generate_seeds includes "outer" seeds at exactly depth == _depth so that
  // sector_reduction_ff_n can count them separately; the bound is inclusive.
  std::vector<bool> lines = {true, true};
  Sector s = SectorTestHelper::make_sector(3, 2, lines, 3, 2);
  SectorTestHelper::generate_seeds(s);
  const auto &seeds = SectorTestHelper::seeds(s);

  REQUIRE(!seeds.empty());
  for (const auto &seed : seeds) {
    REQUIRE(seed.depth() <= 3u);  // at most _depth (outer seeds sit at == _depth)
    REQUIRE(seed.rank()  <= 2u);  // at most _rank
  }
}

TEST(seeds_active_lines_are_positive_or_zero) {
  // Sector 0b01: only propagator 0 is active
  std::vector<bool> lines = {true, false};
  Sector s = SectorTestHelper::make_sector(1, 2, lines, 2, 2);
  SectorTestHelper::generate_seeds(s);
  const auto &seeds = SectorTestHelper::seeds(s);

  for (const auto &seed : seeds) {
    REQUIRE(seed[0] >= 1);  // active propagator starts at 1 (sector bit)
    REQUIRE(seed[1] <= 0);  // inactive propagator is non-positive
  }
}

TEST(seeds_weights_match_indices) {
  std::vector<bool> lines = {true, false, true};
  Sector s = SectorTestHelper::make_sector(5, 3, lines, 2, 2);
  SectorTestHelper::generate_seeds(s);
  const auto &seeds   = SectorTestHelper::seeds(s);
  const auto &weights = SectorTestHelper::weights(s);

  for (unsigned i = 0; i < seeds.size(); ++i) {
    REQUIRE(weights.contains(seeds[i]));
    REQUIRE(weights.at(seeds[i]) == i);
  }
}

TEST(seeds_no_duplicates) {
  std::vector<bool> lines = {true, true, false};
  Sector s = SectorTestHelper::make_sector(3, 3, lines, 3, 2);
  SectorTestHelper::generate_seeds(s);
  const auto &seeds = SectorTestHelper::seeds(s);

  std::set<RawIntegral> unique(seeds.begin(), seeds.end());
  REQUIRE(unique.size() == seeds.size());
}

TEST(seeds_clear_resets_state) {
  std::vector<bool> lines = {true};
  Sector s = SectorTestHelper::make_sector(1, 1, lines, 2, 2);
  SectorTestHelper::generate_seeds(s);
  s.clear();
  REQUIRE(SectorTestHelper::seeds(s).empty());
  REQUIRE(SectorTestHelper::weights(s).empty());
}

// Sector with all-inactive propagators (sector id == 0, all lines false):
// seeds should cover depth=0 only → just the integral {0,...,0} variants.
TEST(seeds_all_inactive_propagators) {
  // sector id=0, nprops=2, lines={false,false}, depth=1, rank=2
  std::vector<bool> lines = {false, false};
  Sector s = SectorTestHelper::make_sector(0, 2, lines, 1, 2);
  SectorTestHelper::generate_seeds(s);
  const auto &seeds = SectorTestHelper::seeds(s);
  // All seeds must have depth 0 (no positive indices)
  for (const auto &seed : seeds)
    REQUIRE(seed.depth() == 0u);
}

// After clear(), seeds can be regenerated independently.
TEST(seeds_regenerate_after_clear) {
  std::vector<bool> lines = {true, true};
  Sector s = SectorTestHelper::make_sector(3, 2, lines, 3, 2);
  SectorTestHelper::generate_seeds(s);
  unsigned first_count = SectorTestHelper::seeds(s).size();
  s.clear();
  SectorTestHelper::generate_seeds(s);
  REQUIRE(SectorTestHelper::seeds(s).size() == first_count);
}

// ---------------------------------------------------------------------------
// GlobalEquation and ReductionContext — new fields for global kira collection
// ---------------------------------------------------------------------------

TEST(global_equation_typedef_construction) {
  // GlobalEquation is vector<pair<RawIntegral, string>>
  GlobalEquation eq;
  eq.emplace_back(RI({1, 0, 0, 0}), "3/2");
  eq.emplace_back(RI({0, 1, 0, 0}), "-1");
  REQUIRE(eq.size() == 2u);
  REQUIRE(eq[0].first  == RI({1, 0, 0, 0}));
  REQUIRE(eq[0].second == "3/2");
  REQUIRE(eq[1].first  == RI({0, 1, 0, 0}));
  REQUIRE(eq[1].second == "-1");
}

TEST(reduction_context_global_equations_initially_empty) {
  ReductionContext ctx;
  REQUIRE(ctx.globalEquationsCount.load() == 0u);
  REQUIRE(ctx.distinctIntegrals.empty());
}

// ---------------------------------------------------------------------------
// system_memory — /proc/meminfo probe
// ---------------------------------------------------------------------------

TEST(system_memory_sanity) {
  SystemMemory sm = system_memory();
  REQUIRE(sm.total > 0);
  REQUIRE(sm.available <= sm.total);
}

TEST(process_used_memory_sanity) {
  // The running test process always has a positive resident set.
  REQUIRE(process_used_memory() > 0);
}

// ---------------------------------------------------------------------------
// ReductionContext memory gate — budget headroom admission (maxMemory − used)
// ---------------------------------------------------------------------------

TEST(reduction_context_memory_defaults) {
  ReductionContext ctx;
  REQUIRE(ctx.maxMemory == 512ULL * 1024 * 1024 * 1024);
  REQUIRE(ctx.memoryReserve == 60ULL * 1024 * 1024 * 1024);
}

TEST(memory_headroom_proceeds_when_idle) {
  // Nothing running → the first sector is admitted even with no headroom, so
  // the reduction can never deadlock at startup.
  ReductionContext ctx;
  ctx.maxMemory = 3000;
  ctx.memoryReserve = 1000;
  ctx.memoryProbe = [] { return 2900u; };  // headroom = 100, far below reserve
  ctx.waitForMemoryHeadroom();             // runningSectors == 0 → proceeds
  REQUIRE(ctx.runningSectors.load() == 1u);
  ctx.releaseMemorySlot();
}

TEST(memory_headroom_admits_when_plentiful) {
  // With headroom above 2*reserve, sectors are admitted freely.
  ReductionContext ctx;
  ctx.maxMemory = 5000;
  ctx.memoryReserve = 1000;
  ctx.memoryProbe = [] { return 500u; };  // headroom = 4500 > 2*reserve
  ctx.waitForMemoryHeadroom();  // idle → running=1
  ctx.waitForMemoryHeadroom();  // plentiful → running=2
  REQUIRE(ctx.runningSectors.load() == 2u);
  ctx.releaseMemorySlot();
  ctx.releaseMemorySlot();
  REQUIRE(ctx.runningSectors.load() == 0u);
}

TEST(memory_headroom_blocks_then_admits) {
  // One running sector + no headroom blocks a second sector until the first
  // releases and the process frees memory (used drops, headroom reappears).
  ReductionContext ctx;
  ctx.maxMemory = 3000;
  ctx.memoryReserve = 1000;
  std::atomic<std::uint64_t> used{2900};
  ctx.memoryProbe = [&] { return used.load(); };

  ctx.waitForMemoryHeadroom();  // idle → first sector admitted
  REQUIRE(ctx.runningSectors.load() == 1u);

  std::atomic<bool> admitted{false};
  std::thread t([&] {
    ctx.waitForMemoryHeadroom();  // headroom ≤ reserve and running > 0 → blocks
    admitted.store(true);
    ctx.releaseMemorySlot();
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  REQUIRE(!admitted.load());  // still blocked

  used.store(500);           // memory freed → headroom = 2500
  ctx.releaseMemorySlot();   // runningSectors → 0, notify_all
  t.join();
  REQUIRE(admitted.load());
  REQUIRE(ctx.runningSectors.load() == 0u);
}

TEST(memory_headroom_one_admission_per_tight_burst) {
  // In the tight regime (reserve < headroom <= 2*reserve) at most one sector is
  // admitted per free event: a second waiter blocks even though headroom is
  // still above the reserve, until it is consumed (headroom <= reserve) and a
  // fresh free event re-arms it.
  ReductionContext ctx;
  ctx.maxMemory = 3000;
  ctx.memoryReserve = 1000;
  std::atomic<std::uint64_t> used{1500};  // headroom = 1500: tight
  ctx.memoryProbe = [&] { return used.load(); };

  ctx.waitForMemoryHeadroom();  // idle escape → running=1
  ctx.waitForMemoryHeadroom();  // first tight admission → running=2, flag set
  REQUIRE(ctx.runningSectors.load() == 2u);

  std::atomic<bool> third{false};
  std::thread t([&] {
    ctx.waitForMemoryHeadroom();  // headroomTaken → blocks
    third.store(true);
    ctx.releaseMemorySlot();
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  REQUIRE(!third.load());  // blocked: the burst is already claimed

  used.store(2900);          // consume headroom (headroom = 100 <= reserve)
  ctx.releaseMemorySlot();   // running 2→1 + notify → waiter re-checks, re-arms
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  REQUIRE(!third.load());    // still blocked: headroom is now <= reserve

  used.store(1500);          // fresh free event (headroom = 1500)
  ctx.releaseMemorySlot();   // running 1→0 + notify → waiter admits (idle escape)
  t.join();
  REQUIRE(third.load());
  REQUIRE(ctx.runningSectors.load() == 0u);
}

TEST(memory_headroom_reams_when_memory_consumed) {
  // A small sector that does not exhaust the headroom must still re-arm the
  // burst flag once avail has fallen at least rearmDrop below the admission
  // snapshot (here rearmDrop=50, so a 100-byte drop qualifies).  The next waiter
  // is then admitted even though avail stays above the reserve — without this,
  // the flag stays set and stalls every later waiter.
  ReductionContext ctx;
  ctx.maxMemory = 3000;
  ctx.memoryReserve = 1000;
  ctx.rearmDrop = 50;                     // small threshold for the test
  std::atomic<std::uint64_t> used{1500};  // headroom = 1500: tight
  ctx.memoryProbe = [&] { return used.load(); };

  ctx.waitForMemoryHeadroom();  // idle escape → running=1
  ctx.waitForMemoryHeadroom();  // tight admission → running=2, availAtAdmit=1500
  REQUIRE(ctx.runningSectors.load() == 2u);

  std::atomic<bool> third{false};
  std::thread t([&] {
    ctx.waitForMemoryHeadroom();
    third.store(true);
    ctx.releaseMemorySlot();
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  REQUIRE(!third.load());  // blocked: the burst is claimed

  used.store(1600);         // consume a little (headroom 1500 → 1400, drop 100 ≥ 50)
  ctx.releaseMemorySlot();  // running 2→1 + notify → waiter re-arms and admits
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  REQUIRE(third.load());    // admitted: flag re-armed via the ≥rearmDrop drop

  t.join();
  REQUIRE(ctx.runningSectors.load() == 1u);
}

TEST(memory_headroom_does_not_ream_below_threshold) {
  // A drop below rearmDrop must NOT re-arm the burst flag: VmRSS noise below
  // the threshold keeps the flag set, so the waiter stays blocked.
  ReductionContext ctx;
  ctx.maxMemory = 3000;
  ctx.memoryReserve = 1000;
  ctx.rearmDrop = 100;
  std::atomic<std::uint64_t> used{1500};  // headroom = 1500: tight
  ctx.memoryProbe = [&] { return used.load(); };

  ctx.waitForMemoryHeadroom();  // idle escape → running=1
  ctx.waitForMemoryHeadroom();  // tight admission → running=2, availAtAdmit=1500
  REQUIRE(ctx.runningSectors.load() == 2u);

  std::atomic<bool> third{false};
  std::thread t([&] {
    ctx.waitForMemoryHeadroom();
    third.store(true);
    ctx.releaseMemorySlot();
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  REQUIRE(!third.load());  // blocked: the burst is claimed

  used.store(1530);         // drop 30 (< rearmDrop 100) → must NOT re-arm
  ctx.releaseMemorySlot();  // running 2→1 + notify → waiter re-checks, stays blocked
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  REQUIRE(!third.load());   // still blocked: drop below threshold

  // Clean up: exhaust the headroom and free the slot so the waiter finally
  // admits via the idle escape.
  used.store(2900);         // avail = 100 <= reserve
  ctx.releaseMemorySlot();  // running 1→0 + notify → idle escape admits
  t.join();
  REQUIRE(third.load());
}

// ---------------------------------------------------------------------------
// ReductionContext raw-equation spill — appendRawEquation + distinctIntegrals
// ---------------------------------------------------------------------------

static void open_raw_spill(ReductionContext &ctx, const std::string &path) {
  ctx.rawEquationsPath = path;
  ctx.rawEqOut = std::make_unique<std::ofstream>(path, std::ios::trunc);
}

TEST(append_raw_equation_spills_to_file) {
  ReductionContext ctx;
  const std::string path = "/tmp/secred_test_raw_eqs.txt";
  open_raw_spill(ctx, path);

  GlobalEquation eq;
  eq.emplace_back(RI({1, 1}), "2");
  eq.emplace_back(RI({1, 0}), "-1");
  ctx.appendRawEquation(eq);

  ctx.rawEqOut.reset();  // close before reading back
  REQUIRE(ctx.globalEquationsCount.load() == 1u);
  REQUIRE(ctx.distinctIntegrals.size() == 2u);
  REQUIRE(ctx.distinctIntegrals.count(RI({1, 1})) == 1u);

  std::ifstream in(path);
  std::string line;
  std::getline(in, line);
  REQUIRE(line == "[1, 1]\t2");
  std::getline(in, line);
  REQUIRE(line == "[1, 0]\t-1");
  std::getline(in, line);
  REQUIRE(line.empty());  // blank equation separator
  std::filesystem::remove(path);
}

TEST(append_raw_equation_thread_safe) {
  ReductionContext ctx;
  const std::string path = "/tmp/secred_test_raw_eqs_thread.txt";
  open_raw_spill(ctx, path);

  const unsigned N = 200;
  std::vector<std::thread> ts;
  for (unsigned i = 0; i < N; ++i) {
    ts.emplace_back([&, i] {
      GlobalEquation eq;
      eq.emplace_back(RI({(int)(i % 7), 1}), "1");
      ctx.appendRawEquation(eq);
    });
  }
  for (auto &t : ts) t.join();
  ctx.rawEqOut.reset();

  REQUIRE(ctx.globalEquationsCount.load() == N);
  std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// select_dum_eq_from_file — picks the equation with the smallest max integral ID
// ---------------------------------------------------------------------------

static void write_raw_eqs(const std::string &path,
                          const std::vector<GlobalEquation> &eqs) {
  std::ofstream out(path, std::ios::trunc);
  for (const auto &eq : eqs) {
    for (const auto &[integ, coeff] : eq)
      out << integ << '\t' << coeff << '\n';
    out << '\n';
  }
}

TEST(select_dum_eq_single_equation) {
  // A single equation: must return 0.
  std::unordered_map<RawIntegral, unsigned> id;
  id[RI({1, 0})] = 0;
  id[RI({0, 1})] = 1;

  std::vector<GlobalEquation> eqs(1);
  eqs[0].emplace_back(RI({1, 0}), "1");
  eqs[0].emplace_back(RI({0, 1}), "-1");

  const std::string path = "/tmp/secred_test_dum_eqs.txt";
  write_raw_eqs(path, eqs);
  REQUIRE(select_dum_eq_from_file(path, id) == 0u);
  std::filesystem::remove(path);
}

TEST(select_dum_eq_picks_smallest_max_id) {
  // eq[0] has max ID 3, eq[1] has max ID 1 → select eq[1].
  std::unordered_map<RawIntegral, unsigned> id;
  id[RI({1, 1, 0})] = 3;
  id[RI({1, 0, 0})] = 1;
  id[RI({0, 1, 0})] = 0;

  std::vector<GlobalEquation> eqs(2);
  eqs[0].emplace_back(RI({1, 1, 0}), "2");   // max ID = 3
  eqs[0].emplace_back(RI({0, 1, 0}), "-1");
  eqs[1].emplace_back(RI({1, 0, 0}), "3");   // max ID = 1
  eqs[1].emplace_back(RI({0, 1, 0}), "1");

  const std::string path = "/tmp/secred_test_dum_eqs.txt";
  write_raw_eqs(path, eqs);
  REQUIRE(select_dum_eq_from_file(path, id) == 1u);
  std::filesystem::remove(path);
}

TEST(select_dum_eq_tie_keeps_first) {
  // Both equations have the same max ID → return the first (index 0).
  std::unordered_map<RawIntegral, unsigned> id;
  id[RI({1, 0})] = 2;
  id[RI({0, 1})] = 0;

  std::vector<GlobalEquation> eqs(2);
  eqs[0].emplace_back(RI({1, 0}), "1");   // max ID = 2
  eqs[1].emplace_back(RI({1, 0}), "-3");  // max ID = 2

  const std::string path = "/tmp/secred_test_dum_eqs.txt";
  write_raw_eqs(path, eqs);
  REQUIRE(select_dum_eq_from_file(path, id) == 0u);
  std::filesystem::remove(path);
}

TEST(select_dum_eq_ignores_unmapped_integrals) {
  // An integral with no entry in id_map must not contribute to max_id.
  std::unordered_map<RawIntegral, unsigned> id;
  id[RI({1, 1, 0})] = 5;
  id[RI({1, 0, 0})] = 2;
  // RI({0,0,1}) intentionally NOT in id_map.

  std::vector<GlobalEquation> eqs(2);
  eqs[0].emplace_back(RI({1, 1, 0}), "1");   // ID = 5
  eqs[0].emplace_back(RI({0, 0, 1}), "-2");  // unmapped → ignored
  eqs[1].emplace_back(RI({1, 0, 0}), "4");   // ID = 2

  const std::string path = "/tmp/secred_test_dum_eqs.txt";
  write_raw_eqs(path, eqs);
  REQUIRE(select_dum_eq_from_file(path, id) == 1u);
  std::filesystem::remove(path);
}

TEST(select_dum_eq_three_equations_middle_wins) {
  // Three equations with max IDs 4, 1, 3 → index 1 wins.
  std::unordered_map<RawIntegral, unsigned> id;
  id[RI({1, 1})] = 4;
  id[RI({1, 0})] = 1;
  id[RI({0, 1})] = 3;

  std::vector<GlobalEquation> eqs(3);
  eqs[0].emplace_back(RI({1, 1}), "1");  // max ID = 4
  eqs[1].emplace_back(RI({1, 0}), "1");  // max ID = 1
  eqs[2].emplace_back(RI({0, 1}), "1");  // max ID = 3

  const std::string path = "/tmp/secred_test_dum_eqs.txt";
  write_raw_eqs(path, eqs);
  REQUIRE(select_dum_eq_from_file(path, id) == 1u);
  std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// OrderingMode enum and SectorGeometry struct
// ---------------------------------------------------------------------------

TEST(ordering_mode_enum_values) {
  REQUIRE(OrderingMode::Normal != OrderingMode::Balance);
  REQUIRE(OrderingMode::Balance != OrderingMode::Complexity);
  REQUIRE(OrderingMode::Normal != OrderingMode::Complexity);
}

// ---------------------------------------------------------------------------
// RawIntegral::maxpos / maxneg — the two fine keys of OrderingMode::Complexity
// ---------------------------------------------------------------------------

TEST(raw_integral_maxpos_returns_largest_denominator_power) {
  REQUIRE(RI({1, 2, 3}).maxpos() == 3);
  REQUIRE(RI({5, 0, 1}).maxpos() == 5);
  REQUIRE(RI({1, 1, 1}).maxpos() == 1);
}

TEST(raw_integral_maxpos_sentinel_is_one) {
  // No positive index at all (pure numerator integral): sentinel 1, so that the
  // key stays comparable with integrals that do have a denominator power.
  REQUIRE(RI({0, -2, -5}).maxpos() == 1);
  REQUIRE(RI({0, 0}).maxpos() == 1);
}

TEST(raw_integral_maxnum_key_is_negated_maxneg) {
  // Largest numerator power = -maxneg(), sentinel 1 when nothing is negative —
  // the mirror image of maxpos().
  REQUIRE(-RI({1, -4, 0}).maxneg() == 4);
  REQUIRE(-RI({1, -1, 0}).maxneg() == 1);
  REQUIRE(-RI({1, 2, 0}).maxneg() == 1);
  REQUIRE(-RI({0, 0}).maxneg() == 1);
}

// ---------------------------------------------------------------------------
// _compare_complexity — key priority
// ---------------------------------------------------------------------------

TEST(compare_complexity_total_power_is_first_key) {
  // [1,-3]: depth 1, rank 3 → total 4.  [2,0]: depth 2, rank 0 → total 2.
  // The total power decides before depth, so the shallower-but-longer integral
  // is NOT the smaller one here.
  REQUIRE(SectorTestHelper::compare_complexity(RI({2, 0}), RI({1, -3})));
  REQUIRE(!SectorTestHelper::compare_complexity(RI({1, -3}), RI({2, 0})));
}

TEST(compare_complexity_depth_breaks_total_power_ties) {
  // [3,0] and [1,-2] both have total power 3; the smaller depth wins.
  REQUIRE(SectorTestHelper::compare_complexity(RI({1, -2}), RI({3, 0})));
  REQUIRE(!SectorTestHelper::compare_complexity(RI({3, 0}), RI({1, -2})));
}

TEST(compare_complexity_rank_never_decides) {
  // rank = total - depth, so two integrals that tie on the first two keys
  // necessarily tie on rank as well: the third key can never break a tie.  This
  // pair ties on all three (total 3, depth 2, rank 1) and is decided by the
  // max-denominator key instead.
  RawIntegral a = RI({2, -1, 0});
  RawIntegral b = RI({1, 1, -1});
  REQUIRE(a.depth() + a.rank() == b.depth() + b.rank());
  REQUIRE(a.depth() == b.depth());
  REQUIRE(a.rank() == b.rank());
  REQUIRE(SectorTestHelper::compare_complexity(b, a));
}

TEST(compare_complexity_max_denominator_is_fourth_key) {
  // [2,-1,0] and [1,1,-1] both have depth 2, rank 1, total 3; the one whose
  // largest denominator power is smaller comes first.
  REQUIRE(SectorTestHelper::compare_complexity(RI({1, 1, -1}), RI({2, -1, 0})));
  REQUIRE(!SectorTestHelper::compare_complexity(RI({2, -1, 0}), RI({1, 1, -1})));
}

TEST(compare_complexity_max_numerator_is_last_key) {
  // [2,-3,0,0] and [2,-1,-1,-1]: depth 2, rank 3, total 5, max denominator 2 —
  // they differ only in the largest numerator power (3 vs 1).
  REQUIRE(SectorTestHelper::compare_complexity(RI({2, -1, -1, -1}),
                                               RI({2, -3, 0, 0})));
  REQUIRE(!SectorTestHelper::compare_complexity(RI({2, -3, 0, 0}),
                                                RI({2, -1, -1, -1})));
}

TEST(compare_complexity_denominator_key_outranks_numerator_key) {
  // The two keys above are tested in isolation; this pair pins down their
  // RELATIVE priority — the two keys point opposite ways, so a comparator that
  // consulted the numerator power before the denominator power would order this
  // pair the other way round.
  //   a = [2,-1,-1,0]: total 4, depth 2, rank 2, max denominator 2, max num 1
  //   b = [1,1,-2,0] : total 4, depth 2, rank 2, max denominator 1, max num 2
  RawIntegral a = RI({2, -1, -1, 0});
  RawIntegral b = RI({1, 1, -2, 0});
  REQUIRE(a.depth() + a.rank() == b.depth() + b.rank());
  REQUIRE(a.depth() == b.depth());
  REQUIRE(a.rank() == b.rank());
  REQUIRE(a.maxpos() == 2 && b.maxpos() == 1);
  REQUIRE(-a.maxneg() == 1 && -b.maxneg() == 2);
  // denominator power decides first → the smaller one (b) comes first
  REQUIRE(SectorTestHelper::compare_complexity(b, a));
  REQUIRE(!SectorTestHelper::compare_complexity(a, b));
}

TEST(compare_complexity_is_strict_weak_ordering_on_ties) {
  // Two distinct integrals with identical keys: neither is less than the other.
  RawIntegral a = RI({1, 1, -1, 0});
  RawIntegral b = RI({1, 1, 0, -1});
  REQUIRE(!SectorTestHelper::compare_complexity(a, b));
  REQUIRE(!SectorTestHelper::compare_complexity(b, a));
  REQUIRE(!SectorTestHelper::compare_complexity(a, a));
}

// ---------------------------------------------------------------------------
// _order_subintegrals — the subsector column order
// ---------------------------------------------------------------------------

TEST(order_subintegrals_complexity_puts_largest_key_in_smallest_column) {
  // Column nvars+j: index 0 gets the smallest subsector column, so the vector
  // must come back in DESCENDING key order.
  std::vector<RawIntegral> in = {RI({2, -1, 0}), RI({1, 0, 0}), RI({1, 1, -1})};
  auto out = SectorTestHelper::order_subintegrals(in, OrderingMode::Complexity);
  REQUIRE(out.size() == 3u);
  REQUIRE(out[0] == RI({2, -1, 0}));   // total 3, depth 2, max pos 2
  REQUIRE(out[1] == RI({1, 1, -1}));   // total 3, depth 2, max pos 1
  REQUIRE(out[2] == RI({1, 0, 0}));    // total 1
}

TEST(order_subintegrals_non_complexity_modes_keep_canonical_order) {
  // Balance and Normal order subsector integrals by the canonical per-index
  // comparator — the new key must not leak into them.
  std::vector<RawIntegral> in = {RI({2, -1, 0}), RI({1, 0, 0}), RI({1, 1, -1})};
  auto bal = SectorTestHelper::order_subintegrals(in, OrderingMode::Balance);
  auto nor = SectorTestHelper::order_subintegrals(in, OrderingMode::Normal);
  // canonical: depth ASC, rank ASC, then sector ASC → [1,0,0], [2,-1,0], [1,1,-1]
  REQUIRE(bal[0] == RI({1, 1, -1}));
  REQUIRE(bal[1] == RI({2, -1, 0}));
  REQUIRE(bal[2] == RI({1, 0, 0}));
  REQUIRE(nor == bal);
  // ... and it differs from the Complexity order on this very input.
  auto cmp = SectorTestHelper::order_subintegrals(in, OrderingMode::Complexity);
  REQUIRE(bal != cmp);
}

TEST(order_subintegrals_complexity_ties_mirror_input_order) {
  // Every one of the 24 integrals is a permutation of the same multiset, so all
  // five keys are identical for all of them: the stable sort must keep the
  // incoming order, and the final reverse must therefore mirror it exactly.
  // The input is deliberately larger than the smallest sorting path of
  // libstdc++ (where std::sort leaves ≤ 16 tied elements in place by luck but
  // permutes 24), so an unstable sort fails here instead of passing silently.
  const std::vector<int> multiset = {-4, -2, -1, -1, 1, 1, 2, 4};
  std::vector<RawIntegral> in;
  std::vector<int> p = multiset;  // lexicographically smallest arrangement first
  for (int i = 0; i < 24; ++i) {
    in.push_back(RawIntegral(p));
    REQUIRE(std::next_permutation(p.begin(), p.end()));
  }
  // Precondition: all 24 are distinct yet tie on every key.
  for (const RawIntegral &x : in) {
    REQUIRE(x.depth() == 8u);
    REQUIRE(x.rank() == 8u);
    REQUIRE(x.maxpos() == 4);
    REQUIRE(-x.maxneg() == 4);
    REQUIRE(!SectorTestHelper::compare_complexity(x, in[0]));
    REQUIRE(!SectorTestHelper::compare_complexity(in[0], x));
  }
  auto out = SectorTestHelper::order_subintegrals(in, OrderingMode::Complexity);
  REQUIRE(out.size() == in.size());
  for (unsigned i = 0; i < in.size(); ++i)
    REQUIRE(out[i] == in[in.size() - 1 - i]);
}

// ---------------------------------------------------------------------------
// _generate_seeds — ordering modes
// ---------------------------------------------------------------------------

// Normal mode: seeds are in the raw generation order (no sort applied).
// The first seed should be [1,0] (excess_depth=0, rank=0) because the
// generation loop starts at depth=0.
TEST(seeds_normal_mode_preserves_generation_order) {
  std::vector<bool> lines = {true, false};
  Sector s = SectorTestHelper::make_sector(1, 2, lines, 2, 1);
  SectorTestHelper::generate_seeds(s, OrderingMode::Normal);
  const auto &seeds = SectorTestHelper::seeds(s);
  REQUIRE(!seeds.empty());
  REQUIRE(seeds[0][0] == 1);
  REQUIRE(seeds[0][1] == 0);
}

// Complexity mode: seeds ordered by (depth+rank ASC, depth ASC, rank ASC,
// max_denominator ASC, max_numerator ASC).
TEST(seeds_complexity_mode_total_power_first) {
  // One active line, one zero line, box depth 3 rank 2 → 9 seeds whose keys
  // differ enough to separate this mode from both Normal and Balance.
  std::vector<bool> lines = {true, false};
  Sector s = SectorTestHelper::make_sector(1, 2, lines, 3, 2);
  SectorTestHelper::generate_seeds(s, OrderingMode::Complexity);
  const auto &seeds = SectorTestHelper::seeds(s);
  REQUIRE(seeds.size() == 9u);
  // total power 1 → [1,0]; total 2 → [1,-1] (depth 1) before [2,0] (depth 2);
  // total 3 → [1,-2], [2,-1], [3,0] by depth.
  REQUIRE(seeds[0] == RI({1, 0}));
  REQUIRE(seeds[1] == RI({1, -1}));
  REQUIRE(seeds[2] == RI({2, 0}));
  REQUIRE(seeds[3] == RI({1, -2}));
  REQUIRE(seeds[4] == RI({2, -1}));
  REQUIRE(seeds[5] == RI({3, 0}));
  // total 4 → [2,-2] (depth 2) before [3,-1] (depth 3); total 5 → [3,-2].
  REQUIRE(seeds[6] == RI({2, -2}));
  REQUIRE(seeds[7] == RI({3, -1}));
  REQUIRE(seeds[8] == RI({3, -2}));
  // The weights are still the positions in this order.
  const auto &w = SectorTestHelper::weights(s);
  for (unsigned i = 0; i < seeds.size(); ++i)
    REQUIRE(w.at(seeds[i]) == i);
}

TEST(seeds_complexity_mode_differs_from_normal_and_balance) {
  // On this one-line sector Normal and Balance coincide: the generation order is
  // already (excess depth, rank)-major, and within a fixed excess depth and rank
  // the largest active power is constant, so Balance's third key never fires.
  // Complexity interleaves the two axes by their sum instead, so [2,0] (total 2)
  // overtakes [1,-2] (total 3).
  std::vector<bool> lines = {true, false};
  Sector sn = SectorTestHelper::make_sector(1, 2, lines, 3, 2);
  Sector sb = SectorTestHelper::make_sector(1, 2, lines, 3, 2);
  Sector sc = SectorTestHelper::make_sector(1, 2, lines, 3, 2);
  SectorTestHelper::generate_seeds(sn, OrderingMode::Normal);
  SectorTestHelper::generate_seeds(sb, OrderingMode::Balance);
  SectorTestHelper::generate_seeds(sc, OrderingMode::Complexity);
  const auto &n = SectorTestHelper::seeds(sn);
  const auto &b = SectorTestHelper::seeds(sb);
  const auto &c = SectorTestHelper::seeds(sc);
  REQUIRE(n == b);
  REQUIRE(n[2] == RI({1, -2}));
  REQUIRE(c[2] == RI({2, 0}));
  REQUIRE(c != n);
  REQUIRE(c != b);
}

// Balance mode: seeds ordered by (excess_depth ASC, rank ASC, max_power ASC).
TEST(seeds_balance_mode_excess_depth_first) {
  std::vector<bool> lines = {true, true};
  Sector s = SectorTestHelper::make_sector(3, 2, lines, 3, 1);
  SectorTestHelper::generate_seeds(s, OrderingMode::Balance);
  const auto &seeds = SectorTestHelper::seeds(s);
  REQUIRE(!seeds.empty());
  REQUIRE(seeds[0].depth() == 2u);
  REQUIRE(seeds[0].rank()  == 0u);
}

// ---------------------------------------------------------------------------
// ATPS timing — per-sector probe/full pass accounting
// ---------------------------------------------------------------------------

TEST(timing_row_format) {
  SectorTiming t;
  t.id = 42;
  t.expanded = true;
  t.probeNanos = 1'500'000'000ULL;  // 1.5 s
  t.fullNanos = 12'000'000'000ULL;  // 12 s
  REQUIRE(format_timing_row(t) == "42,1,1.500,12.000\n");
}

TEST(timing_row_rounds_to_milliseconds) {
  SectorTiming t;
  t.id = 7;
  t.expanded = false;
  t.probeNanos = 1'234'567ULL;  // 1.234567 ms -> 0.001 s
  t.fullNanos = 0;
  REQUIRE(format_timing_row(t) == "7,0,0.001,0.000\n");
}

TEST(timing_counters_start_at_zero) {
  ReductionContext ctx;
  REQUIRE(ctx.probeNanos.load() == 0u);
  REQUIRE(ctx.fullNanos.load() == 0u);
}

TEST(timing_accumulates_across_sectors) {
  ReductionContext ctx;
  SectorTiming a;
  a.id = 1;
  a.expanded = true;
  a.probeNanos = 2'000'000'000ULL;
  a.fullNanos = 5'000'000'000ULL;
  SectorTiming b;
  b.id = 5;
  b.expanded = false;
  b.probeNanos = 500'000'000ULL;
  b.fullNanos = 30'000'000'000ULL;
  ctx.accumulateTiming(a);
  ctx.accumulateTiming(b);
  REQUIRE(ctx.probeNanos.load() == 2'500'000'000ULL);
  REQUIRE(ctx.fullNanos.load() == 35'000'000'000ULL);
}

// ---------------------------------------------------------------------------
// Per-sector target list — complete snapshot on disk as soon as it is written
// ---------------------------------------------------------------------------

TEST(write_target_list_complete_on_disk) {
  const std::string path = "/tmp/secred_test_targets.txt";
  std::set<RawIntegral> targets;
  // Enough entries to overflow the 8 KiB stdio filebuf: a stream left open for
  // the sector's lifetime would show only its first buffer here.  Indices stay
  // within RawIntegral's int8_t range, and (i % 100, i / 100) is injective.
  for (int i = 1; i <= 1000; ++i)
    targets.insert(RI({1, 1, i % 100, i / 100}));
  REQUIRE(targets.size() == 1000u);

  writeTargetList(path, targets);

  REQUIRE(std::filesystem::file_size(path) > 8192);
  std::ifstream in(path);
  std::set<RawIntegral> back;
  std::string line;
  while (std::getline(in, line))
    back.insert(RawIntegral::from_text(line));
  REQUIRE(back.size() == targets.size());
  REQUIRE(back == targets);

  // Newline-terminated: no entry may be cut mid-number.
  in.clear();
  in.seekg(-1, std::ios::end);
  char last = 0;
  in.get(last);
  REQUIRE(last == '\n');
  std::filesystem::remove(path);
}

TEST(write_target_list_empty_is_empty_file) {
  const std::string path = "/tmp/secred_test_targets_empty.txt";
  std::set<RawIntegral> targets;
  writeTargetList(path, targets);
  REQUIRE(std::filesystem::file_size(path) == 0);
  std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// initial_seed_bounds — smallest box covering a sector's requests
// ---------------------------------------------------------------------------

TEST(initial_seed_bounds_no_requests_is_base_layer) {
  // Sector 0b101 (two lines) asked for nothing beyond its base layer: with no
  // minimum-1 restriction the box is the base layer itself, i.e. zero excess
  // depth, not the one-layer-too-big box the old floor forced.
  auto [depth, rank] = initial_seed_bounds(0b101, {}, {});
  REQUIRE(depth == 2u);   // popcount(0b101), not 3
  REQUIRE(rank == 1u);    // the rank floor is unchanged
}

TEST(initial_seed_bounds_uptarget_at_base_depth_keeps_floor_off) {
  // An uptarget is reduced in place, so it needs its own depth only: a base
  // integral (all its lines at power 1) does not raise the box at all.  This is
  // the case the removed floor used to inflate to popcount + 1.
  std::set<RawIntegral> uptargets{RI({1, 0, 1})};
  auto [depth, rank] = initial_seed_bounds(0b101, {}, uptargets);
  REQUIRE(depth == 2u);
  REQUIRE(rank == 1u);
}

TEST(initial_seed_bounds_uptarget_depth_lifts_box) {
  // A deeper uptarget lifts the box to its own depth, without the extra +1 a
  // direct target gets.
  std::set<RawIntegral> uptargets{RI({3, 0, 1})};  // depth 4, rank 0
  auto [depth, rank] = initial_seed_bounds(0b101, {}, uptargets);
  REQUIRE(depth == 4u);
  REQUIRE(rank == 1u);
}

TEST(initial_seed_bounds_target_gets_one_layer_more) {
  // A direct target needs the layer above it, so its depth contributes +1.
  std::set<RawIntegral> targets{RI({2, 0, 0})};  // depth 2, rank 0
  auto [depth, rank] = initial_seed_bounds(0b101, targets, {});
  REQUIRE(depth == 3u);
  REQUIRE(rank == 1u);
}

TEST(initial_seed_bounds_rank_from_negative_indices) {
  // Rank is read off the negative (numerator) powers: the deepest request wins,
  // and every one of them contributes +1.
  std::set<RawIntegral> targets{RI({1, 0, 1}), RI({-2, 0, 1})};
  std::set<RawIntegral> uptargets{RI({1, 0, -3})};
  auto [depth, rank] = initial_seed_bounds(0b101, targets, uptargets);
  REQUIRE(depth == 3u);  // max(base 2, target [1,0,1]: 2 + 1)
  REQUIRE(rank == 4u);   // max(1, target [-2,0,1]: 2 + 1, uptarget [1,0,-3]: 3 + 1)
}

TEST(initial_seed_bounds_never_below_base_layer) {
  // Every request belongs to this sector, so it has at least popcount(id)
  // positive units: the box can never fall below the base layer, which is what
  // keeps the unsigned excess-depth arithmetic in _generate_seeds sound.
  const unsigned id = 0b1011;
  std::set<RawIntegral> uptargets{RI({1, 1, 1, 1})};
  auto [depth, rank] = initial_seed_bounds(id, {}, uptargets);
  REQUIRE(depth >= std::popcount(id));
  REQUIRE(depth == 4u);
  REQUIRE(rank == 1u);
}

TEST(initial_seed_bounds_matches_observed_sector_box) {
  // Reproduce a sector log line: sector 0b111 with target [1,1,1] (depth 3)
  // gets a depth bound of 4, one layer above the target, i.e. an excess depth of
  // 1 -- and the log's seed depth sits one below that bound, so it reads 3.
  std::set<RawIntegral> targets{RI({1, 1, 1})};
  auto [depth, rank] = initial_seed_bounds(0b111, targets, {});
  REQUIRE(depth == 4u);
  REQUIRE(depth - std::popcount(0b111u) == 1u);
  REQUIRE(rank == 1u);
}

TEST(initial_seed_bounds_matches_observed_deep_sector_box) {
  // A deeper target reproduces a two-layer sector: sector 0b111 asked for
  // [1, 1, 2] (depth 4) gets a bound of 5, excess depth 2 (logged seed depth 4).
  std::set<RawIntegral> targets{RI({1, 1, 2})};
  auto [depth, rank] = initial_seed_bounds(0b111, targets, {});
  REQUIRE(depth == 5u);
  REQUIRE(depth - std::popcount(0b111u) == 2u);
}

TEST(initial_seed_bounds_excess_depth_is_zero_only_for_base_layer_box) {
  // The bound's excess depth is 0 iff the box is the base layer, so it is this
  // difference that the old floor (`_depth = popcount(id) + 1`) forced to at
  // least 1.  Such a box admits no seed: the seed depth the log prints is one
  // below the base layer, i.e. popcount(id) - 1.
  const unsigned id = 0b10110;               // five propagators, lines 1, 2, 4
  std::set<RawIntegral> uptargets{RI({0, 1, 1, 0, 1})};  // the base integral
  auto [depth, rank] = initial_seed_bounds(id, {}, uptargets);
  REQUIRE(depth - std::popcount(id) == 0u);  // logged seed depth: popcount - 1
  REQUIRE(rank == 1u);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// SeedBound — the seed-boundary rule behind settings.boundary_mode
// ---------------------------------------------------------------------------

TEST(seedbound_box_is_the_legacy_rule) {
  // Box mode: two independent strict bounds, exactly as before the option
  // existed (limit is Total-only and stays 0 here).  RI({+d, -r}) is depth d,
  // rank r.
  SeedBound b = SeedBound::initial(BoundaryMode::Box, 5, 4);
  REQUIRE(b.mode  == BoundaryMode::Box);
  REQUIRE(b.depth == 5u);
  REQUIRE(b.rank  == 4u);
  REQUIRE(b.limit == 0u);            // not used in Box mode
  REQUIRE(b.boundary() == 5u);       // the depth bound is the boundary here
  REQUIRE(b.quantity(RI({3, -2})) == 3u);
  REQUIRE(b.admits(RI({4, -3})));
  REQUIRE(!b.admits(RI({5, -3})));   // depth cap
  REQUIRE(!b.admits(RI({3, -4})));   // rank cap
  REQUIRE(b.exceeded(RI({5, 0})));
  REQUIRE(!b.exceeded(RI({4, 0})));
  // The criterion is one-sided and depth-only here: a seed the rank cap refuses
  // but whose depth is inside the box is NOT reported, because the Box-mode repair
  // grows the depth bound and would not admit it.  Those are the wide, shallow
  // "outer" integrals a sector hands on as masters.
  REQUIRE(!b.exceeded(RI({4, -4})));
  REQUIRE(!b.admits(RI({4, -4})));
}

TEST(seedbound_box_admits_matches_the_literal_old_expression) {
  // Box mode must be provably the old conjunction, over a grid of integrals.
  SeedBound b = SeedBound::initial(BoundaryMode::Box, 4, 3);
  for (unsigned d = 0; d <= 6; ++d)
    for (unsigned r = 0; r <= 5; ++r)
      REQUIRE(b.admits(RI({(int)d, -(int)r})) == (d < 4u && r < 3u));
}

TEST(seedbound_box_filter_is_the_legacy_rule) {
  // The subsector filter drops an integral whose depth is more than n below the
  // boundary -- the shallow ones -- and keeps the ones at or past boundary - n.
  // Box mode must be the literal historical `depth < depthBound - n` over a
  // grid: inverting the comparison silently rebuilds every sector's system
  // (the smoke test moved from 480/128 to 1167/377 equations when it happened).
  SeedBound b = SeedBound::initial(BoundaryMode::Box, 4, 3);
  for (unsigned d = 0; d <= 6; ++d)
    for (unsigned n = 0; n <= 5; ++n)
      REQUIRE(b.filter_drops(RI({(int)d, 0}), n) == ((int)d < 4 - (int)n));
}

TEST(seedbound_filter_direction_is_boundary_relative) {
  // Depth 1 / rank 1, n = 1: the boundary is 2, so only depth >= 1 survives --
  // the integral below it is dropped, the one on the boundary is kept.
  SeedBound b = SeedBound::initial(BoundaryMode::Box, 2, 2);
  REQUIRE(b.filter_drops(RI({0, -1}), 1));
  REQUIRE(!b.filter_drops(RI({1, 0}), 1));
  REQUIRE(!b.filter_drops(RI({3, 0}), 1));
  // n = 0 drops nothing at or above the boundary but everything below it.
  REQUIRE(!b.filter_drops(RI({2, 0}), 0));
  REQUIRE(b.filter_drops(RI({1, 0}), 0));
  // The subtraction is signed (the old code cast both operands too): a boundary
  // below n must not wrap around into "drop everything" -- with n past the
  // boundary nothing is far enough below it to be dropped.
  SeedBound tiny = SeedBound::initial(BoundaryMode::Box, 1, 1);
  REQUIRE(!tiny.filter_drops(RI({0, 0}), 3));
  REQUIRE(!tiny.filter_drops(RI({5, 0}), 3));
}

TEST(seedbound_total_filter_counts_from_the_seed_box_top) {
  // Total mode: box (6, 5) at its initial cap (11 = depth + rank) admits seeds up
  // to total power 9 -- the box corner -- so the filter's reference is 9, not the
  // cap 11.  At n = 1 it keeps the top layer of what the seeds can generate
  // (total power >= 9); at n = 0 it keeps nothing at all, because no subsector
  // integral a seed generates reaches past that layer.  An integral below the top
  // layer is dropped even though its depth is large.
  SeedBound b = SeedBound::initial(BoundaryMode::Total, 6, 5);
  REQUIRE(b.boundary() == 11u);
  REQUIRE(b.seed_top() == 9);
  REQUIRE(!b.filter_drops(RI({5, -5}), 1));   // total 10, past the top layer
  REQUIRE(!b.filter_drops(RI({9, -2}), 1));   // total 11, past the boundary
  REQUIRE(!b.filter_drops(RI({5, -4}), 1));   // total 9, the top layer itself
  REQUIRE(b.filter_drops(RI({4, -4}), 1));    // total 8
  REQUIRE(b.filter_drops(RI({5, -2}), 1));    // total 7
  // n = 0: everything the seeds can generate is dropped -- at the box-top cap no
  // generated subsector integral reaches past the top layer, so nothing is kept.
  REQUIRE(b.filter_drops(RI({5, -4}), 0));    // total 9, exactly the seed top
  REQUIRE(b.filter_drops(RI({5, -2}), 0));    // total 7
  REQUIRE(b.filter_drops(RI({0, 0}), 0));     // the emptiest integral of all
  // Above the top layer the filter still keeps what it is given (it has no upper
  // end) -- nothing generates such an integral here, but the rule is one-sided.
  REQUIRE(!b.filter_drops(RI({5, -5}), 0));   // total 10
  REQUIRE(!b.filter_drops(RI({6, -5}), 0));   // total 11
  // The subtraction is signed here too: a reference below n must not wrap around
  // into "drop everything" (Box mode's guard, in the total-power quantity).
  SeedBound tiny = SeedBound::initial(BoundaryMode::Total, 1, 0);
  REQUIRE(tiny.boundary() == 1u);
  REQUIRE(!tiny.filter_drops(RI({0, 0}), 3));
  REQUIRE(!tiny.filter_drops(RI({5, 0}), 3));
}

TEST(seedbound_filter_reference_is_the_seed_box_not_the_cap) {
  // The filter counts layers from the admitted seeds, not from the cap: a cap
  // biting into the box (the corner-cut generation) lowers the reference to the
  // corner it leaves, so n = 1 there reads exactly like n = 2 does from the box top
  // the cap originally sat above.  The +1 cap, which bites the corner exactly,
  // still leaves the top layer where the box put it.
  SeedBound top = SeedBound::initial(BoundaryMode::Total, 6, 5);  // cap 11
  REQUIRE(top.seed_top() == 9);
  SeedBound cut = top;
  cut.limit = 9;                           // depth + rank - 2: cuts the corner
  REQUIRE(cut.seed_top() == 8);
  SeedBound plus1 = top;
  plus1.limit = 10;                        // depth + rank - 1: bites the corner
  REQUIRE(plus1.seed_top() == top.seed_top());
  for (int t = 0; t <= 13; ++t) {
    RawIntegral depthOnly = RI({t, 0});    // a depth-t, rank-0 integral: total t
    REQUIRE(cut.filter_drops(depthOnly, 1) == top.filter_drops(depthOnly, 2));
    REQUIRE(plus1.filter_drops(depthOnly, 1) == top.filter_drops(depthOnly, 1));
  }
  // A widening instead raises the reference by one layer -- it grows the box, and
  // in Total mode the cap rides at its top -- so n keeps its meaning as a distance
  // below the top: n = 1 keeps the top layer wherever the top now is.
  SeedBound widened = top;
  widened.widen_depth();                   // box (7, 5), cap 12
  REQUIRE(widened.seed_top() == top.seed_top() + 1);
  for (int t = 0; t <= 13; ++t)
    REQUIRE(widened.filter_drops(RI({t, 0}), 1) == (t < widened.seed_top()));
}

TEST(seedbound_total_admission_equals_box_admission) {
  // The deepest seed the hard caps admit is (depth - 1, rank - 1) = (5, 4),
  // total 9, and the strict total comparison needs the cap above it, so that
  // seed is admitted and the cap cuts nothing.  Total-mode admission must
  // therefore equal Box-mode admission over every integral.
  SeedBound b = SeedBound::initial(BoundaryMode::Total, 6, 5);
  SeedBound box = SeedBound::initial(BoundaryMode::Box, 6, 5);
  REQUIRE(b.depth == 6u);    // deepest admitted seed depth 5, strict bound
  REQUIRE(b.rank  == 5u);    // deepest admitted seed rank 4, strict bound
  REQUIRE(b.limit == 11u);   // = depth + rank, the top of the box's total-power range
  REQUIRE(b.quantity(RI({3, -2})) == 5u);   // depth + rank, not depth
  REQUIRE(b.admits(RI({5, -3})));           // total 8, inside
  REQUIRE(b.admits(RI({5, -4})));           // total 9 = the deepest admitted seed
  REQUIRE(!b.exceeded(RI({5, -4})));        // and not "outside" either
  for (unsigned d = 0; d <= 8; ++d)
    for (unsigned r = 0; r <= 7; ++r)
      REQUIRE(b.admits(RI({(int)d, -(int)r})) ==
              box.admits(RI({(int)d, -(int)r})));
}

TEST(seedbound_total_probe_criterion_is_the_box_top) {
  // The probe's "outside" test reads the seed box, in total power.  Starting at
  // the cap initial() builds (limit == depth + rank) the seeds past the box top
  // (total 9 for a box (6, 5)) are exactly the ones the hard caps reject: the two
  // shells at total 10 and the corner at 11.  The pre-fix criterion compared
  // against the cap (11) instead, which only the corner reaches -- so a sector
  // whose reduction needed one of the shells kept it as a master with no report.
  // refusal() says which cap refuses each of them, and the widening admits it.
  SeedBound b = SeedBound::initial(BoundaryMode::Total, 6, 5);
  REQUIRE(b.seed_top() == 9);
  REQUIRE(!b.exceeded(RI({5, -4})));  // total 9, deepest seed the caps admit
  REQUIRE(b.exceeded(RI({5, -5})));   // total 10, box shell (rank)
  REQUIRE(b.exceeded(RI({6, -4})));   // total 10, box shell (depth)
  REQUIRE(b.exceeded(RI({6, -5})));   // total 11, the box corner
  REQUIRE(b.refusal(RI({5, -5})) == 2u);
  REQUIRE(b.refusal(RI({6, -4})) == 1u);
  REQUIRE(b.refusal(RI({6, -5})) == 3u);
  b.widen_depth();
  REQUIRE(b.admits(RI({6, -4})));     // the depth widening admits the depth shell
  REQUIRE(!b.admits(RI({5, -5})));    // and leaves the rank cap where it was
  b.widen_rank();
  REQUIRE(b.admits(RI({5, -5})));
  REQUIRE(b.admits(RI({6, -5})));
}

TEST(seedbound_total_keeps_the_caps_hard) {
  // Widening one hard bound never lets an integral out of the box through the
  // other: (3, 5) has total 8, passes the cap comfortably and is refused by the
  // rank cap, and a depth widening -- the only repair a Total-mode depth refusal
  // asks for -- leaves that refusal standing.  (The probe reads this: the seed is
  // inside the box top, so it is not reported at all.)
  SeedBound b = SeedBound::initial(BoundaryMode::Total, 6, 5);
  REQUIRE(b.quantity(RI({3, -5})) == 8u);
  REQUIRE(!b.exceeded(RI({3, -5})));
  REQUIRE(!b.admits(RI({3, -5})));
  REQUIRE(!b.admits(RI({6, 0})));   // depth cap
  b.widen_depth();
  REQUIRE(!b.admits(RI({3, -5})));  // the rank cap still refuses it
  REQUIRE(b.admits(RI({6, 0})));    // while the widened depth admits this one
}

TEST(seedbound_widen_undoes_the_narrowing_only) {
  // Box: the widening raises the depth bound (the historical repair) and thereby
  // admits the layer just outside it, leaving the rank bound alone.
  SeedBound box = SeedBound::initial(BoundaryMode::Box, 5, 4);
  REQUIRE(!box.admits(RI({5, -3})));
  box.widen_depth();
  REQUIRE(box.depth == 6u);
  REQUIRE(box.rank  == 4u);
  REQUIRE(box.admits(RI({5, -3})));

  // Total: the same widening, with the cap riding at the top of the box, so the
  // admitted set stays exactly the Box-mode one and the cap keeps admitting
  // nothing the hard caps reject.
  SeedBound tot = SeedBound::initial(BoundaryMode::Total, 6, 5);
  tot.widen_depth();
  REQUIRE(tot.depth == 7u);
  REQUIRE(tot.rank  == 5u);
  REQUIRE(tot.limit == 12u);          // = depth + rank: the top of the new box
  SeedBound boxSame = SeedBound::initial(BoundaryMode::Box, 7, 5);
  for (unsigned d = 0; d <= 8; ++d)
    for (unsigned r = 0; r <= 7; ++r)
      REQUIRE(tot.admits(RI({(int)d, -(int)r})) ==
              boxSame.admits(RI({(int)d, -(int)r})));
  tot.widen_rank();
  REQUIRE(tot.rank  == 6u);
  REQUIRE(tot.limit == 13u);          // and the cap follows it up
}

TEST(seedbound_widens_the_box_a_master_sits_above) {
  // The failure the criterion exists for, on the case that reported it: box.yaml's
  // sector12 under boundary_mode: total with subsector_filter_n: 0.  Its targets
  // [0, 0, 1, 2] and [0, 0, 2, 1] give a box of depth 3, rank 1 -- the sector's four
  // propagators are two lines, so popcount(0b1100) = 2 and the deepest target
  // (depth 3) asks for one layer above itself -- and the master the reduction was
  // left with is [0, 0, 1, 2], depth 3, rank 0, total 3.  The box top is 2, so the
  // pre-fix criterion (against the cap, 4) never fired and the sector kept the
  // master.  The probe reports it now, and one widening admits it: box (4, 1), cap
  // 5, seeds up to total 3.
  SeedBound b = SeedBound::initial(BoundaryMode::Total, 3, 1);
  REQUIRE(b.limit == 4u);
  REQUIRE(b.seed_top() == 2);
  RawIntegral master = RI({0, 0, 1, 2});
  REQUIRE(master.depth() == 3u);
  REQUIRE(master.rank() == 0u);
  REQUIRE(!b.admits(master));
  REQUIRE(b.exceeded(master));
  REQUIRE(b.refusal(master) == 1u);   // the depth cap is what refuses it
  b.widen_depth();
  REQUIRE(b.depth == 4u);
  REQUIRE(b.limit == 5u);
  REQUIRE(b.seed_top() == 3);         // the top of the box rose to the master's total
  REQUIRE(b.admits(master));
}

TEST(seedbound_total_deep_low_rank_seed_is_reported) {
  // The second half of the criterion, on the case that reported it: F234's sector
  // 246 under boundary_mode: total.  Its seed box is (7, 5), so the initial cap is
  // 12 and seed_top() is 10 -- a high rank figure lifts the box top three layers
  // above the depth cap -- and the seed the reduction was left with is
  // [0,1,2,0,1,1,1,1,0]: depth 7, rank 0, total 7.  admits() refuses it on the
  // depth (7 >= 7), so it generates no equation and stays a master, while the
  // total-power half of the criterion stays silent (7 is not past 10).  Box mode,
  // which counts the depth alone, reported that seed and widened; Total mode did
  // not, and kept the master -- the +1 the global master list showed against box
  // mode (375 against 374, sector 246 the only sector that differed).
  SeedBound b = SeedBound::initial(BoundaryMode::Total, 7, 5);
  REQUIRE(b.limit == 12u);
  REQUIRE(b.seed_top() == 10);
  RawIntegral master = RI({0, 1, 2, 0, 1, 1, 1, 1, 0});
  REQUIRE(master.depth() == 7u);
  REQUIRE(master.rank() == 0u);
  REQUIRE(b.quantity(master) == 7u);
  REQUIRE(!b.admits(master));
  REQUIRE(!b.past_top(master));         // the total-power half stays silent
  REQUIRE(b.past_depth_cap(master));    // the depth cap is what refuses it
  REQUIRE(b.exceeded(master));          // and the criterion reports it now
  REQUIRE(b.refusal(master) == 1u);     // the depth figure is the one to grow

  // The repair: deepen the box, leave the cap where it was.  The seed is admitted
  // (total 7 < cap 12) and the box top -- the filter's reference -- rises by one.
  b.widen_depth(/*carry_cap=*/false);
  REQUIRE(b.depth == 8u);
  REQUIRE(b.limit == 12u);
  REQUIRE(b.seed_top() == 11);
  REQUIRE(b.admits(master));
  REQUIRE(!b.exceeded(master));

  // At this first layer nothing is cut yet: the cap sits at the old box's top
  // (depth + rank = 12) and the new corner (7, 4) has total 11, so the deepened box
  // admits exactly what the carrying repair would -- the two differ only from the
  // second layer on, where the corner reaches the cap.
  SeedBound rect = SeedBound::initial(BoundaryMode::Total, 7, 5);
  rect.widen_depth();                   // the carrying repair, for comparison
  REQUIRE(rect.limit == 13u);
  REQUIRE(rect.seed_top() == 11);       // the same reference, from the box top
  REQUIRE(b.admits(RI({7, -4})) == rect.admits(RI({7, -4})));

  // Second layer: the cap now bites.  The deepened box keeps its total-power
  // ceiling (12) and cuts the corner (8, 4) at total 12, where the carrying repair
  // -- a rectangle whose top rides at 14 -- admits it.  That is the equation the
  // repair the depth cap asks for deliberately does not buy.
  b.widen_depth(/*carry_cap=*/false);
  REQUIRE(b.depth == 9u);
  REQUIRE(b.limit == 12u);
  REQUIRE(b.seed_top() == 11);
  REQUIRE(b.admits(RI({7, -4})));        // the deeper layer is admitted
  REQUIRE(!b.admits(RI({8, -4})));       // its total-12 corner is not
  rect.widen_depth();
  REQUIRE(rect.limit == 14u);
  REQUIRE(rect.admits(RI({8, -4})));
}

TEST(seedbound_total_past_top_still_carries_the_cap) {
  // A report the *box top* makes asks for the rectangle (bit 4): the depth
  // widening carries the cap, so the admitted set stays the Box-mode one.  Both
  // halves can fire for the same seed, and then the box top wins -- nothing about
  // a seed past the top is fixed by deepening the box alone.
  SeedBound b = SeedBound::initial(BoundaryMode::Total, 7, 5);
  RawIntegral rankShell = RI({6, -5});  // depth 6, rank 5, total 11 > top 10
  REQUIRE(b.quantity(rankShell) == 11u);
  REQUIRE(b.past_top(rankShell));
  REQUIRE(!b.past_depth_cap(rankShell));  // the rank cap, not the depth cap
  REQUIRE(b.exceeded(rankShell));
  REQUIRE(b.refusal(rankShell) == 2u);    // so widen_rank is the repair
  RawIntegral deepBoth = RI({7, -4});     // depth 7, rank 4, total 11 > top 10
  REQUIRE(b.past_top(deepBoth));
  REQUIRE(b.past_depth_cap(deepBoth));    // both halves fire on this one
  REQUIRE(b.refusal(deepBoth) == 1u);
  b.widen_depth();                        // the carrying repair: limit 12 -> 13
  REQUIRE(b.limit == 13u);
  REQUIRE(b.admits(deepBoth));
  // Had the depth cap been the only report (bit 4 clear) the cap would have stayed
  // at 12 and this seed -- total 11, inside either cap -- would be admitted just the
  // same: the two repairs first differ only at the layer where the box corner
  // reaches the cap (see the test above).  The box top at 12 also lifts, since the
  // largest total power the hard caps admit is now 11.
  SeedBound deep = SeedBound::initial(BoundaryMode::Total, 7, 5);
  deep.widen_depth(/*carry_cap=*/false);
  REQUIRE(deep.limit == 12u);
  REQUIRE(deep.admits(deepBoth));
}

TEST(seedbound_box_mode_ignores_the_depth_cap_half) {
  // In Box mode quantity *is* the depth and seed_top() is depth - 1, so
  // past_depth_cap() is exactly past_top(): the new half adds nothing and Box
  // mode's criterion -- and therefore its seed box, its equations and its global
  // master basis -- is unchanged by it.
  SeedBound b = SeedBound::initial(BoundaryMode::Box, 5, 4);
  for (int d = 0; d <= 8; ++d)
    for (int r = 0; r <= 7; ++r) {
      RawIntegral i = RI({d, -r});
      REQUIRE(b.past_depth_cap(i) == b.past_top(i));
      REQUIRE(b.exceeded(i) == b.past_top(i));
    }
}

TEST(seedbound_total_degenerate_box_is_harmless) {
  // The smallest box initial_seed_bounds can return: the cap is depth + rank, a
  // plain sum, so it cannot underflow, and a degenerate box admits nothing
  // anyway (one of the two hard caps is always 0).
  SeedBound b = SeedBound::initial(BoundaryMode::Total, 1, 0);
  REQUIRE(b.limit == 1u);
  REQUIRE(!b.admits(RI({0, 0, 0})));
  SeedBound c = SeedBound::initial(BoundaryMode::Total, 0, 0);
  REQUIRE(c.limit == 0u);
  REQUIRE(!c.admits(RI({0, 0, 0})));
}

TEST(generate_seeds_is_independent_of_boundary_mode) {
  // Seeds are generated over the whole box under either mode; only admission
  // and the filter read the total cap, so switching mode cannot change nvars.
  Sector box = SectorTestHelper::make_sector(3, 2, {true, true}, 3, 2);
  Sector tot = SectorTestHelper::make_sector(3, 2, {true, true}, 3, 2);
  SectorTestHelper::set_boundary(
      tot, SeedBound::initial(BoundaryMode::Total, 3, 2));
  SectorTestHelper::generate_seeds(box, OrderingMode::Balance);
  SectorTestHelper::generate_seeds(tot, OrderingMode::Balance);
  REQUIRE(!SectorTestHelper::seeds(box).empty());
  REQUIRE(SectorTestHelper::seeds(box) == SectorTestHelper::seeds(tot));
  REQUIRE(SectorTestHelper::weights(box) == SectorTestHelper::weights(tot));
}

// ---------------------------------------------------------------------------
// format_seed_box_line — the seeds a sector was reduced with, and the seeds the
// probe pass started from
// ---------------------------------------------------------------------------

TEST(seed_box_line_reports_probe_start_when_raised) {
  // id 0b111 (three lines): reduced with a boundary of depth 5, whose deepest seed
  // has depth 4 (one inside the bound); the probe started from a boundary of depth
  // 4, i.e. seeds of depth 3, and raised the bound once.
  REQUIRE(format_seed_box_line(0b111, SeedBound::initial(BoundaryMode::Box, 5, 4),
                               SeedBound::initial(BoundaryMode::Box, 4, 4)) ==
          " depth: 4 rank: 3  probe start: depth: 3 rank: 3");
}

TEST(seed_box_line_repeats_start_when_probe_did_not_raise) {
  REQUIRE(format_seed_box_line(0b101, SeedBound::initial(BoundaryMode::Box, 3, 2),
                               SeedBound::initial(BoundaryMode::Box, 3, 2)) ==
          " depth: 2 rank: 1  probe start: depth: 2 rank: 1");
}

TEST(seed_box_line_prints_the_seed_bounds_for_a_base_layer_box) {
  // A box of exactly the sector's base layer (depth == popcount(id), the smallest
  // initial_seed_bounds can return) admits no seed at all -- the removal of the
  // `_depth = popcount(id) + 1` floor made this reachable, and the bound it hands
  // the probe is the one the probe has to raise.  The three figures are the seeds'
  // bounds, one step inside the boundary, so they still describe the box (depth 2
  // for a boundary of depth 3); that this box admits none of its own base-layer
  // integrals is what the probe reports and repairs.
  REQUIRE(format_seed_box_line(0b1011, SeedBound::initial(BoundaryMode::Box, 3, 2),
                               SeedBound::initial(BoundaryMode::Box, 3, 2)) ==
          " depth: 2 rank: 1  probe start: depth: 2 rank: 1");
}

TEST(seed_box_line_total_mode_adds_the_total_power) {
  // Total mode: the line gains the seeds' largest total power.  Box (5, 4) puts
  // the cap at 9, above the seed box, so the figure is the box corner 7 (= 5 + 4
  // - 2).  A widening grows the box by one layer and carries the cap with it, so
  // every figure of the leading triple is one past the probe's start -- which is
  // how a log shows that the probe had to enlarge the box.
  SeedBound widened    = SeedBound::initial(BoundaryMode::Total, 5, 4);
  SeedBound probeStart = widened;
  widened.widen_depth();                     // box (6, 4), cap 10
  REQUIRE(widened.limit == 10u);
  REQUIRE(format_seed_box_line(0b111, widened, probeStart) ==
          " depth: 5 rank: 3 total: 8  probe start: depth: 4 rank: 3 total: 7");
}

TEST(seed_box_line_reports_the_extents_of_the_admitted_seeds) {
  // The figures are the extents of the seeds once all three admission rules have
  // been applied, so they have to agree with admits() itself.  Enumerate the
  // integrals of a sector with three lines and one non-line propagator (every
  // integral therefore has depth >= 3, the base layer) and compare the maxima over
  // the admitted ones with the printed line, for a cap above the box and for one
  // that bites into it.
  const unsigned id = 0b0111;
  using Extents = std::array<int, 3>;  // depth, rank, total
  auto extents = [](const SeedBound &b) {
    Extents e = {-1, -1, -1};
    for (int a = 1; a <= 6; ++a)
      for (int c = 1; c <= 6; ++c)
        for (int e1 = 1; e1 <= 6; ++e1)
          for (int f = 0; f >= -6; --f) {
            RawIntegral ri = RI({a, c, e1, f});
            if (!b.admits(ri))
              continue;
            e[0] = std::max(e[0], (int)ri.depth());
            e[1] = std::max(e[1], (int)ri.rank());
            e[2] = std::max(e[2], (int)ri.depth() + (int)ri.rank());
          }
    return e;
  };
  SeedBound top = SeedBound::initial(BoundaryMode::Total, 5, 4);  // limit 9
  SeedBound cut = top;
  cut.limit = 6;  // bites: no admitted seed may reach total 6
  REQUIRE((extents(top) == Extents{4, 3, 7}));
  REQUIRE((extents(cut) == Extents{4, 2, 5}));
  REQUIRE(format_seed_box_line(id, top, cut) ==
          " depth: 4 rank: 3 total: 7  probe start: depth: 4 rank: 2 total: 5");
}

TEST(seed_box_line_reads_off_the_seed_box_one_layer_inside_the_boundary) {
  // The worked case the field is defined by: a two-line sector (id 0b0011) with a
  // boundary of depth 5 and rank 2, i.e. the cap at 7, admits the seeds up to
  // depth 4, rank 1, total power 5 -- one step inside the boundary in each hard
  // figure, and one below the box corner (depth 4, rank 1) in total power.  The
  // line therefore reads the seeds, not the boundary it came from.
  SeedBound b = SeedBound::initial(BoundaryMode::Total, 5, 2);
  REQUIRE(b.limit == 7u);
  REQUIRE(format_seed_box_line(0b0011, b, b) ==
          " depth: 4 rank: 1 total: 5  probe start: depth: 4 rank: 1 total: 5");
  // The depth figure is absolute -- the target [3, 1, -1, 0] of this sector has
  // depth 4, and depth 4 is what the line reports, not its 2 above the base layer.
  REQUIRE(RI({3, 1, -1, 0}).depth() == 4u);
}

TEST(seed_box_line_box_mode_reports_no_total) {
  // Box mode must not grow the line: its log is unchanged by the new option.
  SeedBound b = SeedBound::initial(BoundaryMode::Box, 5, 4);
  REQUIRE(format_seed_box_line(0b111, b, b).find("total") == std::string::npos);
}

// ---------------------------------------------------------------------------
// Stage S: applying one pass's report to the seed box (SeedBound::widen_for).
// run_reduce_ff_n hands each pass's refusal bits to this single entry point, so
// the widening policy lives here rather than inline in the driver's loops.
// ---------------------------------------------------------------------------

TEST(widen_for_box_mode_ignores_the_cap_bit) {
  // Box mode has no cap for bit 4 to carry and never calls widen_rank (refusal()
  // returns bit 1 alone there), so a report of 1 and one of 5 must land the same.
  SeedBound a = SeedBound::initial(BoundaryMode::Box, 5, 4);
  SeedBound b = a;
  a.widen_for(1);
  b.widen_for(5);  // bits 1|4
  REQUIRE(a.depth == 6u && a.rank == 4u && a.limit == 0u);
  REQUIRE(b.depth == 6u && b.rank == 4u && b.limit == 0u);
  REQUIRE(a.seed_top() == 5);
  REQUIRE(b.seed_top() == 5);
  b.widen_for(2);
  REQUIRE(b.depth == 6u && b.rank == 5u);  // bit 2 widens the rank, nothing else
}

TEST(widen_for_total_mode_carries_the_cap_for_a_box_top_report) {
  // bit 4 set: the report came from past_top(), so the depth widening moves the box
  // and the cap rides at the top of the new one -- seed_top() rises exactly one layer.
  SeedBound b = SeedBound::initial(BoundaryMode::Total, 6, 5);  // limit 11, top 9
  REQUIRE(b.limit == 11u);
  REQUIRE(b.seed_top() == 9);
  b.widen_for(5);  // bits 1|4
  REQUIRE(b.depth == 7u && b.rank == 5u);
  REQUIRE(b.limit == 12u);
  REQUIRE(b.seed_top() == 10);
}

TEST(widen_for_total_mode_depth_cap_alone_leaves_the_cap_where_it_is) {
  // bit 4 clear: the depth hard cap reported on its own, so nothing about the total
  // power was wrong -- the box deepens and keeps its ceiling.  seed_top() still rises
  // one layer, because the biggest admitted corner moved.
  SeedBound b = SeedBound::initial(BoundaryMode::Total, 6, 5);  // limit 11, top 9
  b.widen_for(1);
  REQUIRE(b.depth == 7u && b.rank == 5u);
  REQUIRE(b.limit == 11u);
  REQUIRE(b.seed_top() == 10);
}

TEST(widen_for_rank_bit_always_carries_the_cap_and_both_bits_apply) {
  SeedBound r = SeedBound::initial(BoundaryMode::Total, 6, 5);
  r.widen_for(2);
  REQUIRE(r.depth == 6u && r.rank == 6u);
  REQUIRE(r.limit == 12u);  // widen_rank's max(11, 6 + 6)
  REQUIRE(r.seed_top() == 10);

  SeedBound both = SeedBound::initial(BoundaryMode::Total, 6, 5);
  both.widen_for(3);  // bits 1|2, no bit 4
  REQUIRE(both.depth == 7u && both.rank == 6u);
  REQUIRE(both.limit == 13u);  // the rank widening's max(11, 7 + 6) wins
  REQUIRE(both.seed_top() == 11);
}

TEST(widen_for_a_lone_repair_hint_widens_nothing) {
  // Bit 4 is the caller's cap-policy hint, never a report on its own (a seed that
  // reports it also pins a hard cap), so widen_for(4) must be a no-op.
  SeedBound b = SeedBound::initial(BoundaryMode::Total, 6, 5);
  b.widen_for(4);
  REQUIRE(b.depth == 6u && b.rank == 5u && b.limit == 11u);
  REQUIRE(b.seed_top() == 9);
}

TEST(widen_for_applied_twice_matches_two_widenings) {
  // The driver applies the reports of consecutive rounds in sequence; a sector that
  // widens twice runs with a box two layers deeper and its top two layers higher.
  SeedBound b = SeedBound::initial(BoundaryMode::Total, 6, 5);
  b.widen_for(5);
  b.widen_for(5);
  REQUIRE(b.depth == 8u && b.rank == 5u);
  REQUIRE(b.limit == 13u);  // max(12, 8 + 5)
  REQUIRE(b.seed_top() == 11);
}

// ---------------------------------------------------------------------------
// Stage S: what the driver does after a round (round_action).  Round indices are
// 0-based, and the rule mirrors the probe loop it replaces: the cap is checked
// before a widening, so the last round reports and is dropped without widening.
// ---------------------------------------------------------------------------

TEST(round_action_settles_on_a_clean_pass_at_every_round) {
  for (unsigned round = 0; round < kMaxRounds; ++round)
    REQUIRE(round_action(round, 0) == RoundAction::Settle);
}

TEST(round_action_widens_while_rounds_remain_and_gives_up_at_the_cap) {
  const unsigned reports[] = {1u, 2u, 3u, 5u};
  for (unsigned round = 0; round + 1 < kMaxRounds; ++round)
    for (unsigned bits : reports)
      REQUIRE(round_action(round, bits) == RoundAction::Widen);
  for (unsigned bits : reports)
    REQUIRE(round_action(kMaxRounds - 1, bits) == RoundAction::GiveUp);
}

TEST(round_action_treats_a_lone_repair_hint_as_a_report) {
  // Bit 4 alone cannot arrive, but it is not a clean pass either: the round must
  // not be taken for one that settled.
  REQUIRE(round_action(0, 4) == RoundAction::Widen);
  REQUIRE(round_action(kMaxRounds - 1, 4) == RoundAction::GiveUp);
}
