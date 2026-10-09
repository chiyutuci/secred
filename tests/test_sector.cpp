#include "test_runner.h"
#include "sector.h"

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
    s._depth  = depth;
    s._rank   = rank;
    s._trivialsectors = trivials;
    return s;
  }

  // Proxy for the private static comparator.
  static bool compare_subintegral(const RawIntegral &a, const RawIntegral &b) {
    return Sector::_compare_subintegral(a, b);
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

  static unsigned depth(const Sector &s) { return s._depth; }
  static unsigned rank(const Sector &s)  { return s._rank; }
};

// ---------------------------------------------------------------------------
// SectorConfig — default values
// ---------------------------------------------------------------------------

TEST(sectorconfig_defaults) {
  SectorConfig cfg;
  REQUIRE(cfg.cache_dir      == "cache");
  REQUIRE(cfg.subsector_depth_offset == 1u);
  REQUIRE(cfg.ordering_mode == OrderingMode::Balance);
}

TEST(sectorconfig_subsector_depth_offset_configurable) {
  SectorConfig cfg;
  cfg.subsector_depth_offset = 2;
  REQUIRE(cfg.subsector_depth_offset == 2u);
}

TEST(sectorconfig_custom_values) {
  SectorConfig cfg;
  cfg.cache_dir      = "/tmp/myrun";
  cfg.ordering_mode  = OrderingMode::Normal;
  REQUIRE(cfg.cache_dir     == "/tmp/myrun");
  REQUIRE(cfg.ordering_mode == OrderingMode::Normal);
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
