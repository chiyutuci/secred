#include "test_runner.h"
#include "sector.h"
#include "sector_engine.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <map>
#include <set>
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
    s._bound  = SeedBound::initial(depth, rank);
    s._trivialsectors = trivials;
    return s;
  }

  // Proxy for the private static comparator.
  static bool compare_subintegral(const RawIntegral &a, const RawIntegral &b) {
    return Sector::_compare_subintegral(a, b);
  }

  // Proxy for the private static comparator.
  static bool compare_balance(const RawIntegral &a, const RawIntegral &b) {
    return Sector::_compare_balance(a, b);
  }

  // Proxy for the private subsector column ordering.
  static std::vector<RawIntegral> order_subintegrals(
      const std::vector<RawIntegral> &in, OrderingMode mode,
      const std::set<RawIntegral> *preferred = nullptr) {
    std::vector<RawIntegral> v = in;
    Sector::_order_subintegrals(v, mode, preferred);
    return v;
  }

  static void generate_combinations(Sector &s, int n, int sum) {
    s._generate_combinations(n, sum);
  }

  static void generate_seeds(Sector &s,
                              OrderingMode mode = OrderingMode::Balance,
                              const std::set<RawIntegral> *preferred = nullptr) {
    s._generate_seeds(mode, preferred);
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
  // The probe is scoped to the input targets unless settings.probe_mode says
  // otherwise.
  REQUIRE(cfg.probe_mode    == ProbeMode::Targets);
  // Kira's mandatory list is on unless settings.mandatory_list says otherwise:
  // it reproduces Kira's basis (box 8 masters, F234_N 309).
  REQUIRE(cfg.mandatory_list == true);
}

TEST(sectorconfig_mandatory_list_configurable) {
  SectorConfig cfg;
  cfg.mandatory_list = false;
  REQUIRE(cfg.mandatory_list == false);
  REQUIRE(SectorConfig{}.mandatory_list == true);
}

TEST(sectorconfig_probe_mode_configurable) {
  SectorConfig cfg;
  cfg.probe_mode = ProbeMode::All;
  REQUIRE(cfg.probe_mode == ProbeMode::All);
  REQUIRE(SectorConfig{}.probe_mode == ProbeMode::Targets);
}

TEST(sectorconfig_subsector_filter_n_configurable) {
  SectorConfig cfg;
  cfg.subsector_filter_n = 2;
  REQUIRE(cfg.subsector_filter_n == 2u);
}

TEST(sectorconfig_custom_values) {
  SectorConfig cfg;
  cfg.cache_dir      = "/tmp/myrun";
  cfg.ordering_mode  = OrderingMode::Normal;
  REQUIRE(cfg.cache_dir     == "/tmp/myrun");
  REQUIRE(cfg.ordering_mode == OrderingMode::Normal);
}

TEST(sectorconfig_ordering_mode_configurable) {
  SectorConfig cfg;
  cfg.ordering_mode = OrderingMode::Normal;
  REQUIRE(cfg.ordering_mode == OrderingMode::Normal);
  // Setting the key is what changes the behaviour: the default stays Balance.
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
// GlobalEquation and ReductionContext — the global collection surface
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
  REQUIRE(ctx.spillIntegrals.empty());
  REQUIRE(ctx.spillCoeffs.empty());
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
  REQUIRE(ctx.memoryReserve == 30ULL * 1024 * 1024 * 1024);
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
// ReductionContext raw-equation spill — the binary frame and its tables
// ---------------------------------------------------------------------------

static void open_raw_spill(ReductionContext &ctx, const std::string &path) {
  ctx.rawEquationsPath = path;
  ctx.openSpill();
}

// The whole file, so a test can read the framing back.
static std::string slurp(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

// "SRSP" + version 1, little-endian.
static bool has_spill_header(const std::string &bytes) {
  return bytes.size() >= 8 && bytes.compare(0, 4, "SRSP") == 0 &&
         bytes[4] == '\x01' && bytes[5] == '\0' && bytes[6] == '\0' &&
         bytes[7] == '\0';
}

TEST(append_raw_equation_spills_to_file) {
  ReductionContext ctx;
  const std::string path = "/tmp/secred_test_raw_eqs.bin";
  open_raw_spill(ctx, path);

  GlobalEquation eq;
  eq.emplace_back(RI({1, 1}), "2");
  eq.emplace_back(RI({1, 0}), "-1");
  ctx.appendRawEquation(eq);

  std::vector<RawIntegral> integrals;
  ctx.finishSpill(integrals);  // closes the file
  REQUIRE(ctx.globalEquationsCount.load() == 1u);
  // The tables keep every distinct integral and coefficient once, in first-seen
  // order -- the reader rebuilds the system from the term stream through them.
  REQUIRE(integrals.size() == 2u);
  REQUIRE(integrals[0] == RI({1, 1}));
  REQUIRE(integrals[1] == RI({1, 0}));
  REQUIRE(ctx.spillCoeffs.size() == 2u);
  REQUIRE(ctx.spillCoeffs[0] == "2");
  REQUIRE(ctx.spillCoeffs[1] == "-1");

  const std::string bytes = slurp(path);
  REQUIRE(has_spill_header(bytes));
  // header(8) + one equation: n_terms = 2, then two (integral, coefficient) pairs.
  REQUIRE(bytes.size() > 8u + 4u + 16u);
  REQUIRE(bytes[8] == '\x02');
  REQUIRE(bytes[9] == '\0');
  REQUIRE(bytes[12] == '\0');  // first term's integral index
  REQUIRE(bytes[16] == '\0');  // first term's coefficient index
  REQUIRE(bytes[20] == '\x01');  // second term's integral index
  REQUIRE(bytes[24] == '\x01');  // second term's coefficient index
  // Both tables and the footer follow the term stream.
  REQUIRE(bytes.compare(bytes.size() - 4, 4, "SRSP") == 0);
  std::filesystem::remove(path);
}

// The tables intern: a coefficient a system spells out a million times is stored once.
TEST(append_raw_equation_interns_repeated_terms) {
  ReductionContext ctx;
  const std::string path = "/tmp/secred_test_raw_eqs_intern.bin";
  open_raw_spill(ctx, path);

  GlobalEquation a;
  a.emplace_back(RI({1, 1}), "-2");
  a.emplace_back(RI({0, 1}), "3/2");
  GlobalEquation b;
  b.emplace_back(RI({0, 1}), "3/2");
  b.emplace_back(RI({1, 1}), "-2");
  ctx.appendRawEquation(a);
  ctx.appendRawEquation(b);

  std::vector<RawIntegral> integrals;
  ctx.finishSpill(integrals);
  REQUIRE(ctx.globalEquationsCount.load() == 2u);
  REQUIRE(integrals.size() == 2u);
  REQUIRE(ctx.spillCoeffs.size() == 2u);
  std::filesystem::remove(path);
}

// One lock for a whole batch must write exactly what one lock per equation writes:
// the frames are concatenated in the same order and the tables are filled by the
// same first-seen rule.
TEST(append_raw_equations_batch_matches_single_appends) {
  std::vector<GlobalEquation> eqs(3);
  eqs[0].emplace_back(RI({1, 1}), "-2");
  eqs[0].emplace_back(RI({0, 1}), "3/2");
  eqs[1].emplace_back(RI({2, 1}), "1");
  eqs[2].emplace_back(RI({0, 1}), "3/2");
  eqs[2].emplace_back(RI({2, 1}), "-1/3");
  eqs[2].emplace_back(RI({1, 1}), "-2");

  const std::string p1 = "/tmp/secred_test_raw_eqs_batch.bin";
  const std::string p2 = "/tmp/secred_test_raw_eqs_single.bin";
  {
    ReductionContext ctx;
    open_raw_spill(ctx, p1);
    ctx.appendRawEquations(eqs);
    std::vector<RawIntegral> integrals;
    ctx.finishSpill(integrals);
    REQUIRE(ctx.globalEquationsCount.load() == 3u);
    REQUIRE(integrals.size() == 3u);
  }
  {
    ReductionContext ctx;
    open_raw_spill(ctx, p2);
    for (const GlobalEquation &eq : eqs)
      ctx.appendRawEquation(eq);
    std::vector<RawIntegral> integrals;
    ctx.finishSpill(integrals);
  }
  REQUIRE(slurp(p1) == slurp(p2));
  std::filesystem::remove(p1);
  std::filesystem::remove(p2);
}

TEST(append_raw_equation_thread_safe) {
  ReductionContext ctx;
  const std::string path = "/tmp/secred_test_raw_eqs_thread.bin";
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
  std::vector<RawIntegral> integrals;
  ctx.finishSpill(integrals);

  REQUIRE(ctx.globalEquationsCount.load() == N);
  REQUIRE(integrals.size() == 7u);  // the seven distinct integrals, once each
  REQUIRE(ctx.spillCoeffs.size() == 1u);
  std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// ReductionContext::resetRunStatistics — a retry starts from zero
// ---------------------------------------------------------------------------

TEST(reduction_context_reset_run_statistics_zeroes_the_per_run_counters) {
  // Family::run_reduce's seed-dots refinement reruns the sector phase; every
  // counter the report reads must then describe the new pass only.  The state
  // other owners reset on the same retry -- the spill and its equation count
  // (openSpill) and the sector set (prepare_sectors) -- must survive.
  ReductionContext ctx;
  ctx.narrowingSuccess.store(3);
  ctx.narrowingFail.store(2);
  ctx.truncatedSectors.store(7);
  ctx.truncationLifted.store(1);
  ctx.boxSpillRaised.store(9);
  ctx.boxSpillExtraDepth.store(4);
  ctx.boxSpillExtraRank.store(5);
  ctx.probeScopeSkipped.store(11);
  ctx.rawEquationsTotal.store(13);
  ctx.probeNanos.store(17);
  ctx.fullNanos.store(19);
  ctx.runningSectors.store(2);
  ctx.headroomTaken = true;
  ctx.availAtAdmit = 23;
  ctx.globalEquationsCount.store(29);
  ctx.totalSectors = 31;
  ctx.completedSectors = 3;

  ctx.resetRunStatistics();

  REQUIRE(ctx.narrowingSuccess.load() == 0u);
  REQUIRE(ctx.narrowingFail.load() == 0u);
  REQUIRE(ctx.truncatedSectors.load() == 0u);
  REQUIRE(ctx.truncationLifted.load() == 0u);
  REQUIRE(ctx.boxSpillRaised.load() == 0u);
  REQUIRE(ctx.boxSpillExtraDepth.load() == 0u);
  REQUIRE(ctx.boxSpillExtraRank.load() == 0u);
  REQUIRE(ctx.probeScopeSkipped.load() == 0u);
  REQUIRE(ctx.rawEquationsTotal.load() == 0u);
  REQUIRE(ctx.probeNanos.load() == 0u);
  REQUIRE(ctx.fullNanos.load() == 0u);
  REQUIRE(ctx.runningSectors.load() == 0u);
  REQUIRE(!ctx.headroomTaken);
  REQUIRE(ctx.availAtAdmit == 0u);
  // Another owner's to reset, so this one leaves them alone.
  REQUIRE(ctx.globalEquationsCount.load() == 29u);
  REQUIRE(ctx.totalSectors == 31u);
  REQUIRE(ctx.completedSectors == 3u);
}

// ---------------------------------------------------------------------------
// OrderingMode enum and SectorGeometry struct
// ---------------------------------------------------------------------------

TEST(ordering_mode_enum_values) {
  REQUIRE(OrderingMode::Normal != OrderingMode::Balance);
}

// ---------------------------------------------------------------------------
// RawIntegral::maxpos / maxneg — the two fine keys of OrderingMode::Balance
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
// _compare_balance — key priority
// ---------------------------------------------------------------------------

TEST(compare_balance_total_power_is_first_key) {
  // [1,-3]: depth 1, rank 3 → total 4.  [2,0]: depth 2, rank 0 → total 2.
  // The total power decides before depth, so the shallower-but-longer integral
  // is NOT the smaller one here.
  REQUIRE(SectorTestHelper::compare_balance(RI({2, 0}), RI({1, -3})));
  REQUIRE(!SectorTestHelper::compare_balance(RI({1, -3}), RI({2, 0})));
}

TEST(compare_balance_depth_breaks_total_power_ties) {
  // [3,0] and [1,-2] both have total power 3; the smaller depth wins.
  REQUIRE(SectorTestHelper::compare_balance(RI({1, -2}), RI({3, 0})));
  REQUIRE(!SectorTestHelper::compare_balance(RI({3, 0}), RI({1, -2})));
}

TEST(compare_balance_rank_never_decides) {
  // rank = total - depth, so two integrals that tie on the first two keys
  // necessarily tie on rank as well: the third key can never break a tie.  This
  // pair ties on all three (total 3, depth 2, rank 1) and is decided by the
  // max-denominator key instead.
  RawIntegral a = RI({2, -1, 0});
  RawIntegral b = RI({1, 1, -1});
  REQUIRE(a.depth() + a.rank() == b.depth() + b.rank());
  REQUIRE(a.depth() == b.depth());
  REQUIRE(a.rank() == b.rank());
  REQUIRE(SectorTestHelper::compare_balance(b, a));
}

TEST(compare_balance_max_denominator_is_fourth_key) {
  // [2,-1,0] and [1,1,-1] both have depth 2, rank 1, total 3; the one whose
  // largest denominator power is smaller comes first.
  REQUIRE(SectorTestHelper::compare_balance(RI({1, 1, -1}), RI({2, -1, 0})));
  REQUIRE(!SectorTestHelper::compare_balance(RI({2, -1, 0}), RI({1, 1, -1})));
}

TEST(compare_balance_max_numerator_is_last_key) {
  // [2,-3,0,0] and [2,-1,-1,-1]: depth 2, rank 3, total 5, max denominator 2 —
  // they differ only in the largest numerator power (3 vs 1).
  REQUIRE(SectorTestHelper::compare_balance(RI({2, -1, -1, -1}),
                                               RI({2, -3, 0, 0})));
  REQUIRE(!SectorTestHelper::compare_balance(RI({2, -3, 0, 0}),
                                                RI({2, -1, -1, -1})));
}

TEST(compare_balance_denominator_key_outranks_numerator_key) {
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
  REQUIRE(SectorTestHelper::compare_balance(b, a));
  REQUIRE(!SectorTestHelper::compare_balance(a, b));
}

TEST(compare_balance_is_strict_weak_ordering_on_ties) {
  // Two distinct integrals with identical keys: neither is less than the other.
  RawIntegral a = RI({1, 1, -1, 0});
  RawIntegral b = RI({1, 1, 0, -1});
  REQUIRE(!SectorTestHelper::compare_balance(a, b));
  REQUIRE(!SectorTestHelper::compare_balance(b, a));
  REQUIRE(!SectorTestHelper::compare_balance(a, a));
}

// ---------------------------------------------------------------------------
// _order_subintegrals — the subsector column order
// ---------------------------------------------------------------------------

TEST(order_subintegrals_balance_puts_largest_key_in_smallest_column) {
  // Column nvars+j: index 0 gets the smallest subsector column, so the vector
  // must come back in DESCENDING key order.
  std::vector<RawIntegral> in = {RI({2, -1, 0}), RI({1, 0, 0}), RI({1, 1, -1})};
  auto out = SectorTestHelper::order_subintegrals(in, OrderingMode::Balance);
  REQUIRE(out.size() == 3u);
  REQUIRE(out[0] == RI({2, -1, 0}));   // total 3, depth 2, max pos 2
  REQUIRE(out[1] == RI({1, 1, -1}));   // total 3, depth 2, max pos 1
  REQUIRE(out[2] == RI({1, 0, 0}));    // total 1
}

TEST(order_subintegrals_normal_keeps_canonical_order) {
  // Normal orders subsector integrals by the canonical per-index comparator —
  // the balance key must not leak into it.
  std::vector<RawIntegral> in = {RI({2, -1, 0}), RI({1, 0, 0}), RI({1, 1, -1})};
  auto bal = SectorTestHelper::order_subintegrals(in, OrderingMode::Balance);
  auto nor = SectorTestHelper::order_subintegrals(in, OrderingMode::Normal);
  // canonical: depth ASC, rank ASC, then sector ASC → no reverse of that sort
  // is applied for Normal, so the vector comes back most-complex-first as
  // [1,1,-1], [2,-1,0], [1,0,0].
  REQUIRE(nor[0] == RI({1, 1, -1}));
  REQUIRE(nor[1] == RI({2, -1, 0}));
  REQUIRE(nor[2] == RI({1, 0, 0}));
  // ... and the balance order differs from it on this very input.
  REQUIRE(bal != nor);
}

TEST(order_subintegrals_balance_ties_mirror_input_order) {
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
    REQUIRE(!SectorTestHelper::compare_balance(x, in[0]));
    REQUIRE(!SectorTestHelper::compare_balance(in[0], x));
  }
  auto out = SectorTestHelper::order_subintegrals(in, OrderingMode::Balance);
  REQUIRE(out.size() == in.size());
  for (unsigned i = 0; i < in.size(); ++i)
    REQUIRE(out[i] == in[in.size() - 1 - i]);
}

TEST(order_subintegrals_preferred_go_last) {
  // The simplest column is the LAST index of this vector (the caller reads index
  // i as the column at nvars + i, so index 0 is the smallest column = the largest
  // id = the most complex integral).  A preferred integral is the simplest one
  // and must therefore come back at the end -- the reverse of the seeds' rule.
  std::vector<RawIntegral> in = {RI({2, -1, 0}), RI({1, 0, 0}), RI({1, 1, -1})};
  const std::set<RawIntegral> preferred{RI({2, -1, 0})};
  for (OrderingMode mode : {OrderingMode::Normal, OrderingMode::Balance}) {
    auto plain = SectorTestHelper::order_subintegrals(in, mode);
    auto got = SectorTestHelper::order_subintegrals(in, mode, &preferred);
    REQUIRE(got.size() == 3u);
    REQUIRE(got[2] == RI({2, -1, 0}));
    // The exact statement, mode-independent: the plain order with the preferred
    // integral moved to the end.  (Its plain position differs by mode -- in
    // Normal it is already at index 1 -- so the tail is not a shift of the whole
    // vector.)
    std::vector<RawIntegral> want;
    for (const RawIntegral &x : plain)
      if (x != RI({2, -1, 0}))
        want.push_back(x);
    want.push_back(RI({2, -1, 0}));
    for (unsigned i = 0; i < want.size(); ++i)
      REQUIRE(got[i] == want[i]);
  }
}

TEST(order_subintegrals_foreign_preferred_entries_are_ignored) {
  // Entries that belong to another sector are not in this vector at all, so the
  // partition must be a no-op -- and so must nullptr and an empty set.
  std::vector<RawIntegral> in = {RI({2, -1, 0}), RI({1, 0, 0}), RI({1, 1, -1})};
  const std::set<RawIntegral> foreign{RI({9, 9, 9}), RI({0, 0, 0})};
  const std::set<RawIntegral> empty;
  for (OrderingMode mode : {OrderingMode::Normal, OrderingMode::Balance}) {
    auto plain = SectorTestHelper::order_subintegrals(in, mode);
    // An explicit array: an initializer_list cannot deduce an element type from
    // a pointer and a nullptr_t at once.
    const std::set<RawIntegral> *cases[] = {&foreign, &empty, nullptr};
    for (const std::set<RawIntegral> *p : cases) {
      auto got = SectorTestHelper::order_subintegrals(in, mode, p);
      REQUIRE(got.size() == plain.size());
      for (unsigned i = 0; i < plain.size(); ++i)
        REQUIRE(got[i] == plain[i]);
    }
  }
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

// Balance mode: seeds ordered by (depth+rank ASC, depth ASC, rank ASC,
// max_denominator ASC, max_numerator ASC).
TEST(seeds_balance_mode_total_power_first) {
  // One active line, one zero line, box depth 3 rank 2 → 9 seeds whose keys
  // differ enough to separate this mode from Normal.
  std::vector<bool> lines = {true, false};
  Sector s = SectorTestHelper::make_sector(1, 2, lines, 3, 2);
  SectorTestHelper::generate_seeds(s, OrderingMode::Balance);
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

TEST(seeds_preferred_take_the_smallest_weights) {
  // The one-line 3/2 box: nine seeds, simplest first in both modes, and both
  // modes end on [3,-2].  Marking that last seed preferred must move it to
  // weight 0 and shift nothing else's relative order.
  std::vector<bool> lines = {true, false};
  const std::set<RawIntegral> preferred{RI({3, -2})};
  for (OrderingMode mode : {OrderingMode::Normal, OrderingMode::Balance}) {
    Sector sp = SectorTestHelper::make_sector(1, 2, lines, 3, 2);
    SectorTestHelper::generate_seeds(sp, mode);
    const auto &plain = SectorTestHelper::seeds(sp);
    REQUIRE(plain.size() == 9u);
    REQUIRE(plain.back() == RI({3, -2}));

    Sector s = SectorTestHelper::make_sector(1, 2, lines, 3, 2);
    SectorTestHelper::generate_seeds(s, mode, &preferred);
    const auto &seeds = SectorTestHelper::seeds(s);
    const auto &w = SectorTestHelper::weights(s);
    REQUIRE(seeds.size() == plain.size());
    REQUIRE(seeds[0] == RI({3, -2}));
    REQUIRE(w.at(RI({3, -2})) == 0u);
    // Every other seed keeps its relative order, one rank up.
    for (unsigned i = 1; i < seeds.size(); ++i) {
      REQUIRE(seeds[i] == plain[i - 1]);
      REQUIRE(w.at(seeds[i]) == i);
    }
  }
}

TEST(seeds_two_preferred_take_weights_zero_and_one) {
  // Inside the preferred group the mode's order still decides, not the order the
  // set happens to be built in.
  std::vector<bool> lines = {true, false};
  Sector sp = SectorTestHelper::make_sector(1, 2, lines, 3, 2);
  SectorTestHelper::generate_seeds(sp, OrderingMode::Balance);
  const auto &plain = SectorTestHelper::seeds(sp);
  // Balance order: [1,0] [1,-1] [2,0] [1,-2] [2,-1] [3,0] [2,-2] [3,-1] [3,-2],
  // so [3,0] (total 3) precedes [2,-2] (total 4) -- and the two preferred seeds
  // must come back in that order, ahead of everything else.
  REQUIRE(plain[5] == RI({3, 0}));
  REQUIRE(plain[6] == RI({2, -2}));

  const std::set<RawIntegral> preferred{RI({2, -2}), RI({3, 0})};
  Sector s = SectorTestHelper::make_sector(1, 2, lines, 3, 2);
  SectorTestHelper::generate_seeds(s, OrderingMode::Balance, &preferred);
  const auto &seeds = SectorTestHelper::seeds(s);
  const auto &w = SectorTestHelper::weights(s);
  REQUIRE(seeds.size() == 9u);
  REQUIRE(seeds[0] == RI({3, 0}));
  REQUIRE(seeds[1] == RI({2, -2}));
  REQUIRE(w.at(RI({3, 0})) == 0u);
  REQUIRE(w.at(RI({2, -2})) == 1u);
  REQUIRE(seeds[2] == RI({1, 0}));
}

TEST(seeds_foreign_preferred_entries_are_ignored) {
  // A one-line sector never holds an integral with a positive second index, so
  // [0,1] and [9,9] are legal-length entries of another sector: the weights must
  // be exactly the plain ones.
  std::vector<bool> lines = {true, false};
  const std::set<RawIntegral> foreign{RI({9, 9}), RI({0, 1})};
  for (OrderingMode mode : {OrderingMode::Normal, OrderingMode::Balance}) {
    Sector sp = SectorTestHelper::make_sector(1, 2, lines, 3, 2);
    SectorTestHelper::generate_seeds(sp, mode);
    Sector sg = SectorTestHelper::make_sector(1, 2, lines, 3, 2);
    SectorTestHelper::generate_seeds(sg, mode, &foreign);
    const auto &plain = SectorTestHelper::seeds(sp);
    const auto &got = SectorTestHelper::seeds(sg);
    const auto &w = SectorTestHelper::weights(sg);
    REQUIRE(got.size() == plain.size());
    for (unsigned i = 0; i < plain.size(); ++i) {
      REQUIRE(got[i] == plain[i]);
      REQUIRE(w.at(got[i]) == i);
    }
  }
}

TEST(seeds_balance_mode_differs_from_normal) {
  // On this one-line sector the generation order is already (excess depth,
  // rank)-major, so Normal's [1,-2] (total 3) sits ahead of [2,0] (total 2);
  // Balance interleaves the two axes by their sum and swaps the two.
  std::vector<bool> lines = {true, false};
  Sector sn = SectorTestHelper::make_sector(1, 2, lines, 3, 2);
  Sector sb = SectorTestHelper::make_sector(1, 2, lines, 3, 2);
  SectorTestHelper::generate_seeds(sn, OrderingMode::Normal);
  SectorTestHelper::generate_seeds(sb, OrderingMode::Balance);
  const auto &n = SectorTestHelper::seeds(sn);
  const auto &b = SectorTestHelper::seeds(sb);
  REQUIRE(n[2] == RI({1, -2}));
  REQUIRE(b[2] == RI({2, 0}));
  REQUIRE(b != n);
}

TEST(seeds_balance_mode_keeps_the_simplest_seed_first) {
  // A two-line sector: the simplest seed is the base integral [1,1] (total 2),
  // which the balance key puts first.
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

TEST(initial_seed_bounds_uptarget_takes_the_layer_above_it) {
  // An uptarget asks for the layer above itself exactly like a target: what
  // differs between the two sets is where the request came from -- the input, or
  // a larger sector's spill -- not what the box has to cover.  A base integral
  // (all its lines at power 1, depth = popcount(id) = 2 here) therefore lifts the
  // box to 3, one layer above itself.
  std::set<RawIntegral> uptargets{RI({1, 0, 1})};  // depth 2, rank 0
  auto [depth, rank] = initial_seed_bounds(0b101, {}, uptargets);
  REQUIRE(depth == 3u);
  REQUIRE(rank == 1u);
}

TEST(initial_seed_bounds_uptarget_depth_lifts_box) {
  // A deeper uptarget lifts the box one layer above its own depth, the same +1 a
  // direct target gets.
  std::set<RawIntegral> uptargets{RI({3, 0, 1})};  // depth 4, rank 0
  auto [depth, rank] = initial_seed_bounds(0b101, {}, uptargets);
  REQUIRE(depth == 5u);
  REQUIRE(rank == 1u);
}

TEST(initial_seed_bounds_target_gets_one_layer_more) {
  // A direct target needs the layer above it, so its depth contributes +1 -- the
  // same rule the uptarget test above pins for the other set.
  std::set<RawIntegral> targets{RI({2, 0, 0})};  // depth 2, rank 0
  auto [depth, rank] = initial_seed_bounds(0b101, targets, {});
  REQUIRE(depth == 3u);
  REQUIRE(rank == 1u);
}

TEST(initial_seed_bounds_rank_from_negative_indices) {
  // Rank is read off the negative (numerator) powers: the widest request wins,
  // and every one of them -- from either set -- contributes +1.
  std::set<RawIntegral> targets{RI({1, 0, 1}), RI({-2, 0, 1})};
  std::set<RawIntegral> uptargets{RI({1, 0, -3})};
  auto [depth, rank] = initial_seed_bounds(0b101, targets, uptargets);
  REQUIRE(depth == 3u);  // max(base 2, target [1,0,1]: 2 + 1, uptarget [1,0,-3]: 1 + 1)
  REQUIRE(rank == 4u);   // max(1, target [-2,0,1]: 2 + 1, uptarget [1,0,-3]: 3 + 1)
}

TEST(initial_seed_bounds_never_below_base_layer) {
  // Every request belongs to this sector, so it has at least popcount(id)
  // positive units: the box can never fall below the base layer, which is what
  // keeps the unsigned excess-depth arithmetic in _generate_seeds sound.  A
  // request shallower than the base layer (a spill of this sector need not reach
  // every one of its lines) leaves the base layer standing.
  const unsigned id = 0b1011;
  std::set<RawIntegral> uptargets{RI({1, 0, 0, 0})};  // depth 1, under the base 3
  auto [depth, rank] = initial_seed_bounds(id, {}, uptargets);
  REQUIRE(depth >= std::popcount(id));
  REQUIRE(depth == 3u);
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

TEST(initial_seed_bounds_excess_depth_is_one_for_a_base_layer_request) {
  // With every request asking for the layer above itself, the excess depth
  // (depth - popcount(id)) is 1 for a request sitting at the sector's own base
  // layer -- the smallest it can be, since a request of this sector carries at
  // least one unit per line.  Such a box admits none of the sector's integrals:
  // the log's seed depth is then the printed cap, popcount(id) - 1, and the round
  // loop widens once to admit the request (the format test below pins the line).
  const unsigned id = 0b10110;               // five propagators, lines 1, 2, 4
  std::set<RawIntegral> uptargets{RI({0, 1, 1, 0, 1})};  // the base integral
  auto [depth, rank] = initial_seed_bounds(id, {}, uptargets);
  REQUIRE(depth - std::popcount(id) == 1u);  // logged seed depth: popcount - 1
  REQUIRE(rank == 1u);
}

// ---------------------------------------------------------------------------
// SeedBound — the seed-boundary rule: a depth/rank box with a total-power cap
// ---------------------------------------------------------------------------

TEST(seedbound_initial_builds_the_box_and_the_cap) {
  // initial() takes the box from initial_seed_bounds and puts the cap one layer
  // below the top of the box's total-power range (limit == depth + rank - 2), so
  // the strict total comparison cuts exactly the box corner -- the seed class with
  // both hard caps at their maxima.  RI({+d, -r}) is depth d, rank r.
  SeedBound b = SeedBound::initial(5, 4);
  REQUIRE(b.depth == 5u);
  REQUIRE(b.rank  == 4u);
  REQUIRE(b.limit == 7u);            // = depth + rank - 2
  REQUIRE(b.boundary() == 7u);
  REQUIRE(b.quantity(RI({3, -2})) == 5u);   // depth + rank, not depth
  REQUIRE(b.seed_top() == 6);        // = limit - 1, one below the box corner
  REQUIRE(!b.admits(RI({4, -3})));   // total 7, the box corner: the cap cuts it
  REQUIRE(b.admits(RI({4, -2})));    // total 6, the last admitted layer
  REQUIRE(!b.admits(RI({5, -3})));   // depth cap
  REQUIRE(!b.admits(RI({3, -4})));   // rank cap
  REQUIRE(!b.exceeded(RI({5, 0})));  // at the depth cap, inside the level: not reported
  REQUIRE(!b.exceeded(RI({4, 0})));
  REQUIRE(b.exceeded(RI({4, -3})));  // the corner: reported, not admitted
  REQUIRE(b.exceeded(RI({4, -4})));  // total 8, past the box top
}

TEST(seedbound_admission_is_the_box_cut_by_the_cap) {
  // Over a grid: a seed is admitted iff both hard caps pass and its total power
  // stays under the cap, which starts one layer below the box corner -- so the
  // corner itself is always outside.
  SeedBound b = SeedBound::initial(4, 3);
  for (unsigned d = 0; d <= 6; ++d)
    for (unsigned r = 0; r <= 5; ++r)
      REQUIRE(b.admits(RI({(int)d, -(int)r})) == (d < 4u && r < 3u && d + r < 5u));
  // ... and the clause bites harder still once the cap is lowered into the box.
  SeedBound cut = b;
  cut.limit = 4;
  REQUIRE(!cut.admits(RI({3, -1})));   // total 4, at the cap
  REQUIRE(cut.admits(RI({2, -1})));    // total 3, under it
}

TEST(seedbound_filter_counts_from_the_seed_box_top) {
  // Box (6, 5) at its initial cap (9 = depth + rank - 2) admits seeds up to total
  // power 8 -- one layer below the box corner -- so the filter's reference is 8,
  // not the cap 9.  At n = 1 it keeps the top layer of what the seeds can generate
  // (total power >= 8); at n = 0 it keeps nothing at all, because no subsector
  // integral a seed generates reaches past that layer.  An integral below the top
  // layer is dropped even though its depth is large.
  SeedBound b = SeedBound::initial(6, 5);
  REQUIRE(b.boundary() == 9u);
  REQUIRE(b.seed_top() == 8);
  REQUIRE(!b.filter_drops(RI({5, -5}), 1));   // total 10, past the top layer
  REQUIRE(!b.filter_drops(RI({9, -2}), 1));   // total 11, past the boundary
  REQUIRE(!b.filter_drops(RI({4, -4}), 1));   // total 8, the top layer itself
  REQUIRE(b.filter_drops(RI({4, -3}), 1));    // total 7
  REQUIRE(b.filter_drops(RI({5, -2}), 1));    // total 7
  // n = 0: everything the seeds can generate is dropped -- at the box-top cap no
  // generated subsector integral reaches past the top layer, so nothing is kept.
  REQUIRE(b.filter_drops(RI({4, -4}), 0));    // total 8, exactly the seed top
  REQUIRE(b.filter_drops(RI({5, -2}), 0));    // total 7
  REQUIRE(b.filter_drops(RI({0, 0}), 0));     // the emptiest integral of all
  // Above the top layer the filter still keeps what it is given (it has no upper
  // end) -- nothing generates such an integral here, but the rule is one-sided.
  REQUIRE(!b.filter_drops(RI({5, -5}), 0));   // total 10
  REQUIRE(!b.filter_drops(RI({6, -5}), 0));   // total 11
  // The subtraction is signed: a reference below n must not wrap around into
  // "drop everything" -- with n past the reference nothing is far enough below it.
  SeedBound tiny = SeedBound::initial(1, 0);
  REQUIRE(tiny.boundary() == 0u);
  REQUIRE(!tiny.filter_drops(RI({0, 0}), 3));
  REQUIRE(!tiny.filter_drops(RI({5, 0}), 3));
}

TEST(seedbound_filter_reference_is_the_seed_box_not_the_cap) {
  // The filter counts layers from the admitted seeds, not from the cap: the box
  // initial() builds has its cap one layer below the corner, so raising the cap to
  // the corner itself raises the reference by one, and one layer from the box as
  // built reads like two layers from the raised one.  The offset counts from the
  // seeds, wherever the cap is.
  SeedBound boxed = SeedBound::initial(6, 5);  // cap 9, top 8
  REQUIRE(boxed.seed_top() == 8);
  SeedBound corner = boxed;
  corner.limit = 10;                       // the box corner itself: cuts nothing
  REQUIRE(corner.seed_top() == 9);
  for (int t = 0; t <= 13; ++t) {
    RawIntegral depthOnly = RI({t, 0});    // a depth-t, rank-0 integral: total t
    REQUIRE(boxed.filter_drops(depthOnly, 1) == corner.filter_drops(depthOnly, 2));
  }
  // A widening instead raises the reference by one layer -- it grows the box, and
  // the cap rides one layer below its top -- so n keeps its meaning as a distance
  // below the top: n = 1 keeps the top layer wherever the top now is.
  SeedBound widened = boxed;
  widened.widen_depth();                   // box (7, 5), cap 10
  REQUIRE(widened.seed_top() == boxed.seed_top() + 1);
  for (int t = 0; t <= 13; ++t)
    REQUIRE(widened.filter_drops(RI({t, 0}), 1) == (t < widened.seed_top()));
}

TEST(seedbound_probe_criterion_is_the_box_top) {
  // The probe's "outside" test reads the seed box, in total power.  Starting at
  // the cap initial() builds (limit == depth + rank - 2) the seeds one layer above
  // the box top (total 9 for a box (6, 5)) are the corner layer the cap cuts and
  // the two shells outside it: total 9 as a depth shell and a rank shell, and the
  // box corner at 10.  Comparing against the cap alone would report only what is
  // at or above it (total 9 upward) and miss nothing here, but the corner is the
  // case that matters -- it is admitted by both hard caps and cut by the total
  // power alone.  refusal() names the figure to grow for each of them: the cap
  // itself for the corner, the hard cap the seed crossed otherwise -- and the
  // widening admits it.
  SeedBound b = SeedBound::initial(6, 5);
  REQUIRE(b.seed_top() == 8);
  REQUIRE(!b.exceeded(RI({5, -3})));  // total 8, deepest seed the box admits
  REQUIRE(b.exceeded(RI({5, -4})));   // total 9, the box corner
  REQUIRE(b.refusal(RI({5, -4})) == 8u);  // no hard cap refuses it: the cap alone
  REQUIRE(b.exceeded(RI({5, -5})));   // depth 5, rank 5: the rank shell
  REQUIRE(b.refusal(RI({5, -5})) == 2u);
  REQUIRE(b.exceeded(RI({6, -3})));   // depth 6: the depth shell
  REQUIRE(b.refusal(RI({6, -3})) == 1u);
  REQUIRE(b.exceeded(RI({6, -5})));   // both caps at once
  REQUIRE(b.refusal(RI({6, -5})) == 3u);
  b.widen_depth();
  REQUIRE(b.admits(RI({6, -3})));     // the depth widening admits the depth shell
  REQUIRE(!b.admits(RI({5, -5})));    // and leaves the rank cap where it was
  b.widen_rank();
  REQUIRE(b.admits(RI({5, -5})));
  REQUIRE(!b.admits(RI({6, -5})));    // the new box's own corner stays outside
}

TEST(seedbound_widens_the_cap_alone_for_the_box_corner) {
  // The field case (F234's sector 255): the box (9, 5) has cap 12 and seed top 11,
  // and the request the reduction was left with -- [1,1,1,1,1,1,1,1,-4], depth 8,
  // rank 4 -- has total power 12: both hard caps clear, the total power one layer
  // past the top.  Every seed inside the two hard caps has total power at most
  // depth + rank - 2 = 12, so a report with both caps clear is always the box
  // corner and the cap is the only figure that can take it: one layer admits it and
  // leaves the box the whole box its hard caps allow, with both figures standing.
  SeedBound b = SeedBound::initial(9, 5);
  REQUIRE(b.limit == 12u);
  REQUIRE(b.seed_top() == 11);
  const RawIntegral corner = RI({1, 1, 1, 1, 1, 1, 1, 1, -4});
  REQUIRE(b.quantity(corner) == 12u);
  REQUIRE(!b.admits(corner));
  REQUIRE(b.exceeded(corner));
  REQUIRE(b.past_top(corner));
  REQUIRE(!b.past_depth_cap(corner));
  REQUIRE(b.refusal(corner) == 8u);   // the cap alone: no hard cap refuses it
  b.widen_cap();
  REQUIRE(b.depth == 9u);             // both hard caps stand
  REQUIRE(b.rank == 5u);
  REQUIRE(b.limit == 13u);            // one layer up: depth + rank - 1
  REQUIRE(b.seed_top() == 12);
  REQUIRE(b.admits(corner));
  REQUIRE(!b.exceeded(corner));
  // Nothing but the corner came in: the box is now full, so the next report out of
  // it pins a hard cap and widen_depth / widen_rank is the repair for that one.
  for (unsigned d = 0; d <= 10; ++d)
    for (unsigned r = 0; r <= 6; ++r)
      REQUIRE(b.admits(RI({(int)d, -(int)r})) == (d < 9u && r < 5u));
}

TEST(seedbound_widen_for_raises_the_cap_on_a_corner_report) {
  // The driver's path: probe_criterion turns the corner into refusal()'s bit 8 plus
  // bit 4 (the seed crossed the box top), and widen_for grows the cap one layer with
  // both figures standing -- the level goes from (8, 4, 11) to (8, 4, 12), not to
  // the (9, 4, 12) a depth widening would have printed.
  SeedBound b = SeedBound::initial(9, 5);
  b.widen_for(8u | 4u);
  REQUIRE(b.depth == 9u);
  REQUIRE(b.rank == 5u);
  REQUIRE(b.limit == 13u);
  REQUIRE(b.admits(RI({1, 1, 1, 1, 1, 1, 1, 1, -4})));
}

TEST(seedbound_widen_for_grows_the_cap_before_a_hard_cap_it_also_reported) {
  // A pass can report both at once -- a seed on a hard cap and the box corner --
  // and the cap goes first: the level takes the step its own figure asks for,
  // (8, 4, 11) to (8, 4, 12), while the hard figure the pass also reported waits
  // for the next pass.  Growing both in one pass would take the level to
  // (9, 4, 13), a hard figure grown for a report that never pinned it.
  SeedBound b = SeedBound::initial(9, 5);
  b.widen_for(1u | 4u | 8u);   // the depth-cap seed and the corner, one pass
  REQUIRE(b.depth == 9u);
  REQUIRE(b.rank == 5u);
  REQUIRE(b.limit == 13u);
  b.widen_for(1u | 4u);        // the pass that still finds the depth-cap seed
  REQUIRE(b.depth == 10u);     // now that figure grows, one layer
  REQUIRE(b.limit == 13u);     // the carrying deepening leaves the cap where it is
  REQUIRE(format_seed_box_line(b, SeedBound::initial(9, 5)) ==
          " depth: 9 rank: 4 total: 12  start: depth: 8 rank: 4 total: 11");
}

TEST(seedbound_keeps_the_caps_hard) {
  // Widening one hard bound never lets an integral out of the box through the
  // other: (3, 5) has total 8, passes the cap comfortably and is refused by the
  // rank cap, and a depth widening -- the repair a depth refusal asks for --
  // leaves that refusal standing.  (The probe reads this: the seed is inside the
  // box top, so it is not reported at all.)
  SeedBound b = SeedBound::initial(6, 5);
  REQUIRE(b.quantity(RI({3, -5})) == 8u);
  REQUIRE(!b.exceeded(RI({3, -5})));
  REQUIRE(!b.admits(RI({3, -5})));
  REQUIRE(!b.admits(RI({6, 0})));   // depth cap
  b.widen_depth();
  REQUIRE(!b.admits(RI({3, -5})));  // the rank cap still refuses it
  REQUIRE(b.admits(RI({6, 0})));    // while the widened depth admits this one
}

TEST(seedbound_widen_admits_one_more_layer) {
  // The widening raises the depth bound and carries the cap along one layer below
  // the new box top, so the admitted set grows by exactly the layer just outside
  // the old box.
  SeedBound b = SeedBound::initial(6, 5);  // cap 9: the corner layer is outside
  REQUIRE(!b.admits(RI({6, -4})));
  b.widen_depth();
  REQUIRE(b.depth == 7u);
  REQUIRE(b.rank  == 5u);
  REQUIRE(b.limit == 10u);          // = depth + rank - 2: one below the new corner
  REQUIRE(b.admits(RI({6, -3})));   // the layer the widening bought
  REQUIRE(!b.admits(RI({6, -4})));  // the new corner is outside, as it was before
  for (unsigned d = 0; d <= 8; ++d)
    for (unsigned r = 0; r <= 7; ++r)
      REQUIRE(b.admits(RI({(int)d, -(int)r})) == (d < 7u && r < 5u && d + r < 10u));
  b.widen_rank();
  REQUIRE(b.rank  == 6u);
  REQUIRE(b.limit == 11u);          // and the cap follows it up
}

TEST(seedbound_widens_the_box_a_master_sits_above) {
  // The failure the criterion exists for, on the case that reported it: box.yaml's
  // sector12 with subsector_filter_n: 0.  The master the reduction was left with is
  // [0, 0, 1, 2], depth 3, rank 0, total 3 -- and the sector's four propagators are
  // two lines, so popcount(0b1100) = 2.
  RawIntegral master = RI({0, 0, 1, 2});
  REQUIRE(master.depth() == 3u);
  REQUIRE(master.rank() == 0u);

  // A request asks for the layer above itself, so an uptarget at depth 3 gives the
  // box (4, 1), whose corner is the master (depth 3, rank 0) itself -- the layer the
  // total cap is built to cut, so the master starts out as a column without an
  // equation of its own.  The reduction normally expresses it in terms of the seeds
  // below it and the pass settles; if it does not, the criterion reports it and the
  // cap widening admits it -- the repair that moves the total-power figure alone.
  SeedBound up = SeedBound::initial(4, 1);
  REQUIRE(up.limit == 3u);
  REQUIRE(up.seed_top() == 2);
  REQUIRE(!up.admits(master));
  REQUIRE(up.exceeded(master));       // both hard caps pass: the cap alone cuts it
  REQUIRE(up.past_top(master));
  REQUIRE(!up.past_depth_cap(master));
  REQUIRE(up.refusal(master) == 8u);  // the cap, since no hard cap refuses it
  up.widen_cap();
  REQUIRE(up.depth == 4u);            // the depth figure stands ...
  REQUIRE(up.limit == 4u);            // ... and the cap rises one layer
  REQUIRE(up.admits(master));

  // A master that also pins the depth cap -- the master at the very depth of the box
  // -- takes one pass per figure: the depth widening carries the cap up with the box
  // top and still leaves the master on the new ceiling, and the cap widening then
  // admits it.  The box here is the one the old rule built (the request's own depth,
  // no layer above), which is what makes the master pin the cap.
  SeedBound b = SeedBound::initial(3, 1);
  REQUIRE(b.limit == 2u);
  REQUIRE(b.seed_top() == 1);
  REQUIRE(!b.admits(master));
  REQUIRE(b.exceeded(master));
  REQUIRE(b.past_top(master));
  REQUIRE(b.past_depth_cap(master));
  REQUIRE(b.refusal(master) == 1u);   // the depth cap is what refuses it
  b.widen_depth();
  REQUIRE(b.depth == 4u);
  REQUIRE(b.limit == 3u);
  REQUIRE(b.seed_top() == 2);
  REQUIRE(b.exceeded(master));        // still outside: total 3 is the new ceiling
  REQUIRE(b.past_top(master));
  REQUIRE(!b.past_depth_cap(master));  // and now only the total power refuses it
  REQUIRE(b.refusal(master) == 8u);    // which makes the repair the cap widening
  b.widen_cap();
  REQUIRE(b.depth == 4u);
  REQUIRE(b.limit == 4u);
  REQUIRE(b.admits(master));          // the box top is now above the master's total
}

TEST(seedbound_deep_low_rank_seed_is_left_to_the_masters) {
  // The case that made the criterion read the depth cap too (2026-10-11), now the
  // case the criterion deliberately leaves alone: F234's sector 246, whose box
  // (7, 5) has cap 10 and seed_top() 9 -- a high rank figure lifts the box top two
  // layers above the depth cap -- and whose reduction was left with the seed
  // [0,1,2,0,1,1,1,1,0]: depth 7, rank 0, total 7.  admits() refuses it on the
  // depth (7 >= 7), so it generates no equation and stays a master, and the
  // criterion -- which judges a master by the level's total-power figure alone --
  // stays silent (7 is not past 9): the box is left where it is and the seed is
  // handed on as a master.
  SeedBound b = SeedBound::initial(7, 5);
  REQUIRE(b.limit == 10u);
  REQUIRE(b.seed_top() == 9);
  RawIntegral master = RI({0, 1, 2, 0, 1, 1, 1, 1, 0});
  REQUIRE(master.depth() == 7u);
  REQUIRE(master.rank() == 0u);
  REQUIRE(b.quantity(master) == 7u);
  REQUIRE(!b.admits(master));
  REQUIRE(!b.past_top(master));         // the total power is inside the level
  REQUIRE(b.past_depth_cap(master));    // the depth cap is what refuses it
  REQUIRE(!b.exceeded(master));         // ... and the criterion reports nothing
  REQUIRE(b.refusal(master) == 1u);     // the depth figure, were it ever reported

  // The repair the old criterion asked for, kept as the reference for what
  // widen_depth(false) buys: deepen the box, leave the cap where it was.  The seed
  // is admitted (total 7 < cap 10), while the filter's reference -- the cap lowered
  // by one -- stays where the cap holds it.
  b.widen_depth(/*carry_cap=*/false);
  REQUIRE(b.depth == 8u);
  REQUIRE(b.limit == 10u);
  REQUIRE(b.seed_top() == 9);
  REQUIRE(b.admits(master));
  REQUIRE(!b.exceeded(master));

  // The carrying repair deepens and lifts the ceiling with the box, so the layer
  // the deepening adds is admitted rather than cut: with the cap one layer below
  // the box top, the two repairs differ at once, not only from a later layer on.
  SeedBound rect = SeedBound::initial(7, 5);
  rect.widen_depth();                   // the carrying repair, for comparison
  REQUIRE(rect.limit == 11u);
  REQUIRE(rect.seed_top() == 10);
  REQUIRE(!b.admits(RI({7, -3})));      // total 10, at the deepened box's cap
  REQUIRE(rect.admits(RI({7, -3})));

  // Second layer: the deepened box admits the new layer's seeds under its ceiling
  // (total 9) and not the seeds that reach the cap, where the carrying repair --
  // whose top rides one layer higher again -- admits them.  That is the equation
  // the repair the depth cap asks for deliberately does not buy.
  b.widen_depth(/*carry_cap=*/false);
  REQUIRE(b.depth == 9u);
  REQUIRE(b.limit == 10u);
  REQUIRE(b.admits(RI({8, -1})));        // total 9, the deeper layer is admitted
  REQUIRE(!b.admits(RI({8, -2})));       // its total-10 corner is not
  rect.widen_depth();
  REQUIRE(rect.limit == 12u);
  REQUIRE(rect.admits(RI({8, -2})));
}

// ---------------------------------------------------------------------------
// Kira-style seed truncation: admission caps derived from the input targets
// ---------------------------------------------------------------------------

TEST(seed_truncation_rule_is_derived_from_targets) {
  // Kira 3 §3.1.2 with no hand-typed l / dots: s_max is the largest s among the
  // targets, so l = popcount(top) - s_max; dots is the largest (depth - popcount)
  // of a target; and each sector keeps its own s_exc, the largest s among the
  // targets that live in it (Kira's per-sector `reduce` entry).
  const unsigned top = 0b111;  // three lines
  const std::vector<RawIntegral> targets{
      RI({1, 1, 1}),     // sector 0b111: depth 3, rank 0, dots 0
      RI({1, 0, -2}),    // sector 0b001: depth 1, rank 2, dots 0
      RI({2, 0, 0}),     // sector 0b001: depth 2, rank 0, dots 1
  };
  // 0: the paper's formula alone, before the fixed kSeedLevelOffset.
  const SeedTruncation t = derive_seed_truncation(targets, top, 0);
  REQUIRE(t.active);
  REQUIRE(t.l == 1);              // 3 - s_max(2)
  REQUIRE(t.dots == 1u);          // 2 - 1
  REQUIRE(t.s_exc.at(0b111u) == 0u);
  REQUIRE(t.s_exc.at(0b001u) == 2u);
}

TEST(seed_truncation_keeps_a_zero_dots_limit_at_zero) {
  // A target set made only of denominators derives no dots, and Kira's `d` is
  // used as written (no floor of one): dots 0 keeps only the pure denominators
  // -- depth = popcount(sector), every positive index 1 -- not the one-dot
  // seeds above them.  The strict depth cap is lines + dots + 1, so sector
  // 0b011 admits depth <= 2.  The 2026-10-18 floor of one admitted one extra
  // dot per line here and broke Kira agreement on Jpsi (109 masters against
  // Kira's 106).
  const unsigned top = 0b111;
  const std::vector<RawIntegral> targets{RI({1, 1, 1}), RI({1, 0, 1})};
  const SeedTruncation t = derive_seed_truncation(targets, top, 0);
  REQUIRE(t.active);
  REQUIRE(t.dots == 0u);
  REQUIRE(truncated_seed_caps(0b011u, 32u, 32u, t).first == 3u);
}

TEST(seed_truncation_inactive_keeps_the_dots_limit_at_zero) {
  // With no targets there is nothing to derive, and the rule stays inactive.
  const SeedTruncation t = derive_seed_truncation({}, 0b111);
  REQUIRE(!t.active);
  REQUIRE(t.dots == 0u);
}

TEST(automatic_seed_level_offset_reads_the_derived_dots) {
  // The offset the default picks when the config names none: targets carrying
  // dots (numerator powers) keep the relaxation kSeedLevelOffset, pure
  // denominators use offset 0 -- the paper's tuning formula, tennis's offset --
  // and an input with no targets has nothing to derive.  Offset 0 alone is not
  // what reproduces tennis's 166 masters; the settled basis' feedback on the
  // dots limit is (seed_dots_refinement).
  REQUIRE(kSeedLevelOffset == 1u);
  REQUIRE(seed_dots_from_targets({RI({2, 0, 0})}) == 1u);
  REQUIRE(automatic_seed_level_offset({RI({2, 0, 0})}) == 1u);
  REQUIRE(seed_dots_from_targets({RI({1, 1, 0}), RI({1, 0, 1})}) == 0u);
  REQUIRE(automatic_seed_level_offset({RI({1, 1, 0}), RI({1, 0, 1})}) == 0u);
  REQUIRE(automatic_seed_level_offset({}) == kSeedLevelOffset);
}

TEST(max_seed_dots_reads_the_largest_dot_count) {
  // dots = depth - popcount(sector): the number of positive powers beyond one
  // per line, the figure Kira's `d` bounds.  Negative indices do not enter it,
  // and the target derivation is exactly this over the target set.
  REQUIRE(max_seed_dots({}) == 0u);
  REQUIRE(max_seed_dots({RI({1, 1, 0}), RI({1, 0, 1})}) == 0u);
  REQUIRE(max_seed_dots({RI({2, 0, 0}), RI({1, -3, 0})}) == 1u);
  REQUIRE(max_seed_dots({RI({3, 1, 0}), RI({1, 1, 1})}) == 2u);  // 4 - 2
  REQUIRE(seed_dots_from_targets({RI({3, 1, 0})}) ==
          max_seed_dots({RI({3, 1, 0})}));
}

TEST(seed_dots_refinement_fires_only_above_the_rule_dots) {
  // The feedback Kira's tuning loop applies to a settled basis: only a master
  // carrying MORE dots than the rule allowed raises the limit (strict, exactly
  // as the admission caps are), and the raise is to the largest master dots.
  // An inactive rule has no limit to raise and never fires.  Tennis is the
  // first case below: derived dots 0, a basis with masters at dots 1.
  const SeedTruncation atZero = derive_seed_truncation({RI({1, 1})}, 0b11, 0);
  REQUIRE(atZero.active);
  REQUIRE(atZero.dots == 0u);
  REQUIRE(seed_dots_refinement(atZero, {}) == 0u);
  REQUIRE(seed_dots_refinement(atZero, {RI({1, 1}), RI({0, 1})}) == 0u);
  REQUIRE(seed_dots_refinement(atZero, {RI({2, 1})}) == 1u);
  REQUIRE(seed_dots_refinement(atZero, {RI({2, 1}), RI({3, 0})}) == 2u);

  SeedTruncation atOne = atZero;
  atOne.dots = 1;
  REQUIRE(seed_dots_refinement(atOne, {RI({2, 1})}) == 0u);  // at the limit
  REQUIRE(seed_dots_refinement(atOne, {RI({3, 1})}) == 2u);  // above it

  REQUIRE(seed_dots_refinement(derive_seed_truncation({}, 0b11, 0),
                               {RI({2, 1})}) == 0u);
}

TEST(seed_request_refinement_reports_only_its_own_sector_cut_masters) {
  // The feedback on the request cut: a settled master the rule still cuts is an
  // integral no sector generated an equation for, so the basis keeps it only
  // because nothing reduces it.  A request is reduced in the sector of its
  // positive support, the key the sector phase recorded it under, so the vote
  // reads cutRequests[master.sector()]; the same integral under another
  // sector's key is not the master's request (that lookup misses).
  using CutMap = std::map<unsigned, std::unordered_set<RawIntegral>>;
  const RawIntegral cut = RI({1, 0, -1});   // sector 0b001
  const RawIntegral plain = RI({0, 1, -1}); // sector 0b010, never cut
  REQUIRE(seed_request_refinement({}, {}).empty());
  REQUIRE(seed_request_refinement({plain}, {}).empty());

  const CutMap underOwnSector{{cut.sector(), {cut}}};
  REQUIRE(seed_request_refinement({plain}, underOwnSector).empty());
  REQUIRE(seed_request_refinement({cut}, underOwnSector) ==
          std::vector<RawIntegral>{cut});

  const RawIntegral other = RI({1, 1, -1}); // sector 0b011
  const CutMap underAnotherSector{{other.sector(), {cut}}};
  REQUIRE(seed_request_refinement({cut}, underAnotherSector).empty());

  // Two masters, one offender: only the offender is returned, in master order.
  const CutMap both{{cut.sector(), {cut}}};
  REQUIRE(seed_request_refinement({plain, cut}, both) ==
          std::vector<RawIntegral>{cut});
}

TEST(seed_truncation_level_offset_relaxes_the_derived_l) {
  // The relaxation lowers Kira's derived l by that many levels and never below
  // 0.  Its default is kSeedLevelOffset (1), the default of
  // settings.seed_level_offset; 0 is the paper's formula, kept for the
  // pure-formula test above.
  const unsigned top = 0b11111;  // five lines
  const std::vector<RawIntegral> targets{
      RI({1, 0, 0, 0, -2}),  // s_max 2
  };
  REQUIRE(derive_seed_truncation(targets, top).l == 2);     // default: 5 - 2 - 1
  REQUIRE(derive_seed_truncation(targets, top, 0).l == 3);  // 5 - 2
  REQUIRE(derive_seed_truncation(targets, top, 1).l == 2);
  REQUIRE(derive_seed_truncation(targets, top, 2).l == 1);
  REQUIRE(derive_seed_truncation(targets, top, 3).l == 0);
  REQUIRE(derive_seed_truncation(targets, top, 9).l == 0);  // clamped, no wrap
}

TEST(seed_truncation_records_the_level_offset_it_used) {
  // The offset is part of the rule: the derived record keeps the value the
  // caller passed, so the run's report can print what the config asked for
  // rather than a compile-time constant.
  const unsigned top = 0b11111;
  const std::vector<RawIntegral> targets{RI({1, 0, 0, 0, -2})};
  REQUIRE(derive_seed_truncation(targets, top).level_offset == kSeedLevelOffset);
  REQUIRE(derive_seed_truncation(targets, top, 0).level_offset == 0);
  REQUIRE(derive_seed_truncation(targets, top, 2).level_offset == 2);
  REQUIRE(derive_seed_truncation({}, top, 4).level_offset == 4);
}

TEST(seed_truncation_level_offset_widens_the_rank_cap) {
  // The offset relaxes the rank figure of the per-sector caps (dots, the depth
  // figure, is the other knob and does not move).  Sector 0b011 has two lines and
  // no target of its own: at l = 1 its rank cap is max(1, 2 - 1 + 1) + 1 = 3, and
  // a one-level relaxation lifts it to max(1, 2) + 1 = 4.
  const unsigned top = 0b111;  // three lines
  const std::vector<RawIntegral> targets{RI({1, 0, -2})};  // s_max 2, in 0b001
  const auto plain = truncated_seed_caps(
      0b011u, 32u, 32u, derive_seed_truncation(targets, top, 0));
  const auto relaxed = truncated_seed_caps(
      0b011u, 32u, 32u, derive_seed_truncation(targets, top, 1));
  REQUIRE(plain.first == relaxed.first);
  REQUIRE(plain.second == 3u);
  REQUIRE(relaxed.second == 4u);
}

TEST(seed_truncation_caps_leave_every_target_admitted) {
  // The auto-derived rule can never truncate a target: s_exc is the sector's own
  // largest s and dots the largest dot count, so both caps sit strictly above
  // every target of the sector.  The property is the safety net under the
  // "derive it, do not type it" rule -- a target the caps refused would cost a
  // guaranteed probe round.
  const unsigned top = 0b1111;
  const std::vector<RawIntegral> targets{
      RI({1, 1, 1, 0}),      // depth 3, rank 0
      RI({2, 0, -2, 1}),     // depth 3, rank 2, dots 1
      RI({0, 0, 1, -3}),     // depth 1, rank 3, dots 0
      RI({3, 1, 0, 0}),      // depth 4, rank 0, dots 2
  };
  const SeedTruncation t = derive_seed_truncation(targets, top);
  for (const RawIntegral &target : targets) {
    // A generous box: the rule's caps are the only admission limit here.
    const auto [capDepth, capRank] =
        truncated_seed_caps(target.sector(), 32u, 32u, t);
    REQUIRE(target.depth() < capDepth);
    REQUIRE(target.rank() < capRank);
  }
}

TEST(seedbound_truncation_is_a_second_admission_layer) {
  // Truncation cuts equations, not columns: a seed inside the hard box but past a
  // truncation cap is refused by admits() and -- unlike the hard-cap refusals,
  // which the old criterion deliberately leaves to the masters -- reported by
  // exceeded().  refusal() names the truncation figure with its own bits (16 the
  // depth, 32 the rank), so the driver can tell the repair from a box widening.
  SeedBound b = SeedBound::initial(6, 5);
  b.trunc_depth = 4;
  b.trunc_rank = 2;

  REQUIRE(b.admits(RI({3, 0})));       // inside every cap
  REQUIRE(!b.admits(RI({4, 0})));      // past the truncation depth
  REQUIRE(!b.past_top(RI({4, 0})));    // the box itself is large enough
  REQUIRE(b.exceeded(RI({4, 0})));     // ... but the reduction may still need it
  REQUIRE(b.refusal(RI({4, 0})) == 16u);

  REQUIRE(!b.admits(RI({0, -2})));     // past the truncation rank
  REQUIRE(!b.past_top(RI({0, -2})));
  REQUIRE(b.exceeded(RI({0, -2})));
  REQUIRE(b.refusal(RI({0, -2})) == 32u);

  // The hard-cap refusals keep their old verdict: inside the box top, no report.
  REQUIRE(!b.exceeded(RI({6, 0})));
  REQUIRE(b.refusal(RI({6, 0})) == 1u);
  REQUIRE(!b.exceeded(RI({0, -5})));
  REQUIRE(b.refusal(RI({0, -5})) == 2u);

  // With the truncation caps equal to the hard caps nothing is truncated: the
  // cut is exactly the old box, and the corner still reports the cap (bit 8).
  SeedBound plain = SeedBound::initial(6, 5);
  REQUIRE(plain.trunc_depth == plain.depth);
  REQUIRE(plain.trunc_rank == plain.rank);
  REQUIRE(plain.refusal(RI({5, -4})) == 8u);
}

TEST(seedbound_widen_for_lifts_the_truncation_before_the_box) {
  // The adaptive half: a truncation report is repaired by lifting the sector's
  // exception in one pass -- truncation caps go back to the hard caps, which the
  // generation box already fixed from the requests -- and the box does not grow.
  // A later hard-cap report then widens as before, and the exception stays
  // lifted (the caps follow the box).
  SeedBound b = SeedBound::initial(6, 5);
  b.trunc_depth = 4;
  b.trunc_rank = 2;

  b.widen_for(16u);
  REQUIRE(b.trunc_depth == 6u);
  REQUIRE(b.trunc_rank == 2u);
  REQUIRE(b.depth == 6u);          // the box did not move
  REQUIRE(b.limit == 9u);

  b.widen_for(32u);
  REQUIRE(b.trunc_rank == 5u);
  REQUIRE(b.rank == 5u);

  b.widen_for(1u | 4u);            // a hard depth report after the lift
  REQUIRE(b.depth == 7u);
  REQUIRE(b.limit == 10u);         // the carrying deepening
  REQUIRE(b.trunc_depth == 7u);    // the exception follows the box

  // A truncation bit and the total-power cap in one report: lift first, and the
  // cap waits for the next pass (it may not even be needed once the exception is
  // in place).
  SeedBound c = SeedBound::initial(9, 5);
  c.trunc_rank = 2;
  c.widen_for(32u | 8u);
  REQUIRE(c.trunc_rank == 5u);
  REQUIRE(c.limit == 12u);
}

TEST(seedbound_truncation_line_reports_the_admission_caps) {
  // The sector log's seed-box line names the admission caps only while they cut
  // the box, read as levels like the leading triple: a settling sector that never
  // lifted its exception shows what the truncation was.
  SeedBound b = SeedBound::initial(6, 5);
  b.trunc_depth = 4;
  b.trunc_rank = 2;
  REQUIRE(format_seed_box_line(b, SeedBound::initial(6, 5)) ==
          " depth: 5 rank: 4 total: 8 trunc: depth 3 rank 1"
          "  start: depth: 5 rank: 4 total: 8");
  SeedBound plain = SeedBound::initial(6, 5);
  REQUIRE(format_seed_box_line(plain, SeedBound::initial(6, 5)) ==
          " depth: 5 rank: 4 total: 8  start: depth: 5 rank: 4 total: 8");
}

TEST(probe_criterion_reports_a_truncated_seed_with_the_exception_bit) {
  // A needed seed the truncation cut is reported like a seed past the box top
  // (bit 4 rides along), but with the truncation figure: the driver lifts the
  // exception rather than growing a box that is already large enough.
  SeedBound b = SeedBound::initial(6, 5);
  b.trunc_depth = 4;
  b.trunc_rank = 2;
  REQUIRE(probe_criterion(b, {0u}, {RI({4, 0})}, 1u, false).refusal_bits ==
          20u);  // 16 | 4
  REQUIRE(probe_criterion(b, {0u}, {RI({0, -2})}, 1u, false).refusal_bits ==
          36u);  // 32 | 4
  // Inside the truncation caps: nothing to report.
  REQUIRE(probe_criterion(b, {0u}, {RI({3, -1})}, 1u, false).refusal_bits == 0u);
}

TEST(seedbound_past_top_still_carries_the_cap) {
  // A report the *box top* makes asks for the rectangle (bit 4): the depth
  // widening carries the cap.  Both halves can fire for the same seed, and then
  // the box top wins -- nothing about a seed past the top is fixed by deepening
  // the box alone, because the cap the deepening leaves behind is what refused it.
  SeedBound b = SeedBound::initial(7, 5);   // cap 10, top 9
  RawIntegral rankShell = RI({6, -5});      // depth 6, rank 5, total 11 > top 9
  REQUIRE(b.quantity(rankShell) == 11u);
  REQUIRE(b.past_top(rankShell));
  REQUIRE(!b.past_depth_cap(rankShell));    // the rank cap, not the depth cap
  REQUIRE(b.exceeded(rankShell));
  REQUIRE(b.refusal(rankShell) == 2u);      // so widen_rank is the repair
  RawIntegral deepBoth = RI({7, -3});       // depth 7, rank 3, total 10 > top 9
  REQUIRE(b.past_top(deepBoth));
  REQUIRE(b.past_depth_cap(deepBoth));      // both halves fire on this one
  REQUIRE(b.refusal(deepBoth) == 1u);
  b.widen_depth();                          // the carrying repair: limit 10 -> 11
  REQUIRE(b.limit == 11u);
  REQUIRE(b.admits(deepBoth));
  // The depth cap on its own (bit 4 clear) would have left the cap at 10, and the
  // seed -- whose total is exactly the cap -- would stay outside: the deepening
  // alone cannot readmit what the total-power boundary refused.
  SeedBound deep = SeedBound::initial(7, 5);
  deep.widen_depth(/*carry_cap=*/false);
  REQUIRE(deep.depth == 8u);
  REQUIRE(deep.limit == 10u);
  REQUIRE(!deep.admits(deepBoth));
  REQUIRE(deep.exceeded(deepBoth));
}

TEST(seedbound_degenerate_box_is_harmless) {
  // The smallest boxes initial_seed_bounds can return for a written-by-hand bound:
  // the cap is depth + rank - 2 clamped at 0, so it cannot underflow, the filter's
  // reference goes signed rather than wrapping, and a degenerate box admits nothing
  // anyway (one of the two hard caps is always 0).
  SeedBound b = SeedBound::initial(1, 0);
  REQUIRE(b.limit == 0u);
  REQUIRE(b.seed_top() == -1);
  REQUIRE(!b.admits(RI({0, 0, 0})));
  SeedBound c = SeedBound::initial(0, 0);
  REQUIRE(c.limit == 0u);
  REQUIRE(!c.admits(RI({0, 0, 0})));
}

TEST(generate_seeds_covers_the_whole_box) {
  // Seeds are generated over the whole box; only admission and the filter read the
  // total cap, so the cap cannot change nvars.
  Sector s = SectorTestHelper::make_sector(3, 2, {true, true}, 3, 2);
  SectorTestHelper::generate_seeds(s, OrderingMode::Balance);
  const auto &seeds = SectorTestHelper::seeds(s);
  REQUIRE(!seeds.empty());
  for (const auto &seed : seeds)
    REQUIRE(seed.depth() - std::popcount(3u) < 3u);
  REQUIRE(SectorTestHelper::weights(s).size() == seeds.size());
}

// ---------------------------------------------------------------------------
// format_seed_box_line — the boundary a sector was reduced with, and the one the
// first pass started from, both read as the level a seed range is quoted in
// ---------------------------------------------------------------------------

TEST(seed_box_line_reports_the_start_when_raised) {
  // A boundary of depth 5 and rank 4 is the level depth 4, rank 3, total 6; the run
  // started one layer below it and was raised once, so every figure of the leading
  // triple is one past the start's.  The start box sits on the base layer of a
  // three-line sector (popcount(0b111) = 3 = the depth the request asked for), and
  // its level still reads rank 3 even though no rank-3 seed is reachable there (one
  // would have depth >= 3 and so total power >= 6, the cap); the level is what the
  // requests ask for, and the widening to the leading triple is what reports that
  // the sector could not hold it.
  REQUIRE(format_seed_box_line(SeedBound::initial(5, 4),
                               SeedBound::initial(4, 4)) ==
          " depth: 4 rank: 3 total: 6  start: depth: 3 rank: 3 total: 5");
}

TEST(seed_box_line_repeats_the_start_when_nothing_was_raised) {
  REQUIRE(format_seed_box_line(SeedBound::initial(3, 2),
                               SeedBound::initial(3, 2)) ==
          " depth: 2 rank: 1 total: 2  start: depth: 2 rank: 1 total: 2");
}

TEST(seed_box_line_quotes_the_level_the_requests_ask_for) {
  // The field case (F234's sector 255): the requests sit on the sector's own base
  // layer -- eight lines at +1 with the ninth at -4 -- so they ask for depth 8 and
  // rank 4 and the line reads exactly that, with the total one below their sum.
  // The cap leaves no rank-4 seed reachable here (one would have depth >= 8 and so
  // total power >= 12, above the cap of 12), and the request is the box corner: it
  // is generated but starts out refused, and the cap widening -- the round loop's
  // first repair -- admits it with the depth and rank figures standing, so the
  // leading triple repeats the level the requests asked for with the total-power
  // figure one above it.
  const unsigned id = 0b011111111;
  const RawIntegral request = RI({1, 1, 1, 1, 1, 1, 1, 1, -4});
  auto [d, r] = initial_seed_bounds(id, {}, std::set<RawIntegral>{request});
  REQUIRE(d == 9u);
  REQUIRE(r == 5u);
  const SeedBound b = SeedBound::initial(d, r);
  REQUIRE(!b.admits(request));
  SeedBound widened = b;
  widened.widen_cap();
  REQUIRE(widened.admits(request));
  REQUIRE(format_seed_box_line(widened, b) ==
          " depth: 8 rank: 4 total: 12  start: depth: 8 rank: 4 total: 11");
}

TEST(seed_box_line_prints_the_seed_bounds_for_a_base_layer_box) {
  // A request sitting on the sector's own base layer (popcount(0b1011) = 3) asks
  // for depth 3, so the level reads depth 3, rank 0 and the total one below their
  // sum -- a range holding no integral of this sector, whose depth is at least the
  // base layer's 3.  The round loop widens the cap once to admit the request, which
  // is what makes such a box a starting point rather than a dead end: the level
  // becomes (3, 0, 3) with both figures standing.
  const unsigned id = 0b1011;
  const RawIntegral base = RI({0, 1, 1, 0, 1});  // the base integral, depth 3
  auto [d, r] = initial_seed_bounds(id, {}, std::set<RawIntegral>{base});
  SeedBound b = SeedBound::initial(d, r);
  REQUIRE(d == 4u);
  SeedBound widened = b;
  widened.widen_cap();
  REQUIRE(widened.admits(base));
  REQUIRE(format_seed_box_line(b, b) ==
          " depth: 3 rank: 0 total: 2  start: depth: 3 rank: 0 total: 2");
  REQUIRE(format_seed_box_line(widened, b) ==
          " depth: 3 rank: 0 total: 3  start: depth: 3 rank: 0 total: 2");
}

TEST(seed_box_line_shows_the_widening) {
  // The box's cap sits one layer below its corner, so the total-power figure is
  // limit - 1 -- one less than the depth and rank figures add up to.  A widening
  // grows the box by one layer and carries the cap with it, so the depth and
  // total-power figures rise by one, while the rank figure stays where the rank
  // hard cap holds it.  That is how a log shows the pass had to enlarge the box.
  SeedBound widened    = SeedBound::initial(5, 4);
  SeedBound start = widened;
  widened.widen_depth();                     // box (6, 4), cap 8
  REQUIRE(widened.limit == 8u);
  REQUIRE(format_seed_box_line(widened, start) ==
          " depth: 5 rank: 3 total: 7  start: depth: 4 rank: 3 total: 6");
}

TEST(seed_box_line_grows_the_total_power_before_the_hard_caps) {
  // The order the round loop widens in: a report the box corner makes -- both hard
  // caps clear, the total power one layer past the top -- moves the total-power
  // figure alone, (d, r, d + r - 1) becoming (d, r, d + r), which is the whole box
  // the two figures allow; only a report a hard cap makes grows that figure's axis
  // and leaves the total where the caps hold it.
  SeedBound start = SeedBound::initial(9, 5);   // cap 12: the corner sits at total 12
  SeedBound b = start;
  b.widen_cap();
  REQUIRE(format_seed_box_line(b, start) ==
          " depth: 8 rank: 4 total: 12  start: depth: 8 rank: 4 total: 11");
  b.widen_depth();                              // the repair a depth-cap report asks for
  REQUIRE(b.limit == 13u);                      // the cap rides at depth + rank - 2
  REQUIRE(format_seed_box_line(b, start) ==
          " depth: 9 rank: 4 total: 12  start: depth: 8 rank: 4 total: 11");
}

TEST(seed_box_line_reports_the_boundary_not_the_extents_under_it) {
  // The figures are the boundary's, so they have to bound the seeds admits() lets
  // through -- and the total-power figure is the one that drops when the cap bites,
  // never the depth or rank figures, which stay where the hard caps put them.
  // Enumerate the integrals of a sector with three lines and one non-line
  // propagator (every integral therefore has depth >= 3, the base layer) and
  // compare the maxima over the admitted ones with the printed line, for a cap
  // above the corner and for one that bites into it.
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
  SeedBound top = SeedBound::initial(5, 4);  // limit 7
  SeedBound cut = top;
  cut.limit = 6;  // bites: no admitted seed may reach total 6
  REQUIRE((extents(top) == Extents{4, 3, 6}));
  REQUIRE((extents(cut) == Extents{4, 2, 5}));
  // Cut, the rank figure still reads the boundary's 3 while the admitted seeds
  // reach only 2, and the total figure reports the bite: 5, one below the corner.
  REQUIRE(format_seed_box_line(top, cut) ==
          " depth: 4 rank: 3 total: 6  start: depth: 4 rank: 3 total: 5");
}

TEST(seed_box_line_reads_the_boundary_as_the_level_it_is) {
  // The worked case the field is defined by: a boundary of depth 5 and rank 2, i.e.
  // the cap at 5, admits the seeds up to depth 4, rank 1, total power 4 -- one step
  // inside the boundary in each hard figure, and one below the box corner (depth 4,
  // rank 1, total 5) in total power.  Those maxima are what the level the line
  // prints stands for; where the boundary reaches past what exists (a sector whose
  // integrals cannot carry the rank, or a cap biting into the rank), the line still
  // prints the boundary.
  SeedBound b = SeedBound::initial(5, 2);
  REQUIRE(b.limit == 5u);
  REQUIRE(format_seed_box_line(b, b) ==
          " depth: 4 rank: 1 total: 4  start: depth: 4 rank: 1 total: 4");
  // The depth figure is absolute -- the target [3, 1, -1, 0] of this sector has
  // depth 4, and depth 4 is what the line reports, not its 2 above the base layer.
  REQUIRE(RI({3, 1, -1, 0}).depth() == 4u);
}

// ---------------------------------------------------------------------------
// Applying one pass's report to the seed box (SeedBound::widen_for).
// run_reduce_ff_n hands each pass's refusal bits to this single entry point, so
// the widening policy lives here rather than inline in the driver's loops.
// ---------------------------------------------------------------------------

TEST(widen_for_carries_the_cap_for_a_box_top_report) {
  // bit 4 set: the report came from past_top(), so the depth widening moves the box
  // and the cap rides one layer below the top of the new one -- seed_top() rises
  // exactly one layer.
  SeedBound b = SeedBound::initial(6, 5);  // limit 9, top 8
  REQUIRE(b.limit == 9u);
  REQUIRE(b.seed_top() == 8);
  b.widen_for(5);  // bits 1|4
  REQUIRE(b.depth == 7u && b.rank == 5u);
  REQUIRE(b.limit == 10u);
  REQUIRE(b.seed_top() == 9);
}

TEST(widen_for_depth_cap_alone_leaves_the_cap_where_it_is) {
  // bit 4 clear: the depth hard cap reported on its own, so nothing about the total
  // power was wrong -- the box deepens and keeps its ceiling, and with the cap one
  // layer below the box top it is the cap that holds the filter's reference where it
  // was.  The layer the box gained is bounded by the total power the sector asked
  // for, which is the deliberate trade of the depth cap's own repair.
  SeedBound b = SeedBound::initial(6, 5);  // limit 9, top 8
  b.widen_for(1);
  REQUIRE(b.depth == 7u && b.rank == 5u);
  REQUIRE(b.limit == 9u);
  REQUIRE(b.seed_top() == 8);
}

TEST(widen_for_rank_bit_always_carries_the_cap_and_both_bits_apply) {
  SeedBound r = SeedBound::initial(6, 5);
  r.widen_for(2);
  REQUIRE(r.depth == 6u && r.rank == 6u);
  REQUIRE(r.limit == 10u);  // widen_rank's max(9, 6 + 6 - 2)
  REQUIRE(r.seed_top() == 9);

  SeedBound both = SeedBound::initial(6, 5);
  both.widen_for(3);  // bits 1|2, no bit 4
  REQUIRE(both.depth == 7u && both.rank == 6u);
  REQUIRE(both.limit == 11u);  // the rank widening's max(10, 7 + 6 - 2) wins
  REQUIRE(both.seed_top() == 10);
}

TEST(widen_for_a_lone_repair_hint_widens_nothing) {
  // Bit 4 is the caller's cap-policy hint, never a report on its own (a seed that
  // reports it also pins a hard cap), so widen_for(4) must be a no-op.
  SeedBound b = SeedBound::initial(6, 5);
  b.widen_for(4);
  REQUIRE(b.depth == 6u && b.rank == 5u && b.limit == 9u);
  REQUIRE(b.seed_top() == 8);
}

TEST(widen_for_applied_twice_matches_two_widenings) {
  // The driver applies the reports of consecutive rounds in sequence; a sector that
  // widens twice runs with a box two layers deeper and its top two layers higher.
  SeedBound b = SeedBound::initial(6, 5);
  b.widen_for(5);
  b.widen_for(5);
  REQUIRE(b.depth == 8u && b.rank == 5u);
  REQUIRE(b.limit == 11u);  // max(10, 8 + 5 - 2)
  REQUIRE(b.seed_top() == 10);
}

// ---------------------------------------------------------------------------
// The pass's own report (probe_criterion) and the block sector_debug prints from
// it (format_probe_report).  needed_indep holds columns: below nvars a column is
// a seed of this sector, at or above it a subsector integral.
// ---------------------------------------------------------------------------

TEST(probe_criterion_settles_when_every_own_master_is_inside_the_box) {
  // Box (4, 4): cap 6, seed top 5.  All three own masters are admitted, so the
  // pass reports nothing.
  SeedBound b = SeedBound::initial(4, 4);
  REQUIRE(b.seed_top() == 5);
  const std::vector<RawIntegral> seeds = {RI({1, 1, 0, 0}), RI({2, 1, 0, 0}),
                                          RI({1, 2, -1, 0})};
  REQUIRE(probe_criterion(b, {0, 1, 2}, seeds, 3u, false).refusal_bits == 0u);
}

TEST(probe_criterion_leaves_a_deep_low_rank_master_alone) {
  // The F234 sector-246 shape (SeedBound::past_depth_cap): a deep, low-rank seed
  // sits under the box top yet through the depth hard cap.  The criterion judges by
  // the level's total-power figure, which this seed is inside, so the pass reports
  // nothing: no bit, and the seed stays a master of the sector.
  SeedBound b = SeedBound::initial(7, 5);  // cap 10, top 9, depth cap 7
  REQUIRE(b.seed_top() == 9);
  const RawIntegral deep = RI({2, 2, 2, 1});  // depth 7, rank 0, total 7
  REQUIRE(deep.depth() == 7u);
  REQUIRE(!b.past_top(deep));
  REQUIRE(b.past_depth_cap(deep));
  REQUIRE(probe_criterion(b, {0}, {deep}, 1u, false).refusal_bits == 0u);
}

TEST(probe_criterion_reports_a_seed_past_the_box_top_with_the_cap_hint) {
  // A seed through the top of the box: the total power is what has to move, so
  // bit 4 tells the driver the depth widening carries the cap along.
  SeedBound b = SeedBound::initial(5, 4);   // cap 7, top 6, depth cap 5
  const RawIntegral wide = RI({2, 2, 2, -2});  // depth 6, rank 2, total 8
  REQUIRE(b.past_top(wide));
  REQUIRE(b.past_depth_cap(wide));
  REQUIRE(probe_criterion(b, {0}, {wide}, 1u, false).refusal_bits == 5u);
}

TEST(probe_criterion_reports_a_wide_seed_with_the_rank_bit) {
  // A shallow, wide seed through the top is the rank cap's report: bit 2, and bit
  // 4 from the top it crossed (a rank widening carries the cap anyway).  The
  // depth cap stays clear, which is what makes this the rank case.
  SeedBound b = SeedBound::initial(5, 4);    // cap 7, top 6, rank cap 4
  const RawIntegral wide = RI({1, 1, 1, -5}); // depth 3, rank 5, total 8
  REQUIRE(wide.depth() == 3u && wide.rank() == 5u);
  REQUIRE(b.past_top(wide));
  REQUIRE(!b.past_depth_cap(wide));
  REQUIRE(probe_criterion(b, {0}, {wide}, 1u, false).refusal_bits == 6u);
}

TEST(probe_criterion_reports_the_box_corner_with_the_cap_bit) {
  // The corner: both hard caps clear and the total power one layer past the top.  The
  // report names the cap (bit 8) and the top it crossed (bit 4), so the driver widens
  // the cap alone -- the depth widening the old fallback asked for would have grown
  // the depth figure and bought a layer of seeds nothing reported.
  SeedBound b = SeedBound::initial(9, 5);  // cap 12, top 11
  const RawIntegral corner = RI({1, 1, 1, 1, 1, 1, 1, 1, -4});  // depth 8, rank 4
  REQUIRE(b.quantity(corner) == 12u);
  REQUIRE(probe_criterion(b, {0}, {corner}, 1u, false).refusal_bits == 12u);
}

TEST(probe_criterion_ignores_the_subsector_columns) {
  // A column at or above nvars is a subsector integral: it travels up through
  // ctx.uptargets and generates no equation here, so it may not report.  The two
  // calls differ in those columns alone, so the bits and the collected list must
  // not move.
  SeedBound b = SeedBound::initial(6, 5);  // cap 9, top 8
  const std::vector<RawIntegral> seeds = {RI({5, -4}), RI({1, 1, 0, 0})};
  const ProbeCriterion own = probe_criterion(b, {0, 1}, seeds, 2u, true);
  REQUIRE(own.refusal_bits == 12u);  // seeds[0]: total 9, past the top of 8
  const ProbeCriterion full = probe_criterion(b, {0, 1, 2, 3}, seeds, 2u, true);
  REQUIRE(full.refusal_bits == own.refusal_bits);
  REQUIRE(full.own_masters == own.own_masters);
}

TEST(probe_criterion_scope_reports_only_the_masters_the_input_targets_need) {
  // The probe's fixed target scope.  The walk still visits every own master, but
  // only a master the sector's input targets need may move the box: one that
  // only a spill request needs is accepted where it is, so the scope keeps it
  // out of the report.  With nvars 2, column 0 is seeds[1] and column 1 is
  // seeds[0].
  SeedBound b = SeedBound::initial(5, 4);  // cap 7, top 6
  const std::vector<RawIntegral> seeds = {RI({2, 2, 2, -2}),   // column 1, bits 5
                                          RI({1, 1, 1, -5})};  // column 0, bits 6
  const std::unordered_set<std::size_t> scope{1};
  const ProbeCriterion crit = probe_criterion(b, {0, 1}, seeds, 2u, true, &scope);
  REQUIRE(crit.refusal_bits == 5u);  // column 1 alone
  REQUIRE(crit.scope_skipped == 1u);
  REQUIRE(crit.accepted == std::vector<RawIntegral>{seeds[1]});
  // The unscoped walk still reports both: the switch is the scope alone.
  REQUIRE(probe_criterion(b, {0, 1}, seeds, 2u, true).refusal_bits == 7u);
}

TEST(probe_criterion_scope_accepts_a_cut_master_outside_it) {
  // The scope suppresses the report, and with it the truncation lift: a master
  // outside the scope that the rule cut stays where it is, the way Kira accepts
  // what the smaller box finds.  A cut *request* never reaches the probe -- the
  // per-seed exception (SeedBound::requestExempt) generates its equation -- so
  // this is the cut master nothing requested.  A column the scope names still
  // reports the cut: the input target needs it.
  SeedBound b = SeedBound::initial(6, 5);
  b.trunc_depth = 4;
  b.trunc_rank = 2;
  const std::unordered_set<std::size_t> other{7};
  const ProbeCriterion crit =
      probe_criterion(b, {0u}, {RI({4, 0})}, 1u, true, &other);
  REQUIRE(crit.refusal_bits == 0u);
  REQUIRE(crit.scope_skipped == 1u);
  REQUIRE(crit.accepted == std::vector<RawIntegral>{RI({4, 0})});
  const std::unordered_set<std::size_t> scope{0};
  REQUIRE(probe_criterion(b, {0u}, {RI({4, 0})}, 1u, false, &scope).refusal_bits ==
          20u);  // 16 | 4
}

TEST(seed_bound_request_exemption_admits_a_cut_request) {
  // The truncation rule is derived from the input targets, so a spill request can
  // sit past it.  A request is reduced in its sector: cutting its equation would
  // leave it a master nothing asked for.  The exception is per seed -- the rest
  // of the cut stands -- and a hard cap still outranks it.
  SeedBound b = SeedBound::initial(6, 5);
  b.trunc_depth = 4;
  b.trunc_rank = 2;
  const RawIntegral request = RI({4, 0}); // depth 4: cut by the depth cap
  const RawIntegral rankcut = RI({0, -3}); // rank 3: cut by the rank cap
  REQUIRE(!b.admits(request));
  REQUIRE(b.truncation_refused(request));
  REQUIRE(b.exceeded(request));
  const std::unordered_set<RawIntegral> exempt{request};
  b.requestExempt = &exempt;
  REQUIRE(b.admits(request));
  REQUIRE(!b.truncation_refused(request));
  REQUIRE(!b.exceeded(request));
  REQUIRE(!b.admits(rankcut));
  REQUIRE(b.truncation_refused(rankcut));
  // A seed past a hard cap was never admitted by the box, so no truncation
  // exception can admit it (widen_depth/widen_rank is its repair instead).
  const RawIntegral pastdepth = RI({6, 0});
  const std::unordered_set<RawIntegral> deep{pastdepth};
  b.requestExempt = &deep;
  REQUIRE(!b.admits(pastdepth));
  REQUIRE(!b.truncation_refused(pastdepth));
}

TEST(probe_criterion_does_not_report_an_exempt_request) {
  // An exempted request generates its equation, so a master it leaves behind is
  // genuinely independent: reporting it would lift the sector's truncation for a
  // seed the lift cannot help.
  SeedBound b = SeedBound::initial(6, 5);
  b.trunc_depth = 4;
  b.trunc_rank = 2;
  const std::unordered_set<RawIntegral> exempt{RI({4, 0})};
  b.requestExempt = &exempt;
  const ProbeCriterion crit = probe_criterion(b, {0u}, {RI({4, 0})}, 1u, true);
  REQUIRE(crit.refusal_bits == 0u);
  REQUIRE(crit.own_masters == std::vector<RawIntegral>{RI({4, 0})});
}

TEST(probe_report_marks_a_master_accepted_outside_the_scope) {
  // sector_debug: an accepted master is still printed -- it is part of the basis
  // the pass keeps -- and the verdict says the criterion let it go on purpose.
  SeedBound b = SeedBound::initial(5, 4);  // cap 7, top 6
  const std::vector<RawIntegral> seeds = {RI({2, 2, 2, -2}), RI({1, 1, 1, -5})};
  const std::unordered_set<std::size_t> scope{1};
  const ProbeCriterion crit = probe_criterion(b, {0, 1}, seeds, 2u, true, &scope);
  REQUIRE(format_probe_report(1, b, crit) ==
          "[probe] round 1 failed: depth: 5 rank: 4 limit: 7, seed top: 6\n"
          "[probe] own masters: 2, refused: 1, accepted: 1, refusal bits: 5\n"
          "[probe]   [1, 1, 1, -5]  depth: 3 rank: 5 total: 8  PAST-TOP ACCEPTED\n"
          "[probe]   [2, 2, 2, -2]  depth: 6 rank: 2 total: 8  PAST-TOP"
          " PAST-DEPTH-CAP\n");
}

TEST(probe_criterion_collects_the_own_masters_in_walk_order) {
  // Column 0 is the most complex seed, i.e. seeds[nvars - 1] -- the list the
  // report prints follows that walk.  Only probe_criterion fills it: with the
  // flag off nothing is collected, which is what keeps sector_debug free.
  SeedBound b = SeedBound::initial(4, 4);
  const std::vector<RawIntegral> seeds = {RI({1, 1, 0, 0}), RI({2, 1, 0, 0}),
                                          RI({1, 2, -1, 0})};
  const ProbeCriterion crit = probe_criterion(b, {0, 1, 2}, seeds, 3u, true);
  REQUIRE(crit.own_masters.size() == 3u);
  REQUIRE(crit.own_masters[0] == seeds[2]);
  REQUIRE(crit.own_masters[1] == seeds[1]);
  REQUIRE(crit.own_masters[2] == seeds[0]);
  REQUIRE(probe_criterion(b, {0, 1, 2}, seeds, 3u, false).own_masters.empty());
}

TEST(probe_report_prints_the_box_the_own_masters_and_their_verdicts) {
  // The sector_debug block: the box the pass read, then one line per own master
  // in walk order, with the figures the criterion reads and the verdict it
  // reached.  A master that is fine is marked "ok" too, so a reader can see what
  // the sector holds, not only what it refused.
  SeedBound b = SeedBound::initial(5, 4);  // cap 7, top 6
  const std::vector<RawIntegral> seeds = {RI({1, 1, 1, -5}), RI({2, 2, 2, 1}),
                                          RI({2, 2, 2, -2}), RI({1, 1, 0, 0})};
  const ProbeCriterion crit = probe_criterion(b, {0, 1, 2, 3}, seeds, 4u, true);
  REQUIRE(crit.refusal_bits == 7u);  // 5 (1|4) from each of the two seeds the depth
                                     // cap and the top both refuse, 6 (2|4) from the
                                     // one only the rank cap and the top refuse
  REQUIRE(crit.own_masters.size() == 4u);
  REQUIRE(format_probe_report(3, b, crit) ==
          "[probe] round 3 failed: depth: 5 rank: 4 limit: 7, seed top: 6\n"
          "[probe] own masters: 4, refused: 3, accepted: 0, refusal bits: 7\n"
          "[probe]   [1, 1, 0, 0]  depth: 2 rank: 0 total: 2  ok\n"
          "[probe]   [2, 2, 2, -2]  depth: 6 rank: 2 total: 8  PAST-TOP PAST-DEPTH-CAP\n"
          "[probe]   [2, 2, 2, 1]  depth: 7 rank: 0 total: 7  PAST-TOP PAST-DEPTH-CAP\n"
          "[probe]   [1, 1, 1, -5]  depth: 3 rank: 5 total: 8  PAST-TOP\n");
}

TEST(sector_config_defaults_to_no_sector_debug) {
  // The option is off unless asked for: a run must be identical to the one
  // without it, and the reports cost a formatted string per reporting round.
  const SectorConfig cfg;
  REQUIRE(!cfg.sector_debug);
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
  const unsigned reports[] = {1u, 2u, 3u, 5u, 12u};
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

// ---------------------------------------------------------------------------
// Stage B: which seeds the symmetry expansion walks.  The relations exist to cut
// redundant masters in the global stage, so the seed set is this sector's own
// masters -- not the admitted seed set the IBP system is built on, which expands
// every seed a second time for relations nothing reads.
// ---------------------------------------------------------------------------

using secred::symmetry_seed_masters;

TEST(symmetry_seed_masters_keeps_only_the_sectors_own_masters) {
  // Rank r of a sector's seed list is seeds[nvars-1-r], so with nvars = 5 the ids
  // 4..0 name seeds 0..4.  needed_indep arrives descending in id, the order
  // sector_masters leaves it in.
  const std::vector<RawIntegral> seeds = {RI({1, 0, 0, 0}), RI({0, 1, 0, 0}),
                                          RI({0, 0, 1, 0}), RI({0, 0, 0, 1}),
                                          RI({1, 1, 0, 0})};
  // id 6 is at or above nvars: a subsector integral, which travels up through
  // ctx.uptargets and carries no relation of this sector.
  const std::vector<RawIntegral> got =
      symmetry_seed_masters({6u, 3u, 1u, 0u}, seeds, 5u);
  REQUIRE(got.size() == 3u);
  REQUIRE(got[0] == seeds[1]);  // id 3
  REQUIRE(got[1] == seeds[3]);  // id 1
  REQUIRE(got[2] == seeds[4]);  // id 0

  // The seeds the local reduction already expressed carry no relation.  This
  // exclusion is the rule, so pin it, not only the ids that do come through.
  REQUIRE(std::find(got.begin(), got.end(), seeds[0]) == got.end());
  REQUIRE(std::find(got.begin(), got.end(), seeds[2]) == got.end());
}

TEST(symmetry_seed_masters_follows_needed_indep_order) {
  const std::vector<RawIntegral> seeds = {RI({1, 0}), RI({0, 1}), RI({1, 1})};
  const std::vector<RawIntegral> got =
      symmetry_seed_masters({1u, 0u, 2u}, seeds, 3u);
  REQUIRE(got.size() == 3u);
  REQUIRE(got[0] == seeds[1]);
  REQUIRE(got[1] == seeds[2]);
  REQUIRE(got[2] == seeds[0]);
}

TEST(symmetry_seed_masters_is_empty_when_no_master_is_a_seed) {
  const std::vector<RawIntegral> seeds = {RI({1, 0}), RI({0, 1})};
  REQUIRE(symmetry_seed_masters({}, seeds, 2u).empty());
  // Every entry at or above nvars: subsector integrals only.
  REQUIRE(symmetry_seed_masters({3u, 2u}, seeds, 2u).empty());
}

// ---------------------------------------------------------------------------
// P0-1: a sector that gives up is a failed run, not a quiet short spill
// ---------------------------------------------------------------------------

// A refusal at the last probe round means the sector's seed box never covered
// its reduction.  The context records that as a run failure, and the caller
// checks it after the sector phase; the global stage must never read a spill
// this sector silently left short.
TEST(reduction_context_records_a_sector_give_up_as_a_run_failure) {
  ReductionContext ctx;
  REQUIRE(!ctx.hasFailed());
  REQUIRE_NOTHROW(ctx.throwIfFailed());

  ctx.noteSectorFailure(7u, kMaxRounds, "seed box still refuses masters");
  REQUIRE(ctx.hasFailed());

  bool threw = false;
  try {
    ctx.throwIfFailed();
  } catch (const std::runtime_error &e) {
    threw = true;
    const std::string message = e.what();
    REQUIRE(message.find("sector 7") != std::string::npos);
    REQUIRE(message.find("seed box still refuses masters") != std::string::npos);
  }
  REQUIRE(threw);
}

// Only the first failure is kept, so a later sector cannot overwrite the cause
// while the workers drain.
TEST(reduction_context_keeps_the_first_failure_message) {
  ReductionContext ctx;
  ctx.noteSectorFailure(3u, kMaxRounds, "first");
  ctx.noteSectorFailure(4u, kMaxRounds, "second");
  bool threw = false;
  try {
    ctx.throwIfFailed();
  } catch (const std::runtime_error &e) {
    threw = true;
    REQUIRE(std::string(e.what()).find("first") != std::string::npos);
  }
  REQUIRE(threw);
}
