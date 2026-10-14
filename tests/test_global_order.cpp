#include "test_runner.h"
#include "global_order.h"
#include "sector.h"

#include <algorithm>
#include <initializer_list>
#include <random>
#include <vector>

// Global integral numbering (src/global_order.h).  The number is the index after
// the sort, and the smallest one is what the solver prefers as a master, so
// these orders decide the master basis directly.
namespace {

RawIntegral RI(std::initializer_list<int> il) {
  return RawIntegral(std::vector<int>(il));
}

std::vector<RawIntegral> run(std::vector<RawIntegral> v, OrderingMode mode) {
  secred::order_global_integrals(v, mode);
  return v;
}

// Assert that got equals want element by element (the line number names which).
void expect_order(const std::vector<RawIntegral> &got,
                  const std::vector<RawIntegral> &want) {
  REQUIRE(got.size() == want.size());
  for (unsigned i = 0; i < want.size(); ++i)
    REQUIRE(got[i] == want[i]);
}

// Assert that out is a permutation of in (nothing lost, nothing duplicated).
void expect_permutation(const std::vector<RawIntegral> &in,
                        const std::vector<RawIntegral> &out) {
  std::vector<RawIntegral> a = in, b = out;
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  expect_order(a, b);
}

}  // namespace

// ---------------------------------------------------------------------------
// First key: the layer popcount(sector)
// ---------------------------------------------------------------------------

TEST(global_order_popcount_is_the_first_key) {
  // One representative per layer.  x2's total power (2) is below x1's (5), so a
  // balance key that outranked popcount would order these x0, x2, x3, x1.
  RawIntegral x0 = RI({0, 0, 0});  // popcount 0, total 0
  RawIntegral x1 = RI({5, 0, 0});  // popcount 1, total 5
  RawIntegral x2 = RI({1, 1, 0});  // popcount 2, total 2
  RawIntegral x3 = RI({1, 1, 1});  // popcount 3, total 3
  std::vector<RawIntegral> in = {x3, x1, x2, x0};  // deliberately unordered
  std::vector<RawIntegral> want = {x0, x1, x2, x3};
  expect_order(run(in, OrderingMode::Normal), want);
  expect_order(run(in, OrderingMode::Balance), want);
}

// ---------------------------------------------------------------------------
// Normal: within a layer, (depth, rank, sector, lexicographic)
// ---------------------------------------------------------------------------

TEST(global_order_normal_key_is_depth_then_rank_then_sector) {
  // Within one layer (popcount 1): the two depth-2 integrals come first, then the
  // two depth-3 ones; equal depth falls back to the sector mask ascending
  // (b: sector 1 < c: sector 2; a: 1 < d: 4).
  RawIntegral a = RI({3, 0, 0});  // depth 3, sector 1
  RawIntegral b = RI({2, 0, 0});  // depth 2, sector 1
  RawIntegral c = RI({0, 2, 0});  // depth 2, sector 2
  RawIntegral d = RI({0, 0, 3});  // depth 3, sector 4
  std::vector<RawIntegral> in = {d, a, c, b};
  std::vector<RawIntegral> want = {b, c, a, d};
  auto nor = run(in, OrderingMode::Normal);
  expect_order(nor, want);
}

// ---------------------------------------------------------------------------
// Balance: within a layer, the five keys interleave the sectors
// ---------------------------------------------------------------------------

TEST(global_order_balance_mixes_sectors_within_a_layer) {
  // All popcount 1, ordered by the five balance keys (total power, depth, rank,
  // max denominator power, max numerator power):
  //   c = [2,-1]: (3, 2, 1, 2, 1)
  //   b = [3, 0]: (3, 3, 0, 3, 1)
  //   d = [0, 3]: (3, 3, 0, 3, 1)   <- tied with b on all five
  //   a = [1,-5]: (6, 1, 5, 1, 5)
  // Normal order on the same input is (depth, rank, sector) → a, c, b, d.
  RawIntegral a = RI({1, -5});  // sector 1
  RawIntegral b = RI({3, 0});   // sector 1
  RawIntegral c = RI({2, -1});  // sector 1
  RawIntegral d = RI({0, 3});   // sector 2
  std::vector<RawIntegral> in = {a, b, c, d};
  auto bal = run(in, OrderingMode::Balance);
  // c is the simplest; b and d tie on all five keys, so the sector mask breaks
  // the tie (b: 1 < d: 2); a has the largest total power and comes last.  The
  // sector sequence is {0}, {0}, {1}, {0} -- the sort is not grouped by sector.
  expect_order(bal, {c, b, d, a});
  // Normal gives a completely different numbering on the same input, and both
  // are permutations of it.
  auto nor = run(in, OrderingMode::Normal);
  expect_order(nor, {a, c, b, d});
  REQUIRE(!(bal == nor));
  expect_permutation(in, bal);
  expect_permutation(in, nor);
}

TEST(global_order_balance_depth_key_decides_at_equal_total) {
  // Same layer (popcount 1), same total power (3), different depth: the second
  // balance key (depth ascending) decides, and the sector fallback key points the
  // other way -- a comparator that only compared the total power and then fell
  // through to the sector fallback would swap these two, so this case pins it.
  RawIntegral a = RI({3, 0, 0});   // depth 3, rank 0, sector {0} = 1
  RawIntegral b = RI({0, 2, -1});  // depth 2, rank 1, sector {1} = 2
  REQUIRE(a.depth() + a.rank() == b.depth() + b.rank());
  expect_order(run({a, b}, OrderingMode::Balance), {b, a});
  // Normal's first key is the depth too, so the same pair comes out {b, a} there
  // as well -- for a different reason: it has no total-power key at all.
  expect_order(run({a, b}, OrderingMode::Normal), {b, a});
}

TEST(global_order_balance_maxpos_and_maxneg_keys_decide) {
  // Once the first three keys (total power, depth, rank) tie, the fourth key
  // (maxpos) and the fifth (-maxneg) decide.  In both cases the sector fallback
  // key points the other way, so a comparator that stopped after three keys would
  // order the pair backwards (and when the sectors are equal the lexicographic
  // fallback points the other way as well).
  //
  // (a) maxpos: both popcount 2, total 6, depth 4, rank 2; a's maxpos 3 > b's 2.
  RawIntegral a = RI({3, 1, -1, -1});  // sector {0,1} = 3
  RawIntegral b = RI({2, -1, 2, -1});  // sector {0,2} = 5
  REQUIRE(a.depth() == b.depth());
  REQUIRE(a.rank() == b.rank());
  expect_order(run({a, b}, OrderingMode::Balance), {b, a});  // maxpos 2 < 3

  // (b) -maxneg: both popcount 2, total 4, depth 2, rank 2, maxpos 1, and the same
  //     sector {0,1}; c has maxneg -2 (-maxneg = 2) above d's -1 (-maxneg = 1), so d
  //     comes first -- while the lexicographic fallback (-2 < -1) would put c first.
  RawIntegral c = RI({1, 1, -2, 0});   // sector {0,1} = 3
  RawIntegral d = RI({1, 1, -1, -1});  // sector {0,1} = 3
  REQUIRE(c.maxpos() == d.maxpos());
  expect_order(run({c, d}, OrderingMode::Balance), {d, c});  // -maxneg 1 < 2
}

TEST(global_order_balance_ties_break_by_sector_mask) {
  // [2,0,0] and [0,2,0] tie on all five keys (2, 2, 0, 2, 1); the sector mask
  // breaks the tie (1 < 2).
  auto out = run({RI({0, 2, 0}), RI({2, 0, 0})}, OrderingMode::Balance);
  REQUIRE(out[0] == RI({2, 0, 0}));
  REQUIRE(out[1] == RI({0, 2, 0}));
}

TEST(global_order_balance_ties_break_by_lexicographic_order) {
  // [1,2,-1,0] and [2,1,-1,0] share the sector {0,1} and all five keys
  // (3, 3, 1, 2, 1): the lexicographic fallback decides.
  auto out = run({RI({2, 1, -1, 0}), RI({1, 2, -1, 0})}, OrderingMode::Balance);
  REQUIRE(out[0] == RI({1, 2, -1, 0}));
  REQUIRE(out[1] == RI({2, 1, -1, 0}));
}

// ---------------------------------------------------------------------------
// Determinism: the fallback keys make the order total, independent of the input
// ---------------------------------------------------------------------------

TEST(global_order_is_input_order_independent) {
  std::vector<RawIntegral> in = {
      RI({3, 0, 0}), RI({2, -1}), RI({0, 3}), RI({1, -5}), RI({0, 2, 0}),
      RI({1, 1, 0}), RI({0, 0, 3}), RI({2, 1, -1, 0}), RI({1, 2, -1, 0}),
  };
  for (OrderingMode mode : {OrderingMode::Normal, OrderingMode::Balance}) {
    auto forward = run(in, mode);
    std::vector<RawIntegral> reversed(in.rbegin(), in.rend());
    auto backward = run(reversed, mode);
    expect_order(backward, forward);
    expect_permutation(in, forward);
  }
}

// ---------------------------------------------------------------------------
// Direction: number 0 is the simplest integral of its layer (the solver prefers
// it as a master)
// ---------------------------------------------------------------------------

TEST(global_order_zero_id_is_the_simplest_in_its_layer) {
  // Within a layer, number 0 must be the smallest under both orders (the
  // direction must not be inverted).
  std::vector<RawIntegral> in = {RI({3, 0}), RI({2, -1}), RI({1, -4}),
                                 RI({1, 0})};
  auto bal = run(in, OrderingMode::Balance);
  REQUIRE(bal.front() == RI({1, 0}));
  for (const RawIntegral &x : bal)
    REQUIRE(!balance_less(x, bal.front()));
  auto nor = run(in, OrderingMode::Normal);
  REQUIRE(nor.front() == RI({1, 0}));
}

// ---------------------------------------------------------------------------
// The packed sort key (decorate-sort-undecorate)
//
// The numbered table is the solver's column order, so the key must induce
// exactly the comparator's order -- a key that differs anywhere moves the master
// basis.  These tests pit the key against global_integral_less itself.
// ---------------------------------------------------------------------------

namespace {

// A deterministic corpus: the field extremes of the packed key (the widest
// powers the type admits, the empty integral) plus random integrals over mixed
// propagator counts, so the lexicographic fallback and the nprops tie-break are
// exercised as well.
std::vector<RawIntegral> order_corpus(unsigned count, unsigned seed) {
  std::vector<RawIntegral> v;
  v.reserve(count + 6);
  v.emplace_back(std::vector<int>(32, 127));   // depth 32 * 127, the widest key
  v.emplace_back(std::vector<int>(32, -128));  // rank 32 * 128
  v.emplace_back(std::vector<int>{0});
  v.emplace_back(std::vector<int>{1, 0});
  v.emplace_back(std::vector<int>{0, 1});
  v.emplace_back(std::vector<int>{127, -128});
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> idx(-4, 4);
  std::uniform_int_distribution<unsigned> np(1, 6);
  for (unsigned k = 0; k < count; ++k) {
    std::vector<int> w(np(rng));
    for (int &x : w)
      x = idx(rng);
    v.emplace_back(std::move(w));
  }
  return v;
}

std::vector<secred::GlobalOrderKey> keyed(const std::vector<RawIntegral> &v,
                                          OrderingMode mode) {
  std::vector<secred::GlobalOrderKey> k;
  k.reserve(v.size());
  for (unsigned i = 0; i < v.size(); ++i)
    k.push_back(secred::global_order_key(v[i], mode, i));
  return k;
}

}  // namespace

TEST(global_order_key_orders_exactly_like_the_comparator) {
  auto v = order_corpus(400, 20261014u);
  for (OrderingMode mode : {OrderingMode::Normal, OrderingMode::Balance}) {
    auto k = keyed(v, mode);
    for (unsigned i = 0; i < v.size(); ++i) {
      for (unsigned j = 0; j < v.size(); ++j) {
        // Both directions, so a key that only reproduces "less" but not its
        // negation (a wrong field order, a truncated field) is caught.
        REQUIRE(secred::global_order_key_less(k[i], k[j], v) ==
                secred::global_integral_less(v[i], v[j], mode));
        REQUIRE(secred::global_order_key_less(k[j], k[i], v) ==
                secred::global_integral_less(v[j], v[i], mode));
      }
    }
  }
}

TEST(global_order_decorated_sort_reproduces_the_comparator_sort) {
  // The same sort two ways: the compact key with the gather, and the original
  // comparator in place.  Both are total orders, so the sequences must agree
  // element for element.
  auto v = order_corpus(3000, 7u);
  for (OrderingMode mode : {OrderingMode::Normal, OrderingMode::Balance}) {
    std::vector<RawIntegral> want = v;
    std::sort(want.begin(), want.end(),
              [mode](const RawIntegral &a, const RawIntegral &b) {
                return secred::global_integral_less(a, b, mode);
              });
    std::vector<RawIntegral> got = v;
    secred::order_global_integrals(got, mode);
    expect_order(got, want);
    expect_permutation(v, got);
  }
}
