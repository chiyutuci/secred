#include "test_runner.h"
#include "sector.h"

// Helper: unambiguously construct from integer initializer list
static RawIntegral RI(std::initializer_list<int> il) {
  return RawIntegral(std::vector<int>(il));
}

// ---------------------------------------------------------------------------
// RawIntegral construction
// ---------------------------------------------------------------------------

TEST(rawintegral_default_construct) {
  RawIntegral ri;
  REQUIRE(ri.size() == 0);
}

TEST(rawintegral_n_zeros) {
  RawIntegral ri(4);
  REQUIRE(ri.size() == 4);
  for (unsigned i = 0; i < 4; ++i)
    REQUIRE(ri[i] == 0);
}

TEST(rawintegral_n_value) {
  RawIntegral ri(3, 1);
  REQUIRE(ri.size() == 3);
  for (unsigned i = 0; i < 3; ++i)
    REQUIRE(ri[i] == 1);
}

TEST(rawintegral_from_vector_int) {
  std::vector<int> v = {1, -2, 0, 3};
  RawIntegral ri(v);
  REQUIRE(ri.size() == 4);
  REQUIRE(ri[0] == 1);
  REQUIRE(ri[1] == -2);
  REQUIRE(ri[2] == 0);
  REQUIRE(ri[3] == 3);
}

TEST(rawintegral_from_vector_bool) {
  std::vector<bool> lines = {true, false, true, false};
  RawIntegral ri(lines);
  REQUIRE(ri.size() == 4);
  REQUIRE(ri[0] == 1);
  REQUIRE(ri[1] == 0);
  REQUIRE(ri[2] == 1);
  REQUIRE(ri[3] == 0);
}

// Constructors must never write past indices_[31].  A 33-element input used to write
// indices_[32], which is nprops_ itself, so the integral silently became 3 propagators
// wide; anything longer overwrote whatever followed the object.
TEST(rawintegral_vector_int_is_capped_at_max_props) {
  std::vector<int> v(33, 1);
  v[31] = 2;
  RawIntegral ri(v);
  REQUIRE(ri.size() == RawIntegral::MAX_PROPS);
  REQUIRE(ri[0] == 1);
  REQUIRE(ri[31] == 2);
}

TEST(rawintegral_vector_bool_is_capped_at_max_props) {
  std::vector<bool> lines(33, true);
  RawIntegral ri(lines);
  REQUIRE(ri.size() == RawIntegral::MAX_PROPS);
  REQUIRE(ri[31] == 1);
}

TEST(rawintegral_n_value_is_capped_at_max_props) {
  RawIntegral ri(33u, 1);
  REQUIRE(ri.size() == RawIntegral::MAX_PROPS);
  REQUIRE(ri[31] == 1);
}

TEST(rawintegral_n_zeros_is_capped_at_max_props) {
  RawIntegral ri(33u);
  REQUIRE(ri.size() == RawIntegral::MAX_PROPS);
}

TEST(rawintegral_mutable_index) {
  RawIntegral ri(3, 0);
  ri[1] = 5;
  REQUIRE(ri[1] == 5);
}

// ---------------------------------------------------------------------------
// depth(): sum of positive indices
// ---------------------------------------------------------------------------

TEST(depth_all_positive) {
  REQUIRE(RI({2, 3, 1}).depth() == 6);
}

TEST(depth_mixed) {
  // positives: 1 + 2 = 3
  REQUIRE(RI({1, -3, 2, 0}).depth() == 3);
}

TEST(depth_all_zero) {
  REQUIRE(RawIntegral(4, 0).depth() == 0);
}

TEST(depth_all_negative) {
  REQUIRE(RI({-1, -2, -3}).depth() == 0);
}

// ---------------------------------------------------------------------------
// rank(): sum of |negative| indices
// ---------------------------------------------------------------------------

TEST(rank_all_negative) {
  REQUIRE(RI({-1, -2, -3}).rank() == 6);
}

TEST(rank_mixed) {
  // |-1| + |-2| = 3
  REQUIRE(RI({2, -1, 0, -2}).rank() == 3);
}

TEST(rank_all_positive) {
  REQUIRE(RI({1, 2, 3}).rank() == 0);
}

// ---------------------------------------------------------------------------
// sector(): bitmask of propagators with positive index
// ---------------------------------------------------------------------------

TEST(sector_all_positive) {
  REQUIRE(RI({1, 1, 1, 1}).sector() == 0b1111u);
}

TEST(sector_mixed) {
  // positions 0 and 2 are positive → 0b0101
  REQUIRE(RI({1, -1, 2, 0}).sector() == 0b0101u);
}

TEST(sector_all_zero_or_negative) {
  REQUIRE(RI({0, -1, 0, -2}).sector() == 0u);
}

TEST(sector_single_propagator) {
  REQUIRE(RI({0, 0, 1, 0}).sector() == (1u << 2));
}

// ---------------------------------------------------------------------------
// maxneg(): minimum (most negative) index, sentinel -1 if none
// ---------------------------------------------------------------------------

TEST(maxneg_no_negative) {
  REQUIRE(RI({1, 2, 3}).maxneg() == -1);
}

TEST(maxneg_with_negatives) {
  REQUIRE(RI({1, -3, -1, 2}).maxneg() == -3);
}

// ---------------------------------------------------------------------------
// operator== and operator!=
// ---------------------------------------------------------------------------

TEST(equality_same) {
  RawIntegral a = RI({1, 2, 3});
  RawIntegral b = RI({1, 2, 3});
  REQUIRE(a == b);
  REQUIRE(!(a != b));
}

TEST(equality_different) {
  RawIntegral a = RI({1, 2, 3});
  RawIntegral b = RI({1, 2, 4});
  REQUIRE(!(a == b));
  REQUIRE(a != b);
}

TEST(not_equal_is_symmetric) {
  RawIntegral a = RI({1, 2, 3});
  RawIntegral b = RI({1, 2, 4});
  REQUIRE((a != b) == (b != a));
}

// ---------------------------------------------------------------------------
// operator< (total order — used by std::set / std::map)
// ---------------------------------------------------------------------------

TEST(less_than_lexicographic) {
  RawIntegral a = RI({1, 2, 3});
  RawIntegral b = RI({1, 2, 4});
  REQUIRE(a < b);
  REQUIRE(!(b < a));
}

TEST(less_than_irreflexive) {
  RawIntegral a = RI({1, 2, 3});
  REQUIRE(!(a < a));
}

TEST(less_than_consistent_with_equality) {
  RawIntegral a = RI({1, 2, 3});
  RawIntegral b = RI({1, 2, 3});
  REQUIRE(!(a < b));
  REQUIRE(!(b < a));
}

// ---------------------------------------------------------------------------
// operator+ (index-wise addition)
// ---------------------------------------------------------------------------

TEST(addition_basic) {
  RawIntegral a = RI({1, 0, -1});
  RawIntegral b = RI({0, 1, 1});
  RawIntegral c = a + b;
  REQUIRE(c[0] == 1);
  REQUIRE(c[1] == 1);
  REQUIRE(c[2] == 0);
}

TEST(addition_negative_sum) {
  RawIntegral a = RI({-1, -2, 3});
  RawIntegral b = RI({0, 0, -4});
  RawIntegral c = a + b;
  REQUIRE(c[0] == -1);
  REQUIRE(c[1] == -2);
  REQUIRE(c[2] == -1);
}

TEST(addition_identity) {
  RawIntegral a = RI({3, -1, 2});
  RawIntegral z(3, 0);
  RawIntegral c = a + z;
  REQUIRE(c == a);
}

// ---------------------------------------------------------------------------
// operator- (index-wise subtraction)
// ---------------------------------------------------------------------------

TEST(subtraction_basic) {
  RawIntegral a = RI({3, 2, 1});
  RawIntegral b = RI({1, 1, 1});
  RawIntegral c = a - b;
  REQUIRE(c[0] == 2);
  REQUIRE(c[1] == 1);
  REQUIRE(c[2] == 0);
}

TEST(subtraction_gives_zero) {
  RawIntegral a = RI({3, -1, 2});
  RawIntegral c = a - a;
  REQUIRE(c.depth() == 0);
  REQUIRE(c.rank() == 0);
}

// ---------------------------------------------------------------------------
// std::hash consistency
// ---------------------------------------------------------------------------

TEST(hash_equal_objects) {
  std::hash<RawIntegral> h;
  REQUIRE(h(RI({1, 2, 3})) == h(RI({1, 2, 3})));
}

TEST(hash_different_objects_differ) {
  std::hash<RawIntegral> h;
  REQUIRE(h(RI({1, 2, 3})) != h(RI({3, 2, 1})));
}

// ---------------------------------------------------------------------------
// YAML decode
// ---------------------------------------------------------------------------

TEST(yaml_decode_integers) {
  YAML::Node node = YAML::Load("[1, -2, 0, 3]");
  RawIntegral ri = node.as<RawIntegral>();
  REQUIRE(ri.size() == 4);
  REQUIRE(ri[0] == 1);
  REQUIRE(ri[1] == -2);
  REQUIRE(ri[2] == 0);
  REQUIRE(ri[3] == 3);
}

TEST(yaml_decode_rejects_non_sequence) {
  YAML::Node node = YAML::Load("scalar_value");
  bool threw = false;
  try {
    node.as<RawIntegral>();
  } catch (...) {
    threw = true;
  }
  REQUIRE(threw);
}

// ---------------------------------------------------------------------------
// Usable in STL containers
// ---------------------------------------------------------------------------

TEST(usable_in_set) {
  std::set<RawIntegral> s;
  s.insert(RI({1, 2, 3}));
  s.insert(RI({1, 2, 3}));  // duplicate → ignored
  s.insert(RI({1, 2, 4}));
  REQUIRE(s.size() == 2);
}

TEST(usable_in_unordered_map) {
  std::unordered_map<RawIntegral, int> m;
  m[RI({1, 2})] = 10;
  m[RI({1, 2})] = 20;  // overwrite
  REQUIRE(m.size() == 1);
  REQUIRE(m[RI({1, 2})] == 20);
}
