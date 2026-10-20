
// The univariate linear reconstruction with partially recovered coefficients:
// the state seeded from the first prime's trial function, the per-prime system
// that only carries the coefficients still unknown, and the promotion rule that
// fixes a coefficient once two consecutive primes agree on its rational value.
#include <cstdint>
#include <vector>

#include <fflow/common.hh>
#include <fflow/function_cache.hh>
#include <fflow/primes.hh>
#include <fflow/rational_function.hh>
#include <fflow/univariate_reconstruction.hh>

#include "test_runner.h"
#include "uni_linear_reconstruct.h"

namespace {

namespace ff = fflow;

// Values of the rational function num/den at x modulo p.
ff::UInt eval_rat(const unsigned num[], unsigned nd, const unsigned den[],
                  unsigned dd, ff::UInt x, ff::Mod mod) {
  ff::UInt n = 0, d = 0;
  for (unsigned j = nd + 1; j-- > 0;)
    n = ff::add_mod(ff::mul_mod(n, x, mod), num[j] % mod.n(), mod);
  for (unsigned j = dd + 1; j-- > 0;)
    d = ff::add_mod(ff::mul_mod(d, x, mod), den[j] % mod.n(), mod);
  return ff::mul_mod(n, ff::mul_inv(d, mod), mod);
}

// The samples the univariate path caches for one prime: the first points of
// FiniteFlow's OFFSET_0 walk, keyed on (x, modulus).
void fill_samples(ff::UIntCache &cache, ff::Mod mod, unsigned npoints,
                  const unsigned num[], unsigned nd, const unsigned den[],
                  unsigned dd) {
  ff::UInt x = ff::OFFSET_0_HASH;
  for (unsigned i = 0; i < npoints; ++i) {
    ff::UInt *key = cache.get_new_inptr();
    key[0] = x;
    key[1] = mod.n();
    ff::UInt *val = cache.new_entry(key, 1);
    val[0] = eval_rat(num, nd, den, dd, x, mod);
    x = ff::add_mod(x, ff::SAMPLING_STRIDE, mod);
  }
}

ff::SparseRationalFunction make_function(const unsigned num[], unsigned nd,
                                         const unsigned den[], unsigned dd) {
  ff::URationalFunction u;
  u.setNumDegree(nd);
  u.setDenDegree(dd);
  for (unsigned j = 0; j <= nd; ++j)
    u.num(j) = num[j];
  for (unsigned j = 0; j <= dd; ++j)
    u.den(j) = den[j];
  ff::SparseRationalFunction sf;
  sf.fromUnivariate(u);
  return sf;
}

// (2x^3 + 3x + 1)/(x^2 + 5x + 7): a gap at numerator degree 2, pivot 0.
const unsigned kNum1[] = {1, 3, 0, 2};
const unsigned kDen1[] = {1, 5, 7};

// (x + 2)/(x + 3x^2): the denominator starts at degree 1, pivot 1.
const unsigned kNum2[] = {2, 1};
const unsigned kDen2[] = {0, 1, 3};

}

TEST(uni_linear_state_counts_only_the_unknown_coefficients) {
  const ff::UInt p = ff::BIG_UINT_PRIMES[0];
  secred::UniLinearState st;

  secred::uni_seed(st, make_function(kNum1, 3, kDen1, 2), 3, 2, 0, p);
  // three present numerator coefficients, two non-pivot denominator ones
  REQUIRE_EQ(secred::uni_unknown_count(st, 0), 5u);

  // A denominator that starts at degree 1 still fixes only the pivot.
  secred::UniLinearState low;
  secred::uni_seed(low, make_function(kNum2, 1, kDen2, 2), 1, 2, 1, p);
  REQUIRE_EQ(secred::uni_unknown_count(low, 1), 3u);
}

TEST(uni_linear_step_solves_the_open_coefficients_and_promotes_them) {
  const ff::UInt primes[] = {ff::BIG_UINT_PRIMES[0], ff::BIG_UINT_PRIMES[1],
                             ff::BIG_UINT_PRIMES[2], ff::BIG_UINT_PRIMES[3]};
  ff::UIntCache caches[4];
  ff::SampledRatFun funs[4];
  for (unsigned i = 0; i < 4; ++i) {
    ff::Mod mod(primes[i]);
    caches[i].init(2);
    fill_samples(caches[i], mod, 8, kNum1, 3, kDen1, 2);
    funs[i].set_function_cache(&caches[i]);
    funs[i].idx = 0;
  }

  secred::UniLinearState st;
  secred::uni_seed(st, make_function(kNum1, 3, kDen1, 2), 3, 2, 0, primes[0]);
  std::vector<ff::UInt> num, den;

  REQUIRE(secred::uni_linear_step(st, 3, 2, 0, ff::Mod(primes[1]), 2, 0, funs[1],
                                  num, den) == ff::SUCCESS);
  for (unsigned j = 0; j <= 3; ++j)
    REQUIRE_EQ(num[j], kNum1[j] % primes[1]);
  for (unsigned j = 0; j <= 2; ++j)
    REQUIRE_EQ(den[j], kDen1[j] % primes[1]);
  // One merge is not enough to fix a coefficient: the value has to repeat.
  REQUIRE_EQ(secred::uni_unknown_count(st, 0), 5u);

  REQUIRE(secred::uni_linear_step(st, 3, 2, 0, ff::Mod(primes[2]), 2, 0, funs[2],
                                  num, den) == ff::SUCCESS);
  REQUIRE_EQ(secred::uni_unknown_count(st, 0), 0u);

  // Everything is fixed now: the system still solves, with only its checks.
  REQUIRE(secred::uni_linear_step(st, 3, 2, 0, ff::Mod(primes[3]), 2, 0, funs[3],
                                  num, den) == ff::SUCCESS);
  for (unsigned j = 0; j <= 3; ++j)
    REQUIRE_EQ(num[j], kNum1[j] % primes[3]);
  for (unsigned j = 0; j <= 2; ++j)
    REQUIRE_EQ(den[j], kDen1[j] % primes[3]);
}

TEST(uni_linear_step_keeps_a_higher_pivot_normalization) {
  const ff::UInt primes[] = {ff::BIG_UINT_PRIMES[0], ff::BIG_UINT_PRIMES[1],
                             ff::BIG_UINT_PRIMES[2]};
  ff::UIntCache caches[3];
  ff::SampledRatFun funs[3];
  for (unsigned i = 0; i < 3; ++i) {
    ff::Mod mod(primes[i]);
    caches[i].init(2);
    fill_samples(caches[i], mod, 8, kNum2, 1, kDen2, 2);
    funs[i].set_function_cache(&caches[i]);
    funs[i].idx = 0;
  }

  secred::UniLinearState st;
  secred::uni_seed(st, make_function(kNum2, 1, kDen2, 2), 1, 2, 1, primes[0]);
  std::vector<ff::UInt> num, den;
  for (unsigned i = 1; i < 3; ++i)
    REQUIRE(secred::uni_linear_step(st, 1, 2, 1, ff::Mod(primes[i]), 2, 0,
                                    funs[i], num, den) == ff::SUCCESS);
  for (unsigned j = 0; j <= 1; ++j)
    REQUIRE_EQ(num[j], kNum2[j] % primes[2]);
  REQUIRE_EQ(den[0], 0u);
  REQUIRE_EQ(den[1], 1u);
  REQUIRE_EQ(den[2], 3u);
  REQUIRE_EQ(secred::uni_unknown_count(st, 1), 0u);
}

TEST(uni_linear_step_rejects_samples_that_disagree_with_a_known_output) {
  const ff::UInt primes[] = {ff::BIG_UINT_PRIMES[0], ff::BIG_UINT_PRIMES[1],
                             ff::BIG_UINT_PRIMES[2], ff::BIG_UINT_PRIMES[3]};
  // Two more primes fix every coefficient of kNum1/kDen1.
  ff::UIntCache caches[3];
  ff::SampledRatFun funs[3];
  for (unsigned i = 0; i < 3; ++i) {
    ff::Mod mod(primes[i]);
    caches[i].init(2);
    fill_samples(caches[i], mod, 8, kNum1, 3, kDen1, 2);
    funs[i].set_function_cache(&caches[i]);
    funs[i].idx = 0;
  }
  secred::UniLinearState st;
  secred::uni_seed(st, make_function(kNum1, 3, kDen1, 2), 3, 2, 0, primes[0]);
  std::vector<ff::UInt> num, den;
  for (unsigned i = 1; i < 3; ++i)
    REQUIRE(secred::uni_linear_step(st, 3, 2, 0, ff::Mod(primes[i]), 2, 0,
                                    funs[i], num, den) == ff::SUCCESS);
  REQUIRE_EQ(secred::uni_unknown_count(st, 0), 0u);

  // A fourth prime whose samples come from a different numerator: with nothing
  // left to solve, every row is a consistency check and must reject them.
  const unsigned wrong_num[] = {4, 3, 0, 2};
  ff::UIntCache bad;
  bad.init(2);
  fill_samples(bad, ff::Mod(primes[3]), 4, wrong_num, 3, kDen1, 2);
  ff::SampledRatFun badfun;
  badfun.set_function_cache(&bad);
  badfun.idx = 0;
  REQUIRE(secred::uni_linear_step(st, 3, 2, 0, ff::Mod(primes[3]), 2, 0, badfun,
                                  num, den) == ff::FAILED);
}

TEST(uni_linear_step_rejects_an_output_outside_the_seeded_degrees) {
  const ff::UInt p = ff::BIG_UINT_PRIMES[1];
  // Seeded as x + 1 (denominator degree 0), sampled as (x+1)/(x+2), which no
  // degree-0 denominator can represent.
  const unsigned known_den[] = {1};
  const unsigned samp_num[] = {1, 1};
  const unsigned samp_den[] = {2, 1};
  ff::UIntCache cache;
  cache.init(2);
  fill_samples(cache, ff::Mod(p), 4, samp_num, 1, samp_den, 1);
  ff::SampledRatFun fun;
  fun.set_function_cache(&cache);
  fun.idx = 0;

  secred::UniLinearState st;
  secred::uni_seed(st, make_function(samp_num, 1, known_den, 0), 1, 0, 0, p);
  std::vector<ff::UInt> num, den;
  REQUIRE(secred::uni_linear_step(st, 1, 0, 0, ff::Mod(p), 2, 0, fun, num, den) ==
          ff::FAILED);
}

TEST(uni_linear_step_fails_when_the_cache_lacks_a_point) {
  const ff::UInt p = ff::BIG_UINT_PRIMES[1];
  ff::UIntCache cache;
  cache.init(2);
  // five unknowns plus two checks need seven points; cache only four
  fill_samples(cache, ff::Mod(p), 4, kNum1, 3, kDen1, 2);
  ff::SampledRatFun fun;
  fun.set_function_cache(&cache);
  fun.idx = 0;

  secred::UniLinearState st;
  secred::uni_seed(st, make_function(kNum1, 3, kDen1, 2), 3, 2, 0, p);
  std::vector<ff::UInt> num, den;
  REQUIRE(secred::uni_linear_step(st, 3, 2, 0, ff::Mod(p), 2, 0, fun, num, den) ==
          ff::FAILED);
}
