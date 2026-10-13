#include "test_runner.h"

#include "sym_coeff.h"

#include <atomic>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <ginac/ginac.h>

namespace {

using secred::SymCoeff;
using secred::SymEq;

// A 64-bit prime: the FiniteFlow reconstruction prunes with primes of this size, and
// a product of two residues below it does not fit in uint64.
constexpr uint64_t P64 = 18446744073709551557ULL;
constexpr uint64_t P32 = 4294967291ULL;

const GiNaC::symbol d("d"), s("s"), t("t"), m("m");

std::unordered_map<std::string, unsigned> idx() {
  return {{"d", 0}, {"s", 1}, {"t", 2}, {"m", 3}};
}

// Independent reference: reduce an exact CLN integer mod p.
uint64_t exact_mod(const cln::cl_I &x, uint64_t p) {
  return secred::cl_I_mod_u64(x, p);
}

SymCoeff coeff(const GiNaC::ex &e) { return secred::make_sym_coeff(e.expand(), idx()); }

// One-coefficient system whose power table covers the coefficient's exponents.
std::vector<std::vector<uint64_t>>
powers_for(const SymCoeff &c, const std::vector<uint64_t> &x, uint64_t p) {
  std::vector<SymEq> sys;
  sys.push_back(SymEq{{0u, c}});
  return secred::make_powers(sys, x.data(), 4, p);
}

GiNaC::exmap subs_of(const std::vector<uint64_t> &x) {
  GiNaC::exmap sm;
  sm[d] = GiNaC::numeric(x[0]);
  sm[s] = GiNaC::numeric(x[1]);
  sm[t] = GiNaC::numeric(x[2]);
  sm[m] = GiNaC::numeric(x[3]);
  return sm;
}

// Evaluate one coefficient the way the reconstruction does: reduce it for the prime
// first, then read the value.  The evaluation is only defined on a reduced table, so
// every test goes through here.
uint64_t eval_one(const SymCoeff &c, const std::vector<uint64_t> &x, uint64_t p) {
  std::vector<SymEq> sys;
  sys.push_back(SymEq{{0u, c}});
  secred::reduce_sym_system(sys, p);
  const std::vector<std::vector<uint64_t>> pw =
      secred::make_powers(sys, x.data(), 4, p);
  uint64_t v = 0;
  REQUIRE(secred::eval_sym_coeff_mod(sys[0][0].second, subs_of(x), pw, p, v));
  return v;
}

} // namespace

// The precompiled evaluation must agree with the exact integer the GiNaC expression
// stands for.  The sample values here are large enough that a uint64 multiply of two
// residues would wrap.
TEST(sym_coeff_negative_coefficient_at_64bit_prime) {
  const SymCoeff c = coeff(-2 * GiNaC::pow(m, 2));
  REQUIRE(c.precompiled);
  const std::vector<uint64_t> x{P64 - 3, 17, P64 - 12345, P64 - 7};

  const uint64_t got = eval_one(c, x, P64);
  // -2*m^2 as an exact integer, reduced by CLN rather than by our own arithmetic.
  const uint64_t want = exact_mod(cln::cl_I(-2) * cln::cl_I(x[3]) * cln::cl_I(x[3]),
                                  P64);
  REQUIRE_EQ(got, want);
}

TEST(sym_coeff_matches_substitution_at_64bit_prime) {
  const std::vector<GiNaC::ex> exprs{
      -2 * GiNaC::pow(m, 2),
      -2 * GiNaC::pow(m, 2) + 2 * t,
      d * s * t * m,
      (d - 1) * (s + 2),
      4 * GiNaC::pow(d, 2) * GiNaC::pow(t, 3) - 11 * s * t,
      d / (2 * s * t),
  };
  const std::vector<std::vector<uint64_t>> pts{
      {P64 - 3, 17, P64 - 12345, P64 - 7},
      {1, 2, 3, 4},
      {P64 - 1, P64 - 1, P64 - 1, P64 - 1},
  };
  for (const GiNaC::ex &e : exprs) {
    for (const std::vector<uint64_t> &x : pts) {
      std::vector<SymEq> sys;
      sys.push_back(SymEq{{0u, coeff(e)}});
      secred::reduce_sym_system(sys, P64);
      const std::vector<std::vector<uint64_t>> pw =
          secred::make_powers(sys, x.data(), 4, P64);

      uint64_t fast = 0;
      REQUIRE(secred::eval_sym_coeff_mod(sys[0][0].second, GiNaC::exmap(), pw, P64,
                                        fast));
      // The GiNaC `subs` path is the reference the precompiler must reproduce.
      SymCoeff slow = sys[0][0].second;
      slow.precompiled = false;
      uint64_t ref = 0;
      REQUIRE(secred::eval_sym_coeff_mod(slow, subs_of(x), pw, P64, ref));
      REQUIRE_EQ(fast, ref);
    }
  }
}

TEST(mul_mod_u64_does_not_wrap) {
  // (p-1)^2 == 1, (p-2)^2 == 4, (p-1)*(p-2) == 2  (all mod p)
  REQUIRE_EQ(secred::mul_mod_u64(P64 - 1, P64 - 1, P64), 1ULL);
  REQUIRE_EQ(secred::mul_mod_u64(P64 - 2, P64 - 2, P64), 4ULL);
  REQUIRE_EQ(secred::mul_mod_u64(P64 - 1, P64 - 2, P64), 2ULL);
  REQUIRE_EQ(secred::mul_mod_u64(0, P64 - 1, P64), 0ULL);
}

TEST(build_sym_poly_rejects_unrepresentable_shapes) {
  const std::unordered_map<std::string, unsigned> si = idx();
  secred::SymPoly out;

  // Rational coefficient.
  REQUIRE(!secred::build_sym_poly(GiNaC::ex(1) / 2 * d, si, out));
  // Negative exponent.
  REQUIRE(!secred::build_sym_poly(GiNaC::pow(d, -1), si, out));
  // Operand that is not a symbol power.
  REQUIRE(!secred::build_sym_poly(GiNaC::sin(d), si, out));
  // A symbol outside the parameter list.
  REQUIRE(!secred::build_sym_poly(GiNaC::symbol("z") + 1, si, out));

  // ... and the shapes it does accept.
  REQUIRE(secred::build_sym_poly(GiNaC::ex(1), si, out));
  REQUIRE_EQ(out.terms.size(), std::size_t(1));
  REQUIRE(secred::build_sym_poly((-3 * GiNaC::pow(d, 2) * t + 5).expand(), si, out));
  REQUIRE_EQ(out.terms.size(), std::size_t(2));
}

TEST(make_sym_coeff_falls_back_when_unrepresentable) {
  const SymCoeff c = coeff(GiNaC::sin(d) + 1);
  REQUIRE(!c.precompiled);
  // The GiNaC form is still carried, so the coefficient can be evaluated via subs.
  REQUIRE(!c.slow_num.is_zero());
}

TEST(sym_coeff_zero_denominator_mod_p_evaluates_to_zero) {
  const SymCoeff c = coeff(d / (2 * s * t));
  REQUIRE(c.precompiled);
  const std::vector<uint64_t> x{P64 - 3, 0, 5, 7}; // s == 0 mod p
  REQUIRE_EQ(eval_one(c, x, P64), 0ULL);
}

TEST(make_powers_covers_every_exponent_used) {
  SymEq eq0;
  eq0.push_back({0u, coeff(GiNaC::pow(d, 3) * GiNaC::pow(m, 2))});
  eq0.push_back({1u, coeff(t)});
  std::vector<SymEq> sys;
  sys.push_back(std::move(eq0));
  const std::vector<uint64_t> x{2, 3, 5, 7};
  const std::vector<std::vector<uint64_t>> pw =
      secred::make_powers(sys, x.data(), 4, P32);

  const std::vector<unsigned> mx = secred::sym_max_exps(sys, 4);
  REQUIRE_EQ(mx[0], 3u);
  REQUIRE_EQ(mx[1], 0u);
  REQUIRE_EQ(mx[2], 1u);
  REQUIRE_EQ(mx[3], 2u);
  for (unsigned i = 0; i < 4; ++i) {
    REQUIRE_EQ(pw[i].size(), std::size_t(mx[i]) + 1);
    uint64_t want = 1;
    for (unsigned k = 0; k < pw[i].size(); ++k) {
      REQUIRE_EQ(pw[i][k], want);
      want = want * x[i] % P32;
    }
  }
}

TEST(eval_sym_poly_mod_is_linear_in_the_monomial_sum) {
  // 3*d + 4*d must evaluate to 7*d, exercising the 128-bit accumulator.
  const std::vector<uint64_t> x{P64 - 1, 0, 0, 0};
  REQUIRE_EQ(eval_one(coeff((3 * d + 4 * d).expand()), x, P64),
             exact_mod(cln::cl_I(7) * cln::cl_I(x[0]), P64));
}

// The reduced coefficients belong to one prime.  Reading them at another prime would
// silently return the first prime's residue, so the evaluation must refuse instead:
// the reconstruction reduces the system once per prime, and a missed reduce has to be
// a loud failure rather than a wrong number.
TEST(sym_coeff_evaluation_at_an_unreduced_prime_fails) {
  std::vector<SymEq> sys;
  sys.push_back(SymEq{{0u, coeff(-2 * GiNaC::pow(m, 2))}});
  secred::reduce_sym_system(sys, P64);

  const std::vector<uint64_t> x{1, 2, 3, 4};
  uint64_t v = 0;
  REQUIRE(!secred::eval_sym_coeff_mod(sys[0][0].second, subs_of(x),
                                     powers_for(sys[0][0].second, x, P32), P32, v));
  // ... and the prime it was reduced for still evaluates.
  REQUIRE(secred::eval_sym_coeff_mod(sys[0][0].second, subs_of(x),
                                     powers_for(sys[0][0].second, x, P64), P64, v));
}

// A reduce must cover both halves of a coefficient and every equation: the
// reconstruction reduces the whole system once and then only reads.
TEST(reduce_sym_system_covers_every_precompiled_polynomial) {
  std::vector<SymEq> sys;
  sys.push_back(SymEq{{0u, coeff(d / (2 * s * t))}, {1u, coeff(GiNaC::pow(d, 3) - 1)}});
  sys.push_back(SymEq{{2u, coeff(6 * t * m)}});
  sys.push_back(SymEq{{3u, coeff(GiNaC::sin(d))}}); // not precompiled: no table

  secred::reduce_sym_system(sys, P32);
  for (const SymEq &eq : sys)
    for (const auto &pr : eq) {
      const secred::SymCoeff &c = pr.second;
      if (!c.precompiled) {
        REQUIRE_EQ(c.num.mod_vals.size(), std::size_t(0));
        continue;
      }
      REQUIRE_EQ(c.num.mod_vals.size(), c.num.terms.size());
      REQUIRE_EQ(c.den.mod_vals.size(), c.den.terms.size());
      for (std::size_t i = 0; i < c.num.terms.size(); ++i)
        REQUIRE_EQ(c.num.mod_vals[i], exact_mod(c.num.terms[i].coef, P32));
      REQUIRE_EQ(c.num.mod_p, P32);
    }

  // Reducing for another prime overwrites the table rather than adding a second one.
  secred::reduce_sym_system(sys, P64);
  REQUIRE_EQ(sys[0][0].second.num.mod_p, P64);
  REQUIRE_EQ(sys[0][0].second.num.mod_vals.size(),
             sys[0][0].second.num.terms.size());
}

// Many threads read one reduced system at once.  The coefficient tables are written
// only by `reduce_sym_system`, so evaluation is a pure read and the results must not
// depend on how the points are spread over threads.
TEST(reduced_system_is_read_only_during_evaluation) {
  std::vector<SymEq> sys;
  const std::vector<GiNaC::ex> exprs{
      -2 * GiNaC::pow(m, 2) + 2 * t,
      (d - 1) * (s + 2) * m,
      4 * GiNaC::pow(d, 2) * GiNaC::pow(t, 3) - 11 * s * t,
      d / (2 * s * t) + GiNaC::pow(m, 5),
  };
  for (std::size_t i = 0; i < exprs.size(); ++i)
    sys.push_back(SymEq{{static_cast<uint32_t>(i), coeff(exprs[i])},
                        {static_cast<uint32_t>(i + 1), coeff(exprs[i] * d + 1)}});
  secred::reduce_sym_system(sys, P64);

  const unsigned npts = 400;
  std::vector<std::vector<uint64_t>> pts(npts, std::vector<uint64_t>(4));
  std::mt19937_64 rng(0x243F6A8885A308D3ULL);
  for (auto &x : pts)
    for (uint64_t &v : x)
      v = rng() % P64;

  // Reference: one thread, in order.
  std::vector<std::vector<uint64_t>> want(npts);
  for (unsigned k = 0; k < npts; ++k) {
    const std::vector<std::vector<uint64_t>> pw =
        secred::make_powers(sys, pts[k].data(), 4, P64);
    for (const SymEq &eq : sys)
      for (const auto &pr : eq) {
        uint64_t v = 0;
        REQUIRE(secred::eval_sym_coeff_mod(pr.second, GiNaC::exmap(), pw, P64, v));
        want[k].push_back(v);
      }
  }

  // The same reads from 8 threads over the same system.
  const unsigned nthreads = 8;
  std::vector<std::vector<uint64_t>> got(npts);
  std::atomic<unsigned> failures{0};
  {
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < nthreads; ++t) {
      threads.emplace_back([&, t]() {
        for (unsigned k = t; k < npts; k += nthreads) {
          const std::vector<std::vector<uint64_t>> pw =
              secred::make_powers(sys, pts[k].data(), 4, P64);
          for (const SymEq &eq : sys)
            for (const auto &pr : eq) {
              uint64_t v = 0;
              if (!secred::eval_sym_coeff_mod(pr.second, GiNaC::exmap(), pw, P64, v))
                failures.fetch_add(1, std::memory_order_relaxed);
              got[k].push_back(v);
            }
        }
      });
    }
    for (std::thread &th : threads)
      th.join();
  }
  REQUIRE_EQ(failures.load(), 0u);
  for (unsigned k = 0; k < npts; ++k)
    REQUIRE(got[k] == want[k]);
}
