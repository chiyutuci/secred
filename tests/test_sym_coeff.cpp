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

// ---------------------------------------------------------------------------
// Coefficient interning
// ---------------------------------------------------------------------------
//
// A global system spells the same rational function out again and again: Kira's own log
// for F234 reads "Parsed 8040387 functions of which 1617 are distinct".  Evaluating each
// of the reconstructed system's ~11 million (integral, coefficient) entries at every
// sample point -- with a modular inverse each -- is what makes a point cost seconds, so
// the reconstruction evaluates each *distinct* function once and lets the entries read
// the answer out of the table.
TEST(sym_coeff_pool_interns_equal_coefficients) {
  const std::vector<SymEq> sys = {
      SymEq{{3u, coeff(1 - d / s)}, {1u, coeff(d / (2 * s * t))}},
      SymEq{{2u, coeff(d / (2 * s * t))}, {0u, coeff(6 * t * m)}},
      SymEq{{1u, coeff(1 - d / s)}},
  };
  secred::SymCoeffPool pool;
  secred::build_sym_coeff_pool(sys, {0u, 1u, 2u}, pool);

  REQUIRE_EQ(pool.n_equations(), std::size_t(3));
  REQUIRE_EQ(pool.n_terms(), std::size_t(5));
  REQUIRE_EQ(pool.coeffs.size(), std::size_t(3)); // five entries, three functions
  REQUIRE_EQ(pool.eq_begin, (std::vector<uint32_t>{0u, 2u, 4u, 5u}));
  REQUIRE_EQ(pool.terms[0].second, pool.terms[4].second); // 1 - d/s, two equations
  REQUIRE_EQ(pool.terms[1].second, pool.terms[2].second); // d/(2*s*t), two equations
  REQUIRE(pool.terms[3].second != pool.terms[0].second);  // 6*t*m is its own function
  REQUIRE(pool.terms[3].second != pool.terms[1].second);
  // the pool keeps each equation's terms in the system's descending-id order
  REQUIRE_EQ(pool.terms[0].first, 3u);
  REQUIRE_EQ(pool.terms[1].first, 1u);
  REQUIRE_EQ(pool.terms[4].first, 1u);
  REQUIRE_EQ(pool.terms[3].first, 0u);
}

// The strict grammar and GiNaC's parser order a monomial's factors their own way, so the
// same function has to intern whichever spelling produced it.
TEST(sym_coeff_pool_interns_a_monomial_spelled_another_way) {
  const SymCoeff a = coeff(d * s);
  REQUIRE_EQ(a.num.terms[0].vars.size(), std::size_t(2));
  SymCoeff b = a;
  std::swap(b.num.terms[0].vars[0], b.num.terms[0].vars[1]);

  const std::vector<SymEq> sys = {SymEq{{1u, a}}, SymEq{{0u, b}}};
  secred::SymCoeffPool pool;
  secred::build_sym_coeff_pool(sys, {0u, 1u}, pool);
  REQUIRE_EQ(pool.coeffs.size(), std::size_t(1));
}

// The row of `pool` at one point, gathered the way the elimination gathers it: the terms
// whose coefficient did not vanish here, descending id.
secred::ResEq gather_row(const secred::SymCoeffPool &pool, std::size_t e,
                         const std::vector<uint64_t> &values) {
  secred::ResEq row;
  for (std::size_t k = pool.eq_begin[e]; k < pool.eq_begin[e + 1]; ++k) {
    const uint64_t v = values[pool.terms[k].second];
    if (v)
      row.emplace_back(pool.terms[k].first, v);
  }
  return row;
}

// A point fills the table once per distinct function: every residue must be exactly what
// the per-entry evaluation produces, and a table reduced for another prime must be refused
// rather than read.
TEST(sym_coeff_pool_evaluates_every_distinct_function_once) {
  const std::vector<SymEq> sys = {
      SymEq{{3u, coeff(1 - d / s)}, {1u, coeff(d / (2 * s * t))}},
      SymEq{{2u, coeff(d / (2 * s * t))}, {0u, coeff(6 * t * m)}},
      SymEq{{1u, coeff(1 - d / s)}},
  };
  const std::vector<uint64_t> x{12345, 6789, 4321, 987};
  const std::vector<std::vector<uint64_t>> pw =
      secred::make_powers(sys, x.data(), 4, P64);

  secred::SymCoeffPool pool;
  secred::build_sym_coeff_pool(sys, {0u, 1u, 2u}, pool);
  secred::reduce_pool_for_prime(pool, P64);

  std::vector<uint64_t> values;
  REQUIRE(secred::eval_pool_point(pool, subs_of(x), pw, P64, values));
  REQUIRE_EQ(values.size(), pool.coeffs.size());

  // Reference: the same equations through the per-entry path.
  std::vector<SymEq> ref = sys;
  secred::reduce_sym_system(ref, P64);
  for (std::size_t e = 0; e < ref.size(); ++e) {
    secred::ResEq want;
    for (const auto &pr : ref[e]) {
      uint64_t v = 0;
      REQUIRE(secred::eval_sym_coeff_mod(pr.second, subs_of(x), pw, P64, v));
      if (v)
        want.emplace_back(pr.first, v);
    }
    REQUIRE(gather_row(pool, e, values) == want);
  }
  // The interned row really is read from the table: the shared coefficient d/(2*s*t)
  // appears in two equations and takes the same residue in both.
  REQUIRE_EQ(pool.terms[pool.eq_begin[0] + 1].second,
             pool.terms[pool.eq_begin[1]].second);

  std::vector<uint64_t> other_values;
  REQUIRE(!secred::eval_pool_point(pool, subs_of(x), pw, P32, other_values));
}

// The slot array is rebuilt for every point, so a point never sees the previous point's
// residues -- and a term whose coefficient vanishes here must simply be absent from the
// gathered row rather than keep the value it had before.
TEST(sym_coeff_pool_evaluates_each_point_from_scratch) {
  const std::vector<SymEq> sys = {
      SymEq{{2u, coeff(d / s)}, {1u, coeff(3 * t)}},
      SymEq{{0u, coeff(m + 1)}},
  };
  secred::SymCoeffPool pool;
  secred::build_sym_coeff_pool(sys, {0u, 1u}, pool);
  secred::reduce_pool_for_prime(pool, P64);

  std::vector<uint64_t> values;
  // m = p - 1: the second equation is 0 = 0 at this point.
  const std::vector<uint64_t> x1{2, 3, 4, P64 - 1}, x2{6, 7, 8, 9};
  REQUIRE(secred::eval_pool_point(pool, subs_of(x1),
                                  secred::make_powers(sys, x1.data(), 4, P64), P64,
                                  values));
  REQUIRE_EQ(gather_row(pool, 0, values).size(), std::size_t(2));
  REQUIRE(gather_row(pool, 1, values).empty());

  REQUIRE(secred::eval_pool_point(pool, subs_of(x2),
                                  secred::make_powers(sys, x2.data(), 4, P64), P64,
                                  values));
  const secred::ResEq row0 = gather_row(pool, 0, values);
  REQUIRE_EQ(row0.size(), std::size_t(2));
  REQUIRE_EQ(row0[0].first, 2u); // d/s first: the system's descending id order
  REQUIRE_EQ(row0[1].first, 1u);
  const secred::ResEq row1 = gather_row(pool, 1, values);
  REQUIRE_EQ(row1.size(), std::size_t(1));
  REQUIRE_EQ(row1[0].first, 0u);
}

// Interning a whole system is the reader's other half: the reader interns term by term
// as it parses, and this is the shape a caller that already holds a system gets.  Both
// must assign the same slots in the same order, or a pool read one way would not be the
// pool read the other -- which is the whole point of sharing the interner.
TEST(sym_coeff_pool_interns_a_whole_system_the_way_the_reader_does) {
  const std::vector<SymEq> sys = {
      SymEq{{3u, coeff(1 - d / s)}, {1u, coeff(d / (2 * s * t))}},
      SymEq{{2u, coeff(d / (2 * s * t))}, {0u, coeff(6 * t * m)}},
      SymEq{{1u, coeff(1 - d / s)}},
  };
  secred::SymCoeffInterner in;
  for (const SymEq &eq : sys) {
    in.begin_row();
    for (const auto &pr : eq)
      in.add_term(pr.first, pr.second);
  }
  in.finish();

  secred::SymCoeffPool whole;
  secred::intern_sym_system(sys, whole);
  REQUIRE(whole.terms == in.pool.terms);
  REQUIRE(whole.eq_begin == in.pool.eq_begin);
  REQUIRE_EQ(whole.coeffs.size(), in.pool.coeffs.size());

  // And selecting rows out of the whole-system pool is what building the pool from the
  // selection does, up to the slot numbering the selection implies.
  secred::SymCoeffPool selected;
  secred::build_sym_coeff_pool(sys, {1u, 2u}, selected);
  REQUIRE_EQ(selected.eq_begin, (std::vector<uint32_t>{0u, 2u, 3u}));
  REQUIRE_EQ(selected.n_terms(), std::size_t(3));
}

// Retaining a subset of the rows: the pool keeps those rows in the order the subset
// names them, renumbers the surviving coefficients densely, and releases everything
// else -- including the row boundaries of the rows it dropped and the coefficient tables
// no remaining row reads.
TEST(sym_coeff_pool_retains_only_the_named_rows) {
  const std::vector<SymEq> sys = {
      SymEq{{3u, coeff(1 - d / s)}, {1u, coeff(6 * t * m)}},
      SymEq{{2u, coeff(d / (2 * s * t))}},
      SymEq{{1u, coeff(1 - d / s)}, {0u, coeff(m + 1)}},
  };
  secred::SymCoeffPool pool;
  secred::intern_sym_system(sys, pool);
  REQUIRE_EQ(pool.coeffs.size(), std::size_t(4)); // four distinct functions

  secred::retain_pool_rows(pool, {2u, 1u});
  REQUIRE_EQ(pool.n_equations(), std::size_t(2));
  REQUIRE_EQ(pool.eq_begin, (std::vector<uint32_t>{0u, 2u, 3u}));
  REQUIRE_EQ(pool.n_terms(), std::size_t(3));
  // Row 0 is the old row 2's, in its order, and row 1 the old row 1's -- read through
  // the rows' own contents rather than their slots, since the slots were renumbered.
  REQUIRE_EQ(pool.terms[0].first, 1u);
  REQUIRE_EQ(pool.terms[1].first, 0u);
  REQUIRE_EQ(pool.terms[2].first, 2u);
  // The dropped rows' coefficients are gone: their slots named nothing that survives.
  REQUIRE_EQ(pool.coeffs.size(), std::size_t(3));

  const std::vector<uint64_t> x{12345, 6789, 4321, 987};
  secred::reduce_pool_for_prime(pool, P64);
  std::vector<uint64_t> values;
  REQUIRE(secred::eval_pool_point(pool, subs_of(x),
                                  secred::make_powers(pool, x.data(), 4, P64), P64,
                                  values));
  REQUIRE_EQ(gather_row(pool, 0, values).size(), std::size_t(2));
  REQUIRE_EQ(gather_row(pool, 1, values).size(), std::size_t(1));
}

// The rows of a pool at one point, as numeric equations: the pool form of
// `eval_sym_eqs_mod`, with the same rows in the same order and the same residues.
TEST(sym_coeff_pool_rows_match_the_per_entry_evaluation) {
  const std::vector<SymEq> sys = {
      SymEq{{3u, coeff(1 - d / s)}, {1u, coeff(d / (2 * s * t))}},
      SymEq{{2u, coeff(d / (2 * s * t))}, {0u, coeff(6 * t * m)}},
      SymEq{{0u, coeff(m + 1)}},
  };
  const std::vector<uint64_t> x{12345, 6789, 4321, P64 - 1}; // row 2 vanishes
  secred::SymCoeffPool pool;
  secred::intern_sym_system(sys, pool);
  secred::reduce_pool_for_prime(pool, P64);

  std::vector<secred::ResEq> got;
  std::vector<uint32_t> got_idx;
  REQUIRE(secred::eval_pool_rows(pool, {0u, 1u, 2u}, subs_of(x),
                                 secred::make_powers(pool, x.data(), 4, P64), P64, got,
                                 got_idx));

  std::vector<SymEq> ref = sys;
  secred::reduce_sym_system(ref, P64);
  std::vector<secred::ResEq> want;
  std::vector<uint32_t> want_idx;
  REQUIRE(secred::eval_sym_eqs_mod(ref, {0u, 1u, 2u}, subs_of(x),
                                   secred::make_powers(ref, x.data(), 4, P64), P64, want,
                                   want_idx));
  REQUIRE(got == want);
  REQUIRE(got_idx == want_idx);
  REQUIRE_EQ(got_idx, (std::vector<uint32_t>{0u, 1u})); // the vanishing row is dropped
}
