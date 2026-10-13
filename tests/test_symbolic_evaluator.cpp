// The shared evaluator: the selection inputs, the point evaluation, the shared learn,
// and the counters the FireFly backend reports.
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "sector.h"       // RawIntegral, the integral type the learn indexes into
#include "symbolic_ffred.h"
#include "test_runner.h"

namespace {

// 4 integrals (ids 0..3), 3 equations, one master (id 0), one target (id 3):
//   E1: [1,1] + (-2+d)/d [1,0] + 1/d [0,1] + 3 [0,0] = 0
//   E2: [0,1] + (d-4)/s [0,0] = 0
//   E3: [1,0] + 2d/s [0,0] = 0
// Each equation is a sum equal to zero, and the solver stores the master's coefficient
// in the reduction (ffred_solver.h normalises the pivot to "coefficient -> -1"), so E3
// gives [1,0] = -(2d/s) [0,0] and E2 gives [0,1] = -((d-4)/s) [0,0].  Substituting both
// into E1 gives [1,1] = -((-2+d)/d)[1,0] - (1/d)[0,1] - 3[0,0]
//                     = 2(d-2)/s + (d-4)/(d*s) - 3,
// a rational function of d and s.
secred::SymEq eq_of(std::vector<std::pair<std::uint32_t, secred::SymCoeff>> terms) {
  return secred::SymEq(terms.begin(), terms.end());
}

secred::SymCoeff coeff(const std::string &text,
                       const std::vector<GiNaC::possymbol> &symbols) {
  GiNaC::symtab tab;
  std::unordered_map<std::string, unsigned> index;
  for (unsigned i = 0; i < symbols.size(); ++i) {
    tab[symbols[i].get_name()] = symbols[i];
    index[symbols[i].get_name()] = i;
  }
  GiNaC::parser p(tab);
  return secred::make_sym_coeff(p(text).expand(), index);
}

}  // namespace

// The system the evaluator tests share, with the master set given by `masters`
// (master id -> column).  The default is the correct one: [0,0] alone.
secred::SymbolicFFEvaluator evaluator_of(
    std::unordered_map<unsigned, unsigned> master_index = {{0, 0}}) {
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d"),
                                                 GiNaC::possymbol("s")};
  std::vector<secred::SymEq> system = {
      eq_of({{3, coeff("1", symbols)}, {1, coeff("(-2+d)/d", symbols)},
             {2, coeff("1/d", symbols)}, {0, coeff("3", symbols)}}),
      eq_of({{2, coeff("1", symbols)}, {0, coeff("(d-4)/s", symbols)}}),
      eq_of({{1, coeff("1", symbols)}, {0, coeff("(2*d)/s", symbols)}}),
  };
  std::vector<std::uint32_t> eq_idx = {0, 1, 2};
  std::vector<std::uint32_t> bwd_chain = {1, 2, 3};
  std::vector<unsigned> dep_targets = {3};
  std::vector<GiNaC::ex> symbol_ex = {GiNaC::ex(symbols[0]), GiNaC::ex(symbols[1])};
  const unsigned nnindeps = master_index.empty() ? 1u
                             : static_cast<unsigned>(master_index.size());
  return secred::SymbolicFFEvaluator(std::move(system), std::move(eq_idx),
                                     std::move(bwd_chain), std::move(dep_targets),
                                     std::move(master_index), nnindeps,
                                     std::move(symbol_ex));
}

// One target one symbol-power deeper than `evaluator_of()`'s system:
//   E1: [1,0] + (d^4/s)[0,0] = 0,  master [0,0] (id 0), target [1,0] (id 1),
// so the reduced target is -(d^4/s) times the master.  Its numerators carry d^4, where
// `evaluator_of()`'s system carries d^1 and s^1 only.
secred::SymbolicFFEvaluator big_evaluator() {
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d"),
                                                 GiNaC::possymbol("s")};
  std::vector<secred::SymEq> system = {
      eq_of({{1, coeff("1", symbols)}, {0, coeff("d^4/s", symbols)}})};
  std::vector<GiNaC::ex> symbol_ex = {GiNaC::ex(symbols[0]), GiNaC::ex(symbols[1])};
  return secred::SymbolicFFEvaluator(std::move(system), {0}, {1}, {1}, {{0, 0}}, 1,
                                     std::move(symbol_ex));
}

TEST(symbolic_evaluator_flattens_the_target_reduction) {
  secred::SymbolicFFEvaluator ev = evaluator_of();
  REQUIRE_EQ(ev.n_outputs(), 1u);
  REQUIRE_EQ(ev.n_params(), 2u);
  REQUIRE(ev.all_precompiled());

  const std::uint64_t p = 1125899906842597ULL;   // a 50-bit prime
  ev.reduce_for_prime(p, 1);
  secred::SymbolicFFEvaluator::Scratch scratch;
  const std::uint64_t x[2] = {5, 7};
  std::uint64_t out[1] = {0};
  REQUIRE(ev.evaluate_point(p, x, scratch, out) ==
          secred::SymbolicFFEvaluator::PointResult::Ok);
  // 2*(d-2)/s + (d-4)/(d*s) - 3 at d = 5, s = 7, i.e. -74/35 mod p
  const std::uint64_t expected = [&] {
    auto mul = [&](std::uint64_t a, std::uint64_t b) {
      return static_cast<std::uint64_t>((static_cast<__int128>(a) * b) % p);
    };
    auto inv = [&](std::uint64_t a) {
      __int128 t = 0, newt = 1, r = p, newr = a;
      while (newr) {
        const __int128 q = r / newr;
        std::swap(t -= q * newt, newt);
        std::swap(r -= q * newr, newr);
      }
      return static_cast<std::uint64_t>((t % p + p) % p);
    };
    const std::uint64_t a = mul(2, mul(p + 5 - 2, inv(7)));      // 2(d-2)/s
    const std::uint64_t b = mul(p + 5 - 4, inv(mul(5, 7)));      // (d-4)/(d*s)
    return (a + b + p - 3) % p;
  }();
  REQUIRE_EQ(out[0], expected);
  REQUIRE_EQ(scratch.zero_dens, std::uint64_t(0));
}

// A Scratch holds the power table for one point; when a worker keeps one across
// evaluators (firefly_global.h's `scratch_of` is thread_local), the rows must be sized
// for the evaluator that is about to read them.  Sizing them once, for whichever
// evaluator ran first, leaves a smaller system's rows behind for a deeper one and reads
// the power table out of bounds.
TEST(symbolic_evaluator_resizes_a_reused_scratch_for_a_deeper_system) {
  const std::uint64_t p = 1125899906842597ULL;
  secred::SymbolicFFEvaluator small = evaluator_of();
  secred::SymbolicFFEvaluator big = big_evaluator();
  small.reduce_for_prime(p, 1);
  big.reduce_for_prime(p, 1);
  REQUIRE(small.all_precompiled());
  REQUIRE(big.all_precompiled());

  secred::SymbolicFFEvaluator::Scratch scratch;   // one scratch, both evaluators
  const std::uint64_t x[2] = {5, 7};
  std::uint64_t out[1] = {0};
  REQUIRE(small.evaluate_point(p, x, scratch, out) ==
          secred::SymbolicFFEvaluator::PointResult::Ok);

  out[0] = 0;
  REQUIRE(big.evaluate_point(p, x, scratch, out) ==
          secred::SymbolicFFEvaluator::PointResult::Ok);
  // -(d^4/s) at d = 5, s = 7, i.e. -625/7 mod p
  const std::uint64_t expected = [&] {
    __int128 t = 0, newt = 1, r = p, newr = 7;
    while (newr) {
      const __int128 q = r / newr;
      std::swap(t -= q * newt, newt);
      std::swap(r -= q * newr, newr);
    }
    const std::uint64_t inv7 = static_cast<std::uint64_t>((t % p + p) % p);
    const std::uint64_t q625 = static_cast<std::uint64_t>(
        (static_cast<__int128>(625) * inv7) % p);
    return (p - q625) % p;
  }();
  REQUIRE_EQ(out[0], expected);

  // `evaluator_of()` again, with the deeper rows now in the scratch: the shallow
  // evaluator must keep working on the table it shares.
  out[0] = 0;
  REQUIRE(small.evaluate_point(p, x, scratch, out) ==
          secred::SymbolicFFEvaluator::PointResult::Ok);
  REQUIRE(out[0] != std::uint64_t(0));
}

// Spec §7: a point evaluated at a prime the tables were not reduced for must report
// NotReduced, never a residue computed against the wrong modulus.  (The guard is
// `eval_sym_poly_mod`'s `poly.mod_p != p`, sym_coeff.h:170.)
TEST(symbolic_evaluator_reports_a_prime_it_was_not_reduced_for) {
  secred::SymbolicFFEvaluator ev = evaluator_of();
  const std::uint64_t p = 1125899906842597ULL;
  ev.reduce_for_prime(p, 1);
  secred::SymbolicFFEvaluator::Scratch scratch;
  const std::uint64_t x[2] = {5, 7};
  std::uint64_t out[1] = {0};
  REQUIRE(ev.evaluate_point(p - 2, x, scratch, out) ==
          secred::SymbolicFFEvaluator::PointResult::NotReduced);
  // Reducing for the new prime makes the same point evaluable.
  ev.reduce_for_prime(p - 2, 1);
  REQUIRE(ev.evaluate_point(p - 2, x, scratch, out) ==
          secred::SymbolicFFEvaluator::PointResult::Ok);
}

// Spec §7: a right-hand side that is not a master is reported, not silently dropped --
// a truncated reduction is a wrong answer, and this is the check that catches it.
TEST(symbolic_evaluator_reports_a_non_master_rhs) {
  // `[0,0]` (id 0) sits on every right-hand side here; declaring no masters at all
  // makes each of them a non-master term.
  secred::SymbolicFFEvaluator ev = evaluator_of({});
  const std::uint64_t p = 1125899906842597ULL;
  ev.reduce_for_prime(p, 1);
  secred::SymbolicFFEvaluator::Scratch scratch;
  const std::uint64_t x[2] = {5, 7};
  std::uint64_t out[1] = {0};
  REQUIRE(ev.evaluate_point(p, x, scratch, out) ==
          secred::SymbolicFFEvaluator::PointResult::NonMasterRhs);
  REQUIRE_EQ(scratch.fail_target, 3u);
  REQUIRE_EQ(scratch.fail_integral, 0u);
}

TEST(symbolic_global_learn_picks_the_same_selection_every_time) {
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d"),
                                                 GiNaC::possymbol("s")};
  auto build = [&]() {
    return std::vector<secred::SymEq>{
        eq_of({{3, coeff("1", symbols)}, {1, coeff("(-2+d)/d", symbols)},
               {2, coeff("1/d", symbols)}, {0, coeff("3", symbols)}}),
        eq_of({{2, coeff("1", symbols)}, {0, coeff("(d-4)/s", symbols)}}),
        eq_of({{1, coeff("1", symbols)}, {0, coeff("(2*d)/s", symbols)}})};
  };
  const std::vector<RawIntegral> integrals = {RawIntegral(std::vector<int>{0, 0}),
                                              RawIntegral(std::vector<int>{1, 0}),
                                              RawIntegral(std::vector<int>{0, 1}),
                                              RawIntegral(std::vector<int>{1, 1})};
  std::vector<secred::SymEq> a = build(), b = build();
  const secred::SymbolicGlobalLearn la =
      secred::symbolic_global_learn(a, {3}, integrals, symbols, 1125899906842597ULL,
                                    "/tmp/secred_test_masters_final", "[test]", 0, 0, 0, 0);
  const secred::SymbolicGlobalLearn lb =
      secred::symbolic_global_learn(b, {3}, integrals, symbols, 1125899906842597ULL,
                                    "/tmp/secred_test_masters_final", "[test]", 0, 0, 0, 0);
  REQUIRE(la.ok);
  REQUIRE_EQ(la.nnindeps, 1u);                        // one master: [0,0]
  REQUIRE_EQ(la.dep_targets.size(), std::size_t(1));
  REQUIRE_EQ(la.needed_eq_idx.size(), std::size_t(3));
  REQUIRE_EQ(la.masters, lb.masters);                 // deterministic, no clock, no rng
  REQUIRE_EQ(la.needed_eq_idx, lb.needed_eq_idx);
  REQUIRE_EQ(la.bwd_chain, lb.bwd_chain);
}

TEST(symbolic_global_learn_reports_a_target_that_is_a_master) {
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d")};
  std::vector<secred::SymEq> system = {eq_of({{1, coeff("1", symbols)},
                                              {0, coeff("d", symbols)}})};
  const std::vector<RawIntegral> integrals = {RawIntegral(std::vector<int>{0}),
                                              RawIntegral(std::vector<int>{1})};
  const secred::SymbolicGlobalLearn l =
      secred::symbolic_global_learn(system, {0}, integrals, symbols, 1125899906842597ULL,
                                    "/tmp/secred_test_masters_final", "[test]", 0, 0, 0, 0);
  REQUIRE(l.ok);
  REQUIRE_EQ(l.dep_targets.size(), std::size_t(0));   // [0] never becomes a pivot
  REQUIRE_EQ(l.nnindeps, 0u);
}
