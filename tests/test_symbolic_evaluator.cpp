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

// The item table the learn builds for one target: an entry for every master on the
// target's right-hand side, in descending id, with the row boundaries in `begin`.
secred::ReductionItems items_of(
    const std::unordered_map<unsigned, unsigned> &master_index) {
  secred::ReductionItems items;
  items.begin.push_back(0);
  if (master_index.count(0)) {
    items.master.push_back(0);
    items.col.push_back(master_index.at(0));
  }
  items.begin.push_back(static_cast<std::uint32_t>(items.master.size()));
  return items;
}

// The system the evaluator tests share, kept separate from the evaluator so the tests
// can drive the per-entry evaluation path over the same equations.
std::vector<secred::SymEq> raw_system() {
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d"),
                                                 GiNaC::possymbol("s")};
  return {
      eq_of({{3, coeff("1", symbols)}, {1, coeff("(-2+d)/d", symbols)},
             {2, coeff("1/d", symbols)}, {0, coeff("3", symbols)}}),
      eq_of({{2, coeff("1", symbols)}, {0, coeff("(d-4)/s", symbols)}}),
      eq_of({{1, coeff("1", symbols)}, {0, coeff("(2*d)/s", symbols)}}),
  };
}

// The same system with the master set given by `masters` (master id -> column).  The
// default is the correct one: [0,0] alone.  `carry_items` false gives the target an
// empty item row -- the table a learn point that carried nothing on that right-hand
// side would leave behind.
secred::SymbolicFFEvaluator evaluator_of(
    std::unordered_map<unsigned, unsigned> master_index = {{0, 0}},
    bool carry_items = true) {
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d"),
                                                 GiNaC::possymbol("s")};
  std::vector<secred::SymEq> system = raw_system();
  std::vector<std::uint32_t> eq_idx = {0, 1, 2};
  std::vector<std::uint32_t> bwd_chain = {1, 2, 3};
  std::vector<unsigned> dep_targets = {3};
  std::vector<GiNaC::ex> symbol_ex = {GiNaC::ex(symbols[0]), GiNaC::ex(symbols[1])};
  const unsigned nnindeps = master_index.empty() ? 1u
                             : static_cast<unsigned>(master_index.size());
  secred::ReductionItems items = carry_items ? items_of(master_index)
                                             : secred::ReductionItems{{}, {}, {0, 0}};
  secred::SymCoeffPool pool;   // the whole system; eq_idx names the rows to keep
  secred::intern_sym_system(system, pool);
  return secred::SymbolicFFEvaluator(std::move(pool), std::move(eq_idx),
                                     std::move(bwd_chain), std::move(dep_targets),
                                     std::move(master_index), nnindeps,
                                     std::move(items), std::move(symbol_ex));
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
  secred::SymCoeffPool pool;
  secred::intern_sym_system(system, pool);
  return secred::SymbolicFFEvaluator(std::move(pool), {0}, {1}, {1}, {{0, 0}}, 1,
                                     items_of({{0, 0}}), std::move(symbol_ex));
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

// A master with a non-zero coefficient that the learn point did not carry on that
// right-hand side: the item list has no slot for the term, so the output would be a
// truncated reduction.  Reported, never silently dropped.
TEST(symbolic_evaluator_reports_an_item_the_learn_point_missed) {
  secred::SymbolicFFEvaluator ev =
      evaluator_of({{0, 0}}, /*carry_items=*/false);   // [0,0] is a master, but no item
  REQUIRE_EQ(ev.n_outputs(), 0u);
  const std::uint64_t p = 1125899906842597ULL;
  ev.reduce_for_prime(p, 1);
  secred::SymbolicFFEvaluator::Scratch scratch;
  const std::uint64_t x[2] = {5, 7};
  std::uint64_t out[1] = {0};
  REQUIRE(ev.evaluate_point(p, x, scratch, out) ==
          secred::SymbolicFFEvaluator::PointResult::MissingItem);
  REQUIRE_EQ(scratch.fail_target, 3u);
  REQUIRE_EQ(scratch.fail_integral, 0u);
}

// The output block is the item list, not the target x master block: n_outputs() is its
// length, and an empty list means the reconstruction has nothing to do.  (Every cell the
// item list leaves out is a structurally zero coefficient, which FireFly would otherwise
// interpolate an explicit zero for.)
TEST(symbolic_evaluator_outputs_are_the_item_list) {
  REQUIRE_EQ(evaluator_of().n_outputs(), 1u);                     // one master, one item
  REQUIRE_EQ(evaluator_of({{0, 0}}, false).n_outputs(), 0u);      // no items
}

// The three per-phase counters the driver turns into its "probe solution steps"
// report line.  A scratch outlives one evaluator, so the counters accumulate and the
// caller reads deltas.
TEST(symbolic_evaluator_counts_the_three_probe_phases) {
  secred::SymbolicFFEvaluator ev = evaluator_of();
  const std::uint64_t p = 1125899906842597ULL;
  ev.reduce_for_prime(p, 1);
  secred::SymbolicFFEvaluator::Scratch scratch;
  const std::uint64_t x[2] = {5, 7};
  std::uint64_t out[1] = {0};
  REQUIRE_EQ(scratch.eval_seconds, 0.0);
  REQUIRE_EQ(scratch.fwd_seconds, 0.0);
  REQUIRE_EQ(scratch.bwd_seconds, 0.0);
  REQUIRE(ev.evaluate_point(p, x, scratch, out) ==
          secred::SymbolicFFEvaluator::PointResult::Ok);
  REQUIRE(scratch.eval_seconds > 0.0);   // fill + substitute + reduce
  REQUIRE(scratch.fwd_seconds > 0.0);    // forward elimination
  REQUIRE(scratch.bwd_seconds > 0.0);    // back substitution
  const double first = scratch.eval_seconds;
  REQUIRE(ev.evaluate_point(p, x, scratch, out) ==
          secred::SymbolicFFEvaluator::PointResult::Ok);
  REQUIRE(scratch.eval_seconds > first);   // the second point adds to the same counters
}

// The reconstruction evaluates each *distinct* function once per sample point and reads
// every other entry out of the table (Kira's model: "Parsed 8040387 functions of which
// 1617 are distinct"), so the evaluator reports what the table costs: one entry per
// equation term, one slot per distinct function.  The tiny system has 8 terms -- the
// constant `1` stands in all three equations -- and six distinct functions.
TEST(symbolic_evaluator_interns_the_system_coefficients) {
  secred::SymbolicFFEvaluator ev = evaluator_of();
  REQUIRE_EQ(ev.n_equations(), std::size_t(3));
  REQUIRE_EQ(ev.n_coefficient_terms(), std::size_t(8));
  REQUIRE_EQ(ev.n_distinct_coeffs(), std::size_t(6));
}

// The interned path must be the per-entry path: the table holds exactly the residues the
// entries would evaluate, and the elimination that reads them solves to the same map.
TEST(symbolic_evaluator_matches_the_per_entry_evaluation) {
  const std::uint64_t p = 1125899906842597ULL;
  const std::uint64_t x[2] = {5, 7};

  // Reference: every entry evaluated on its own, eliminated, back-substituted.
  std::vector<secred::SymEq> raw = raw_system();
  secred::reduce_sym_system(raw, p);
  const std::vector<std::vector<uint64_t>> pw = secred::make_powers(raw, x, 2, p);
  std::vector<secred::ResEq> res_eqs;
  std::vector<std::uint32_t> sys_idx;
  REQUIRE(secred::eval_sym_eqs_mod(raw, {0u, 1u, 2u}, GiNaC::exmap(), pw, p, res_eqs,
                                   sys_idx));
  std::unordered_map<std::uint32_t, secred::ResEq> want;
  secred::forward_eliminate(res_eqs, sys_idx, p, want, nullptr, nullptr);
  const std::vector<std::uint32_t> chain = {1, 2, 3};   // the evaluator's bwd_chain_
  secred::backward_eliminate(want, p, nullptr, &chain);

  secred::SymbolicFFEvaluator ev = evaluator_of();
  ev.reduce_for_prime(p, 1);
  secred::SymbolicFFEvaluator::Scratch scratch;
  std::uint64_t out[1] = {0};
  REQUIRE(ev.evaluate_point(p, x, scratch, out) ==
          secred::SymbolicFFEvaluator::PointResult::Ok);
  // Same pivots, same right-hand sides: the evaluator's store against the reference map.
  std::size_t n_rows = 0;
  for (std::uint32_t id = 0; id < 4u; ++id) {
    const auto it = want.find(id);
    REQUIRE_EQ(scratch.store.has_row(id), it != want.end());
    if (it == want.end())
      continue;
    ++n_rows;
    secred::ResEq row;
    for (std::uint32_t i = 0; i < scratch.store.row_size(id); ++i)
      row.emplace_back(scratch.store.row_ids(id)[i], scratch.store.row_vals(id)[i]);
    REQUIRE(row == it->second);
  }
  REQUIRE_EQ(n_rows, want.size());
  // The point holds one residue per distinct function -- the pool's -- and no numeric copy
  // of the system: the elimination reads the rows out of the pool's table.
  REQUIRE_EQ(scratch.values.size(), ev.n_distinct_coeffs());
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
  secred::SymCoeffPool pa, pb;
  secred::intern_sym_system(a, pa);
  secred::intern_sym_system(b, pb);
  const secred::SymbolicGlobalLearn la =
      secred::symbolic_global_learn(pa, {3}, integrals, symbols, 1125899906842597ULL,
                                    "/tmp/secred_test_masters_global", "[test]", 0, 0, 0, 0);
  const secred::SymbolicGlobalLearn lb =
      secred::symbolic_global_learn(pb, {3}, integrals, symbols, 1125899906842597ULL,
                                    "/tmp/secred_test_masters_global", "[test]", 0, 0, 0, 0);
  REQUIRE(la.ok);
  REQUIRE_EQ(la.nnindeps, 1u);                        // one master: [0,0]
  REQUIRE_EQ(la.dep_targets.size(), std::size_t(1));
  REQUIRE_EQ(la.needed_eq_idx.size(), std::size_t(3));
  REQUIRE_EQ(la.masters, lb.masters);                 // deterministic, no clock, no rng
  REQUIRE_EQ(la.needed_eq_idx, lb.needed_eq_idx);
  REQUIRE_EQ(la.bwd_chain, lb.bwd_chain);
  REQUIRE(la.items.master == lb.items.master);
  REQUIRE(la.items.col == lb.items.col);
  REQUIRE(la.items.begin == lb.items.begin);
}

// The item table the learn leaves for the reconstruction: one entry per master on the
// target's reduced right-hand side, at the learning point, read straight off the solved
// map the selection check already built.  Its length is what the driver reconstructs.
TEST(symbolic_global_learn_builds_the_item_list) {
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d"),
                                                 GiNaC::possymbol("s")};
  std::vector<secred::SymEq> system = {
      eq_of({{3, coeff("1", symbols)}, {1, coeff("(-2+d)/d", symbols)},
             {2, coeff("1/d", symbols)}, {0, coeff("3", symbols)}}),
      eq_of({{2, coeff("1", symbols)}, {0, coeff("(d-4)/s", symbols)}}),
      eq_of({{1, coeff("1", symbols)}, {0, coeff("(2*d)/s", symbols)}})};
  secred::SymCoeffPool pool;
  secred::intern_sym_system(system, pool);
  const std::vector<RawIntegral> integrals = {RawIntegral(std::vector<int>{0, 0}),
                                              RawIntegral(std::vector<int>{1, 0}),
                                              RawIntegral(std::vector<int>{0, 1}),
                                              RawIntegral(std::vector<int>{1, 1})};
  const secred::SymbolicGlobalLearn l =
      secred::symbolic_global_learn(pool, {3}, integrals, symbols, 1125899906842597ULL,
                                    "/tmp/secred_test_masters_global", "[test]", 0, 0, 0, 0);
  REQUIRE(l.ok);
  REQUIRE_EQ(l.dep_targets.size(), std::size_t(1));
  REQUIRE_EQ(l.items.begin.size(), l.dep_targets.size() + 1);   // one row boundary each
  REQUIRE_EQ(l.items.size(), std::size_t(1));   // [1,1] reduces to [0,0] alone
  REQUIRE_EQ(l.items.master[0], 0u);
  REQUIRE_EQ(l.items.col[0], 0u);               // the column [0,0] has in the basis
  REQUIRE_EQ(l.items.begin[0], 0u);
  REQUIRE_EQ(l.items.begin[1], 1u);
}

TEST(symbolic_global_learn_reports_a_target_that_is_a_master) {
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d")};
  std::vector<secred::SymEq> system = {eq_of({{1, coeff("1", symbols)},
                                              {0, coeff("d", symbols)}})};
  secred::SymCoeffPool pool;
  secred::intern_sym_system(system, pool);
  const std::vector<RawIntegral> integrals = {RawIntegral(std::vector<int>{0}),
                                              RawIntegral(std::vector<int>{1})};
  const secred::SymbolicGlobalLearn l =
      secred::symbolic_global_learn(pool, {0}, integrals, symbols, 1125899906842597ULL,
                                    "/tmp/secred_test_masters_global", "[test]", 0, 0, 0, 0);
  REQUIRE(l.ok);
  REQUIRE_EQ(l.dep_targets.size(), std::size_t(0));   // [0] never becomes a pivot
  REQUIRE_EQ(l.nnindeps, 0u);
}

// The whole-pool planned solve the learn runs: eval_pool_point's slot array plus the
// pool's own term table, against the reference forward/backward elimination on the same
// evaluated rows -- all four maps, at a generic point and at one where a coefficient
// vanishes (the plan's order is then rebuilt from the evaluated rows).
TEST(pool_planned_solve_matches_the_reference) {
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d"),
                                                 GiNaC::possymbol("s")};
  std::vector<secred::SymEq> system = raw_system();
  secred::SymCoeffPool pool;
  secred::intern_sym_system(system, pool);
  const std::uint64_t p = 1125899906842597ULL;
  secred::reduce_pool_for_prime(pool, p);
  const std::vector<std::uint32_t> all = {0, 1, 2};
  const std::uint64_t points[2][2] = {{5, 7}, {4, 7}};   // d = 4 kills (d-4)/s
  for (const auto &x : points) {
    GiNaC::exmap subs;
    subs[symbols[0]] = GiNaC::numeric(x[0]);
    subs[symbols[1]] = GiNaC::numeric(x[1]);
    const std::vector<std::vector<std::uint64_t>> powers =
        secred::make_powers(pool, x, 2, p);
    std::vector<std::uint64_t> values;
    REQUIRE(secred::eval_pool_point(pool, subs, powers, p, values));

    secred::SubSolve got;
    secred::solve_pool_planned(pool, values, p, got);

    std::vector<secred::ResEq> res_eqs;
    std::vector<std::uint32_t> sys_idx;
    REQUIRE(secred::eval_pool_rows(pool, all, subs, powers, p, res_eqs, sys_idx));
    secred::SubSolve want;
    secred::forward_eliminate(res_eqs, sys_idx, p, want.solmap, &want.eq_index,
                              &want.fwd_deps);
    secred::backward_eliminate(want.solmap, p, &want.bwd_deps);

    REQUIRE(got.solmap == want.solmap);
    REQUIRE(got.eq_index == want.eq_index);
    REQUIRE(got.fwd_deps == want.fwd_deps);
    REQUIRE(got.bwd_deps == want.bwd_deps);
  }
}

// A row whose only coefficient vanishes at the point is dropped by the reference before
// the elimination; the planned solve must not register it either (it walks every plan
// row, including the empty ones).
TEST(pool_planned_solve_drops_a_row_that_vanishes) {
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d")};
  std::vector<secred::SymEq> system = {
      eq_of({{1, coeff("d-4", symbols)}}),
      eq_of({{1, coeff("1", symbols)}, {0, coeff("3", symbols)}})};
  secred::SymCoeffPool pool;
  secred::intern_sym_system(system, pool);
  const std::uint64_t p = 1125899906842597ULL;
  secred::reduce_pool_for_prime(pool, p);
  const std::uint64_t x[1] = {4};   // the first row's only coefficient vanishes
  GiNaC::exmap subs;
  subs[symbols[0]] = GiNaC::numeric(x[0]);
  const std::vector<std::vector<std::uint64_t>> powers =
      secred::make_powers(pool, x, 1, p);
  std::vector<std::uint64_t> values;
  REQUIRE(secred::eval_pool_point(pool, subs, powers, p, values));

  secred::SubSolve got;
  secred::solve_pool_planned(pool, values, p, got);

  std::vector<secred::ResEq> res_eqs;
  std::vector<std::uint32_t> sys_idx;
  REQUIRE(secred::eval_pool_rows(pool, {0u, 1u}, subs, powers, p, res_eqs, sys_idx));
  secred::SubSolve want;
  secred::forward_eliminate(res_eqs, sys_idx, p, want.solmap, &want.eq_index,
                            &want.fwd_deps);
  secred::backward_eliminate(want.solmap, p, &want.bwd_deps);
  REQUIRE(got.solmap == want.solmap);
  REQUIRE(got.eq_index == want.eq_index);
  REQUIRE(got.fwd_deps == want.fwd_deps);
  REQUIRE(got.bwd_deps == want.bwd_deps);
}

// The elimination counters behind the probe report's fill-in line.  They accumulate
// exactly like the phase timers -- a scratch outlives one evaluator -- and the row count
// is the equations the point eliminated, so the driver can divide by the point count.
TEST(symbolic_evaluator_counts_the_elimination_fill_in) {
  secred::SymbolicFFEvaluator ev = evaluator_of();
  const std::uint64_t p = 1125899906842597ULL;
  ev.reduce_for_prime(p, 1);
  secred::SymbolicFFEvaluator::Scratch scratch;
  const std::uint64_t x[2] = {5, 7};
  std::uint64_t out[1] = {0};
  REQUIRE_EQ(scratch.fwd_stats.rows, std::uint64_t(0));
  REQUIRE(ev.evaluate_point(p, x, scratch, out) ==
          secred::SymbolicFFEvaluator::PointResult::Ok);
  REQUIRE_EQ(scratch.fwd_stats.rows, std::uint64_t(3));   // the three equations
  REQUIRE_EQ(scratch.fwd_stats.pivots, std::uint64_t(3)); // and their three solutions
  REQUIRE(scratch.fwd_stats.gathered > 0);                // E1's terms, then E2's, E3's
  REQUIRE(scratch.fwd_stats.merges > 0);  // E1 substitutes the other two solutions
  REQUIRE(scratch.fwd_stats.moved > 0);   // and pays their fill-in
  const std::uint64_t first = scratch.fwd_stats.moved;
  REQUIRE(ev.evaluate_point(p, x, scratch, out) ==
          secred::SymbolicFFEvaluator::PointResult::Ok);
  REQUIRE(scratch.fwd_stats.moved > first); // the second point adds to the same counters
}
