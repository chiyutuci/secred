// The FireFly black box: the templated probe path, the per-prime table reduction, the
// threading contract, and the degenerate systems the driver must not hand to FireFly.
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ginac/ginac.h>

#include <firefly/FFInt.hpp>
#include <firefly/FFIntVec.hpp>

#include "firefly_global.h"
#include "globalreduce.h"
#include "spill_reader.h"
#include "symbolic_ffred.h"
#include "test_runner.h"
#include "tiny_symbolic_system.h"

namespace {

// The evaluator over the tiny system, built through the shared learn exactly as the
// driver builds it.  `masters_override` replaces the learned master map, which is how
// the failure test makes a target reduce to something that is not a master.
secred::SymbolicFFEvaluator
evaluator_of(std::unordered_map<unsigned, unsigned> *masters_override = nullptr) {
  const std::string spill = tiny::write_spill("/tmp/secred_test_firefly_tiny");
  secred::SymCoeffPool pool;
  secred::SpillReadStats stats;
  std::string error;
  const auto ids = tiny::integral_ids();
  const auto symbols = tiny::symbols();
  const auto integrals = tiny::integrals();
  if (!secred::read_spill_interned(spill, ids, symbols, pool, stats, error))
    throw std::runtime_error(error);
  secred::SymbolicGlobalLearn learn =
      secred::symbolic_global_learn(pool, {3u}, integrals, symbols,
                                    1125899906842597ULL,
                                    "/tmp/secred_test_firefly_masters", "[test]", 0, 0, 0, 0);
  if (!learn.ok)
    throw std::runtime_error(learn.error);
  std::unordered_map<unsigned, unsigned> master_index =
      masters_override ? *masters_override : learn.master_index;
  std::vector<GiNaC::ex> symbol_ex;
  for (const auto &s : symbols)
    symbol_ex.push_back(GiNaC::ex(s));
  const unsigned nnindeps = masters_override ? 1u : learn.nnindeps;
  // The item table comes from the learn exactly as the driver passes it; with overridden
  // masters the learned one would not match, so the (single) learned item is kept as is.
  secred::ReductionItems items = learn.items;
  return secred::SymbolicFFEvaluator(std::move(pool), std::move(learn.needed_eq_idx),
                                     std::move(learn.bwd_chain), std::move(learn.dep_targets),
                                     std::move(master_index), nnindeps,
                                     std::move(items), std::move(symbol_ex));
}

// `prime_changed()` reads `firefly::FFInt::p`, so the tests set the prime the way
// FireFly does: `set_new_prime` on the global, then the box's hook.
void promote_prime(std::uint64_t p) {
  firefly::FFInt::set_new_prime(p);
}

}  // namespace

TEST(firefly_box_returns_the_evaluators_values) {
  const std::uint64_t p = firefly::FFInt::p;   // what FireFly starts from
  secred::SymbolicFFEvaluator ev = evaluator_of();
  secred::SecRedFireFlyBlackBox box(ev, 1);
  box.prime_changed();
  std::vector<firefly::FFInt> x = {firefly::FFInt(5), firefly::FFInt(7)};
  const std::vector<firefly::FFInt> got = box(x);
  REQUIRE_EQ(got.size(), std::size_t(1));
  secred::SymbolicFFEvaluator::Scratch scratch;
  std::uint64_t ref[1] = {0};
  const std::uint64_t xv[2] = {5, 7};
  REQUIRE(ev.evaluate_point(p, xv, scratch, ref) ==
          secred::SymbolicFFEvaluator::PointResult::Ok);
  REQUIRE_EQ(got[0].n, ref[0]);
  REQUIRE_EQ(box.probes(), std::uint64_t(1));
  REQUIRE_EQ(box.points(), std::uint64_t(1));
  REQUIRE_EQ(box.primes(), std::uint64_t(1));
  REQUIRE(!box.hard_failure());
  REQUIRE(box.fail_reason() == secred::SymbolicFFEvaluator::PointResult::Ok);
}

// The box adds each probe's share of the three solve phases (the scratch is thread-local
// and outlives the probe, so what it reads is a delta), which the driver averages into
// the Kira-style "probe solution steps" line.
TEST(firefly_box_reports_the_probe_phase_seconds) {
  secred::SymbolicFFEvaluator ev = evaluator_of();
  secred::SecRedFireFlyBlackBox box(ev, 1);
  box.prime_changed();
  REQUIRE_EQ(box.eval_seconds(), 0.0);
  const std::vector<firefly::FFInt> x = {firefly::FFInt(5), firefly::FFInt(7)};
  box(x);
  REQUIRE(box.eval_seconds() > 0.0);   // fill + substitute + reduce
  REQUIRE(box.fwd_seconds() > 0.0);    // forward elimination
  REQUIRE(box.bwd_seconds() > 0.0);    // back substitution
}

// The lane trait: a bunched call must return, lane by lane, what N single calls return.
// FireFly only ever uses this path with bunch_size 1, but the template is instantiated
// for every bunch size, and a trait that silently mis-indexes would reconstruct garbage
// if a future driver ever asked for a bunch.
TEST(firefly_box_bunch_lanes_match_single_calls) {
  const std::uint64_t p = firefly::FFInt::p;
  secred::SymbolicFFEvaluator ev = evaluator_of();
  secred::SecRedFireFlyBlackBox box(ev, 1);
  box.prime_changed();
  // Four points for one output: lane l of the vector must equal the scalar probe.
  const std::uint64_t pts[4][2] = {{5, 7}, {11, 13}, {17, 19}, {23, 29}};
  const std::vector<firefly::FFIntVec<4>> bunched = box(
      std::vector<firefly::FFIntVec<4>>{
          firefly::FFIntVec<4>(std::array<firefly::FFInt, 4>{
              firefly::FFInt(pts[0][0]), firefly::FFInt(pts[1][0]),
              firefly::FFInt(pts[2][0]), firefly::FFInt(pts[3][0])}),
          firefly::FFIntVec<4>(std::array<firefly::FFInt, 4>{
              firefly::FFInt(pts[0][1]), firefly::FFInt(pts[1][1]),
              firefly::FFInt(pts[2][1]), firefly::FFInt(pts[3][1])})});
  REQUIRE_EQ(bunched.size(), std::size_t(1));
  for (unsigned l = 0; l < 4; ++l) {
    std::vector<firefly::FFInt> point = {firefly::FFInt(pts[l][0]),
                                         firefly::FFInt(pts[l][1])};
    // A scalar probe re-reads the same prime, so no re-reduction happens between them.
    REQUIRE_EQ(box(point)[0].n, bunched[0].vec[l].n);
  }
  REQUIRE_EQ(box.probes(), std::uint64_t(5));
}

// Spec 7: probes running while the prime moves must each see one prime's tables.
// Phase 1 hammers the box at one prime through 8 threads; phase 2 promotes the prime
// underneath them; every probe in both phases is compared against the reference value at
// one of the two primes -- a probe equal to neither is the failure this test exists for.
TEST(firefly_box_probes_never_mix_two_primes) {
  secred::SymbolicFFEvaluator ev = evaluator_of();
  secred::SecRedFireFlyBlackBox box(ev, 4);
  const std::uint64_t p1 = firefly::FFInt::p - 1000;   // force a promotion below
  promote_prime(p1);
  box.prime_changed();

  std::uint64_t ref1[1] = {0}, ref2[1] = {0};
  const std::uint64_t xv[2] = {5, 7};
  {
    secred::SymbolicFFEvaluator::Scratch s;
    REQUIRE(ev.evaluate_point(p1, xv, s, ref1) ==
            secred::SymbolicFFEvaluator::PointResult::Ok);
  }
  const std::uint64_t p2 = p1 - 1000;
  promote_prime(p2);
  {
    // A second evaluator so the reference for p2 is computed without disturbing `ev`'s
    // tables -- `reduce_for_prime` is exactly what a promotion does.
    secred::SymbolicFFEvaluator ev2 = evaluator_of();
    ev2.reduce_for_prime(p2, 1);
    secred::SymbolicFFEvaluator::Scratch s;
    REQUIRE(ev2.evaluate_point(p2, xv, s, ref2) ==
            secred::SymbolicFFEvaluator::PointResult::Ok);
  }
  promote_prime(p1);
  box.prime_changed();   // back to p1: the threads below start from a known state

  std::vector<firefly::FFInt> x = {firefly::FFInt(5), firefly::FFInt(7)};
  std::atomic<bool> stop{false};
  std::atomic<int> bad{0};
  std::vector<std::thread> ts;
  for (unsigned t = 0; t < 8; ++t)
    ts.emplace_back([&]() {
      while (!stop.load(std::memory_order_relaxed)) {
        const std::uint64_t v = box(x)[0].n;
        if (v != ref1[0] && v != ref2[0])
          ++bad;
      }
    });
  for (unsigned i = 0; i < 20; ++i) {   // promote back and forth under the probes
    promote_prime(p2);
    box.prime_changed();
    promote_prime(p1);
    box.prime_changed();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  stop.store(true);
  for (auto &t : ts)
    t.join();
  REQUIRE_EQ(bad.load(), 0);
  REQUIRE(!box.hard_failure());
}

TEST(firefly_box_reports_a_stale_selection_as_a_hard_failure) {
  // No masters at all: [0,0] sits on every right-hand side and matches none of them, so
  // the reduction would be truncated.  The box must latch the failure and return zeros.
  std::unordered_map<unsigned, unsigned> no_masters;
  secred::SymbolicFFEvaluator ev = evaluator_of(&no_masters);
  secred::SecRedFireFlyBlackBox box(ev, 1);
  box.prime_changed();
  std::vector<firefly::FFInt> x = {firefly::FFInt(5), firefly::FFInt(7)};
  const std::vector<firefly::FFInt> got = box(x);
  REQUIRE(box.hard_failure());
  REQUIRE_EQ(got[0].n, std::uint64_t(0));
  REQUIRE(box.fail_reason() == secred::SymbolicFFEvaluator::PointResult::NonMasterRhs);
  REQUIRE_EQ(box.fail_target(), 3u);
  REQUIRE_EQ(box.fail_integral(), 0u);
}

TEST(firefly_box_handles_a_zero_output_system) {
  // n_outputs() == 0 (every target is a master): the probe returns an empty vector and
  // the driver never constructs a Reconstructor with n == 0.
  const std::string spill = tiny::write_spill("/tmp/secred_test_firefly_zero");
  secred::SymCoeffPool pool;
  secred::SpillReadStats stats;
  std::string error;
  const auto ids = tiny::integral_ids();
  const auto symbols = tiny::symbols();
  const auto integrals = tiny::integrals();
  REQUIRE(secred::read_spill_interned(spill, ids, symbols, pool, stats, error));
  secred::SymbolicGlobalLearn learn =
      secred::symbolic_global_learn(pool, {0u}, integrals, symbols,
                                    1125899906842597ULL,
                                    "/tmp/secred_test_firefly_masters0", "[test]", 0, 0, 0, 0);
  REQUIRE(learn.ok);
  REQUIRE_EQ(learn.dep_targets.size(), std::size_t(0));
  std::vector<GiNaC::ex> symbol_ex;
  for (const auto &s : symbols)
    symbol_ex.push_back(GiNaC::ex(s));
  secred::SymbolicFFEvaluator ev(std::move(pool), std::move(learn.needed_eq_idx),
                                 std::move(learn.bwd_chain), std::move(learn.dep_targets),
                                 std::move(learn.master_index), 0u,
                                 std::move(learn.items), std::move(symbol_ex));
  REQUIRE_EQ(ev.n_outputs(), 0u);
  secred::SecRedFireFlyBlackBox box(ev, 1);
  box.prime_changed();
  const std::vector<firefly::FFInt> x = {firefly::FFInt(5), firefly::FFInt(7)};
  REQUIRE_EQ(box(x).size(), std::size_t(0));
  REQUIRE(!box.hard_failure());
}

// G1 at unit scale: the two symbolic backends must return the same reduction of the same
// system.  The selection is shared and deterministic (Task 6), so a difference here is a
// difference between reconstruction engines.
TEST(firefly_and_finiteflow_agree_on_the_tiny_system) {
  const std::string spill = tiny::write_spill("/tmp/secred_test_tiny_spill");
  const auto integrals = tiny::integrals();
  const auto ids = tiny::integral_ids();
  const auto symbols = tiny::symbols();
  const std::vector<RawIntegral> targets = {integrals[3]};
  const auto run = [&](auto &&fn) {
    return fn(spill, integrals, ids, targets, symbols, 0u, 0u, 0u, std::size_t(0),
              "/tmp/secred_test_tiny_masters", 2u);
  };
  const GlobalReductionResult ff = run(
      [](auto &&...args) { return reduce_global_finiteflow_symbolic(args...); });
  const GlobalReductionResult fly = run(
      [](auto &&...args) { return secred::reduce_global_firefly_symbolic(args...); });
  REQUIRE(ff.ok);
  REQUIRE(fly.ok);
  REQUIRE_EQ(ff.masters.size(), fly.masters.size());
  REQUIRE(ff.masters == fly.masters);
  REQUIRE_EQ(ff.reductions.size(), fly.reductions.size());
  // The two engines spell a rational function differently ((1/2*d^2+3/2)/(1/2*d+1) vs
  // (d^2+3)/(d+2)), so compare values: exact rational arithmetic at a few points.
  GiNaC::symtab tab;
  for (const auto &s : symbols)
    tab[s.get_name()] = s;
  GiNaC::parser parser(tab);
  for (std::size_t i = 0; i < ff.reductions.size(); ++i) {
    REQUIRE(ff.reductions[i].first == fly.reductions[i].first);
    if (ff.reductions[i].second == "-1")
      continue;
    const GiNaC::ex a = parser(ff.reductions[i].second);
    const GiNaC::ex b = parser(fly.reductions[i].second);
    for (const auto &pt : {std::pair<int, int>{3, 5}, {7, 11}, {-2, 13}}) {
      GiNaC::exmap sub;
      sub[symbols[0]] = GiNaC::numeric(pt.first);
      sub[symbols[1]] = GiNaC::numeric(pt.second);
      REQUIRE(GiNaC::ex_to<GiNaC::numeric>(a.subs(sub))
                  .is_equal(GiNaC::ex_to<GiNaC::numeric>(b.subs(sub))));
    }
  }
}

// Review Focus 4: every target is already a master -> zero outputs.  FireFly's
// Reconstructor is fatal with n == 0, so the driver must return before constructing it.
TEST(firefly_driver_handles_a_zero_output_system) {
  const std::string spill = tiny::write_spill("/tmp/secred_test_tiny_spill0");
  const auto integrals = tiny::integrals();
  const auto ids = tiny::integral_ids();
  const auto symbols = tiny::symbols();
  const std::vector<RawIntegral> targets = {integrals[0]};   // never a pivot
  const GlobalReductionResult r = secred::reduce_global_firefly_symbolic(
      spill, integrals, ids, targets, symbols, 0u, 0u, 0u, std::size_t(0),
      "/tmp/secred_test_tiny_masters0", 2u);
  REQUIRE(r.ok);
  REQUIRE_EQ(r.reductions.size(), std::size_t(1));   // just the (target, "-1") marker
  REQUIRE_EQ(r.reductions[0].second, std::string("-1"));
}

// A global phase that did not complete (`ok == false`) returns an empty or truncated
// table.  Written to cache/reductions and exited 0, that is a silently wrong result:
// the run failed, so it has to stop the run.  family.cpp calls this before it writes
// anything, and main.cpp turns the throw into a non-zero exit.
TEST(a_failed_global_reduction_is_fatal_not_an_empty_table) {
  const auto integrals = tiny::integrals();
  const auto ids = tiny::integral_ids();
  const auto symbols = tiny::symbols();
  const std::vector<RawIntegral> targets = {integrals[3]};

  // A spill that is not there: the FireFly backend reports the read failure the way a
  // refused probe is reported, as ok == false with nothing reconstructed.
  const GlobalReductionResult failed = secred::reduce_global_firefly_symbolic(
      "/tmp/secred_test_no_such_spill", integrals, ids, targets, symbols, 0u, 0u, 0u,
      std::size_t(0), "/tmp/secred_test_tiny_masters_req", 2u);
  REQUIRE(!failed.ok);
  REQUIRE(failed.reductions.empty());

  std::string message;
  try {
    secred::require_global_reduction(failed, "firefly");
  } catch (const std::runtime_error &e) {
    message = e.what();
  }
  REQUIRE(message.find("firefly") != std::string::npos);        // names the tool asked for
  REQUIRE(message.find("did not complete cleanly") != std::string::npos);

  // A phase that did complete passes the check unchanged.  (The FiniteFlow symbolic
  // path on the same tiny system: one FireFly reconstruction is all this process does,
  // since FireFly's sampling stream continues from the previous Reconstructor and the
  // tiny system has a degenerate sampling line -- see the ledger note.)
  const std::string spill = tiny::write_spill("/tmp/secred_test_tiny_spill_req");
  const GlobalReductionResult done = reduce_global_finiteflow_symbolic(
      spill, integrals, ids, targets, symbols, 0u, 0u, 0u, std::size_t(0),
      "/tmp/secred_test_tiny_masters_req", 2u);
  REQUIRE(done.ok);
  secred::require_global_reduction(done, "firefly");   // must not throw
  REQUIRE(!done.reductions.empty());   // what the failed phase does not have to offer
}
