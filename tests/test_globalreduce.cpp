// The numeric reconstruction's hoisted plan and its per-prime solve: the table
// NumericEvalPlan builds out of the selected equations, and the guarantee that one prime
// through forward_eliminate_planned + backward_eliminate_planned is the reference
// reduce_mod + forward_eliminate + backward_eliminate, term for term.
#include <algorithm>
#include <cstdint>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ffred_solver.h"
#include "globalreduce.h"
#include "test_runner.h"

namespace {

// A small system with ids 0..3 and masters {0, 1}.  Row 3 carries an identically zero
// coefficient, which is what reduce_mod drops at every prime and what the plan must
// therefore drop at build time.
//
//   E0: 1 [3] + 1 [2] + 1 [0]        (id 0 is a master)
//   E1: 1/2 [3] + 1 [1]
//   E2: 1 [2] + 3/2 [1] + 2 [0]
//   E3: 1 [3] + 0/5 [2]
std::vector<secred::Eq> toy_system() {
  auto term = [](std::uint32_t id, std::int64_t num, std::int64_t den) {
    return std::make_pair(id, secred::Rat64{num, den});
  };
  std::vector<secred::Eq> system(4);
  system[0] = {term(3, 1, 1), term(2, 1, 1), term(0, 1, 1)};
  system[1] = {term(3, 1, 2), term(1, 1, 1)};
  system[2] = {term(2, 1, 1), term(1, 3, 2), term(0, 2, 1)};
  system[3] = {term(3, 1, 1), term(2, 0, 5)};
  return system;
}

const std::vector<std::uint32_t> kEqIdx = {0, 1, 2, 3};
const std::vector<unsigned> kTargets = {3, 2};
const std::vector<unsigned> kMasters = {0, 1};

std::unordered_map<unsigned, unsigned> master_index_of() {
  return {{0, 0}, {1, 1}};
}

// The backward chain as globalreduce builds it off the learn prime: the targets plus the
// derived integrals reachable from them, ascending.
std::vector<std::uint32_t> chain_of(
    const std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> &bwd_deps) {
  std::unordered_set<std::uint32_t> set(kTargets.begin(), kTargets.end());
  std::vector<std::uint32_t> stack(kTargets.begin(), kTargets.end());
  while (!stack.empty()) {
    const std::uint32_t w = stack.back();
    stack.pop_back();
    auto it = bwd_deps.find(w);
    if (it != bwd_deps.end())
      for (std::uint32_t u : it->second)
        if (set.insert(u).second)
          stack.push_back(u);
  }
  std::vector<std::uint32_t> chain(set.begin(), set.end());
  std::sort(chain.begin(), chain.end());
  return chain;
}

// 2^63 - 25, the first of fflow's big primes -- the top of the range every solver here
// assumes.  The small ones below it are chosen so a row's coefficient vanishes mod p.
const std::uint64_t kBigPrime = 9223372036854775783ull;

// Reference reduction rows: reduce_mod + forward_eliminate + backward_eliminate over
// kEqIdx at `p`, restricted to the lazy chain, flattened as the numeric driver reads them.
std::unordered_map<std::uint32_t, secred::ResEq> reference_rows(std::uint64_t p,
                                                               bool with_chain) {
  const std::vector<secred::Eq> system = toy_system();
  secred::SubSolve ref;
  secred::solve_subset(system, kEqIdx, p, ref, false, false, nullptr);
  std::vector<std::uint32_t> chain;
  if (with_chain)
    chain = chain_of(ref.bwd_deps);
  secred::SubSolve restricted;
  secred::solve_subset(system, kEqIdx, p, restricted, false, false,
                       with_chain ? &chain : nullptr);
  return restricted.solmap;
}

} // namespace

// ---------------------------------------------------------------------------
// Task 1: the table
// ---------------------------------------------------------------------------

// The rows are the system's, in eq_idx order, with the identically zero coefficients
// dropped -- the terms reduce_mod keeps at every prime.
TEST(numeric_plan_rows_drop_zero_coefficients) {
  const std::vector<secred::Eq> system = toy_system();
  NumericEvalPlan plan;
  plan.build(system, kEqIdx);
  REQUIRE(plan.n_rows() == kEqIdx.size());
  REQUIRE(plan.sys_idx == kEqIdx);
  REQUIRE(plan.eq_begin.size() == kEqIdx.size() + 1);
  REQUIRE(plan.eq_begin.front() == 0);

  // Row 3's zero term is gone: 3 terms + 2 + 3 + 1 = 9.
  REQUIRE(plan.eq_begin.back() == 9);
  REQUIRE(plan.n_slots() <= 9);
  // One past the largest id in the table.
  REQUIRE(plan.store_ids == 4);

  for (std::size_t r = 0; r < kEqIdx.size(); ++r) {
    std::vector<std::pair<std::uint32_t, secred::Rat64>> expect;
    for (const auto &t : system[kEqIdx[r]])
      if (t.second.num != 0)
        expect.push_back(t);
    REQUIRE(plan.eq_begin[r + 1] - plan.eq_begin[r] == expect.size());
    for (std::size_t k = 0; k < expect.size(); ++k) {
      const auto &got = plan.terms[plan.eq_begin[r] + k];
      REQUIRE(got.first == expect[k].first);
      const secred::Rat64 &c = plan.coeffs[got.second];
      REQUIRE(c.num == expect[k].second.num);
      REQUIRE(c.den == expect[k].second.den);
    }
  }
}

// fill_values is reduce_mod's arithmetic once per distinct coefficient instead of once
// per term: every slot's residue, and every row rebuilt out of the slots, must equal
// what reduce_mod produces.
TEST(numeric_plan_fill_values_match_reduce_mod) {
  const std::vector<secred::Eq> system = toy_system();
  NumericEvalPlan plan;
  plan.build(system, kEqIdx);
  std::vector<std::uint64_t> values;
  plan.fill_values(kBigPrime, values);
  REQUIRE(values.size() == plan.n_slots());
  for (std::size_t s = 0; s < plan.n_slots(); ++s)
    REQUIRE(values[s] ==
            secred::rat_mod_u64(plan.coeffs[s].num, plan.coeffs[s].den, kBigPrime));

  const secred::SymRowTable rows = plan.rows();
  REQUIRE(rows.n_rows == plan.n_rows());
  for (std::size_t r = 0; r < plan.n_rows(); ++r) {
    secred::ResEq expect;
    secred::reduce_mod(system[kEqIdx[r]], kBigPrime, expect);
    const std::size_t off = plan.eq_begin[r];
    const std::size_t size = plan.eq_begin[r + 1] - off;
    REQUIRE(size == expect.size());
    for (std::size_t k = 0; k < size; ++k) {
      REQUIRE(plan.terms[off + k].first == expect[k].first);
      REQUIRE(values[plan.terms[off + k].second] == expect[k].second);
    }
    REQUIRE(rows.begin(static_cast<std::uint32_t>(r)) == off);
    REQUIRE(rows.row_size(static_cast<std::uint32_t>(r)) == size);
  }
  // The plan is the reference's order, and the check the evaluation runs against a
  // point's values passes at a prime where no slot vanishes.
  REQUIRE(plan.plan.order.size() == plan.n_rows());
}

// ---------------------------------------------------------------------------
// Task 2: one prime == the reference elimination
// ---------------------------------------------------------------------------

// Every target's row, through the planned path and through the reference, is the same
// list of (id, residue) -- including at a prime where a coefficient vanishes and the
// row that carries it shortens.
TEST(numeric_prime_solve_matches_reference) {
  const std::vector<secred::Eq> system = toy_system();
  NumericEvalPlan plan;
  plan.build(system, kEqIdx);
  const auto master_index = master_index_of();
  const std::vector<std::uint32_t> chain = chain_of([] {
    secred::SubSolve ref;
    secred::solve_subset(toy_system(), kEqIdx, kBigPrime, ref, false, true, nullptr);
    return ref.bwd_deps;
  }());

  // The big prime plus primes small enough that 2, 3 or 1/2 vanish: p = 2 kills the
  // 2 [0] term of E2 and the 1/2 of E1, p = 3 kills the 3/2 of E2, and p = 5 kills
  // nothing so it doubles as the plain case.
  const std::vector<std::uint64_t> primes = {kBigPrime, 5, 3, 2};
  for (std::uint64_t p : primes) {
    const auto ref_solmap = reference_rows(p, true);
    secred::FwdStore store;
    std::vector<std::uint64_t> values;
    std::vector<std::uint64_t> xout;
    std::uint32_t fail_target = 0, fail_integral = 0;
    const bool ok = solve_prime_numeric(plan, p, chain, kTargets, kMasters,
                                        master_index, store, values, xout,
                                        fail_target, fail_integral);
    REQUIRE(ok);

    // Row for row, against the reference's restricted solve.
    for (unsigned t : kTargets) {
      auto it = ref_solmap.find(t);
      const bool reduced = it != ref_solmap.end();
      REQUIRE(store.has_row(t) == reduced);
      if (!reduced)
        continue;
      const secred::ResEq &expect = it->second;
      REQUIRE(store.row_size(t) == expect.size());
      const std::uint32_t *ids = store.row_ids(t);
      const std::uint64_t *vals = store.row_vals(t);
      for (std::size_t k = 0; k < expect.size(); ++k) {
        REQUIRE(ids[k] == expect[k].first);
        REQUIRE(vals[k] == expect[k].second);
      }
    }

    // The flattened block the CRT consumes: the same numbers, placed by master index.
    REQUIRE(xout.size() == kTargets.size() * kMasters.size());
    for (std::size_t i = 0; i < kTargets.size(); ++i) {
      std::vector<std::uint64_t> expect(kMasters.size(), 0);
      auto it = ref_solmap.find(kTargets[i]);
      if (it != ref_solmap.end())
        for (std::size_t k = 1; k < it->second.size(); ++k)
          expect[master_index.at(it->second[k].first)] = it->second[k].second;
      for (std::size_t j = 0; j < kMasters.size(); ++j)
        REQUIRE(xout[i * kMasters.size() + j] == expect[j]);
    }
  }
}

// A right-hand side that carries an integral the master table does not know is a stale
// selection: the solve must refuse and name the pair rather than emit a truncated block.
TEST(numeric_prime_solve_rejects_non_master_rhs) {
  // One equation, I3 + I1 = 0, so id 3 pivots on id 1 -- and id 1 is neither a master
  // (the table knows only id 0) nor a pivot (no equation reduces it), which is exactly
  // the stale-selection case.
  std::vector<secred::Eq> system(1);
  system[0] = {std::make_pair(std::uint32_t(3), secred::Rat64{1, 1}),
               std::make_pair(std::uint32_t(1), secred::Rat64{1, 1})};
  const std::vector<std::uint32_t> eq_idx = {0};
  NumericEvalPlan plan;
  plan.build(system, eq_idx);
  secred::FwdStore store;
  std::vector<std::uint64_t> values, xout;
  std::uint32_t fail_target = 0, fail_integral = 0;
  const std::unordered_map<unsigned, unsigned> only_master0 = {{0, 0}};
  const std::vector<unsigned> masters0 = {0};
  const bool ok = solve_prime_numeric(plan, kBigPrime, std::vector<std::uint32_t>{},
                                      kTargets, masters0, only_master0, store, values,
                                      xout, fail_target, fail_integral);
  REQUIRE(!ok);
  REQUIRE(fail_target == 3);
  REQUIRE(fail_integral == 1);
}

// The plan's table is read-only and every evaluation owns its store, so the same plan
// solves correctly from several threads at once -- the workers of the reconstruction
// pool share exactly one.
TEST(numeric_plan_solves_concurrently) {
  const std::vector<secred::Eq> system = toy_system();
  NumericEvalPlan plan;
  plan.build(system, kEqIdx);
  const auto master_index = master_index_of();
  const std::vector<std::uint32_t> chain =
      chain_of([&] {
        secred::SubSolve ref;
        secred::solve_subset(system, kEqIdx, kBigPrime, ref, false, true, nullptr);
        return ref.bwd_deps;
      }());
  const std::vector<std::uint64_t> primes = {kBigPrime, 5, 3, 2};

  std::vector<std::vector<std::uint64_t>> serial(primes.size());
  for (std::size_t i = 0; i < primes.size(); ++i) {
    secred::FwdStore store;
    std::vector<std::uint64_t> values;
    std::uint32_t ft = 0, fi = 0;
    REQUIRE(solve_prime_numeric(plan, primes[i], chain, kTargets, kMasters,
                                master_index, store, values, serial[i], ft, fi));
  }

  std::vector<std::vector<std::uint64_t>> parallel(primes.size());
  std::vector<std::thread> workers;
  for (std::size_t i = 0; i < primes.size(); ++i)
    workers.emplace_back([&, i] {
      secred::FwdStore store;
      std::vector<std::uint64_t> values;
      std::uint32_t ft = 0, fi = 0;
      if (!solve_prime_numeric(plan, primes[i], chain, kTargets, kMasters,
                               master_index, store, values, parallel[i], ft, fi))
        parallel[i].clear();
    });
  for (auto &t : workers)
    t.join();

  for (std::size_t i = 0; i < primes.size(); ++i) {
    REQUIRE(!serial[i].empty());
    REQUIRE(parallel[i] == serial[i]);
  }
}
