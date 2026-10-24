// The numeric reconstruction's hoisted plan and its per-prime solve: the table
// NumericEvalPlan builds out of the selected equations, and the guarantee that one prime
// through forward_eliminate_planned + backward_eliminate_planned is the reference
// reduce_mod + forward_eliminate + backward_eliminate, term for term.
#include <algorithm>
#include <cstdint>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fflow/primes.hh>

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

// The reconstruction's prime sequence: fflow's 201 primes first, unchanged, then
// generated further down the same range, each distinct and strictly decreasing.  A cap
// above 201 must not wrap into fflow's table: a repeated modulus is not coprime to the
// CRT's and would silently produce a wrong reconstruction.
TEST(globalreduce_reconstruction_primes_extend_fflows_table) {
  for (unsigned i = 0; i < fflow::BIG_UINT_PRIMES_SIZE; ++i)
    REQUIRE_EQ(reconstruction_prime(i), fflow::BIG_UINT_PRIMES[i]);

  // 100 past the table is well inside any cap this setting is meant for, and cheap:
  // the generation is lazy and cached, so this pays for the primes it asks for once.
  std::set<std::uint64_t> seen;
  std::uint64_t prev = ~std::uint64_t(0);
  for (unsigned i = 0; i <= fflow::BIG_UINT_PRIMES_SIZE + 100; ++i) {
    const std::uint64_t p = reconstruction_prime(i);
    REQUIRE(p < (std::uint64_t(1) << 63));              // 2p < 2^64
    REQUIRE(static_cast<unsigned __int128>(p) * 3 > (static_cast<unsigned __int128>(1) << 64));
    REQUIRE(p < prev);                                  // strictly decreasing
    REQUIRE(seen.insert(p).second);                     // and never repeated
    prev = p;
  }
  // Cached: a second call is the same prime, not a fresh search from the top.
  REQUIRE_EQ(reconstruction_prime(250), reconstruction_prime(250));
}

// A coefficient beyond int64 is an ordinary term to the plan: it is not zero, so it
// keeps its slot (tagged {handle, 0}), fill_values reduces it through the intern table,
// and the planned solve agrees with the reference -- the pure-numeric path of a family
// whose s, t and masses are wider than 64 bits.
TEST(numeric_plan_handles_a_big_coefficient) {
  const std::string text = "18446744073709551616"; // 2^64 = 2 * kBigPrime + 50
  int64_t handle = 0;
  REQUIRE(secred::intern_big_rational(text.data(), text.data() + text.size(),
                                      handle));

  std::vector<secred::Eq> system = toy_system();
  system[0][0].second = secred::Rat64{handle, 0}; // E0's id-3 coefficient

  NumericEvalPlan plan;
  plan.build(system, kEqIdx);
  REQUIRE_EQ(plan.n_slots(), std::size_t(5)); // 4 inline + the big one
  bool saw_big = false;
  for (std::size_t s = 0; s < plan.n_slots(); ++s)
    if (plan.coeffs[s].den == 0) {
      REQUIRE_EQ(plan.coeffs[s].num, handle);
      saw_big = true;
    }
  REQUIRE(saw_big);

  std::vector<std::uint64_t> values;
  plan.fill_values(kBigPrime, values);
  REQUIRE_EQ(values[plan.terms[plan.eq_begin[0]].second], std::uint64_t(50));

  // Reference rows at the same prime: reduce_mod reads the tagged coefficient through
  // rat_mod_u64, so the two paths must agree row for row.
  secred::SubSolve ref;
  secred::solve_subset(system, kEqIdx, kBigPrime, ref, false, true, nullptr);
  const std::vector<std::uint32_t> chain = chain_of(ref.bwd_deps);
  secred::SubSolve restricted;
  secred::solve_subset(system, kEqIdx, kBigPrime, restricted, false, false, &chain);

  secred::FwdStore store;
  std::vector<std::uint64_t> xout;
  std::uint32_t fail_target = 0, fail_integral = 0;
  REQUIRE(solve_prime_numeric(plan, kBigPrime, chain, kTargets, kMasters,
                              master_index_of(), store, values, xout, fail_target,
                              fail_integral));
  for (unsigned t : kTargets) {
    auto it = restricted.solmap.find(t);
    REQUIRE(store.has_row(t) == (it != restricted.solmap.end()));
    if (it == restricted.solmap.end())
      continue;
    REQUIRE_EQ(store.row_size(t), it->second.size());
    for (std::size_t k = 0; k < it->second.size(); ++k) {
      REQUIRE_EQ(store.row_ids(t)[k], it->second[k].first);
      REQUIRE_EQ(store.row_vals(t)[k], it->second[k].second);
    }
  }
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

// The reconstruction pool runs prime 0 as a serial probe, then hands every worker store
// the fill sizes that probe reached.  A store with that hint must produce the same block,
// and because the hint is exactly what the point fills, must not reallocate while filling.
TEST(numeric_prime_solve_with_the_probe_fill_reserve_matches) {
  const std::vector<secred::Eq> system = toy_system();
  NumericEvalPlan plan;
  plan.build(system, kEqIdx);
  const auto master_index = master_index_of();
  const std::vector<std::uint32_t> chain = chain_of([] {
    secred::SubSolve ref;
    secred::solve_subset(toy_system(), kEqIdx, kBigPrime, ref, false, true, nullptr);
    return ref.bwd_deps;
  }());

  secred::FwdStore probe;
  std::vector<std::uint64_t> values, probe_xout;
  std::uint32_t fail_target = 0, fail_integral = 0;
  REQUIRE(solve_prime_numeric(plan, kBigPrime, chain, kTargets, kMasters, master_index,
                              probe, values, probe_xout, fail_target, fail_integral));
  // The hint is only taken when the probe replayed the plan rather than rebuilding an
  // order of its own.
  REQUIRE(probe.order_ptr == &plan.plan.order);

  secred::FwdStore hinted;
  hinted.set_fill_reserve(probe.ids.size(), probe.begin.size());
  hinted.reset(plan.store_ids);
  const std::size_t ids_capacity = hinted.ids.capacity();
  const std::size_t rows_capacity = hinted.begin.capacity();
  REQUIRE(ids_capacity >= probe.ids.size());
  REQUIRE(rows_capacity >= probe.begin.size());

  std::vector<std::uint64_t> values2, xout;
  fail_target = fail_integral = 0;
  REQUIRE(solve_prime_numeric(plan, kBigPrime, chain, kTargets, kMasters, master_index,
                              hinted, values2, xout, fail_target, fail_integral));
  REQUIRE(xout == probe_xout);
  REQUIRE(hinted.ids == probe.ids);
  REQUIRE(hinted.vals == probe.vals);
  REQUIRE(hinted.begin == probe.begin);
  REQUIRE(hinted.len == probe.len);
  // The point filled the reserved capacity exactly, so the fill reallocated nothing.
  REQUIRE_EQ(hinted.ids.capacity(), ids_capacity);
  REQUIRE_EQ(hinted.begin.capacity(), rows_capacity);
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

// ---------------------------------------------------------------------------
// Task 3: the subset solve through the planned store
// ---------------------------------------------------------------------------

namespace {

// A SubSolve flattened to sorted lines, one per entry of the four maps, so two solves
// compare exactly -- including whether a key is present with an empty dependency list.
std::vector<std::string> flattened(const secred::SubSolve &s) {
  std::vector<std::string> out;
  auto ids = [](const std::vector<std::uint32_t> &v) {
    std::ostringstream oss;
    for (std::uint32_t u : v)
      oss << " " << u;
    return oss.str();
  };
  for (const auto &kv : s.solmap) {
    std::ostringstream oss;
    oss << "solmap " << kv.first << ":";
    for (const auto &t : kv.second)
      oss << " " << t.first << "=" << t.second;
    out.push_back(oss.str());
  }
  for (const auto &kv : s.eq_index)
    out.push_back("eq_index " + std::to_string(kv.first) + "=" +
                  std::to_string(kv.second));
  for (const auto &kv : s.fwd_deps)
    out.push_back("fwd_deps " + std::to_string(kv.first) + ":" + ids(kv.second));
  for (const auto &kv : s.bwd_deps)
    out.push_back("bwd_deps " + std::to_string(kv.first) + ":" + ids(kv.second));
  std::sort(out.begin(), out.end());
  return out;
}

// A deterministic system wide enough for multi-step substitution chains: `nvars` ids,
// `neqs` rows, each row a subset of the ids below a biased front, descending and unique,
// with small rational coefficients (identically zero ones dropped, as the reader drops
// them).  A numerator like 3 or -3 vanishes at p = 3, which is what drives the plan check
// and the rebuilt order.
std::vector<secred::Eq> generated_system(unsigned neqs, unsigned nvars,
                                         std::uint64_t seed) {
  std::uint64_t s = seed;
  auto next = [&s]() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return s >> 33;
  };
  std::vector<secred::Eq> system(neqs);
  for (unsigned e = 0; e < neqs; ++e) {
    const unsigned front = nvars - 1 - static_cast<unsigned>(next() % (nvars / 2));
    std::vector<std::pair<std::uint32_t, secred::Rat64>> row;
    row.emplace_back(front, secred::Rat64{1, 1});
    for (unsigned id = 0; id < front; ++id)
      if (next() % 3 == 0)
        row.emplace_back(id, secred::Rat64{static_cast<std::int64_t>(next() % 7) - 3,
                                           static_cast<std::int64_t>(next() % 5) + 1});
    row.erase(std::remove_if(row.begin(), row.end(),
                             [](const auto &t) { return t.second.num == 0; }),
              row.end());
    std::sort(row.begin(), row.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });
    system[e] = std::move(row);
  }
  return system;
}

std::vector<std::uint32_t> all_idx(unsigned n) {
  std::vector<std::uint32_t> idx(n);
  for (unsigned i = 0; i < n; ++i)
    idx[i] = i;
  return idx;
}

// The first line where two flattened solves disagree, with what each side holds -- the
// comparison is a whole-structure one, so a bare false says nothing about which map or
// which pivot went its own way.
std::string solve_difference(const secred::SubSolve &got,
                             const secred::SubSolve &ref) {
  const std::vector<std::string> a = flattened(got), b = flattened(ref);
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    if (a[i] != b[i])
      return "line " + std::to_string(i) + ": got [" + a[i] + "] want [" + b[i] + "]";
  if (a.size() != b.size())
    return "sizes " + std::to_string(a.size()) + " vs " + std::to_string(b.size());
  return "equal";
}

} // namespace

// solve_subset_planned must hand back exactly what solve_subset does -- the same solution
// map, the same pivot -> equation index map and the same two dependency maps -- because
// the learn and the selection read either interchangeably while the reconstruction keeps
// solving through the store.  The primes include ones where a coefficient vanishes (3 and
// 2 on the toy system), which is where the planned elimination leaves the hoisted order.
TEST(planned_subset_solve_matches_solve_subset) {
  const std::vector<secred::Eq> system = toy_system();
  const std::vector<std::uint64_t> primes = {kBigPrime, 5, 3, 2};
  const std::vector<std::pair<bool, bool>> wants = {
      {true, true}, {true, false}, {false, true}, {false, false}};
  for (std::uint64_t p : primes)
    for (const auto &w : wants) {
      secred::SubSolve ref, got;
      secred::solve_subset(system, kEqIdx, p, ref, w.first, w.second, nullptr);
      secred::solve_subset_planned(system, kEqIdx, p, got, w.first, w.second);
      REQUIRE(flattened(got) == flattened(ref));
    }
}

// The same equivalence on a generated system, so the comparison covers chains several
// substitutions deep and rows a vanishing coefficient shortens.
TEST(planned_subset_solve_matches_solve_subset_on_generated_systems) {
  const std::vector<std::uint64_t> primes = {kBigPrime, 7, 5, 3};
  for (std::uint64_t seed : {1ull, 2ull, 3ull}) {
    const std::vector<secred::Eq> system = generated_system(40, 24, seed);
    const std::vector<std::uint32_t> idx = all_idx(40);
    for (std::uint64_t p : primes) {
      secred::SubSolve ref, got;
      secred::solve_subset(system, idx, p, ref, true, true, nullptr);
      secred::solve_subset_planned(system, idx, p, got, true, true);
      const std::string diff = solve_difference(got, ref);
      REQUIRE_MSG(flattened(got) == flattened(ref),
                  "seed " + std::to_string(seed) + " p " + std::to_string(p) + ": " +
                      diff);
      if (ref.solmap.empty())
        continue; // nothing to restrict
      // The reference's lazy chain (one pivot is enough to exercise the restriction),
      // solved through both paths.
      const std::vector<std::uint32_t> chain{ref.solmap.begin()->first};
      secred::SubSolve ref_chain, got_chain;
      secred::solve_subset(system, idx, p, ref_chain, false, true, &chain);
      secred::solve_subset_planned(system, idx, p, got_chain, false, true, &chain);
      REQUIRE(flattened(got_chain) == flattened(ref_chain));
    }
  }
}

// solmap_keys is the caller's own read set: only those pivots' rows are materialised, and
// each of them is the row the reference solve holds.  The selection reads nothing else --
// it looks a target up and walks bwd_deps -- so this is what keeps its two solves off the
// 570k-row map.
TEST(planned_subset_solve_restricts_the_solmap_to_the_read_set) {
  const std::vector<secred::Eq> system = toy_system();
  const std::vector<std::uint64_t> primes = {kBigPrime, 5, 3, 2};
  for (std::uint64_t p : primes) {
    secred::SubSolve ref;
    secred::solve_subset(system, kEqIdx, p, ref, false, true, nullptr);

    std::vector<std::uint32_t> read_set = kTargets;
    read_set.push_back(0); // an id the caller reads whether or not it is a pivot
    secred::SubSolve got;
    secred::solve_subset_planned(system, kEqIdx, p, got, false, true, nullptr,
                                 &read_set);

    for (std::uint32_t id : read_set) {
      auto rit = ref.solmap.find(id);
      auto git = got.solmap.find(id);
      REQUIRE((git != got.solmap.end()) == (rit != ref.solmap.end()));
      if (rit != ref.solmap.end())
        REQUIRE(git->second == rit->second);
    }
    REQUIRE(got.solmap.size() <= read_set.size());
    // Everything else is still the reference's.
    REQUIRE(got.bwd_deps == ref.bwd_deps);
    REQUIRE(got.fwd_deps.empty());
    REQUIRE(got.eq_index.empty());
  }
}
