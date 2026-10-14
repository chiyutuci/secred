#include "test_runner.h"
#include "global_select.h"
#include "ffred_solver.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fflow/primes.hh>

namespace {

using secred::Eq;
using secred::Rat64;
using secred::ResEq;

constexpr uint64_t P = 1000000007ULL;

Eq eq(std::initializer_list<std::pair<uint32_t, std::pair<int64_t, int64_t>>> terms) {
  Eq e;
  for (const auto &t : terms)
    e.push_back({t.first, Rat64{t.second.first, t.second.second}});
  return e;
}

void full_solve(const std::vector<Eq> &system, uint64_t p,
                std::unordered_map<uint32_t, ResEq> &solmap,
                std::unordered_map<uint32_t, uint32_t> &eq_index,
                std::unordered_map<uint32_t, std::vector<uint32_t>> &fwd_deps,
                std::unordered_map<uint32_t, std::vector<uint32_t>> &bwd_deps) {
  std::vector<ResEq> res_eqs;
  std::vector<uint32_t> sys_idx;
  for (std::size_t i = 0; i < system.size(); ++i) {
    ResEq re;
    secred::reduce_mod(system[i], p, re);
    if (!re.empty()) {
      res_eqs.push_back(std::move(re));
      sys_idx.push_back(static_cast<uint32_t>(i));
    }
  }
  secred::forward_eliminate(res_eqs, sys_idx, p, solmap, &eq_index, &fwd_deps);
  secred::backward_eliminate(solmap, p, &bwd_deps);
}

std::unordered_set<uint32_t> mark_sweep(
    const std::vector<uint32_t> &targets,
    const std::unordered_map<uint32_t, std::vector<uint32_t>> &fwd_deps,
    const std::unordered_map<uint32_t, std::vector<uint32_t>> &bwd_deps) {
  std::unordered_set<uint32_t> needed(targets.begin(), targets.end());
  std::vector<uint32_t> stack(targets.begin(), targets.end());
  while (!stack.empty()) {
    uint32_t w = stack.back();
    stack.pop_back();
    auto fit = fwd_deps.find(w);
    if (fit != fwd_deps.end())
      for (uint32_t u : fit->second)
        if (needed.insert(u).second)
          stack.push_back(u);
    auto bit = bwd_deps.find(w);
    if (bit != bwd_deps.end())
      for (uint32_t u : bit->second)
        if (needed.insert(u).second)
          stack.push_back(u);
  }
  return needed;
}

} // namespace

// Simple chain: I_1 = I_0, I_2 = I_1 + I_0, I_3 = I_2 + I_0 (I_3 is unrelated to the target).
// Target I_2 -> I_2 = 2 I_0; mark-and-sweep should select only {I_2, I_1}, excluding I_3.
TEST(ffred_forward_backward_basic) {
  std::vector<Eq> system;
  system.push_back(eq({{1, {1, 1}}, {0, {-1, 1}}}));
  system.push_back(eq({{2, {1, 1}}, {1, {-1, 1}}, {0, {-1, 1}}}));
  system.push_back(eq({{3, {1, 1}}, {2, {-1, 1}}, {0, {-1, 1}}}));

  std::unordered_map<uint32_t, ResEq> solmap;
  std::unordered_map<uint32_t, uint32_t> eq_index;
  std::unordered_map<uint32_t, std::vector<uint32_t>> fwd_deps, bwd_deps;
  full_solve(system, P, solmap, eq_index, fwd_deps, bwd_deps);

  // I_2 = 2 I_0
  REQUIRE(solmap.find(2) != solmap.end());
  const ResEq &r2 = solmap.at(2);
  REQUIRE_EQ(r2.size(), 2u);
  REQUIRE_EQ(r2[0].first, 2u);
  REQUIRE_EQ(r2[0].second, P - 1);
  REQUIRE_EQ(r2[1].first, 0u);
  REQUIRE_EQ(r2[1].second, 2u);

// mark-and-sweep from {2} -> {2, 1}, excluding 3.
  auto needed = mark_sweep({2}, fwd_deps, bwd_deps);
  REQUIRE(needed.count(2));
  REQUIRE(needed.count(1));
  REQUIRE(!needed.count(3));
  REQUIRE_EQ(needed.size(), 2u);
}

// Derived integral (the box [1,2,1,2] case): I_125 has no equation of its own; it is
// registered late as the pivot of the duplicated-front=127 equation and shows up on
// I_127's right-hand side, so back substitution must push it out.
//   I_126 = I_0
//   I_127 = 2 I_125 + 2 I_0           (E_b, registers 127)
//   I_127 = I_125 + I_126 + I_1       (E_a, registers 125 after substituting 127)
//   I_128 = I_127 + I_0               (unrelated, must not be selected)
// Target I_127 -> I_127 = 2 I_1.
TEST(ffred_forward_backward_derived_integral) {
  std::vector<Eq> system;
  system.push_back(eq({{126, {1, 1}}, {0, {-1, 1}}}));
  system.push_back(eq({{127, {1, 1}}, {125, {-2, 1}}, {126, {-1, 1}}, {0, {-1, 1}}}));
  system.push_back(eq({{127, {1, 1}}, {125, {-1, 1}}, {126, {-1, 1}}, {1, {-1, 1}}}));
  system.push_back(eq({{128, {1, 1}}, {127, {-1, 1}}, {0, {-1, 1}}}));

  std::unordered_map<uint32_t, ResEq> solmap;
  std::unordered_map<uint32_t, uint32_t> eq_index;
  std::unordered_map<uint32_t, std::vector<uint32_t>> fwd_deps, bwd_deps;
  full_solve(system, P, solmap, eq_index, fwd_deps, bwd_deps);

// I_127 = 2 I_1 (after back substitution removes I_125)
  REQUIRE(solmap.find(127) != solmap.end());
  const ResEq &r127 = solmap.at(127);
  REQUIRE_EQ(r127.size(), 2u);
  REQUIRE_EQ(r127[0].first, 127u);
  REQUIRE_EQ(r127[0].second, P - 1);
  REQUIRE_EQ(r127[1].first, 1u);
  REQUIRE_EQ(r127[1].second, 2u);

// Dependency record: fwd_deps[125] = [127, 126] (E_a substitutes 127, then 126),
//                   bwd_deps[127] = [125] (backward substitution of I_125).
  REQUIRE(fwd_deps.find(125) != fwd_deps.end());
  REQUIRE_EQ(fwd_deps[125].size(), 2u);
  REQUIRE_EQ(fwd_deps[125][0], 127u);
  REQUIRE_EQ(fwd_deps[125][1], 126u);
  REQUIRE(bwd_deps.find(127) != bwd_deps.end());
  REQUIRE_EQ(bwd_deps[127].size(), 1u);
  REQUIRE_EQ(bwd_deps[127][0], 125u);

// mark-and-sweep from {127} -> {127, 126, 125}, excluding 128.
  auto needed = mark_sweep({127}, fwd_deps, bwd_deps);
  REQUIRE(needed.count(127));
  REQUIRE(needed.count(126));
  REQUIRE(needed.count(125));
  REQUIRE(!needed.count(128));
  REQUIRE_EQ(needed.size(), 3u);

// The trimmed reconstruction (only the needed equations) must match the full result.
  std::vector<uint32_t> needed_eq_idx;
  for (uint32_t w : needed)
    needed_eq_idx.push_back(eq_index.at(w));
  std::vector<ResEq> trim_eqs;
  std::vector<uint32_t> trim_idx;
  for (uint32_t i : needed_eq_idx) {
    ResEq re;
    secred::reduce_mod(system[i], P, re);
    if (!re.empty()) {
      trim_eqs.push_back(std::move(re));
      trim_idx.push_back(i);
    }
  }
  std::unordered_map<uint32_t, ResEq> trim_solmap;
  secred::forward_eliminate(trim_eqs, trim_idx, P, trim_solmap, nullptr, nullptr);
  secred::backward_eliminate(trim_solmap, P, nullptr);

  REQUIRE(trim_solmap.find(127) != trim_solmap.end());
  const ResEq &t127 = trim_solmap.at(127);
  REQUIRE_EQ(t127.size(), 2u);
  REQUIRE_EQ(t127[0].first, 127u);
  REQUIRE_EQ(t127[0].second, P - 1);
  REQUIRE_EQ(t127[1].first, 1u);
  REQUIRE_EQ(t127[1].second, 2u);
}

// A stale key in only_keys (absent from solmap) must not insert an empty entry: that would
// later misread the genuine master I_0 as derived and inline its empty tail, silently
// dropping I_0's coefficient.
//   I_1 = I_0,  I_2 = I_1 + I_0  =>  I_2 = 2 I_0  (I_0 is no pivot)
TEST(ffred_backward_ignores_stale_only_keys) {
  std::vector<Eq> system;
  system.push_back(eq({{1, {1, 1}}, {0, {-1, 1}}}));
  system.push_back(eq({{2, {1, 1}}, {1, {-1, 1}}, {0, {-1, 1}}}));

  std::vector<ResEq> res_eqs;
  std::vector<uint32_t> sys_idx;
  for (std::size_t i = 0; i < system.size(); ++i) {
    ResEq re;
    secred::reduce_mod(system[i], P, re);
    res_eqs.push_back(std::move(re));
    sys_idx.push_back(static_cast<uint32_t>(i));
  }
  std::unordered_map<uint32_t, ResEq> solmap;
  secred::forward_eliminate(res_eqs, sys_idx, P, solmap, nullptr, nullptr);
  REQUIRE(solmap.find(2) != solmap.end());
  REQUIRE_EQ(solmap.at(2).size(), 2u);

// key 0 is not a pivot (it is a master), key 2 is; passed in ascending order.
  const std::vector<uint32_t> stale_keys{0u, 2u};
  secred::backward_eliminate(solmap, P, nullptr, &stale_keys);

// No empty entry may be inserted for the stale key 0.
  REQUIRE(solmap.find(0) == solmap.end());

// I_2 = 2 I_0 must survive intact: master I_0's coefficient must not be lost as an empty tail.
  const ResEq &r2 = solmap.at(2);
  REQUIRE_EQ(r2.size(), 2u);
  REQUIRE_EQ(r2[0].first, 2u);
  REQUIRE_EQ(r2[0].second, P - 1);
  REQUIRE_EQ(r2[1].first, 0u);
  REQUIRE_EQ(r2[1].second, 2u);
}

// solve_subset (the subset solve shared by learn, the repair and reconstruction) must agree
// term by term with hand-written reduce_mod + forward_eliminate + backward_eliminate.
TEST(ffred_solve_subset_matches_inline) {
  std::vector<Eq> system;
  system.push_back(eq({{126, {1, 1}}, {0, {-1, 1}}}));
  system.push_back(eq({{127, {1, 1}}, {125, {-2, 1}}, {126, {-1, 1}}, {0, {-1, 1}}}));
  system.push_back(eq({{127, {1, 1}}, {125, {-1, 1}}, {126, {-1, 1}}, {1, {-1, 1}}}));
  system.push_back(eq({{128, {1, 1}}, {127, {-1, 1}}, {0, {-1, 1}}}));

  std::unordered_map<uint32_t, ResEq> solmap;
  std::unordered_map<uint32_t, uint32_t> eq_index;
  std::unordered_map<uint32_t, std::vector<uint32_t>> fwd_deps, bwd_deps;
  full_solve(system, P, solmap, eq_index, fwd_deps, bwd_deps);

  std::vector<uint32_t> all{0u, 1u, 2u, 3u};
  secred::SubSolve sub;
  secred::solve_subset(system, all, P, sub, true, true);

  REQUIRE_EQ(sub.solmap.size(), solmap.size());
  for (const auto &kv : solmap) {
    REQUIRE(sub.solmap.find(kv.first) != sub.solmap.end());
    REQUIRE(sub.solmap.at(kv.first) == kv.second);
  }
  REQUIRE_EQ(sub.eq_index.size(), eq_index.size());
  for (const auto &kv : eq_index)
    REQUIRE_EQ(sub.eq_index.at(kv.first), kv.second);
  REQUIRE_EQ(sub.fwd_deps.size(), fwd_deps.size());
  for (const auto &kv : fwd_deps)
    REQUIRE(sub.fwd_deps.at(kv.first) == kv.second);
  REQUIRE_EQ(sub.bwd_deps.size(), bwd_deps.size());
  for (const auto &kv : bwd_deps)
    REQUIRE(sub.bwd_deps.at(kv.first) == kv.second);

// The recording switches must not change any value: with want_fwd/want_bwd off the solmap
// is identical and the dependency graphs stay empty.
  secred::SubSolve bare;
  secred::solve_subset(system, all, P, bare, false, false);
  REQUIRE_EQ(bare.solmap.size(), solmap.size());
  for (const auto &kv : solmap)
    REQUIRE(bare.solmap.at(kv.first) == kv.second);
  REQUIRE(bare.eq_index.empty());
  REQUIRE(bare.fwd_deps.empty());
  REQUIRE(bare.bwd_deps.empty());
}

// ---------------------------------------------------------------------------
// Equation selection for the global stage: forward closure plus numeric sufficiency repair
// ---------------------------------------------------------------------------

namespace {

// Derived-integral fixture (as in ffred_forward_backward_derived_integral):
//   eq0: I_126 = I_0
//   eq1: I_127 = 2 I_125 + I_126 + I_0   (registers 127)
//   eq2: I_127 = I_125 + I_126 + I_1     (registers 125 after substituting 127, fwd_deps[125]={127,126})
//   eq3: I_128 = I_127 + I_0             (unrelated)
// Target I_127 -> master {I_1}; the forward-only closure selects eq1 alone, and its
// right-hand side carries I_125/I_126 -- neither a master, exactly the hidden zero
// Kira puts back with the numeric repair.
std::vector<Eq> derived_fixture() {
  std::vector<Eq> system;
  system.push_back(eq({{126, {1, 1}}, {0, {-1, 1}}}));
  system.push_back(
      eq({{127, {1, 1}}, {125, {-2, 1}}, {126, {-1, 1}}, {0, {-1, 1}}}));
  system.push_back(
      eq({{127, {1, 1}}, {125, {-1, 1}}, {126, {-1, 1}}, {1, {-1, 1}}}));
  system.push_back(eq({{128, {1, 1}}, {127, {-1, 1}}, {0, {-1, 1}}}));
  return system;
}

// What learn (full system) produces: the selection only reads these; masters take no part.
struct LearnCtx {
  std::vector<Eq> system = derived_fixture();
  std::unordered_map<uint32_t, ResEq> solmap;
  std::unordered_map<uint32_t, uint32_t> eq_index;
  std::unordered_map<uint32_t, std::vector<uint32_t>> fwd_deps, bwd_deps;
  std::vector<unsigned> dep_targets{127u};
std::vector<unsigned> masters;                       // descending id
std::unordered_map<unsigned, unsigned> master_index; // master -> output column

  LearnCtx() {
    full_solve(system, P, solmap, eq_index, fwd_deps, bwd_deps);
std::set<unsigned> ms; // masters = right-hand sides of the target reductions (same rule as production)
    for (unsigned id : dep_targets) {
      auto it = solmap.find(id);
      if (it == solmap.end())
        continue;
      for (std::size_t k = 1; k < it->second.size(); ++k)
        ms.insert(it->second[k].first);
    }
    masters.assign(ms.rbegin(), ms.rend());
    for (unsigned j = 0; j < masters.size(); ++j)
      master_index[masters[j]] = j;
  }
};

// "Re-solve the subsystem at the learn prime", isomorphic to the callback production injects.
void solve_at_learn(const std::vector<Eq> &system, uint64_t p,
                    const std::vector<uint32_t> &idx, secred::SubSolve &out) {
  secred::solve_subset(system, idx, p, out, false, true);
}

// Sufficiency test (= the per-prime hard check in production): a target's right-hand side
// may hold masters only.
bool rhs_only_masters(
    const secred::SubSolve &sub, const std::vector<unsigned> &dep_targets,
    const std::unordered_map<unsigned, unsigned> &master_index) {
  for (unsigned w : dep_targets) {
    auto it = sub.solmap.find(w);
    if (it == sub.solmap.end())
      return false;
    for (std::size_t k = 1; k < it->second.size(); ++k)
      if (!master_index.count(it->second[k].first))
        return false;
  }
  return true;
}

GlobalSelection run_selection(const LearnCtx &c, unsigned max_rounds = 64) {
  return select_global_forward(
      c.eq_index, c.fwd_deps, c.solmap, c.dep_targets, c.master_index,
      [&](const std::vector<uint32_t> &idx, secred::SubSolve &sub) {
        solve_at_learn(c.system, P, idx, sub);
      },
      max_rounds);
}

} // namespace

// The forward closure alone is not enough; the repair adds the missed equation back and
// the right-hand side keeps masters only.
TEST(global_forward_only_then_repair) {
  LearnCtx c;

// I_1 is the only master (I_127 = 2 I_1).
  REQUIRE_EQ(c.masters.size(), 1u);
  REQUIRE_EQ(c.masters[0], 1u);

// Forward dependency: registering 127 substituted 126 (fwd_deps[127]={126}); 125 is a backward edge.
  REQUIRE_EQ(c.fwd_deps.at(127u).size(), 1u);
  REQUIRE_EQ(c.fwd_deps.at(127u)[0], 126u);
REQUIRE(c.bwd_deps.find(125u) == c.bwd_deps.end()); // 125 is reachable only through bwd_deps[127]

// Forward closure only (repair off): {127} plus fwd_deps[127] = {127,126} -> equations
// {eq0,eq1}; re-solving still leaves I_125 (reducible, but no equation registers it) and
// I_0 (zero difference throughout, irreducible) on the right-hand side.
  GlobalSelection fwd_only = run_selection(c, 0u);
  REQUIRE(!fwd_only.ok);
  REQUIRE(fwd_only.fallback);
  REQUIRE_EQ(fwd_only.n_fwd_only, 2u);
  REQUIRE(fwd_only.needed_eq_idx == (std::vector<uint32_t>{0u, 1u}));
  secred::SubSolve sub_fwd;
  solve_at_learn(c.system, P, fwd_only.needed_eq_idx, sub_fwd);
  REQUIRE(!rhs_only_masters(sub_fwd, c.dep_targets, c.master_index));
  bool saw_125 = false;
  for (std::size_t k = 1; k < sub_fwd.solmap.at(127).size(); ++k)
    saw_125 |= sub_fwd.solmap.at(127)[k].first == 125u;
  REQUIRE(saw_125);

// The legacy closure (forward and backward) selects {0,1,2}.
  auto legacy_needed = mark_sweep({127u}, c.fwd_deps, c.bwd_deps);
  REQUIRE_EQ(legacy_needed.size(), 3u);

// The repair: one round, adding I_125's equation (eq2), giving the same set as legacy.
  GlobalSelection sel = run_selection(c);
  REQUIRE(sel.ok);
  REQUIRE(!sel.fallback);
  REQUIRE_EQ(sel.n_repair_rounds, 1u);
  REQUIRE_EQ(sel.n_repair_added, 1u);
  REQUIRE(sel.needed_eq_idx == (std::vector<uint32_t>{0u, 1u, 2u}));

// The final selection is sufficient: masters only on the right-hand side, with coefficient I_127 = 2 I_1.
  secred::SubSolve fin;
  solve_at_learn(c.system, P, sel.needed_eq_idx, fin);
  REQUIRE(rhs_only_masters(fin, c.dep_targets, c.master_index));
  REQUIRE_EQ(fin.solmap.at(127).size(), 2u);
  REQUIRE_EQ(fin.solmap.at(127)[1].first, 1u);
  REQUIRE_EQ(fin.solmap.at(127)[1].second, 2u);
}

// The selection is deterministic: the same input twice gives the same equations, rounds,
// additions and back-substitution chain.
TEST(global_forward_selection_is_deterministic) {
  LearnCtx c;
  GlobalSelection a = run_selection(c);
  GlobalSelection b = run_selection(c);
  REQUIRE(a.needed_eq_idx == b.needed_eq_idx);
  REQUIRE(a.bwd_chain == b.bwd_chain);
  REQUIRE_EQ(a.n_repair_rounds, b.n_repair_rounds);
  REQUIRE_EQ(a.n_repair_added, b.n_repair_added);
  REQUIRE_EQ(a.n_fwd_only, b.n_fwd_only);
}

// Hitting the repair round cap sets fallback explicitly (the caller then returns to the
// legacy closure) rather than failing hard: the forward closure is already computed and
// needed_eq_idx is usable, it simply never entered the repair.
TEST(global_forward_repair_cap_falls_back) {
  LearnCtx c;
  GlobalSelection capped = run_selection(c, 0u);
  REQUIRE(!capped.ok);
  REQUIRE(capped.fallback);
  REQUIRE_EQ(capped.n_fwd_only, 2u);
  REQUIRE_EQ(capped.n_repair_rounds, 0u);
  REQUIRE_EQ(capped.n_repair_added, 0u);
  REQUIRE(capped.needed_eq_idx == (std::vector<uint32_t>{0u, 1u}));
}

// Chain invariants: bwd_chain lies among the subset solmap's keys (INV-1) and is closed
// under the subset's backward dependencies (INV-2); and restricting back substitution to
// only_keys = bwd_chain matches the unrestricted run term by term on every chain key --
// which is what lets reconstruction rebuild just this chain per prime.  Holds at several primes.
TEST(subsolve_chain_invariant) {
  LearnCtx c;
  GlobalSelection sel = run_selection(c);
  REQUIRE(sel.ok);
  REQUIRE(!sel.bwd_chain.empty());

  const uint64_t primes[] = {P, 1000000009ULL, 998244353ULL};
  for (uint64_t p : primes) {
    secred::SubSolve sub;
    solve_at_learn(c.system, p, sel.needed_eq_idx, sub);

// INV-1: every key in the chain is a pivot of the selected subsystem.
    for (uint32_t k : sel.bwd_chain)
      REQUIRE(sub.solmap.find(k) != sub.solmap.end());

// INV-2: the chain is closed under the subset's recorded backward dependencies.
    for (uint32_t k : sel.bwd_chain) {
      auto bit = sub.bwd_deps.find(k);
      if (bit == sub.bwd_deps.end())
        continue;
      for (uint32_t u : bit->second)
        REQUIRE(std::find(sel.bwd_chain.begin(), sel.bwd_chain.end(), u) !=
                sel.bwd_chain.end());
    }

// Restricted vs unrestricted: on the same forward-only sub-solve both back-substitution
// paths must give the same reduction (otherwise reconstruction reads wrong coefficients).
    std::vector<secred::ResEq> res_eqs, res_eqs2;
    std::vector<uint32_t> sys_idx, sys_idx2;
    for (uint32_t i : sel.needed_eq_idx) {
      secred::ResEq re, re2;
      secred::reduce_mod(c.system[i], p, re);
      secred::reduce_mod(c.system[i], p, re2);
      if (!re.empty()) {
        res_eqs.push_back(std::move(re));
        res_eqs2.push_back(std::move(re2));
        sys_idx.push_back(i);
        sys_idx2.push_back(i);
      }
    }
    std::unordered_map<uint32_t, secred::ResEq> restr, full;
    secred::forward_eliminate(res_eqs, sys_idx, p, restr, nullptr, nullptr);
    secred::forward_eliminate(res_eqs2, sys_idx2, p, full, nullptr, nullptr);
    secred::backward_eliminate(restr, p, nullptr, &sel.bwd_chain);
    secred::backward_eliminate(full, p, nullptr, nullptr);

    for (uint32_t k : sel.bwd_chain)
      REQUIRE(restr.at(k) == full.at(k));
    REQUIRE(restr.at(127u) == full.at(127u));
  }
}

// ---------------------------------------------------------------------------
// The sample-point path: plan + flat store + Shoup products
// ---------------------------------------------------------------------------

namespace {

// The store's row for `id` as a ResEq.
bool store_row(const secred::FwdStore &store, uint32_t id, ResEq &out) {
  if (!store.has_row(id))
    return false;
  const uint32_t *ids = store.row_ids(id);
  const uint64_t *vals = store.row_vals(id);
  const uint32_t n = store.row_size(id);
  ResEq row;
  row.reserve(n);
  for (uint32_t i = 0; i < n; ++i)
    row.emplace_back(ids[i], vals[i]);
  out = std::move(row);
  return true;
}

// Every row of the reference solmap must be a row of the store, term for term.
void compare_rows(const std::unordered_map<uint32_t, ResEq> &ref,
                  const secred::FwdStore &store, std::size_t n_ids) {
  ResEq row;
  for (const auto &kv : ref) {
    REQUIRE(store.has_row(kv.first));
    REQUIRE(store_row(store, kv.first, row));
    REQUIRE(row == kv.second);
  }
  // ... and no row the reference does not have (a rewritten chain row replaces the row it
  // came from, so the count is the pivot count either way).
  std::size_t n = 0;
  for (uint32_t id = 0; id < n_ids; ++id)
    n += store.has_row(id) ? 1 : 0;
  REQUIRE_EQ(n, ref.size());
}

// The pool's term table for `system` at one prime: every system term gets its own slot and
// the slot's value is that term's residue -- 0 for a coefficient that vanished here, which
// is how the elimination drops it.  The table holds the *unreduced* rows, as the
// coefficient pool does, so it is what the plan was built from.
void pool_of(const std::vector<Eq> &system, uint64_t p,
             std::vector<std::pair<uint32_t, uint32_t>> &terms,
             std::vector<uint32_t> &eq_begin, std::vector<uint64_t> &values,
             std::vector<uint32_t> &sys_idx) {
  terms.clear();
  eq_begin.assign(1, 0u);
  values.clear();
  sys_idx.clear();
  for (std::size_t i = 0; i < system.size(); ++i) {
    for (const auto &t : system[i]) {
      terms.emplace_back(t.first, static_cast<uint32_t>(values.size()));
      values.push_back(secred::rat_mod_u64(t.second.num, t.second.den, p));
    }
    eq_begin.push_back(static_cast<uint32_t>(terms.size()));
    sys_idx.push_back(static_cast<uint32_t>(i));
  }
}

// Run the reference and the planned path over the same reduced subsystem and compare
// every pivot and every row, before and after back substitution.  The plan is built from
// the *unreduced* system's structure (as the coefficient pool's term table holds it),
// since a plan that followed the residues would differ between primes.
void compare_paths(const std::vector<Eq> &system, uint64_t p, std::size_t n_ids,
                   const std::vector<uint32_t> *only_keys) {
  // The pool's row order is the system's, dropping nothing: rows that evaluate to zero at
  // this prime are empty in the plan's input and sort last there, and the reference drops
  // them outright -- the surviving rows keep their relative order in both.
  std::vector<std::pair<uint32_t, uint32_t>> terms;
  std::vector<uint32_t> eq_begin, sys_idx;
  std::vector<uint64_t> values;
  pool_of(system, p, terms, eq_begin, values, sys_idx);
  secred::FwdPlan plan;
  secred::build_fwd_plan(terms, eq_begin, sys_idx, plan);
  REQUIRE_EQ(plan.order.size(), system.size());
  secred::SymRowTable table{terms.data(), eq_begin.data(), system.size()};

  std::vector<ResEq> ref_eqs;
  std::vector<uint32_t> ref_idx;
  for (std::size_t i = 0; i < system.size(); ++i) {
    ResEq re;
    secred::reduce_mod(system[i], p, re);
    if (!re.empty()) {
      ref_eqs.push_back(std::move(re));
      ref_idx.push_back(static_cast<uint32_t>(i));
    }
  }

  std::unordered_map<uint32_t, ResEq> ref;
  secred::forward_eliminate(ref_eqs, ref_idx, p, ref, nullptr, nullptr);

  secred::FwdStore store;
  store.reset(n_ids);
  secred::forward_eliminate_planned(table, values.data(), values.size(), plan, p, store);
  REQUIRE_EQ(store.rows(), ref.size());
  compare_rows(ref, store, n_ids);

// Back substitution: restricted to the given chain, or over every pivot (as the
// reference does with only_keys == nullptr).
  if (only_keys) {
    secred::backward_eliminate_planned(store, p, *only_keys);
  } else {
    std::vector<uint32_t> all_keys(ref.size());
    std::size_t k = 0;
    for (uint32_t id = 0; id < n_ids; ++id)
      if (store.has_row(id))
        all_keys[k++] = id;
    all_keys.resize(k);
    secred::backward_eliminate_planned(store, p, all_keys);
  }
  secred::backward_eliminate(ref, p, nullptr, only_keys);
  compare_rows(ref, store, n_ids);
}

const std::vector<Eq> &planned_fixture() {
  static const std::vector<Eq> system = [] {
    std::vector<Eq> s;
    s.push_back(eq({{126, {1, 1}}, {0, {-1, 1}}}));
    s.push_back(eq({{127, {1, 1}}, {126, {-1, 1}}, {125, {-2, 1}}, {0, {-1, 1}}}));
    s.push_back(eq({{127, {1, 1}}, {126, {-1, 1}}, {125, {-1, 1}}, {1, {-1, 1}}}));
    s.push_back(eq({{128, {1, 1}}, {127, {-1, 1}}, {0, {-1, 1}}}));
    return s;
  }();
  return system;
}

} // namespace

// The sample-point path inverts one coefficient per pivot row -- ~547k inverses per point
// at F234's size -- so its normalisation cannot pay the reference's 128-bit Euclid loop.
// The 64-bit form has to agree with the reference on every residue the reconstruction will
// see, and its boundaries matter more than its random cases: 1 and p-1 are the ends of the
// range, and 0 (not invertible) is what both forms document as 0.
TEST(ffred_fast_inverse_matches_reference) {
  std::mt19937_64 rng(20261014ULL);
  for (unsigned pi = 1; pi <= 4; ++pi) {
    const uint64_t p = fflow::BIG_UINT_PRIMES[pi];
    REQUIRE_EQ(secred::mod_inverse_u64_fast(1, p), uint64_t(1));
    REQUIRE_EQ(secred::mod_inverse_u64_fast(p - 1, p), p - 1);
    REQUIRE_EQ(secred::mod_inverse_u64_fast(0, p), uint64_t(0));
    for (int k = 0; k < 2000; ++k) {
      const uint64_t a = rng() % p;
      const uint64_t inv = secred::mod_inverse_u64_fast(a, p);
      REQUIRE_EQ(inv, secred::mod_inverse_u64(a, p));
      if (a)
        REQUIRE_EQ(static_cast<uint64_t>((static_cast<unsigned __int128>(a) * inv) % p),
                   uint64_t(1));
    }
  }
  // The small prime the fixture tests use is not one of fflow's, so cover it too.
  REQUIRE_EQ(secred::mod_inverse_u64_fast(1, P), uint64_t(1));
  REQUIRE_EQ(secred::mod_inverse_u64_fast(P - 1, P), P - 1);
  REQUIRE_EQ(secred::mod_inverse_u64_fast(7, P),
             secred::mod_inverse_u64(7, P));
  // Exhaustive on small odd primes: every residue, even ones included, since a halving
  // step on an even operand is where a binary extended Euclid goes wrong.
  for (uint64_t p : {uint64_t(3), uint64_t(5), uint64_t(101), uint64_t(65537)}) {
    for (uint64_t a = 0; a < p; ++a) {
      const uint64_t inv = secred::mod_inverse_u64_fast(a, p);
      REQUIRE_EQ(inv, secred::mod_inverse_u64(a, p));
      if (a)
        REQUIRE_EQ((a * inv) % p, uint64_t(1));
    }
  }
}

// Shoup's division-free product must equal the plain 128-bit product at every prime the
// reconstruction draws, for arbitrary residues.
TEST(ffred_shoup_mul_matches_reference_product) {
  std::mt19937_64 rng(12345);
  for (unsigned pi = 1; pi <= 4; ++pi) {
    const uint64_t p = fflow::BIG_UINT_PRIMES[pi];
    for (int k = 0; k < 2000; ++k) {
      const uint64_t c = rng() % p;
      const uint64_t x = rng() % p;
      const uint64_t cp = secred::shoup_precomp(c, p);
      const uint64_t want =
          static_cast<uint64_t>((static_cast<unsigned __int128>(x) * c) % p);
      REQUIRE_EQ(secred::shoup_mul(x, c, cp, p), want);
    }
    // The second factor may be anything below 2^64, not only a residue.
    const uint64_t c = p - 1, cp = secred::shoup_precomp(c, p);
    for (uint64_t x : {uint64_t(0), uint64_t(1), ~uint64_t(0), ~uint64_t(0) - 1, p}) {
      const uint64_t want =
          static_cast<uint64_t>((static_cast<unsigned __int128>(x) * c) % p);
      REQUIRE_EQ(secred::shoup_mul(x, c, cp, p), want);
    }
  }
}

// The plan is structure only: rows order by front id, then length, then the id list, then
// sys_idx -- and rows that are structurally empty sort last (the elimination skips them).
TEST(ffred_fwd_plan_is_structure_order) {
  // Two rows with the same front and length but different ids, two identical in structure
  // (tie broken by sys_idx), one shorter row with the same front, one empty row.
  std::vector<Eq> system = planned_fixture();
  system.push_back(eq({{127, {1, 1}}, {126, {-1, 1}}, {125, {-3, 1}}, {0, {-1, 1}}}));
  // An equation with no terms at all: an empty row in the pool.
  system.push_back(Eq{});

  std::vector<std::pair<uint32_t, uint32_t>> terms;
  std::vector<uint32_t> eq_begin{0u}, sys_idx;
  for (std::size_t i = 0; i < system.size(); ++i) {
    for (const auto &t : system[i])
      terms.emplace_back(t.first, 0u);
    eq_begin.push_back(static_cast<uint32_t>(terms.size()));
    sys_idx.push_back(static_cast<uint32_t>(i));
  }
  secred::FwdPlan plan;
  secred::build_fwd_plan(terms, eq_begin, sys_idx, plan);
  REQUIRE_EQ(plan.order.size(), 6u);

  // Rows 0..3 are the fixture, row 4 is a second front-127 row with rows 1's exact ids
  // (and a different coefficient), row 5 is empty:
  //   front 126 first, then the front-127 rows -- ids decide rows 1/4 against row 2, and
  //   sys_idx decides rows 1 and 4 against each other -- then front 128, empty last.
  const std::vector<uint32_t> want{0u, 1u, 4u, 2u, 3u, 5u};
  REQUIRE(plan.order == want);
}

// The plan is built once and shared by every prime and every point, so a point's pivots
// must not depend on the residues: two primes that differ in the coefficients give the
// same pivot set (the reference's coefficient tie-break never fires on these systems).
TEST(ffred_planned_pivots_are_prime_independent) {
  const std::vector<Eq> &system = planned_fixture();
  // One pool, one plan, three primes: only the slot values change between the rounds, as
  // they do between the sample points of one reconstruction.
  std::vector<std::pair<uint32_t, uint32_t>> terms;
  std::vector<uint32_t> eq_begin, sys_idx;
  for (std::size_t i = 0; i < system.size(); ++i) {
    for (const auto &t : system[i])
      terms.emplace_back(t.first, static_cast<uint32_t>(terms.size()));
    eq_begin.push_back(static_cast<uint32_t>(terms.size()));
    sys_idx.push_back(static_cast<uint32_t>(i));
  }
  eq_begin.insert(eq_begin.begin(), 0u);
  secred::FwdPlan plan;
  secred::build_fwd_plan(terms, eq_begin, sys_idx, plan);
  secred::SymRowTable table{terms.data(), eq_begin.data(), system.size()};

  std::vector<uint32_t> first;
  std::vector<uint64_t> values(terms.size(), 0);
  for (unsigned pi = 1; pi <= 3; ++pi) {
    const uint64_t p = fflow::BIG_UINT_PRIMES[pi];
    std::size_t v = 0;
    for (const Eq &e : system)
      for (const auto &t : e)
        values[v++] = secred::rat_mod_u64(t.second.num, t.second.den, p);

    secred::FwdStore store;
    store.reset(129u);
    secred::forward_eliminate_planned(table, values.data(), values.size(), plan, p, store);
    std::vector<uint32_t> pivots;
    for (uint32_t id = 0; id < 129u; ++id)
      if (store.has_row(id))
        pivots.push_back(id);
    if (first.empty())
      first = pivots;
    REQUIRE(pivots == first);
  }
  REQUIRE_EQ(first.size(), 4u); // 125, 126, 127, 128
}

// Forward and back substitution on the planned path must reproduce the reference pivots
// and rows exactly, at several primes, with and without a restricted back substitution.
TEST(ffred_planned_matches_reference) {
  const std::vector<Eq> &system = planned_fixture();
  const uint64_t primes[] = {P, 1000000009ULL, 998244353ULL, fflow::BIG_UINT_PRIMES[1]};
  const std::vector<uint32_t> chain{1u, 125u, 126u, 127u};
  for (uint64_t p : primes) {
    compare_paths(system, p, 129u, nullptr);
    compare_paths(system, p, 129u, &chain);
  }
}

// The whole path on random systems: many structures, many primes, both directions.  The
// plan is shared between the primes, as it is in production.
TEST(ffred_planned_matches_reference_on_random_systems) {
  std::mt19937_64 rng(987654321ULL);
  const uint64_t primes[] = {P, 1000000009ULL, fflow::BIG_UINT_PRIMES[2]};
  for (int trial = 0; trial < 24; ++trial) {
    std::vector<Eq> system;
    const unsigned nrows = 4 + static_cast<unsigned>(rng() % 10);
    const uint32_t n_ids = 40;
    for (unsigned r = 0; r < nrows; ++r) {
      const uint32_t front = 1 + static_cast<uint32_t>(rng() % (n_ids - 1));
      Eq e;
      e.emplace_back(front, Rat64{1, 1});
      std::unordered_set<uint32_t> used{front};
      const unsigned tail = 1 + static_cast<unsigned>(rng() % 4);
      std::vector<uint32_t> ids;
      for (unsigned k = 0; k < tail; ++k) {
        const uint32_t id = static_cast<uint32_t>(rng() % front);
        if (used.insert(id).second)
          ids.push_back(id);
      }
      std::sort(ids.begin(), ids.end(), std::greater<uint32_t>()); // descending, as the reader emits
      for (uint32_t id : ids) {
        const int64_t num = 1 + static_cast<int64_t>(rng() % 7);
        e.emplace_back(id, Rat64{rng() % 2 ? num : -num, 1});
      }
      system.push_back(std::move(e));
    }
    for (uint64_t p : primes)
      compare_paths(system, p, n_ids, nullptr);
    // Restricted back substitution over an ascending chain of the fixture's ids.
    const std::vector<uint32_t> chain{1u, 2u, 3u, 5u, 8u, 13u, 21u};
    for (uint64_t p : primes)
      compare_paths(system, p, n_ids, &chain);
  }
}

// A coefficient that vanishes at the prime drops out of its row, so the row the reference
// orders is not the pool row the plan was built from.  The planned path must notice and
// order the rows the reference's way -- otherwise it eliminates in a different order, and
// the chain-restricted back substitution then leaves a pivot on a chain row's right-hand
// side where the reference leaves none (in production: a NonMasterRhs abort reporting a
// stale selection that is fine).
TEST(ffred_planned_reorders_rows_when_a_coefficient_vanishes) {
  // The last row's front coefficient is 7/1, i.e. 0 at p = 7: its evaluated row is {0, -3}
  // where the pool row is {5, 0}.  Every other coefficient is below 7 in absolute value,
  // so this is the only vanishing one.
  std::vector<Eq> system;
  system.push_back(eq({{3, {1, 1}}, {2, {-1, 1}}}));
  system.push_back(eq({{3, {1, 1}}, {2, {3, 1}}, {0, {-3, 1}}}));
  system.push_back(eq({{5, {7, 1}}, {0, {-3, 1}}}));
  const uint64_t p = 7;
  const std::size_t n_ids = 8u;

  // The two orders really do differ here, so the fallback is what this test exercises.
  secred::FwdPlan plan;
  std::vector<std::pair<uint32_t, uint32_t>> terms;
  std::vector<uint32_t> eq_begin{0u}, sys_idx;
  for (std::size_t i = 0; i < system.size(); ++i) {
    for (const auto &t : system[i])
      terms.emplace_back(t.first, 0u);
    eq_begin.push_back(static_cast<uint32_t>(terms.size()));
    sys_idx.push_back(static_cast<uint32_t>(i));
  }
  secred::build_fwd_plan(terms, eq_begin, sys_idx, plan);
  std::vector<ResEq> rows(system.size());
  for (std::size_t i = 0; i < system.size(); ++i)
    secred::reduce_mod(system[i], p, rows[i]);
  std::vector<uint32_t> from_rows;
  secred::fwd_order_from_rows(rows, sys_idx, from_rows);
  REQUIRE_EQ(rows[2].size(), 1u); // the vanishing coefficient left {0, -3}
  REQUIRE(plan.order != from_rows);

  // Both passages must still agree with the reference, with and without a restricted chain.
  compare_paths(system, p, n_ids, nullptr);
  const std::vector<uint32_t> chain{3u};
  compare_paths(system, p, n_ids, &chain);
}

// The plan check reads the slot array, not the rows, so it must catch a coefficient that
// two rows share -- one slot, two rows reordered, which is what interning produces: at
// F234's size 7.2M terms are 1329 distinct functions, and a row's stored form is never
// materialised just to be checked against the plan.
TEST(ffred_planned_sees_a_vanishing_shared_coefficient) {
  // Row 0's front and row 1's second term are the *same* pool slot, 7/1, i.e. 0 at p = 7:
  // both rows are shorter there, and row 0's front moves from 6 down to 4, which swaps the
  // two rows in the reference's order.
  std::vector<Eq> system;
  system.push_back(eq({{6, {7, 1}}, {4, {-1, 1}}}));
  system.push_back(eq({{5, {1, 1}}, {4, {7, 1}}}));
  const uint64_t p = 7;
  const std::size_t n_ids = 8u;

  // The pool: one slot for the shared 7/1, one for each of the other two coefficients.
  const std::vector<std::pair<uint32_t, uint32_t>> terms{{6u, 0u}, {4u, 1u},
                                                         {5u, 2u}, {4u, 0u}};
  const std::vector<uint32_t> eq_begin{0u, 2u, 4u};
  const std::vector<uint32_t> sys_idx{0u, 1u};
  const std::vector<uint64_t> values{secred::rat_mod_u64(7, 1, p),
                                     secred::rat_mod_u64(-1, 1, p),
                                     secred::rat_mod_u64(1, 1, p)};
  REQUIRE_EQ(values[0], uint64_t(0)); // the shared coefficient vanishes here

  secred::FwdPlan plan;
  secred::build_fwd_plan(terms, eq_begin, sys_idx, plan);
  secred::SymRowTable table{terms.data(), eq_begin.data(), system.size()};

  // What the reference orders at this point: both rows lost the shared term.
  std::vector<ResEq> rows(system.size());
  std::vector<uint32_t> from_rows;
  for (std::size_t i = 0; i < system.size(); ++i)
    secred::reduce_mod(system[i], p, rows[i]);
  REQUIRE_EQ(rows[0].size(), std::size_t(1));
  REQUIRE_EQ(rows[1].size(), std::size_t(1));
  secred::fwd_order_from_rows(rows, sys_idx, from_rows);
  REQUIRE(plan.order != from_rows); // the two orders really do differ here

  secred::FwdStore store;
  store.reset(n_ids);
  secred::forward_eliminate_planned(table, values.data(), values.size(), plan, p, store);
  REQUIRE(store.order() == from_rows);

  // The reference over the same evaluated rows: same pivots, same rows, term for term.
  std::vector<ResEq> ref_eqs;
  std::vector<uint32_t> ref_idx;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (!rows[i].empty()) {
      ref_eqs.push_back(rows[i]);
      ref_idx.push_back(static_cast<uint32_t>(i));
    }
  }
  std::unordered_map<uint32_t, ResEq> ref;
  secred::forward_eliminate(ref_eqs, ref_idx, p, ref, nullptr, nullptr);
  compare_rows(ref, store, n_ids);

  // ... and the same table at a prime where nothing vanishes replays the plan, unchanged.
  const uint64_t q = fflow::BIG_UINT_PRIMES[1];
  const std::vector<uint64_t> clean{secred::rat_mod_u64(7, 1, q),
                                    secred::rat_mod_u64(-1, 1, q),
                                    secred::rat_mod_u64(1, 1, q)};
  secred::FwdStore at_plan;
  at_plan.reset(n_ids);
  secred::forward_eliminate_planned(table, clean.data(), clean.size(), plan, q, at_plan);
  REQUIRE(at_plan.order() == plan.order);

  // Both passages agree with the reference, with and without a restricted chain.
  compare_paths(system, p, n_ids, nullptr);
  const std::vector<uint32_t> chain{3u};
  compare_paths(system, p, n_ids, &chain);
}

// A store must forget every row on reset: a second point at another prime may not see the
// first point's memberships.
TEST(ffred_planned_store_reset_forgets_rows) {
  const std::vector<Eq> &system = planned_fixture();
  std::vector<std::pair<uint32_t, uint32_t>> terms;
  std::vector<uint32_t> eq_begin{0u}, sys_idx;
  for (std::size_t i = 0; i < system.size(); ++i) {
    for (const auto &t : system[i])
      terms.emplace_back(t.first, static_cast<uint32_t>(terms.size()));
    eq_begin.push_back(static_cast<uint32_t>(terms.size()));
    sys_idx.push_back(static_cast<uint32_t>(i));
  }
  secred::FwdPlan plan;
  secred::build_fwd_plan(terms, eq_begin, sys_idx, plan);
  secred::SymRowTable table{terms.data(), eq_begin.data(), system.size()};

  std::vector<uint64_t> values(terms.size(), 0);
  secred::FwdStore store;
  for (uint64_t p : {fflow::BIG_UINT_PRIMES[1], P}) {
    std::size_t v = 0;
    for (const Eq &e : system)
      for (const auto &t : e)
        values[v++] = secred::rat_mod_u64(t.second.num, t.second.den, p);
    store.reset(129u);
    REQUIRE(!store.has_row(127u));
    secred::forward_eliminate_planned(table, values.data(), values.size(), plan, p, store);
    REQUIRE(store.has_row(127u));

    secred::FwdStore fresh;
    fresh.reset(129u);
    secred::forward_eliminate_planned(table, values.data(), values.size(), plan, p, fresh);
    REQUIRE_EQ(store.rows(), fresh.rows());
    ResEq a, b;
    for (uint32_t id = 0; id < 129u; ++id) {
      REQUIRE_EQ(store.has_row(id), fresh.has_row(id));
      if (store.has_row(id)) {
        REQUIRE(store_row(store, id, a));
        REQUIRE(store_row(fresh, id, b));
        REQUIRE(a == b);
      }
    }
  }
}

// The plan's order is one vector shared by every prime and every point, so a point must
// read it rather than copy it: at F234's size the copy is 547k entries, 2.3 MB, once per
// point.  Only the fallback -- a point where some coefficient vanished -- gives the store
// an order of its own, and then it is the rebuilt one.
TEST(ffred_planned_reads_the_plan_order_without_copying_it) {
  const std::vector<Eq> &system = planned_fixture();
  std::vector<std::pair<uint32_t, uint32_t>> terms;
  std::vector<uint32_t> eq_begin{0u}, sys_idx;
  for (std::size_t i = 0; i < system.size(); ++i) {
    for (const auto &t : system[i])
      terms.emplace_back(t.first, static_cast<uint32_t>(terms.size()));
    eq_begin.push_back(static_cast<uint32_t>(terms.size()));
    sys_idx.push_back(static_cast<uint32_t>(i));
  }
  secred::FwdPlan plan;
  secred::build_fwd_plan(terms, eq_begin, sys_idx, plan);
  secred::SymRowTable table{terms.data(), eq_begin.data(), system.size()};

  // A prime no coefficient of the fixture vanishes at: the plan applies as it stands.
  const uint64_t p = fflow::BIG_UINT_PRIMES[1];
  std::vector<uint64_t> values(terms.size(), 0);
  std::size_t v = 0;
  for (const Eq &e : system)
    for (const auto &t : e)
      values[v++] = secred::rat_mod_u64(t.second.num, t.second.den, p);

  secred::FwdStore store;
  store.reset(129u);
  secred::forward_eliminate_planned(table, values.data(), values.size(), plan, p, store);
  REQUIRE(store.order() == plan.order);
  REQUIRE(store.order_ptr == &plan.order);
  REQUIRE(store.order_own.empty());

  // The vanishing case: the store's own order, and the pointer follows it there.
  const std::vector<std::pair<uint32_t, uint32_t>> shared{{6u, 0u}, {4u, 1u},
                                                          {5u, 2u}, {4u, 0u}};
  const std::vector<uint32_t> shared_begin{0u, 2u, 4u};
  const std::vector<uint32_t> shared_idx{0u, 1u};
  const uint64_t q = 7;
  const std::vector<uint64_t> vanishing{secred::rat_mod_u64(7, 1, q),
                                        secred::rat_mod_u64(-1, 1, q),
                                        secred::rat_mod_u64(1, 1, q)};
  secred::FwdPlan shared_plan;
  secred::build_fwd_plan(shared, shared_begin, shared_idx, shared_plan);
  secred::SymRowTable shared_table{shared.data(), shared_begin.data(), 2u};
  secred::FwdStore fallback;
  fallback.reset(8u);
  secred::forward_eliminate_planned(shared_table, vanishing.data(), vanishing.size(),
                                    shared_plan, q, fallback);
  REQUIRE(fallback.order_ptr == &fallback.order_own);
  REQUIRE(fallback.order() != shared_plan.order);
}

// The fallback order is the risky half of the sample-point path: a point where a
// coefficient vanishes orders the rows the reference's way instead of replaying the plan.
// The two tests above drive one vanishing coefficient each; this drives many at once, over
// random structures, so the merge buffers and the fallback order are exercised together.
TEST(ffred_planned_matches_reference_when_many_coefficients_vanish) {
  std::mt19937_64 rng(20261014ULL);
  const uint64_t p = 7; // small enough that "a multiple of p" is common in the draw below
  const uint32_t n_ids = 40;
  for (int trial = 0; trial < 24; ++trial) {
    std::vector<Eq> system;
    const unsigned nrows = 4 + static_cast<unsigned>(rng() % 10);
    for (unsigned r = 0; r < nrows; ++r) {
      const uint32_t front = 1 + static_cast<uint32_t>(rng() % (n_ids - 1));
      Eq e;
      e.emplace_back(front, Rat64{1, 1});
      std::unordered_set<uint32_t> used{front};
      const unsigned tail = 1 + static_cast<unsigned>(rng() % 4);
      std::vector<uint32_t> ids;
      for (unsigned k = 0; k < tail; ++k) {
        const uint32_t id = static_cast<uint32_t>(rng() % front);
        if (used.insert(id).second)
          ids.push_back(id);
      }
      std::sort(ids.begin(), ids.end(), std::greater<uint32_t>());
      for (uint32_t id : ids) {
        int64_t num = 1 + static_cast<int64_t>(rng() % 7);
        if (rng() % 2)
          num = -num;
        if (rng() % 3 == 0)
          num *= 7; // 0 at p = 7: this row loses the term at this point
        e.emplace_back(id, Rat64{num, 1});
      }
      system.push_back(std::move(e));
    }
    compare_paths(system, p, n_ids, nullptr);
    const std::vector<uint32_t> chain{1u, 2u, 3u, 5u};
    compare_paths(system, p, n_ids, &chain);
  }
}
