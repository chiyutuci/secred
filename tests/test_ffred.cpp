#include "test_runner.h"
#include "global_select.h"
#include "ffred_solver.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
