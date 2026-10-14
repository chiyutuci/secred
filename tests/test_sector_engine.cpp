#include "test_runner.h"
#include "global_select.h"
#include "sector_engine.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using secred::ResEq;
using secred::SubSolve;

constexpr uint64_t P = 1000000007ULL;

// Column convention of the per-sector matrix: a smaller column number is deeper / more
// complex; seeds use col = nvars-1-weight and sub-sectors col = nvars+j.  This fixture
// takes ncols = 129, so id = ncols-1-col puts every integral on the seed side (id = weight),
// matching derived_fixture in tests/test_ffred.cpp one to one.
constexpr unsigned NCOLS = 129;

unsigned col_of(unsigned id) { return NCOLS - 1 - id; }

// Write a row by id and convert to columns internally, which also exercises
// res_eq_from_row's column -> id mapping.
ResEq row(std::initializer_list<std::pair<unsigned, int64_t>> terms) {
  std::vector<unsigned> cols;
  std::vector<uint64_t> vals;
  for (const auto &t : terms) {
    cols.push_back(col_of(t.first));
    int64_t v = t.second % static_cast<int64_t>(P);
    if (v < 0)
      v += static_cast<int64_t>(P);
    vals.push_back(static_cast<uint64_t>(v));
  }
  return secred::res_eq_from_row(cols.data(), vals.data(),
                                 static_cast<unsigned>(cols.size()), NCOLS, P);
}

// Derived-integral fixture (isomorphic to derived_fixture in test_ffred.cpp, written with
// the per-sector column convention):
//   eq0: I_126 = I_0
//   eq1: I_127 = 2 I_125 + I_126 + I_0   (registers 127)
//   eq2: I_127 = I_125 + I_126 + I_1     (registers 125 after substituting 127/126)
//   eq3: I_128 = I_127 + I_0             (unrelated)
// eq0 gives its terms in ascending id while eq1/eq2 are unordered: res_eq_from_row must
// sort them descending itself.
std::vector<ResEq> derived_fixture() {
  return {
      row({{0, -1}, {126, 1}}),
      row({{126, -1}, {127, 1}, {0, -1}, {125, -2}}),
      row({{125, -1}, {1, -1}, {127, 1}, {126, -1}}),
      row({{128, 1}, {127, -1}, {0, -1}}),
  };
}

// Full-system learn plus master extraction, the same calls sector.cpp's ffred branch makes.
struct SectorCtx {
  std::vector<ResEq> eqs = derived_fixture();
  SubSolve learn;
  std::vector<unsigned> target_ids{127u};
  std::vector<unsigned> masters;                       // descending id = ascending column
  std::unordered_map<unsigned, unsigned> master_index; // master -> output column
  std::vector<unsigned> dep_targets;

  explicit SectorCtx(std::vector<unsigned> targets = {127u})
      : target_ids(std::move(targets)) {
    std::vector<uint32_t> idx;
    idx.reserve(eqs.size());
    for (uint32_t i = 0; i < eqs.size(); ++i)
      idx.push_back(i);
    secred::solve_res_subset(eqs, idx, P, learn, /*want_fwd=*/true,
                             /*want_bwd=*/true);
    masters = secred::sector_masters(learn, target_ids);
    for (unsigned j = 0; j < masters.size(); ++j)
      master_index[masters[j]] = j;
    for (unsigned id : target_ids)
      if (learn.solmap.find(id) != learn.solmap.end())
        dep_targets.push_back(id);
  }

  // Isomorphic to the callback sector.cpp injects: re-solve the subset at the learn prime,
  // forward part only.
  void solve(const std::vector<uint32_t> &idx, SubSolve &out) const {
    secred::solve_res_subset(eqs, idx, P, out, /*want_fwd=*/true,
                             /*want_bwd=*/false);
  }

  GlobalSelection select(unsigned max_rounds = 64,
                         unsigned min_gain_pct = kRepairMinGainPct) const {
    return select_global_forward(
        learn.eq_index, learn.fwd_deps, learn.bwd_deps, learn.solmap, dep_targets,
        master_index,
        [&](const std::vector<uint32_t> &idx, SubSolve &out) { solve(idx, out); },
        max_rounds, min_gain_pct);
  }
};

// Sufficiency test: a reducible target's right-hand side may hold masters only.
bool rhs_only_masters(
    const SubSolve &sub, const std::vector<unsigned> &dep_targets,
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

} // namespace

// Column -> id mapping, descending order, duplicate columns merged mod p, resulting zeros dropped.
TEST(sector_res_eq_from_row_maps_and_normalizes) {
  // Column -> id = ncols-1-col; the output is always descending in id.
  {
    const unsigned cols[] = {col_of(0), col_of(128), col_of(2)};
    const uint64_t vals[] = {7, 3, 5};
    ResEq re = secred::res_eq_from_row(cols, vals, 3, NCOLS, P);
    REQUIRE_EQ(re.size(), 3u);
    REQUIRE_EQ(re[0].first, 128u);
    REQUIRE_EQ(re[0].second, 3u);
    REQUIRE_EQ(re[1].first, 2u);
    REQUIRE_EQ(re[1].second, 5u);
    REQUIRE_EQ(re[2].first, 0u);
    REQUIRE_EQ(re[2].second, 7u);
  }

  // The same column twice: coefficients are summed mod p.  fill_matrix does not deduplicate,
  // so this step is required.
  {
    const unsigned cols[] = {col_of(127), col_of(0), col_of(127)};
    const uint64_t vals[] = {3, 4, P - 2};
    ResEq re = secred::res_eq_from_row(cols, vals, 3, NCOLS, P);
    REQUIRE_EQ(re.size(), 2u);
    REQUIRE_EQ(re[0].first, 127u);
    REQUIRE_EQ(re[0].second, 1u); // 3 + (p-2) = 1
    REQUIRE_EQ(re[1].first, 0u);
    REQUIRE_EQ(re[1].second, 4u);
  }

  // Exact cancellation after merging drops the term (ResEq forbids zeros).
  {
    const unsigned cols[] = {col_of(127), col_of(127)};
    const uint64_t vals[] = {1, P - 1};
    ResEq re = secred::res_eq_from_row(cols, vals, 2, NCOLS, P);
    REQUIRE(re.empty());
  }

  // An input zero coefficient leaves no term either; an all-zero row gives an empty ResEq,
  // which the caller skips.
  {
    const unsigned cols[] = {col_of(5), col_of(0)};
    const uint64_t vals[] = {0, 0};
    ResEq re = secred::res_eq_from_row(cols, vals, 2, NCOLS, P);
    REQUIRE(re.empty());
  }

  // Coefficients are reduced mod p before merging (alg.c already holds residues; this just
  // does not trust the input).
  {
    const unsigned cols[] = {col_of(3)};
    const uint64_t vals[] = {P + 9};
    ResEq re = secred::res_eq_from_row(cols, vals, 1, NCOLS, P);
    REQUIRE_EQ(re.size(), 1u);
    REQUIRE_EQ(re[0].second, 9u);
  }
}

// Masters = union over reducible targets of RHS(solmap[t]), plus every irreducible target,
// in descending id.
TEST(sector_masters_matches_global_formula) {
  SectorCtx c;

  // Reducible target I_127: after full back substitution I_127 = 2 I_1, so I_1 is the only
  // master.  (I_0's contribution cancels exactly -- the hidden zero.)
  REQUIRE_EQ(c.masters.size(), 1u);
  REQUIRE_EQ(c.masters[0], 1u);
  REQUIRE_EQ(c.learn.solmap.at(127u).size(), 2u);
  REQUIRE_EQ(c.learn.solmap.at(127u)[0].first, 127u);
  REQUIRE_EQ(c.learn.solmap.at(127u)[0].second, P - 1);
  REQUIRE_EQ(c.learn.solmap.at(127u)[1].first, 1u);
  REQUIRE_EQ(c.learn.solmap.at(127u)[1].second, 2u);

  // Irreducible targets (no equation of their own, e.g. integrals outside this stage's top
  // layer) must be counted as they are: they are the legacy "needed variables of non-pivot
  // columns" and travel up through uptargets.
  SectorCtx u({127u, 3u});
  REQUIRE_EQ(u.masters.size(), 2u);
  REQUIRE_EQ(u.masters[0], 3u); // descending id
  REQUIRE_EQ(u.masters[1], 1u);

  // Irreducible targets are not dep_targets (no reduction, so no right-hand side to read).
  REQUIRE_EQ(u.dep_targets.size(), 1u);
  REQUIRE_EQ(u.dep_targets[0], 127u);
}

// Selection: forward closure plus numeric repair, ending with masters only on the
// right-hand side and the same solution as the legacy closure.
TEST(sector_selection_forward_then_repair) {
  SectorCtx c;

  // The same dependency shape as the global stage: registering 127 substituted 126, and 125
  // is reachable only through bwd_deps[127].
  REQUIRE_EQ(c.learn.fwd_deps.at(127u).size(), 1u);
  REQUIRE_EQ(c.learn.fwd_deps.at(127u)[0], 126u);
  REQUIRE(c.learn.bwd_deps.find(125u) == c.learn.bwd_deps.end());

  // Forward closure only: {127} plus fwd_deps[127] = {127,126} -> equations {eq0,eq1}; after
  // re-solving the right-hand side still carries I_125 (reducible, but its equation was not
  // selected), so the selection is insufficient.
  GlobalSelection fwd_only = c.select(0u);
  REQUIRE(!fwd_only.ok);
  REQUIRE(fwd_only.fallback);
  REQUIRE_EQ(fwd_only.n_fwd_rows, 2u);
  REQUIRE(fwd_only.needed_eq_idx == (std::vector<uint32_t>{0u, 1u}));
  SubSolve sub_fwd;
  c.solve(fwd_only.needed_eq_idx, sub_fwd);
  REQUIRE(!rhs_only_masters(sub_fwd, c.dep_targets, c.master_index));

  // One repair round adds eq2, giving the same equations as the legacy (forward and backward)
  // closure.
  GlobalSelection sel = c.select();
  REQUIRE(sel.ok);
  REQUIRE(!sel.fallback);
  REQUIRE_EQ(sel.n_repair_rounds, 1u);
  REQUIRE_EQ(sel.n_repair_added, 1u);
  REQUIRE(sel.needed_eq_idx == (std::vector<uint32_t>{0u, 1u, 2u}));

  // The selection is a subset of all equations and sufficient after the repair.
  for (uint32_t i : sel.needed_eq_idx)
    REQUIRE(i < c.eqs.size());
  SubSolve fin;
  c.solve(sel.needed_eq_idx, fin);
  REQUIRE(rhs_only_masters(fin, c.dep_targets, c.master_index));
  REQUIRE_EQ(fin.solmap.at(127u).size(), 2u);
  REQUIRE_EQ(fin.solmap.at(127u)[1].first, 1u);
  REQUIRE_EQ(fin.solmap.at(127u)[1].second, 2u);
}

// The selection and the row mapping are deterministic: the same input twice is identical,
// which cross-run reproducibility requires.
TEST(sector_selection_deterministic) {
  SectorCtx c;
  GlobalSelection a = c.select();
  GlobalSelection b = c.select();
  REQUIRE(a.needed_eq_idx == b.needed_eq_idx);
  REQUIRE_EQ(a.n_repair_rounds, b.n_repair_rounds);
  REQUIRE_EQ(a.n_repair_added, b.n_repair_added);
  REQUIRE_EQ(a.n_fwd_rows, b.n_fwd_rows);

  const auto ea = derived_fixture();
  const auto eb = derived_fixture();
  REQUIRE_EQ(ea.size(), eb.size());
  for (std::size_t i = 0; i < ea.size(); ++i)
    REQUIRE(ea[i] == eb[i]);

  // The master list is deterministic too (built from a std::set, independent of iteration order).
  SectorCtx c2;
  REQUIRE(c.masters == c2.masters);
}

// Hitting the repair cap gives fallback; the fallback closure (forward and backward, today's
// mark_and_sweep_eqs semantics) must cover the forward set, or the fallback would lose equations.
TEST(sector_fallback_covers_legacy_closure) {
  SectorCtx c;

  GlobalSelection capped = c.select(0u);
  REQUIRE(capped.fallback);
  REQUIRE_EQ(capped.n_repair_rounds, 0u);
  REQUIRE_EQ(capped.n_repair_added, 0u);

  std::vector<uint32_t> closure =
      secred::legacy_closure(c.dep_targets, c.learn.fwd_deps, c.learn.bwd_deps,
                             c.learn.eq_index);
  REQUIRE(closure == (std::vector<uint32_t>{0u, 1u, 2u}));

  // The fallback is a superset of the forward set.
  for (uint32_t i : capped.needed_eq_idx)
    REQUIRE(std::find(closure.begin(), closure.end(), i) != closure.end());

  // The fallback is sufficient: same solution as a successful repair selection.
  SubSolve sub;
  c.solve(closure, sub);
  REQUIRE(rhs_only_masters(sub, c.dep_targets, c.master_index));

  // The closure is closed under forward and backward edges (no reachable edge leaves the
  // seed set) -- re-checked with another seed set, which also covers the degenerate case of
  // an empty dep_targets (a sector with no reducible target).
  std::vector<unsigned> none;
  std::vector<uint32_t> empty_closure = secred::legacy_closure(
      none, c.learn.fwd_deps, c.learn.bwd_deps, c.learn.eq_index);
  REQUIRE(empty_closure.empty());
}
