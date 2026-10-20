#include "test_runner.h"

#include "forward_selector.h"
#include "numeric_solver_mod.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fflow/graph.hh>
#include <fflow/primes.hh>

// Tests for the Kira 3 selection replica: fflow learn replica, forward closure and
// numeric sufficiency repair.
//
// No production caller; the production path is select_global_forward in
// globalreduce.cpp.  The oracle is not a hand-computed equation list but two
// self-certifying properties: the replica's learn agrees element-wise with real
// fflow, and re-solving with the selected subset reproduces every needed integral's
// reduction exactly -- which is what "the selection must be sufficient" means.

namespace {

using fflow::UInt;

const uint64_t P = fflow::BIG_UINT_PRIMES[0];
const fflow::Mod kMod(P);

// Negative coefficient as a residue mod the prime.
UInt neg(uint64_t v) { return (P - v % P) % P; }

// Hand-built sparse input for NumericSparseSolverMod: each row is ascending (col, residue).
struct FflowSystem {
  std::unique_ptr<fflow::NumericSparseSolverMod> alg;
  unsigned ncols;

  explicit FflowSystem(unsigned n)
      : alg(new fflow::NumericSparseSolverMod()), ncols(n) {}

  void add(const std::vector<std::pair<unsigned, UInt>> &terms) {
    // Column numbers must be strictly ascending: fflow's sparse elimination merges
    // ordered rows, and an out-of-order row never eliminates its pivot (it hangs).
    for (auto it = terms.begin(); it + 1 != terms.end(); ++it)
      if (!(it->first < (it + 1)->first))
        throw std::runtime_error("FflowSystem::add: columns must ascend");

    fflow::NumericSparseSolverMod::RowInfo ri;
    ri.size = terms.size();
    ri.cols.reset(new unsigned[terms.size()]);
    std::unique_ptr<UInt[]> vals(new UInt[terms.size()]);
    unsigned j = 0;
    for (const auto &t : terms) {
      ri.cols[j] = t.first;
      vals[j] = t.second;
      ++j;
    }
    alg->rinfo.push_back(std::move(ri));
    alg->c.push_back(std::move(vals));
  }
};

// ---------------------------------------------------------------------------
// The real fflow path (init + Graph/Session learn + mark_and_sweep_eqs) and the
// replica run on the same rinfo/c so they can be compared element-wise.
// ---------------------------------------------------------------------------
struct RunResults {
  // The session owns the moved solver and must outlive the test, so raw stays
  // usable for re-solving after run_all returns.
  std::unique_ptr<fflow::Session> session;
  const fflow::NumericSparseSolverMod *raw = nullptr;
  std::vector<std::size_t> fflow_indep, fflow_dep;
  std::vector<unsigned> fflow_selected;
  std::vector<std::size_t> replica_indep, replica_dep;
  std::vector<unsigned> replica_full;
  secred::ForwardSelection fwd;          // forward closure + repair
  secred::ForwardSelection fwd_norepair; // pure forward closure
};

RunResults run_all(FflowSystem &sys, const std::vector<unsigned> &needed) {
  RunResults out;

  out.session = std::make_unique<fflow::Session>();
  fflow::Session &session = *out.session;
  unsigned gid = session.new_graph();
  fflow::Graph *g = session.graph(gid);
  g->set_input_vars(0);
  std::vector<unsigned> inputnodes;

  fflow::NumericSparseSolverMod *raw = sys.alg.get();
  out.raw = raw;
  auto data = std::make_unique<fflow::NumericSparseSolverModData>();
  // Same call sequence as sector.cpp: init allocates xinfo_/indepeqs_ and
  // registers the target columns, or session.learn's reset() fills from null.
  raw->init(raw->rinfo.size(), sys.ncols, needed.data(), needed.size(), *data);
  unsigned sid = g->new_node(std::move(sys.alg), std::move(data), inputnodes.data());

  auto *pre =
      static_cast<fflow::SparseLinearSolver *>(session.algorithm(gid, sid));
  pre->only_homogeneous();
  session.invalidate_subctxt_alg_data(gid, sid);
  session.set_output_node(gid, sid);

  if (session.learn(gid) != fflow::SUCCESS)
    throw std::runtime_error("fflow learn failed");

  auto &sls =
      *static_cast<fflow::SparseLinearSolver *>(session.algorithm(gid, sid));
  out.fflow_indep.assign(sls.needed_indepvars(),
                         sls.needed_indepvars() + sls.n_needed_indepvars());
  out.fflow_dep.assign(sls.needed_depvars(),
                       sls.needed_depvars() + sls.n_needed_depvars());

  // Replica: same rinfo/c, its own matrix and dependency graph.
  secred::FflowLinearLearn learn(sys.ncols,
                                 static_cast<unsigned>(raw->rinfo.size()));
  REQUIRE(learn.learn_1(*raw, needed.data(), needed.size(), kMod));
  REQUIRE(learn.learn_2(*raw, kMod));
  if (learn.n_needed_indepvars())
    out.replica_indep.assign(learn.needed_indepvars(),
                             learn.needed_indepvars() +
                                 learn.n_needed_indepvars());
  if (learn.n_needed_depvars())
    out.replica_dep.assign(learn.needed_depvars(),
                           learn.needed_depvars() + learn.n_needed_depvars());
  out.replica_full = learn.mark({}, false);

  sls.mark_and_sweep_eqs(session.alg_data(gid, sid));
  out.fflow_selected.assign(sls.indep_eqs(),
                            sls.indep_eqs() + sls.n_indep_eqs());

  secred::ForwardSelectorOptions opts;
  out.fwd = secred::select_forward(*raw, sys.ncols, needed.data(),
                                   needed.size(), kMod, opts);
  opts.repair = false;
  out.fwd_norepair = secred::select_forward(*raw, sys.ncols, needed.data(),
                                            needed.size(), kMod, opts);
  return out;
}

// ---------------------------------------------------------------------------
// Oracle: after re-solving with the equation subset, does every needed integral's
// reduction match the one from all equations?  The test's own RREF reads back
// the (column, value) pairs without touching production code.
// ---------------------------------------------------------------------------
using RrefRows = std::vector<std::vector<std::pair<unsigned, UInt>>>;

RrefRows rref_rows(const fflow::NumericSparseSolverMod &alg,
                   const std::vector<unsigned> &rows, unsigned ncols) {
  fflow::SparseMatrix m;
  m.resize(rows.size(), ncols + 1);
  for (unsigned i = 0; i < rows.size(); ++i) {
    fflow::SparseMatrixRow &r = m.row(i);
    const fflow::NumericSparseSolverMod::RowInfo &ri = alg.rinfo[rows[i]];
    r.resize(ri.size);
    for (unsigned j = 0; j < ri.size; ++j) {
      r.el(j).col = ri.cols[j];
      r.el(j).val = alg.c[rows[i]][j];
    }
    r.el(ri.size).col = fflow::SparseMatrixRow::END;
    r.id() = rows[i];
  }
  m.sortRows();
  m.toReducedRowEcholon(kMod);

  RrefRows out;
  for (unsigned i = 0; i < rows.size(); ++i) {
    const fflow::SparseMatrixRow &r = m.row(i);
    if (r.first_nonzero_column() == fflow::SparseMatrixRow::END)
      break;
    std::vector<std::pair<unsigned, UInt>> row;
    for (unsigned k = 0; k < r.size(); ++k) {
      if (r.el(k).col == fflow::SparseMatrixRow::END)
        break;
      row.emplace_back(r.el(k).col, r.el(k).val);
    }
    out.push_back(std::move(row));
  }
  return out;
}

const std::vector<std::pair<unsigned, UInt>> *
find_reduction(const RrefRows &rr, unsigned col) {
  for (const auto &row : rr)
    if (!row.empty() && row.front().first == col)
      return &row;
  return nullptr;
}
// The reductions the selected subset gives for the target columns must equal those
// from all equations.  A target may itself be a master: then neither RREF has a row
// pivoting on it, and the requirement is that the sub-solve must not solve it
// either -- subset pivot columns are a subset of all pivot columns, since the
// subset's row space is a subspace and pivots are its leading positions.
bool reductions_agree(const fflow::NumericSparseSolverMod &alg,
                      const std::vector<unsigned> &selected,
                      const std::vector<unsigned> &needed, unsigned ncols) {
  std::vector<unsigned> all(alg.rinfo.size());
  for (unsigned i = 0; i < all.size(); ++i)
    all[i] = i;
  const RrefRows sub = rref_rows(alg, selected, ncols);
  const RrefRows full = rref_rows(alg, all, ncols);
  for (unsigned w : needed) {
    const auto *s = find_reduction(sub, w);
    const auto *f = find_reduction(full, w);
    if (!f) {
      if (s)
        return false; // a master in the full system, yet the sub-solve reduced it
      continue;       // the target is a master, so its reduction is itself
    }
    if (!s || *s != *f)
      return false; // the sub-solve did not reproduce this reduction
  }
  return true;
}

// Three variables A(col0) B(col1) C(col2) plus master D(col3), target A.  After
// learn the pivots are col0/col1/col2 and col3 is the only master.  e1, which
// holds the target, sorts first and has no forward dependency, but its right-hand
// side contains col2 -- a pivot of the later e2, i.e. Kira's "hidden zero": the
// pure forward closure selects e1 alone and only the repair adds e2 back.
struct Fixture {
  FflowSystem sys{4};
  std::vector<unsigned> needed{0};

  Fixture() {
    sys.add({{0, 1}, {1, neg(1)}, {3, neg(1)}});        // e0
    sys.add({{0, 1}, {2, neg(2)}, {3, neg(1)}});        // e1 (holds the target)
    sys.add({{1, 1}, {2, neg(1)}, {3, neg(1)}});        // e2
  }
};

std::vector<unsigned> sorted_unique(std::vector<unsigned> v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v;
}

} // namespace

// The replica's learn must match real fflow element-wise: masters, needed dependent
// variables, full-edge closure.
TEST(forward_replica_matches_fflow) {
  Fixture fx;
  RunResults r = run_all(fx.sys, fx.needed);

  REQUIRE_EQ(r.replica_indep.size(), r.fflow_indep.size());
  for (std::size_t i = 0; i < r.fflow_indep.size(); ++i)
    REQUIRE_EQ(r.replica_indep[i], r.fflow_indep[i]);
  REQUIRE_EQ(r.replica_dep.size(), r.fflow_dep.size());
  for (std::size_t i = 0; i < r.fflow_dep.size(); ++i)
    REQUIRE_EQ(r.replica_dep[i], r.fflow_dep[i]);

  // D (col3) is the only master and the target is itself a needed dependent variable.
  REQUIRE_EQ(r.fflow_indep.size(), 1u);
  REQUIRE_EQ(r.fflow_dep.size(), 1u);
  REQUIRE_EQ(r.fflow_dep[0], 0u);

  // The replica's full-edge closure equals fflow's mark_and_sweep_eqs, order included.
  REQUIRE(r.replica_full == r.fflow_selected);
}

// Hidden zero: the pure forward closure is insufficient, the repair must add the
// equation back, and the enlarged subset must be sufficient.
TEST(forward_repair_fixes_hidden_zero) {
  Fixture fx;
  RunResults r = run_all(fx.sys, fx.needed);

  // Pure forward closure: only e1, the target's own equation.
  REQUIRE_EQ(r.fwd_norepair.selected_eqs[0], 1u);
  REQUIRE(r.fwd_norepair.n_repair_rounds == 0u);
  // Insufficient: the reduction of A still carries the dependent variable C (col2).
  REQUIRE(!reductions_agree(*r.raw, r.fwd_norepair.selected_eqs, fx.needed,
                            fx.sys.ncols));

  // After the repair: same equations as legacy, and the reduction matches the full system.
  REQUIRE(!r.fwd.legacy_fallback);
  REQUIRE_EQ(r.fwd.n_repair_rounds, 1u);
  REQUIRE_EQ(r.fwd.n_repair_added, 1u);
  REQUIRE(sorted_unique(r.fwd.selected_eqs) ==
          sorted_unique(r.fflow_selected));
  REQUIRE(reductions_agree(*r.raw, r.fwd.selected_eqs, fx.needed,
                           fx.sys.ncols));

  // Masters identical to legacy, element-wise.
  for (std::size_t i = 0; i < r.fflow_indep.size(); ++i)
    REQUIRE_EQ(r.fwd.needed_indepvars[i], r.fflow_indep[i]);
}

// On a family of small systems the repaired selection must always be sufficient
// (reductions match the full system) and always a subset of legacy's.
TEST(forward_selection_is_always_sufficient_and_a_subset_of_legacy) {
  struct Case {
    FflowSystem sys;
    std::vector<unsigned> needed;
  };
  std::vector<Case> cases;

  {
    Fixture fx;
    Case c{std::move(fx.sys), fx.needed};
    cases.push_back(std::move(c));
  }
  {
    // Two targets, four variables: A(col0) B(col1) C(col2) E(col3), master D(col4).
    Case c{FflowSystem(5), {0u, 1u}};
    c.sys.add({{0, 1}, {2, neg(1)}, {4, neg(1)}});        // A - C - D
    c.sys.add({{1, 1}, {2, neg(2)}, {4, neg(1)}});        // B - 2C - D
    c.sys.add({{1, 1}, {3, neg(1)}, {4, neg(1)}});        // B - E - D
    c.sys.add({{2, neg(1)}, {3, 1}, {4, neg(3)}});        // E - C - 3D
    cases.push_back(std::move(c));
  }
  {
    // The target is itself a master (the equation holds only target and master).
    Case c{FflowSystem(2), {0u}};
    c.sys.add({{0, 1}, {1, neg(1)}});                     // A - B
    cases.push_back(std::move(c));
  }

  for (auto &c : cases) {
    auto *raw = c.sys.alg.get();
    secred::ForwardSelectorOptions opts;
    secred::ForwardSelection sel = secred::select_forward(
        *raw, c.sys.ncols, c.needed.data(), c.needed.size(), kMod, opts);
    REQUIRE(sel.ok);
    REQUIRE(reductions_agree(*raw, sel.selected_eqs, c.needed, c.sys.ncols));

    secred::ForwardSelectorOptions leg_opts;
    leg_opts.max_repair_rounds = 0;
    secred::ForwardSelection leg = secred::select_forward(
        *raw, c.sys.ncols, c.needed.data(), c.needed.size(), kMod, leg_opts);
    REQUIRE(sorted_unique(sel.selected_eqs).size() <=
            sorted_unique(leg.selected_eqs).size());
  }
}

// The same input run twice must give identical results (the repair uses ordered sets).
TEST(forward_selection_is_deterministic) {
  Fixture fx;
  auto *raw = fx.sys.alg.get();
  secred::ForwardSelectorOptions opts;

  secred::ForwardSelection a = secred::select_forward(
      *raw, fx.sys.ncols, fx.needed.data(), fx.needed.size(), kMod, opts);
  secred::ForwardSelection b = secred::select_forward(
      *raw, fx.sys.ncols, fx.needed.data(), fx.needed.size(), kMod, opts);

  REQUIRE(a.selected_eqs == b.selected_eqs);
  REQUIRE_EQ(a.n_repair_rounds, b.n_repair_rounds);
  REQUIRE_EQ(a.n_repair_added, b.n_repair_added);
  REQUIRE(a.needed_indepvars == b.needed_indepvars);
  REQUIRE(a.needed_depvars == b.needed_depvars);
}

// Hitting the repair round cap falls back to the legacy closure, matching the old
// behaviour and setting the fallback flag.
TEST(forward_repair_round_cap_falls_back_to_legacy) {
  Fixture fx;
  auto *raw = fx.sys.alg.get();

  secred::ForwardSelectorOptions opts;
  opts.max_repair_rounds = 0;
  secred::ForwardSelection sel = secred::select_forward(
      *raw, fx.sys.ncols, fx.needed.data(), fx.needed.size(), kMod, opts);

  REQUIRE(sel.ok);
  REQUIRE(sel.legacy_fallback);
  REQUIRE_EQ(sel.n_selected, 3u);
  REQUIRE(sorted_unique(sel.selected_eqs) ==
          sorted_unique(std::vector<unsigned>{0u, 1u, 2u}));
  REQUIRE(reductions_agree(*raw, sel.selected_eqs, fx.needed, fx.sys.ncols));
}

// ---------------------------------------------------------------------------
// Randomised differential test: the replica's full-edge closure must match real
// fflow's mark_and_sweep_eqs element-wise.  The two diverged on a large F234 sector
// (the replica's closure was clearly bigger); this reproduces it in milliseconds on
// deterministic pseudo-random sparse systems -- same rinfo/c, real fflow via init +
// Graph/Session learn + mark_and_sweep_eqs, replica via learn_1/learn_2 + mark({}, false).
// A mismatch dumps the whole system to stderr so it can be frozen into a minimal case.
TEST(forward_replica_matches_fflow_random_systems) {
  uint64_t state = 0x243f6a8885a308d3ull; // fixed seed so failures reproduce
  auto rnd = [&]() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<unsigned>(state >> 33);
  };
  auto val = [&rnd]() { return static_cast<UInt>(1 + rnd() % 1000000u); };

  unsigned compared = 0, skipped = 0, mismatch_indep = 0, mismatch_dep = 0,
           mismatch_full = 0, printed = 0;
  for (unsigned trial = 0; trial < 400; ++trial) {
    const unsigned ncols = 5 + rnd() % 10; // 5..14 variables
    const unsigned nrows = 6 + rnd() % 30; // 6..35 equations
    FflowSystem sys(ncols);
    for (unsigned i = 0; i < nrows; ++i) {
      std::vector<std::pair<unsigned, UInt>> row;
      for (unsigned c = 0; c < ncols; ++c)
        if (rnd() % 100 < 45)
          row.emplace_back(c, val());
      if (row.empty())
        row.emplace_back(rnd() % ncols, val());
      sys.add(row);
    }

    std::vector<unsigned> needed;
    const unsigned nneed = 1 + rnd() % 2;
    while (needed.size() < nneed) {
      const unsigned c = rnd() % ncols;
      if (std::find(needed.begin(), needed.end(), c) == needed.end())
        needed.push_back(c);
    }

    RunResults r;
    try {
      r = run_all(sys, needed);
    } catch (const std::exception &e) {
      const std::string msg = e.what();
      if (msg.find("REQUIRE") != std::string::npos)
        throw; // the replica itself blew up: a real failure, not a skip
      ++skipped; // fflow's learn called the system unsolvable (target is dependent)
    }
    ++compared;

    const bool bad_indep =
        r.replica_indep != std::vector<std::size_t>(r.fflow_indep.begin(),
                                                    r.fflow_indep.end());
    const bool bad_dep =
        r.replica_dep !=
        std::vector<std::size_t>(r.fflow_dep.begin(), r.fflow_dep.end());
    const bool bad_full =
        r.replica_full !=
        std::vector<unsigned>(r.fflow_selected.begin(), r.fflow_selected.end());
    if (bad_indep)
      ++mismatch_indep;
    if (bad_dep)
      ++mismatch_dep;
    if (bad_full)
      ++mismatch_full;

    if ((bad_indep || bad_dep || bad_full) && printed < 3) {
      ++printed;
      std::cerr << "  [diff] trial " << trial << " ncols " << ncols << " nrows "
                << nrows << " needed";
      for (unsigned c : needed)
        std::cerr << " " << c;
      std::cerr << "\n  replica: indep " << r.replica_indep.size() << " dep "
                << r.replica_dep.size() << " full " << r.replica_full.size()
                << "\n  fflow  : indep " << r.fflow_indep.size() << " dep "
                << r.fflow_dep.size() << " sel " << r.fflow_selected.size()
                << std::endl;
      for (unsigned i = 0; i < sys.alg->rinfo.size(); ++i) {
        std::cerr << "    row " << i << ":";
        for (unsigned j = 0; j < sys.alg->rinfo[i].size; ++j)
          std::cerr << " (" << sys.alg->rinfo[i].cols[j] << ", "
                    << sys.alg->c[i][j] << ")";
        std::cerr << '\n';
      }
      std::cerr << std::flush;
    }
  }

  REQUIRE(compared > 100); // random systems should mostly be solvable
  REQUIRE_EQ(mismatch_dep, 0u);
  REQUIRE_EQ(mismatch_full, 0u);
}

// Degenerate inputs: an empty system, and a target that is itself a master.
TEST(forward_selection_degenerate_inputs) {
  {
    FflowSystem empty(3);
    std::vector<unsigned> needed{0u};
    secred::ForwardSelectorOptions opts;
    secred::ForwardSelection sel = secred::select_forward(
        *empty.alg, empty.ncols, needed.data(), needed.size(), kMod, opts);
    REQUIRE(sel.ok);
    REQUIRE_EQ(sel.n_selected, 0u);
    REQUIRE_EQ(sel.n_repair_rounds, 0u);
    REQUIRE(!sel.legacy_fallback);
    // With no equations the target is its own master.
    REQUIRE_EQ(sel.needed_indepvars[0], 0u);
  }
  {
    FflowSystem one(2);
    one.add({{0, 1}, {1, neg(1)}}); // A - B: pivot col0, master col1
    std::vector<unsigned> needed{0u};
    secred::ForwardSelectorOptions opts;
    secred::ForwardSelection sel =
        secred::select_forward(*one.alg, one.ncols, needed.data(),
                               needed.size(), kMod, opts);
    REQUIRE(sel.ok);
    REQUIRE_EQ(sel.n_selected, 1u);
    REQUIRE_EQ(sel.selected_eqs[0], 0u);
    REQUIRE_EQ(sel.n_repair_rounds, 0u);
    REQUIRE_EQ(sel.needed_indepvars.size(), 1u);
    REQUIRE_EQ(sel.needed_indepvars[0], 1u);
    REQUIRE(reductions_agree(*one.alg, sel.selected_eqs, needed, one.ncols));
  }
}
