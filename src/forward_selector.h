#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <fflow/alg_linear_solver.hh>
#include <fflow/matrix.hh>

#include "numeric_solver_mod.h"

namespace secred {

// ---------------------------------------------------------------------------
// Differential reference for the global stage's equation selection (forward
// closure plus numeric sufficiency repair -- see global_select.h), built on
// fflow's SparseLinearSolver.
//
// fflow keeps forward and backward substitution edges in one private array and
// only exposes their union, so the split is reconstructed here: run
// toRowEcholon once to record the forward edge length per row, then
// toReducedRowEcholon -- on an already triangular matrix it performs no
// substitution, so eqdeps[i] starts with the forward edges and ends with the
// backward ones.  Total elimination cost is unchanged.
//
// No production caller: production goes through global_select.h, which does not
// depend on fflow.  This class serves the element-wise differential check
// against real fflow in tests/test_forward_selector.cpp.
// ---------------------------------------------------------------------------

class FflowLinearLearn {
public:
  static constexpr unsigned kNoRow = ~0u;

  FflowLinearLearn(unsigned nvars, unsigned neqs);

  // Stage 1 of SparseLinearSolver::learn_1_.  needed_vars are target columns
  // (dense column numbers, not raw ids).
  bool learn_1(const fflow::NumericSparseSolverMod &alg,
               const unsigned *needed_vars, unsigned n_needed, fflow::Mod mod);

  // Stage 2 of SparseLinearSolver::learn_2_, recording the forward edge
  // lengths.
  bool learn_2(const fflow::NumericSparseSolverMod &alg, fflow::Mod mod);

  bool failed() const { return failed_; }

  // ---- Accessors mirroring SparseLinearSolver ----
  std::size_t n_needed_depvars() const { return needed_dep_.size(); }
  std::size_t n_needed_indepvars() const { return needed_indep_.size(); }
  const std::size_t *needed_depvars() const { return needed_dep_.data(); }
  const std::size_t *needed_indepvars() const { return needed_indep_.data(); }
  const fflow::LSVar::flag_t *xinfo() const { return xinfo_.get(); }

  // ---- Reconstructed dependency graph (used by the repair loop) ----
  const std::vector<std::size_t> &depv() const { return depv_; }
  unsigned n_dep_rows() const { return static_cast<unsigned>(depv_.size()); }
  unsigned row_id(unsigned row) const { return m_.row(row).id(); }
  unsigned pivot_row(unsigned col) const {
    return col <= nvars_ ? pivot_to_row_[col] : kNoRow;
  }
  unsigned fwd_len(unsigned row) const { return fwd_len_[row]; }
  unsigned n_forward_edges(unsigned row) const { return fwd_len_[row]; }
  unsigned n_total_edges(unsigned row) const {
    return static_cast<unsigned>(eqdeps_[row].size());
  }

  std::vector<unsigned> mark(const std::vector<unsigned> &extra_pivots,
                             bool forward_only) const;

private:
  bool _reset(const unsigned *needed_vars, unsigned n_needed);
  void _fill(const fflow::NumericSparseSolverMod &alg, const std::size_t *rows,
             unsigned n_rows);
  void _number_eqs();
  void _get_dependent_variables();
  void _get_independent_eqs();
  void _learn_needed();
  bool _check_dependent_variables();
  bool _is_learning_impossible() const {
    return depv_.empty() && indepv_.empty();
  }

  fflow::SparseMatrix m_;
  std::unique_ptr<fflow::LSVar::flag_t[]> xinfo_;
  std::unique_ptr<std::size_t[]> indepeqs_;
  std::vector<std::size_t> depv_, indepv_, needed_dep_, needed_indep_;
  std::vector<fflow::SparseMatrix::EqDeps> eqdeps_;
  std::vector<unsigned> fwd_len_, pivot_to_row_;
  std::size_t neqs_, nvars_;
  std::size_t nndeps_ = 0, nnindeps_ = 0, nnindepeqs_ = 0;
  bool failed_ = false;
};

// ---------------------------------------------------------------------------
// Subset re-solve for the numeric sufficiency repair: one independent full RREF
// over the given equations, reading each retained row's pivot column and the
// non-pivot columns on its right-hand side.  The matrix lives only for the
// call.
// ---------------------------------------------------------------------------
struct FflowSubsolve {
  unsigned n_retained = 0;
  std::vector<unsigned> pivot;  // pivot column of retained row i
  std::vector<unsigned> row_id; // raw equation index of retained row i
  std::vector<unsigned>
      pivot_to_row; // column -> retained row, kNoRow if non-pivot
  std::vector<std::vector<unsigned>> rhs; // non-pivot columns of retained row i
};

FflowSubsolve fflow_rref_subset(const fflow::NumericSparseSolverMod &alg,
                                const std::vector<unsigned> &rows,
                                unsigned ncols, fflow::Mod mod);

// ---------------------------------------------------------------------------
// Selection entry point.
// ---------------------------------------------------------------------------
struct ForwardSelectorOptions {
  bool repair = true;
  // Hard cap on repair rounds (unreachable in practice: the seed set grows
  // strictly every round and is bounded).
  unsigned max_repair_rounds = 64;
};

struct ForwardSelection {
  bool ok = true;
  // Set when the selection stalls and falls back to the legacy forward-and-
  // backward closure, which reproduces the old behaviour.
  bool legacy_fallback = false;
  unsigned n_selected = 0;
  std::vector<unsigned> selected_eqs; // raw equation indices, row order
  unsigned n_repair_rounds = 0;
  unsigned n_repair_added = 0;
  // Masters: same logic as learn_1_/learn_needed_, element-wise identical to
  // the legacy path.
  std::vector<std::size_t> needed_indepvars, needed_depvars;
};

// alg.rinfo/alg.c must already be filled; ncols is the number of variable
// columns (all columns except the right-hand side).
ForwardSelection select_forward(const fflow::NumericSparseSolverMod &alg,
                                unsigned ncols, const unsigned *needed_vars,
                                unsigned n_needed, fflow::Mod mod,
                                const ForwardSelectorOptions &opts);

} // namespace secred
