#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <fflow/alg_linear_solver.hh>
#include <fflow/matrix.hh>

#include "numeric_solver_mod.h"

namespace secred {

// FiniteFlow's linear learn plus dependency bookkeeping.
class FflowLinearLearn {
public:
  static constexpr unsigned kNoRow = ~0u;

  FflowLinearLearn(unsigned nvars, unsigned neqs);

  bool learn_1(const fflow::NumericSparseSolverMod &alg,
               const unsigned *needed_vars, unsigned n_needed, fflow::Mod mod);

  bool learn_2(const fflow::NumericSparseSolverMod &alg, fflow::Mod mod);

  bool failed() const { return failed_; }

  std::size_t n_needed_depvars() const { return needed_dep_.size(); }
  std::size_t n_needed_indepvars() const { return needed_indep_.size(); }
  const std::size_t *needed_depvars() const { return needed_dep_.data(); }
  const std::size_t *needed_indepvars() const { return needed_indep_.data(); }
  const fflow::LSVar::flag_t *xinfo() const { return xinfo_.get(); }

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

// Result of solving a row subset in isolation.
struct FflowSubsolve {
  unsigned n_retained = 0;
  std::vector<unsigned> pivot;
  std::vector<unsigned> row_id;
  std::vector<unsigned>
      pivot_to_row;
  std::vector<std::vector<unsigned>> rhs;
};

// Row-reduced subset solve on the modified solver.
FflowSubsolve fflow_rref_subset(const fflow::NumericSparseSolverMod &alg,
                                const std::vector<unsigned> &rows,
                                unsigned ncols, fflow::Mod mod);

// Forward-selection tuning.
struct ForwardSelectorOptions {
  bool repair = true;

  unsigned max_repair_rounds = 64;
};

// Result of a forward selection.
struct ForwardSelection {
  bool ok = true;

  bool legacy_fallback = false;
  unsigned n_selected = 0;
  std::vector<unsigned> selected_eqs;
  unsigned n_repair_rounds = 0;
  unsigned n_repair_added = 0;

  std::vector<std::size_t> needed_indepvars, needed_depvars;
};

// Choose the equation subset that forward-eliminates the needed variables.
ForwardSelection select_forward(const fflow::NumericSparseSolverMod &alg,
                                unsigned ncols, const unsigned *needed_vars,
                                unsigned n_needed, fflow::Mod mod,
                                const ForwardSelectorOptions &opts);

}
