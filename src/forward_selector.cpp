#include "forward_selector.h"

#include <algorithm>
#include <set>

namespace secred {

namespace {

// Same filling as NumericSparseSolverMod::fill_matrix (coefficients are already
// residues mod the learn prime).
void fill_rows(fflow::SparseMatrix &m, const fflow::NumericSparseSolverMod &alg,
               const std::size_t *rows, unsigned n_rows) {
  for (unsigned i = 0; i < n_rows; ++i) {
    fflow::SparseMatrixRow &r = m.row(i);
    const std::size_t ridx = rows[i];
    const std::size_t row_size = alg.rinfo[ridx].size;
    const unsigned *cols = alg.rinfo[ridx].cols.get();
    const fflow::UInt *f = alg.c[ridx].get();

    r.resize(row_size);
    unsigned j = 0;
    for (; j < row_size; ++j) {
      r.el(j).col = cols[j];
      r.el(j).val = f[j];
    }
    r.el(j).col = fflow::SparseMatrixRow::END;
  }
}

} // namespace

FflowLinearLearn::FflowLinearLearn(unsigned nvars, unsigned neqs)
    : neqs_(neqs), nvars_(nvars) {
  m_.resize(neqs, nvars + 1);
  xinfo_.reset(new fflow::LSVar::flag_t[nvars + 1]());
  indepeqs_.reset(new std::size_t[neqs]);
}

bool FflowLinearLearn::_reset(const unsigned *needed_vars, unsigned n_needed) {
  const unsigned nv = nvars_;
  depv_.clear();
  indepv_.clear();

  std::fill(xinfo_.get(), xinfo_.get() + nv + 1,
            static_cast<fflow::LSVar::flag_t>(fflow::LSVar::IS_NON_ZERO));
  for (unsigned i = 0; i < n_needed; ++i) {
    if (needed_vars[i] >= nv + 1 ||
        (xinfo_[needed_vars[i]] & fflow::LSVar::IS_NEEDED))
      return false;
    xinfo_[needed_vars[i]] |= fflow::LSVar::IS_NEEDED;
  }

  nnindepeqs_ = neqs_;
  for (std::size_t i = 0; i < nnindepeqs_; ++i)
    indepeqs_[i] = i;

  return true;
}

void FflowLinearLearn::_fill(const fflow::NumericSparseSolverMod &alg,
                             const std::size_t *rows, unsigned n_rows) {
  fill_rows(m_, alg, rows, n_rows);
}

void FflowLinearLearn::_number_eqs() {
  for (unsigned i = 0; i < nnindepeqs_; ++i)
    m_.row(i).id() = i;
}

void FflowLinearLearn::_get_dependent_variables() {
  m_.dependent_vars(depv_);
  for (std::size_t dep : depv_)
    xinfo_[dep] |= fflow::LSVar::IS_DEP;
  for (unsigned i = 0; i < nvars_; ++i)
    if (!(xinfo_[i] & fflow::LSVar::IS_DEP))
      indepv_.push_back(i);
}

void FflowLinearLearn::_get_independent_eqs() {
  nnindepeqs_ = depv_.size();
  for (unsigned eq = 0; eq < nnindepeqs_; ++eq)
    indepeqs_[eq] = m_.row(eq).id();
}

// Replica of SparseLinearSolver::learn_needed_: in a needed row, columns simpler than
// the pivot are needed too, which yields the needed dependent / independent tables.
// The constant column (nvars) is not a variable and takes no part.
void FflowLinearLearn::_learn_needed() {
  unsigned eq = 0;
  nndeps_ = 0;
  for (std::size_t dep : depv_) {
    if (xinfo_[dep] & fflow::LSVar::IS_NEEDED) {
      for (unsigned j = static_cast<unsigned>(dep) + 1; j < nvars_; ++j)
        if (m_(eq, j))
          xinfo_[j] |= fflow::LSVar::IS_NEEDED;
      ++nndeps_;
    }
    ++eq;
  }

  nnindeps_ = 0;
  for (unsigned j = 0; j < nvars_; ++j)
    if ((xinfo_[j] & fflow::LSVar::IS_NEEDED) &&
        !(xinfo_[j] & fflow::LSVar::IS_DEP))
      ++nnindeps_;

  needed_dep_.assign(nndeps_, 0);
  needed_indep_.assign(nnindeps_, 0);
  unsigned nic = 0;
  for (unsigned j = 0; j < nvars_; ++j)
    if ((xinfo_[j] & fflow::LSVar::IS_NEEDED) &&
        !(xinfo_[j] & fflow::LSVar::IS_DEP))
      needed_indep_[nic++] = j;
  unsigned ndc = 0;
  for (std::size_t dep : depv_)
    if (xinfo_[dep] & fflow::LSVar::IS_NEEDED)
      needed_dep_[ndc++] = dep;
}

bool FflowLinearLearn::_check_dependent_variables() {
  const std::size_t ndeps = depv_.size();
  m_.dependent_vars(depv_); // clears and refills in place
  if (ndeps != depv_.size()) {
    depv_.resize(ndeps);
    return false;
  }
  for (std::size_t dep : depv_)
    if (!(xinfo_[dep] & fflow::LSVar::IS_DEP) ||
        (xinfo_[dep] & fflow::LSVar::IS_SWEEPED))
      return false;
  return true;
}

bool FflowLinearLearn::learn_1(const fflow::NumericSparseSolverMod &alg,
                               const unsigned *needed_vars, unsigned n_needed,
                               fflow::Mod mod) {
  if (!_reset(needed_vars, n_needed)) {
    failed_ = true;
    return false;
  }

  _fill(alg, indepeqs_.get(), static_cast<unsigned>(nnindepeqs_));
  _number_eqs();
  m_.sortRows();
  m_.toReducedRowEcholon(mod);

  if (m_.isImpossibleSystem())
    return true; // as in fflow: depv_/indepv_ stay empty

  _get_dependent_variables();
  _get_independent_eqs();
  _learn_needed();
  m_.restrict_rows(nnindepeqs_);
  return true;
}

bool FflowLinearLearn::learn_2(const fflow::NumericSparseSolverMod &alg,
                               fflow::Mod mod) {
  _fill(alg, indepeqs_.get(), static_cast<unsigned>(nnindepeqs_));

  eqdeps_.clear();
  eqdeps_.resize(nnindepeqs_);

  // Forward elimination: one explicit pass, recording each row's forward edge count.
  m_.toRowEcholon(mod, eqdeps_.data());
  fwd_len_.assign(nnindepeqs_, 0);
  for (std::size_t i = 0; i < nnindepeqs_; ++i)
    fwd_len_[i] = static_cast<unsigned>(eqdeps_[i].size());

  // Back substitution: the matrix is already triangular, so the internal
  // toRowEcholon substitutes nothing and what eqdeps_ gains are the backward edges.
  m_.toReducedRowEcholon(mod, xinfo_.get(), eqdeps_.data());

  if (m_.isImpossibleSystem()) {
    if (_is_learning_impossible())
      return true;
    failed_ = true;
    return false;
  }

  if (!_check_dependent_variables()) {
    failed_ = true;
    return false;
  }

  pivot_to_row_.assign(nvars_ + 1, kNoRow);
  for (std::size_t i = 0; i < depv_.size(); ++i)
    pivot_to_row_[depv_[i]] = static_cast<unsigned>(i);
  return true;
}

std::vector<unsigned>
FflowLinearLearn::mark(const std::vector<unsigned> &extra_pivots,
                       bool forward_only) const {
  std::vector<unsigned> selected;
  if (nnindepeqs_ == 0 || eqdeps_.empty())
    return selected;

  std::vector<char> extra(nvars_ + 1, 0);
  for (unsigned c : extra_pivots)
    if (c <= nvars_)
      extra[c] = 1;

  std::vector<char> marked(nnindepeqs_, 0);
  std::vector<unsigned> stack;
  for (std::size_t i = 0; i < depv_.size(); ++i) {
    const std::size_t dep = depv_[i];
    if ((xinfo_[dep] & fflow::LSVar::IS_NEEDED) || extra[dep])
      stack.push_back(static_cast<unsigned>(i));
  }
  // Explicit stack rather than recursion (chains reach thousands of rows); same
  // result as fflow's recursive version.
  while (!stack.empty()) {
    const unsigned row = stack.back();
    stack.pop_back();
    if (marked[row])
      continue;
    marked[row] = 1;
    const fflow::SparseMatrix::EqDeps &deps = eqdeps_[row];
    const unsigned n =
        forward_only ? fwd_len_[row] : static_cast<unsigned>(deps.size());
    for (unsigned k = 0; k < n; ++k)
      stack.push_back(deps[k]);
  }

  selected.reserve(nnindepeqs_);
  for (std::size_t i = 0; i < nnindepeqs_; ++i)
    if (marked[i])
      selected.push_back(m_.row(i).id());
  return selected;
}

FflowSubsolve fflow_rref_subset(const fflow::NumericSparseSolverMod &alg,
                                const std::vector<unsigned> &rows,
                                unsigned ncols, fflow::Mod mod) {
  FflowSubsolve out;
  const unsigned n = static_cast<unsigned>(rows.size());

  fflow::SparseMatrix m;
  m.resize(n, ncols + 1);
  std::vector<std::size_t> idx(n);
  for (unsigned i = 0; i < n; ++i)
    idx[i] = rows[i];
  fill_rows(m, alg, idx.data(), n);
  for (unsigned i = 0; i < n; ++i)
    m.row(i).id() = rows[i];
  m.sortRows();
  m.toReducedRowEcholon(mod); // no flags, no deps = full RREF

  out.pivot_to_row.assign(ncols + 1, FflowLinearLearn::kNoRow);
  for (unsigned i = 0; i < n; ++i) {
    const fflow::SparseMatrixRow &r = m.row(i);
    const unsigned col = r.first_nonzero_column();
    if (col == fflow::SparseMatrixRow::END)
      break; // RREF puts the zero rows last
    out.pivot_to_row[col] = i;
    out.pivot.push_back(col);
    out.row_id.push_back(r.id());
    std::vector<unsigned> rhs;
    for (unsigned k = 1; k < r.size(); ++k) {
      if (r.el(k).col == fflow::SparseMatrixRow::END)
        break;
      rhs.push_back(r.el(k).col);
    }
    out.rhs.push_back(std::move(rhs));
  }
  out.n_retained = static_cast<unsigned>(out.pivot.size());
  return out;
}

ForwardSelection select_forward(const fflow::NumericSparseSolverMod &alg,
                                unsigned ncols, const unsigned *needed_vars,
                                unsigned n_needed, fflow::Mod mod,
                                const ForwardSelectorOptions &opts) {
  ForwardSelection sel;
  FflowLinearLearn learn(ncols, static_cast<unsigned>(alg.rinfo.size()));
  if (!learn.learn_1(alg, needed_vars, n_needed, mod) ||
      !learn.learn_2(alg, mod)) {
    sel.ok = false;
    return sel;
  }

  if (learn.n_needed_indepvars())
    sel.needed_indepvars.assign(learn.needed_indepvars(),
                                learn.needed_indepvars() +
                                    learn.n_needed_indepvars());
  if (learn.n_needed_depvars())
    sel.needed_depvars.assign(learn.needed_depvars(),
                              learn.needed_depvars() +
                                  learn.n_needed_depvars());

  // Master columns are the variable columns that are no pivot; the constant column
  // (ncols) is not a variable.
  std::vector<char> is_master(ncols, 1);
  for (std::size_t c : learn.depv())
    if (c < ncols)
      is_master[c] = 0;

  std::vector<unsigned> extra; // pivot columns the repair pushed back in
  std::set<unsigned> added;
  std::vector<unsigned> rows = learn.mark(extra, true);

  unsigned rounds = 0;
  while (opts.repair) {
    if (rounds == opts.max_repair_rounds) {
      sel.legacy_fallback = true;
      break;
    }

    const FflowSubsolve sub = fflow_rref_subset(alg, rows, ncols, mod);

    std::set<unsigned> missing;
    for (unsigned j = 0; j < sub.n_retained; ++j) {
      const unsigned w = sub.pivot[j];
      if (!(learn.xinfo()[w] & fflow::LSVar::IS_NEEDED))
        continue;
      for (unsigned c : sub.rhs[j])
        if (c < ncols && !is_master[c])
          missing.insert(c);
    }
    // Needed integral never solved as a pivot: the selected equations cannot reduce it.
    for (std::size_t w : sel.needed_depvars)
      if (w <= ncols && sub.pivot_to_row[w] == FflowLinearLearn::kNoRow)
        missing.insert(static_cast<unsigned>(w));

    if (missing.empty())
      break; // sufficient

    bool progressed = false;
    bool unaddable = false;
    for (unsigned c : missing) {
      if (learn.pivot_row(c) == FflowLinearLearn::kNoRow) {
        unaddable = true; // not a dependent variable here, cannot be added
        break;
      }
      if (added.insert(c).second) {
        extra.push_back(c);
        progressed = true;
      }
    }
    if (unaddable || !progressed) {
      // No progress: fall back to the legacy forward-and-backward closure.
      sel.legacy_fallback = true;
      break;
    }
    ++rounds;
    rows = learn.mark(extra, true);
  }

  sel.n_repair_rounds = rounds;
  sel.n_repair_added = static_cast<unsigned>(extra.size());
  if (sel.legacy_fallback) {
    extra.clear();
    rows = learn.mark(extra, false);
  }

  sel.n_selected = static_cast<unsigned>(rows.size());
  sel.selected_eqs = std::move(rows);
  return sel;
}

} // namespace secred
