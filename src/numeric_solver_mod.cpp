#include "numeric_solver_mod.h"

namespace fflow {

  Ret NumericSparseSolverMod::fill_matrix(Context *, std::size_t n_rows,
                                          const std::size_t rows[],
                                          AlgInput[], Mod,
                                          AlgorithmData *,
                                          SparseMatrix & m) const {
    for (unsigned i = 0; i < n_rows; ++i) {
      SparseMatrixRow & r = m.row(i);
      const std::size_t row_size = rinfo[rows[i]].size;
      const unsigned * cols = rinfo[rows[i]].cols.get();
      const UInt * f = c[rows[i]].get();

      r.resize(row_size);
      unsigned j = 0;
      for (; j < row_size; ++j) {
        r.el(j).col = cols[j];
        r.el(j).val = f[j]; // already a residue modulo the learn() prime
      }
      r.el(j).col = SparseMatrixRow::END;
    }
    return SUCCESS;
  }

  void NumericSparseSolverMod::delete_unneeded_eqs(AlgorithmData *) {
    std::size_t n_rows = rinfo.size();
    std::unique_ptr<bool[]> needed(new bool[n_rows]());

    std::size_t nindepeqs = n_indep_eqs();
    const std::size_t * ieq = indep_eqs();

    for (unsigned i = 0; i < nindepeqs; ++i)
      needed[ieq[i]] = true;

    for (unsigned i = 0; i < n_rows; ++i)
      if (!needed[i]) {
        rinfo[i] = RowInfo();
        c[i].reset(nullptr);
      }
  }

  AlgorithmData::Ptr
  NumericSparseSolverMod::clone_data(const AlgorithmData * datain) const {
    std::unique_ptr<NumericSparseSolverModData> ptr(
        new NumericSparseSolverModData());
    copy_data(datain, ptr.get());
    return std::move(ptr);
  }

} // namespace fflow
