#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <fflow/alg_linear_solver.hh>
#include <fflow/matrix.hh>

namespace fflow {

struct NumericSparseSolverModData : public SparseLinearSolverData {};

// FiniteFlow sparse solver that records each row's column layout.
class NumericSparseSolverMod : public SparseLinearSolver {
public:
  void delete_unneeded_eqs(AlgorithmData * data);

  Ret fill_matrix(Context * ctxt, std::size_t n_rows,
                  const std::size_t rows[], AlgInput xi[], Mod mod,
                  AlgorithmData * data, SparseMatrix & m) const override;

  AlgorithmData::Ptr clone_data(const AlgorithmData * data) const override;

public:
  struct RowInfo {
    std::size_t size;
    std::unique_ptr<unsigned[]> cols;
  };

  std::vector<RowInfo> rinfo;
  std::vector<std::unique_ptr<UInt[]>> c;
};

}
