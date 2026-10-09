#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <fflow/alg_linear_solver.hh>
#include <fflow/matrix.hh>

namespace fflow {

// Drop-in replacement for NumericSparseSolver whose input coefficients are
// already finite-field residues (UInt) instead of full arbitrary-precision
// rationals (MPRational).  SecRed pre-reduces every coefficient modulo
// FiniteFlow's learn() prime (BIG_UINT_PRIMES[0], see arith/ffint.h), so
// fill_matrix() can copy the residue straight into the sparse matrix without
// the rat_mod rational→modular conversion — and, more importantly, without
// materialising the ~6x-larger MPRational array in the first place.

struct NumericSparseSolverModData : public SparseLinearSolverData {};

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
  std::vector<std::unique_ptr<UInt[]>> c; // finite-field residues (mod learn prime)
};

} // namespace fflow
