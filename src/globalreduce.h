// In-process global reduction: a pure-numeric and a symbolic FiniteFlow path.
// Both select their equations with the built-in modular Gaussian elimination
// (ffred_solver) and hand the system to FiniteFlow for the coefficient
// reconstruction; no external tool is involved.
#pragma once

#include <atomic>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ginac/ginac.h>

#include "ffred_solver.h" // Eq, FwdPlan, SymRowTable, FwdStore, rat_mod_u64
#include "run_report.h"   // format_duration, the statistics blocks
#include "sector.h"       // RawIntegral, GlobalEquation

struct GlobalReductionResult {
  std::vector<RawIntegral> masters;                            // masters
  std::vector<std::pair<RawIntegral, std::string>> reductions; // target -> coefficient string
  std::uint64_t raw_equations = 0;                             // equations sent to the global system
  std::uint64_t selected_equations = 0;                        // equations the selection kept
  bool ok = false;
  // Wall-clock split of the global phase: what the learn pass cost and what the
  // reconstruction cost.
  double learn_seconds = 0.0;
  double reconstruction_seconds = 0.0;
};

// The numeric reconstruction's system, hoisted out of the per-prime loop.
//
// The numeric path solves the selected subset of one vector<Eq> once per prime, and the
// prime only changes the *values*: the rows, their order and their pivots are properties
// of the structure, exactly as they are on the symbolic side (§5.9 of ARCHITECTURE.md).
// Materialising a ResEq per equation per prime and re-deriving the order through an
// unordered_map therefore repeats, 31 times at F234's size, work the structure already
// answers — and each of the 31 primes is independent of the other 30.
//
// This holds what is prime-independent: every term of the selected rows as
// (integral id, coefficient slot), the distinct rational coefficients those slots name,
// and the forward plan built from the table.  A prime then costs one `fill_values` and
// one planned elimination.
//
// Ids stay descending within a row, as the spill reader wrote them.
struct NumericEvalPlan {
  std::vector<std::pair<std::uint32_t, std::uint32_t>> terms; // (integral id, slot)
  std::vector<std::uint32_t> eq_begin;                        // n_rows + 1 boundaries
  std::vector<secred::Rat64> coeffs;                          // slot -> num/den
  secred::FwdPlan plan;
  std::vector<std::uint32_t> sys_idx; // row -> raw equation index (the order's last key)
  std::size_t store_ids = 0;          // one past the largest integral id in the table

  // Rows of `eq_idx`, in that order, with identically zero coefficients dropped — the
  // terms `reduce_mod` would keep at every prime.  `eq_idx` must be the raw indices into
  // `system` of the equations the selection kept, and is what the plan's `sys_idx` gets.
  void build(const std::vector<secred::Eq> &system,
             const std::vector<std::uint32_t> &eq_idx);

  std::size_t n_rows() const { return sys_idx.size(); }
  std::size_t n_slots() const { return coeffs.size(); }

  // The table as the planned elimination reads it.  Built on demand rather than stored,
  // so the plan stays movable: the pointers would dangle through a move.
  secred::SymRowTable rows() const;

  // `values[s]` = coefficient s modulo p, for every slot.  Exactly what reduce_mod would
  // compute term by term, once per distinct coefficient instead of once per term.
  void fill_values(std::uint64_t p, std::vector<std::uint64_t> &values) const;
};

// One prime's solution of the planned elimination: the residues of the target x master
// block, flattened as xout[i * nnindeps + j] (i over dep_targets, j over masters).
//
// Fails when a target's right-hand side carries an integral the master table does not
// know, which means the selection is stale at this prime; the caller reports it rather
// than emit a truncated reduction.  `fail_target` / `fail_integral` name the pair.
//
// `abort`, when given, is polled inside the elimination: the reconstruction pool runs
// primes ahead of its consumer, and a prime the consumer no longer needs is dropped
// half-way instead of being finished.  The false return then means "abandoned", not
// "failed" -- the caller distinguishes them by the flag itself.
bool solve_prime_numeric(const NumericEvalPlan &plan, std::uint64_t p,
                         const std::vector<std::uint32_t> &bwd_chain,
                         const std::vector<unsigned> &dep_targets,
                         const std::vector<unsigned> &masters,
                         const std::unordered_map<unsigned, unsigned> &master_index,
                         secred::FwdStore &store, std::vector<std::uint64_t> &values,
                         std::vector<std::uint64_t> &xout,
                         std::uint32_t &fail_target, std::uint32_t &fail_integral,
                         const std::atomic<bool> *abort = nullptr);

// Pure-numeric in-process global reduction: the elimination runs on FiniteFlow's
// sparse solver and the coefficients are reconstructed as rational numbers with
// algorithm_reconstruct_numeric, so no auxiliary symbol is needed.
// raw_equations_path is the spilled system file (<RawIntegral>\t<coeff> per line,
// a blank line between equations); all_integrals / integral_id are the global
// integral order and numbering; targets are the target integrals.  The last four
// statistics feed the "Reduction Statistics" summary printed after the learn step
// and before the reconstruction (narrowing counts, the equation count after learn
// and the final reduction counts).
// masters_final_path receives the final master basis (one integral per line, the same
// format as masters_sector) as soon as the learn pass has run — before the
// reconstruction, which can be the longest phase of the run.
// The reconstruction set is always the forward closure plus the numeric
// sufficiency repair (see global_select.h); when the repair does not converge the
// selection falls back to the forward-and-backward closure.
GlobalReductionResult reduce_global_finiteflow(
    const std::string &raw_equations_path,
    const std::vector<RawIntegral> &all_integrals,
    const std::unordered_map<RawIntegral, unsigned> &integral_id,
    const std::vector<RawIntegral> &targets,
    unsigned narrowing_success, unsigned narrowing_fail,
    unsigned raw_equations_total, std::size_t global_equations_count,
    const std::string &masters_final_path, unsigned nthreads = 64);

// Symbolic in-process global reduction: the coefficients are GiNaC polynomials in
// `symbols`, the elimination uses the built-in modular solver (the same machinery
// as the pure-numeric path, with a different coefficient evaluation) and the
// coefficients are rebuilt with FiniteFlow's multivariate rational-function
// reconstruction (algorithm_reconstruct).  `symbols` are the variables of the
// rational function (d plus the kinematic invariants), in the order FiniteFlow's
// nparsin expects.
// masters_final_path is written as in the numeric path, and the sample-point
// evaluation that dominates the reconstruction reports its progress.
GlobalReductionResult reduce_global_finiteflow_symbolic(
    const std::string &raw_equations_path,
    const std::vector<RawIntegral> &all_integrals,
    const std::unordered_map<RawIntegral, unsigned> &integral_id,
    const std::vector<RawIntegral> &targets,
    const std::vector<GiNaC::possymbol> &symbols,
    unsigned narrowing_success, unsigned narrowing_fail,
    unsigned raw_equations_total, std::size_t global_equations_count,
    const std::string &masters_final_path, unsigned nthreads = 64);

namespace secred {

// A global phase that did not complete (`ok == false`) leaves `reductions` empty or
// truncated, so it is not a reduction table: writing one to cache/reductions and
// exiting 0 would report a silently wrong result.  Every in-process caller checks the
// result through this before writing anything (family.cpp, right after the phase and
// before the masters file and the table); main.cpp turns the throw into EXIT_FAILURE.
// `tool` names the backend the run selected, so the message says which one stopped.
inline void require_global_reduction(const GlobalReductionResult &result,
                                     const std::string &tool) {
  if (!result.ok)
    throw std::runtime_error(tool + " global reduction did not complete cleanly");
}

// The `global_tool: firefly` backend: the same learn and the same evaluator as the
// FiniteFlow symbolic path, with FireFly's Reconstructor as the engine and the system
// living inside the black box (src/firefly_global.cpp, src/firefly_global.h).  The
// signature and the returned result are deliberately identical to the FiniteFlow
// symbolic path's, so the two are interchangeable and can be compared output for
// output.
GlobalReductionResult reduce_global_firefly_symbolic(
    const std::string &raw_equations_path,
    const std::vector<RawIntegral> &all_integrals,
    const std::unordered_map<RawIntegral, unsigned> &integral_id,
    const std::vector<RawIntegral> &targets,
    const std::vector<GiNaC::possymbol> &symbols,
    unsigned narrowing_success, unsigned narrowing_fail,
    unsigned raw_equations_total, std::size_t global_equations_count,
    const std::string &masters_final_path, unsigned nthreads = 64);

} // namespace secred
