// In-process global reduction: a pure-numeric and a symbolic FiniteFlow path.
// Both select their equations with the built-in modular Gaussian elimination
// (ffred_solver) and hand the system to FiniteFlow for the coefficient
// reconstruction; no external tool is involved.
#pragma once

#include <cstdint>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ginac/ginac.h>

#include "run_report.h" // format_duration, the statistics blocks
#include "sector.h"     // RawIntegral, GlobalEquation

struct GlobalReductionResult {
  std::vector<RawIntegral> masters;                            // masters
  std::vector<std::pair<RawIntegral, std::string>> reductions; // target -> coefficient string
  std::uint64_t raw_equations = 0;                             // equations sent to the global system
  std::uint64_t selected_equations = 0;                        // equations the selection kept
  bool ok = false;
  // Wall-clock split of the in-process FiniteFlow phase.  The Kira path has no
  // separate learn and reconstruction steps, so it leaves phase_split false and
  // the end-of-run Timing Statistics block prints only its one global phase.
  bool phase_split = false;
  double learn_seconds = 0.0;
  double reconstruction_seconds = 0.0;
  // Kira path only: the equations Kira selected out of the system it was handed
  // and submitted to FireFly, read back from its log.  This is the count the
  // reduction is measured by — not the number SecRed sent.  nullopt when the log
  // could not be read, and the statistics block then leaves the row out rather
  // than reporting a zero it did not observe.
  std::optional<std::uint64_t> firefly_equations;
  // Kira path only: peak resident memory of the Kira subprocess including what it
  // ran, in bytes.  The kernel accounts for a child separately, so SecRed's own
  // high-water mark would otherwise hide the global stage's memory.
  std::uint64_t child_peak_bytes = 0;
};

// Pure-numeric in-process global reduction: the elimination runs on FiniteFlow's
// sparse solver and the coefficients are reconstructed as rational numbers with
// algorithm_reconstruct_numeric, so neither Kira nor a "dum" symbol is needed.
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
