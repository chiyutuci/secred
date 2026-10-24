
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ginac/ginac.h>

#include "ffred_solver.h"
#include "run_report.h"
#include "sector.h"

// Outcome of a global reduction: masters, reductions and counters.
struct GlobalReductionResult {
  std::vector<RawIntegral> masters;
  std::vector<std::pair<RawIntegral, std::string>> reductions;
  // Index in `reductions` of each target block's own `(target, "-1")` row.  The
  // coefficient text is not a boundary marker: a master can also reduce to
  // exactly "-1", so the writer must use these recorded starts.
  std::vector<std::size_t> target_starts;
  std::uint64_t raw_equations = 0;
  std::uint64_t selected_equations = 0;
  bool ok = false;

  double learn_seconds = 0.0;
  double reconstruction_seconds = 0.0;

  bool reconstruction_skipped = false;

  // Start one target block: its own row is the block's first row.
  void add_target(const RawIntegral &target) {
    target_starts.push_back(reductions.size());
    reductions.emplace_back(target, std::string("-1"));
  }
};

// Settled-basis check on a reconstructed master list.
using BasisVote = std::function<bool(const std::vector<RawIntegral> &)>;

// Prebuilt numeric evaluation plan for repeated prime solves.
struct NumericEvalPlan {
  std::vector<std::pair<std::uint32_t, std::uint32_t>> terms;
  std::vector<std::uint32_t> eq_begin;
  std::vector<secred::Rat64> coeffs;
  secred::FwdPlan plan;
  std::vector<std::uint32_t> sys_idx;
  std::size_t store_ids = 0;

  void build(const std::vector<secred::Eq> &system,
             const std::vector<std::uint32_t> &eq_idx);

  std::size_t n_rows() const { return sys_idx.size(); }
  std::size_t n_slots() const { return coeffs.size(); }

  secred::SymRowTable rows() const;

  void fill_values(std::uint64_t p, std::vector<std::uint64_t> &values) const;
};

// Solve the planned numeric system at one prime.
bool solve_prime_numeric(const NumericEvalPlan &plan, std::uint64_t p,
                         const std::vector<std::uint32_t> &bwd_chain,
                         const std::vector<unsigned> &dep_targets,
                         const std::vector<unsigned> &masters,
                         const std::unordered_map<unsigned, unsigned> &master_index,
                         secred::FwdStore &store, std::vector<std::uint64_t> &values,
                         std::vector<std::uint64_t> &xout,
                         std::uint32_t &fail_target, std::uint32_t &fail_integral,
                         const std::atomic<bool> *abort = nullptr,
                         secred::FwdStats *stats = nullptr);

// The index-th reconstruction prime.
std::uint64_t reconstruction_prime(unsigned index);

// Global reduction on the FiniteFlow numeric path.
GlobalReductionResult reduce_global_finiteflow(
    const std::string &raw_equations_path,
    const std::vector<RawIntegral> &all_integrals,
    const std::unordered_map<RawIntegral, unsigned> &integral_id,
    const std::vector<RawIntegral> &targets,
    unsigned narrowing_success, unsigned narrowing_fail,
    unsigned raw_equations_total, std::size_t global_equations_count,
    const std::string &masters_global_path, unsigned nthreads,
    std::uint64_t max_memory, std::uint64_t memory_reserve,
    unsigned max_primes = 500, const BasisVote &basis_vote = BasisVote{});

// Shared per-prime interpolation plan of the univariate path: the smallest
// Thiele degree whose sample set covers the current sample need of every
// unfinished output (one equation per unknown coefficient on the linear
// solver, one term per continued-fraction step on Thiele), with n_extra
// standing in for the check points a prime samples beyond the interpolation.
unsigned finiteflow_symbolic_univariate_plan_degree(const unsigned need[],
                                                    const char unfinished[],
                                                    unsigned noutputs,
                                                    unsigned n_extra);

// How many primes of the univariate path share one sample sweep: the largest
// batch that still fills the threads, capped at eight and at primes_left.
unsigned finiteflow_symbolic_univariate_batch_size(unsigned nthreads,
                                                   unsigned pts_per_prime,
                                                   unsigned primes_left);

// Global reduction on the FiniteFlow symbolic path.
GlobalReductionResult reduce_global_finiteflow_symbolic(
    const std::string &raw_equations_path,
    const std::vector<RawIntegral> &all_integrals,
    const std::unordered_map<RawIntegral, unsigned> &integral_id,
    const std::vector<RawIntegral> &targets,
    const std::vector<GiNaC::possymbol> &symbols,
    unsigned narrowing_success, unsigned narrowing_fail,
    unsigned raw_equations_total, std::size_t global_equations_count,
    const std::string &masters_global_path, unsigned nthreads = 64,
    unsigned max_primes = 500, const BasisVote &basis_vote = BasisVote{});

namespace secred {

// Planned subset solve used by the repair loop.
void solve_subset_planned(const std::vector<Eq> &system,
                          const std::vector<std::uint32_t> &eq_idx, std::uint64_t p,
                          SubSolve &out, bool want_fwd, bool want_bwd,
                          const std::vector<std::uint32_t> *only_keys = nullptr,
                          const std::vector<std::uint32_t> *solmap_keys = nullptr,
                          FwdStats *fwd_stats = nullptr);

inline void require_global_reduction(const GlobalReductionResult &result,
                                     const std::string &tool) {
  if (!result.ok)
    throw std::runtime_error(tool + " global reduction did not complete cleanly");
}

// Write `reductions` in the README's block format.  The write goes to a
// temporary file in the same directory and is renamed into place only after it
// is complete and durable, so a failed run cannot leave a truncated result.
void write_reductions_file(const std::string &path,
                           const GlobalReductionResult &result);

// Global reduction through FireFly.
GlobalReductionResult reduce_global_firefly_symbolic(
    const std::string &raw_equations_path,
    const std::vector<RawIntegral> &all_integrals,
    const std::unordered_map<RawIntegral, unsigned> &integral_id,
    const std::vector<RawIntegral> &targets,
    const std::vector<GiNaC::possymbol> &symbols,
    unsigned narrowing_success, unsigned narrowing_fail,
    unsigned raw_equations_total, std::size_t global_equations_count,
    const std::string &masters_global_path, unsigned nthreads = 64,
    unsigned max_primes = 500, const BasisVote &basis_vote = BasisVote{});

}
