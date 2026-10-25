
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

namespace secred {
struct ResumeControl;
}

// The global engine a system's symbol count selects: none -> the numeric
// FiniteFlow path, one or more -> FireFly.
enum class GlobalEngine { FiniteFlowNumeric, FireFlySymbolic };

inline GlobalEngine global_engine_for(std::size_t nsymbols) {
  if (nsymbols == 0)
    return GlobalEngine::FiniteFlowNumeric;
  return GlobalEngine::FireFlySymbolic;
}

// The name the engine's failures carry.
inline const char *global_engine_name(GlobalEngine engine) {
  return engine == GlobalEngine::FireFlySymbolic ? "firefly" : "finiteflow";
}

namespace secred {

// Everything both global backends consume: the spilled system after the sector
// phase, the target block and the run's limits.  The backends differ in how
// they solve and reconstruct; this is the contract they share, and
// run_global_stage() is the single entry the sector phase hands the system to.
struct GlobalStageInput {
  const std::string &raw_equations_path;
  const std::vector<RawIntegral> &all_integrals;
  const std::unordered_map<RawIntegral, unsigned> &integral_id;
  const std::vector<RawIntegral> &targets;
  unsigned narrowing_success = 0;
  unsigned narrowing_fail = 0;
  unsigned raw_equations_total = 0;
  std::size_t global_equations_count = 0;
  const std::string &masters_global_path;
  unsigned nthreads = 64;
  unsigned max_primes = 500;
  // Consulted once the learn has settled the basis; a true return stops before
  // the reconstruction.
  BasisVote basis_vote;
  const ResumeControl *resume = nullptr;
};

// Map the requested targets into the system's integral ids, in request order.
// Prints "<backend>: no targets" to stderr and returns empty when none is known.
std::vector<unsigned> global_target_ids(const GlobalStageInput &input,
                                        const char *backend);

} // namespace secred

// The reconstruction shares one forward schedule among every worker and prime, and
// a schedule costs several times the table's own bytes per term.  Plans above these
// caps keep the planned forward pass, so a spill far larger than the schedule is
// designed for (F1's) does not put the whole schedule in memory.
inline constexpr std::size_t kNumericDagMaxTerms = 16'000'000;
inline constexpr std::size_t kNumericDagMaxRows = 1'000'000;
bool numeric_dag_fits(std::size_t n_terms, std::size_t n_rows);

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

  // Compact the plan to the rows in `keep` (ascending, unique, in range),
  // keeping the referenced coefficient values (renumbered in first-use order),
  // then rebuild the elimination plan.  This is how the numeric driver drops
  // the equations its selection did not need without keeping a second copy of
  // the table alive.
  void select_rows(const std::vector<std::uint32_t> &keep);

  // Row-subset view of this plan.  The terms keep this plan's coefficient
  // slots, so a caller must evaluate fill_values() on the full plan once and
  // pass those residues to solve_plan().
  NumericEvalPlan subset(const std::vector<std::uint32_t> &eq_idx) const;

  std::size_t n_rows() const { return sys_idx.size(); }
  std::size_t n_slots() const { return coeffs.size(); }

  secred::SymRowTable rows() const;

  void fill_values(std::uint64_t p, std::vector<std::uint64_t> &values) const;
};

// Solve the planned numeric system at one prime.
//
// `replay_dag` replays a schedule the builder thread published, with `dag_acc` as the
// caller-owned accumulator.  A replay that departs from the schedule resets the store
// and finishes on the planned forward pass.  Pass `dag_acc` whenever `replay_dag` is
// used; it is left all-zero.
bool solve_prime_numeric(const NumericEvalPlan &plan, std::uint64_t p,
                         const std::vector<std::uint32_t> &bwd_chain,
                         const std::vector<unsigned> &dep_targets,
                         const std::vector<unsigned> &masters,
                         const std::unordered_map<unsigned, unsigned> &master_index,
                         secred::FwdStore &store, std::vector<std::uint64_t> &values,
                         std::vector<std::uint64_t> &xout,
                         std::uint32_t &fail_target, std::uint32_t &fail_integral,
                         const std::atomic<bool> *abort = nullptr,
                         secred::FwdStats *stats = nullptr,
                         const secred::FwdDag *replay_dag = nullptr,
                         std::vector<std::uint64_t> *dag_acc = nullptr);

// Build the shared numeric forward schedule from the plan's coefficients at `p`.  The
// build needs every coefficient non-zero at its point -- a vanishing one would make the
// schedule structural only up to cancellation -- so such a prime is refused before any
// work and the caller tries another.  `values` and `store` are the caller's scratch;
// the store receives the build's own elimination and is reset here.  Returns false,
// with `dag` empty, when no build ran, including when `abort` stops it between rows.
bool build_numeric_schedule(const NumericEvalPlan &plan, std::uint64_t p,
                            secred::FwdDag &dag, secred::FwdStore &store,
                            std::vector<std::uint64_t> &values,
                            const std::atomic<bool> *abort = nullptr);

// The index-th reconstruction prime.
std::uint64_t reconstruction_prime(unsigned index);

namespace secred {

// Global reduction on the FiniteFlow numeric path.
GlobalReductionResult reduce_global_finiteflow(const GlobalStageInput &input,
                                               std::uint64_t max_memory,
                                               std::uint64_t memory_reserve);

// Global reduction through FireFly.
GlobalReductionResult reduce_global_firefly_symbolic(
    const GlobalStageInput &input,
    const std::vector<GiNaC::possymbol> &symbols);

// The one global-stage entry: selects the backend from the symbol list and
// runs it on the shared input.
GlobalReductionResult run_global_stage(const GlobalStageInput &input,
                                       const std::vector<GiNaC::possymbol> &symbols,
                                       std::uint64_t max_memory,
                                       std::uint64_t memory_reserve);

// Planned subset solve used by the repair loop.
void solve_subset_planned(const std::vector<Eq> &system,
                          const std::vector<std::uint32_t> &eq_idx, std::uint64_t p,
                          SubSolve &out, bool want_fwd, bool want_bwd,
                          const std::vector<std::uint32_t> *only_keys = nullptr,
                          const std::vector<std::uint32_t> *solmap_keys = nullptr,
                          FwdStats *fwd_stats = nullptr);

// Solve a prebuilt plan from coefficient residues already in the field.
// `values` is the full plan's fill_values() output; a subset() keeps the same
// slots, so the same residues serve it.
void solve_plan(const NumericEvalPlan &plan,
                const std::vector<std::uint64_t> &values, std::uint64_t p,
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

}
