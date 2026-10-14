// The evaluation half of the symbolic global stage, shared by both reconstruction
// backends: the coefficient tables reduced per prime, and one sample point turned into
// the flattened target x master coefficient matrix.  The two backends must differ only
// in their reconstruction engine and sample-point policy, so this is the one place a
// point is evaluated.
//
// Header-only so tests can drive the real evaluator.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ginac/ginac.h>

#include "ffred_solver.h"
#include "global_select.h"
#include "masters_file.h"
#include "run_report.h"
#include "sym_coeff.h"

namespace secred {

// Simple parallel for: [0,n) split statically over nthreads threads.
template <typename F> void parallel_for(unsigned nthreads, std::size_t n, F f) {
  if (nthreads <= 1 || n <= 1) {
    for (std::size_t i = 0; i < n; ++i)
      f(i);
    return;
  }
  const unsigned nt = static_cast<unsigned>(std::min<std::size_t>(nthreads, n));
  std::vector<std::thread> workers;
  workers.reserve(nt);
  for (unsigned t = 0; t < nt; ++t) {
    const std::size_t begin = n * t / nt;
    const std::size_t end = n * (t + 1) / nt;
    if (begin >= end)
      continue;
    workers.emplace_back([&f, begin, end]() {
      for (std::size_t i = begin; i < end; ++i)
        f(i);
    });
  }
  for (auto &w : workers)
    w.join();
}

// The items a reconstruction reconstructs: one entry per (target, master) pair whose
// coefficient the reduction actually carries.  A coefficient that is identically zero is
// not an item -- FireFly would spend per-item interpolation state proving it is zero and
// the table would then carry an explicit `(((0))/(1))` line, which is exactly what a
// reconstruction that mirrors Kira's (Kira interpolates the 24,223 functions of its
// answer, not a full target x master block) does not do.
//
// `master` (the integral id, descending inside a row) and `begin` (row -> first item)
// let the evaluator pair a right-hand side with its row in one walk; `col` is the item's
// column in the master basis, which is what the driver writes the coefficient against.
struct ReductionItems {
  std::vector<std::uint32_t> master;
  std::vector<std::uint32_t> col;
  std::vector<std::uint32_t> begin;   // dep_targets.size() + 1 entries
  std::size_t size() const { return master.size(); }
};

class SymbolicFFEvaluator {
public:
  // `pool` is the whole system as the reader interned it (every equation of the spill);
  // `eq_idx` names the rows the learn selected.  The two are the reader's and the learn's
  // outputs, so nothing materialises the system on the way here -- not even once.
  SymbolicFFEvaluator(SymCoeffPool pool, std::vector<uint32_t> eq_idx,
                      std::vector<uint32_t> bwd_chain,
                      std::vector<unsigned> dep_targets,
                      std::unordered_map<unsigned, unsigned> master_index,
                      unsigned nnindeps, ReductionItems items,
                      std::vector<GiNaC::ex> symbols)
      : pool_(std::move(pool)), eq_idx_(std::move(eq_idx)),
        bwd_chain_(std::move(bwd_chain)), dep_targets_(std::move(dep_targets)),
        master_index_(std::move(master_index)), nnindeps_(nnindeps),
        items_(std::move(items)), symbols_(std::move(symbols)) {
    // The reconstruction only ever evaluates the selected equations, so the unselected
    // rows (and the coefficients only they used) are released here: for F234 that is the
    // 931k-equation read, of which 547k equations are selected.  Everything below (the
    // max exponents, the fallback flag) is a property of what the pool will evaluate.
    retain_pool_rows(pool_, eq_idx_);
    mx_ = sym_max_exps(pool_, symbols_.size());
    for (const SymCoeff &c : pool_.coeffs)
      if (!c.precompiled)
        all_precompiled_ = false;

    // The elimination order, once.  forward_eliminate sorts by structure (front id, row
    // length, id list, sys_idx) and never by a coefficient, so this is the same order at
    // every prime: a point replays the structure the learn recorded instead of deriving
    // its own from the residues.  It also removes a per-point sort of 547k rows.
    //
    // `eq_idx_` outlives the compaction for this one call: it is the key's last component,
    // and the values here are the system rows the learn sorted by, so the plan is the
    // order the learn would have derived rather than a renumbered copy of it.
    build_fwd_plan(pool_.terms, pool_.eq_begin, eq_idx_, plan_);

    // The same rows as a table the elimination can walk: the plan's structure check and the
    // per-point gather both read the pool's term table directly, so a point holds no copy
    // of the system (see SymRowTable).
    rows_.terms = pool_.terms.data();
    rows_.eq_begin = pool_.eq_begin.data();
    rows_.n_rows = pool_.n_equations();

    // The id space the store's membership bitset must cover: every integral the selected
    // equations mention, plus the targets and the chain (both already appear among them,
    // but the store must be sized for whatever is asked of it).
    std::uint32_t max_id = 0;
    for (const auto &t : pool_.terms)
      max_id = std::max(max_id, t.first);
    for (std::uint32_t k : bwd_chain_)
      max_id = std::max(max_id, k);
    for (unsigned k : dep_targets_)
      max_id = std::max(max_id, k);
    store_ids_ = static_cast<std::size_t>(max_id) + 1;
  }

  // Per-thread scratch for one sample point.  One per worker: nothing here is shared
  // between threads, and the coefficient tables (which are) live in the pool.
  struct Scratch {
    std::vector<std::vector<uint64_t>> powers; // powers[i][k] = x_i^k at this point
    std::vector<uint64_t> values;              // one per distinct function, per point
    FwdStore store;                // this point's solutions (see ffred_solver.h)
    std::uint64_t zero_dens = 0;   // coefficients whose denominator vanished here
    // Per-phase seconds accumulated over this scratch's whole life: the probe's share of
    // coefficient evaluation, forward elimination and back substitution.  The caller
    // counts deltas (a scratch is thread-local and outlives one evaluator), and the
    // firefly driver prints the three as per-probe averages, the way Kira's probe report
    // does.  Three clock reads per point against ~a second of work: free.
    double eval_seconds = 0.0, fwd_seconds = 0.0, bwd_seconds = 0.0;
    // Set by a NonMasterRhs result, for the caller's message.
    unsigned fail_target = 0;
    unsigned fail_integral = 0;
  };

  // MissingItem: a master with a non-zero coefficient that the learn point did not carry
  // on that right-hand side.  The item list would miss the term, so the driver stops
  // rather than emit a truncated reduction (see the walk in evaluate_point).
  enum class PointResult { Ok, NonMasterRhs, MissingItem, NotReduced };

  // Reduce every distinct coefficient mod p.  One prime's tables live in the pool at a
  // time, so this runs between evaluation phases and never while a thread is evaluating.
  void reduce_for_prime(uint64_t p, unsigned nthreads) {
    parallel_for(nthreads, pool_.coeffs.size(),
                 [&](std::size_t i) { reduce_pool_coeff(pool_.coeffs[i], p); });
  }

  // Evaluate one sample point: substitute, reduce mod p, forward-eliminate,
  // back-substitute along the lazy chain, and flatten the targets' reductions into
  // `xout[i * nnindeps + j]` (i over dep_targets_, j over the masters).
  PointResult evaluate_point(uint64_t p, const uint64_t *x, Scratch &scratch,
                             uint64_t *xout) const {
    // A scratch is per-thread and may pass between evaluators (firefly_global.h's
    // scratch_of is thread_local, and one worker reconstructs several parts), so size
    // the power table for THIS evaluator's system every point: a smaller earlier system
    // must not leave rows too short for a deeper one.  Resizing to the size it already
    // has allocates nothing.
    const unsigned nvars = symbols_.size();
    scratch.powers.resize(nvars);
    for (unsigned i = 0; i < nvars; ++i)
      scratch.powers[i].resize(mx_[i] + 1, 1);   // powers[i][0] stays x_i^0 = 1
    const auto t_begin = std::chrono::steady_clock::now();
    fill_powers(scratch.powers, x, p);

    GiNaC::exmap subs;
    if (!all_precompiled_)
      for (unsigned i = 0; i < nvars; ++i)
        subs[symbols_[i]] = GiNaC::numeric(x[i]);

    if (!eval_pool_point(pool_, subs, scratch.powers, p, scratch.values,
                         &scratch.zero_dens)) {
      scratch.eval_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t_begin)
              .count();
      return PointResult::NotReduced;
    }
    const auto t_eval = std::chrono::steady_clock::now();

    // One plan, one flat store: the pivots and rows forward_eliminate would produce from
    // these equations, without the per-pivot heap row and hash entry, and with the
    // division-free product (see ffred_solver.h).  The plan was built in the constructor
    // from the pool's structure and eq_idx_ as sys_idx, which is exactly the key
    // forward_eliminate sorts by, so the learn's structure replays here term for term --
    // and where a coefficient of these rows vanished at this point, the elimination
    // notices the structure moved and reorders the rows the way the reference would.
    // The rows are the pool's, read on demand against this point's slot values, so the
    // scratch holds the solutions rather than a second copy of the system.
    scratch.store.reset(store_ids_);
    forward_eliminate_planned(rows_, scratch.values.data(), scratch.values.size(), plan_, p,
                              scratch.store);
    const auto t_fwd = std::chrono::steady_clock::now();
    backward_eliminate_planned(scratch.store, p, bwd_chain_);
    const auto t_bwd = std::chrono::steady_clock::now();
    scratch.eval_seconds += std::chrono::duration<double>(t_eval - t_begin).count();
    scratch.fwd_seconds += std::chrono::duration<double>(t_fwd - t_eval).count();
    scratch.bwd_seconds += std::chrono::duration<double>(t_bwd - t_fwd).count();

    const unsigned nndeps = static_cast<unsigned>(dep_targets_.size());
    std::fill(xout, xout + items_.size(), std::uint64_t(0));
    for (unsigned i = 0; i < nndeps; ++i) {
      const std::uint32_t target = dep_targets_[i];
      if (!scratch.store.has_row(target))
        continue;
      const std::uint32_t *red_ids = scratch.store.row_ids(target);
      const std::uint64_t *red_vals = scratch.store.row_vals(target);
      const std::uint32_t red_size = scratch.store.row_size(target);
      // The row's items and the right-hand side are both descending by integral id, so
      // one walk pairs them: an item the row does not carry at this point is skipped and
      // keeps its zero.
      std::size_t q = items_.begin[i];
      const std::size_t q_end = items_.begin[i + 1];
      for (std::uint32_t k = 1; k < red_size; ++k) {
        const uint32_t id = red_ids[k];
        if (master_index_.find(id) == master_index_.end()) {
          // A non-master on a right-hand side means the selection is stale at this
          // sample point and prime, so the output would miss a term.  The caller aborts
          // rather than emit a truncated reduction.
          scratch.fail_target = dep_targets_[i];
          scratch.fail_integral = id;
          return PointResult::NonMasterRhs;
        }
        while (q < q_end && items_.master[q] > id)
          ++q;                        // zero at this point: not carried on this side
        if (q < q_end && items_.master[q] == id) {
          xout[q] = red_vals[k];
          ++q;
        } else {
          // A master with a non-zero coefficient that the learn point did not carry on
          // this right-hand side.  The item list would miss the term, so the run stops
          // rather than emit a truncated reduction; for generic points the support of a
          // reduction is a property of the reduction, not of the point, so this fires
          // only when the learn point was special.
          scratch.fail_target = dep_targets_[i];
          scratch.fail_integral = id;
          return PointResult::MissingItem;
        }
      }
    }
    return PointResult::Ok;
  }

  // The number of items the reconstruction reconstructs -- one per (target, master) pair
  // the reduction carries, not one per cell of the target x master block.
  unsigned n_outputs() const { return static_cast<unsigned>(items_.size()); }
  unsigned n_params() const { return static_cast<unsigned>(symbols_.size()); }
  // The equations a sample point evaluates and eliminates (the selected subset, not the
  // whole system that was read).
  std::size_t n_equations() const { return pool_.n_equations(); }
  // Kira's own accounting for the same reconstruction ("Parsed N functions of which M are
  // distinct"): the equation terms that must be filled per sample point, and the distinct
  // functions the pool evaluates once each -- the rest of the terms are table reads.
  std::size_t n_coefficient_terms() const { return pool_.n_terms(); }
  std::size_t n_distinct_coeffs() const { return pool_.coeffs.size(); }
  bool all_precompiled() const { return all_precompiled_; }

private:
  SymCoeffPool pool_;
  // The pool's rows as the elimination walks them.  Built after the pool, and only the
  // pool's coefficient tables change per prime, so the view stays valid for the
  // evaluator's whole life.
  SymRowTable rows_;
  std::vector<uint32_t> eq_idx_;
  std::vector<uint32_t> bwd_chain_;
  std::vector<unsigned> dep_targets_;
  std::unordered_map<unsigned, unsigned> master_index_;
  unsigned nnindeps_;
  ReductionItems items_;
  std::vector<GiNaC::ex> symbols_;
  std::vector<unsigned> mx_;   // highest power each symbol carries
  bool all_precompiled_ = true;
  FwdPlan plan_;               // the elimination order, rebuilt per evaluator
  std::size_t store_ids_ = 0;  // ids a scratch's store must cover
};

// ---------------------------------------------------------------------------
// The learn and selection the two symbolic backends share
// ---------------------------------------------------------------------------

// Solve one subset of the evaluated system: forward eliminate, then back-substitute.
// The learn's numeric sufficiency repair re-solves a candidate subset through this.
inline void solve_res_eqs(std::vector<ResEq> &res_eqs, std::vector<uint32_t> &sys_idx,
                          uint64_t p, SubSolve &out) {
  out.solmap.clear();
  out.eq_index.clear();
  out.fwd_deps.clear();
  out.bwd_deps.clear();
  forward_eliminate(res_eqs, sys_idx, p, out.solmap, nullptr, nullptr);
  backward_eliminate(out.solmap, p, &out.bwd_deps);
}

// Write the final master basis (one integral per line, as in masters_sector).  It is
// known as soon as the learn pass has run, so it lands before the reconstruction —
// which can be the longest phase of the run.
template <typename Integral>
void write_masters_global(const std::string &path,
                         const std::vector<Integral> &all_integrals,
                         const std::vector<unsigned> &masters,
                         const char *tag = kFlowTag) {
  std::vector<Integral> basis;
  basis.reserve(masters.size());
  for (unsigned m : masters)
    basis.push_back(all_integrals[m]);
  if (!write_masters_file(path, basis)) {
    std::cerr << tag << " masters_global: cannot write " << path << "\n";
    return;
  }
  std::cout << "  " << tag << " masters_global: " << basis.size()
            << " masters written to " << path << "\n";
}

// What the learn hands the reconstruction: the master basis, the selection, and the
// inputs the evaluator is built from.
struct SymbolicGlobalLearn {
  bool ok = false;
  std::string error;
  std::vector<unsigned> masters;          // ids into all_integrals
  std::uint64_t selected_equations = 0;   // learn_solmap.size()
  unsigned nnindeps = 0;
  double learn_seconds = 0.0;
  std::vector<std::uint32_t> needed_eq_idx;
  std::vector<std::uint32_t> bwd_chain;
  std::vector<unsigned> dep_targets;
  std::unordered_map<unsigned, unsigned> master_index;
  ReductionItems items;   // what the reconstruction reconstructs
};

// The selection the two symbolic backends share, so that a difference between them is
// a difference between reconstruction engines and nothing else.
//
// Learn at one deterministic generic sample point (a fixed seed, no clock and no
// entropy, so the same input picks the same equations in every run and in both
// backends), forward/backward eliminate, take the masters as the union of the targets'
// right-hand sides, mark-and-sweep from the targets with the numeric sufficiency repair
// (global_select.h), write masters_global, and print the Learn Summary and Reduction
// Statistics blocks.
//
// `system` is mutated (its tables are reduced for the learn prime) and is meant to be
// moved into the evaluator afterwards; `all_integrals` is templated so this header does
// not have to pull in the sector machinery for one `push_back`.
template <typename Integral>
SymbolicGlobalLearn symbolic_global_learn(
    SymCoeffPool &pool, const std::vector<unsigned> &target_ids,
    const std::vector<Integral> &all_integrals,
    const std::vector<GiNaC::possymbol> &symbols, std::uint64_t learn_prime,
    const std::string &masters_global_path, const char *tag,
    unsigned narrowing_success, unsigned narrowing_fail,
    unsigned raw_equations_total, std::size_t global_equations_count) {
  SymbolicGlobalLearn out;
  const unsigned nparams = symbols.size();
  const unsigned neqs = static_cast<unsigned>(pool.n_equations());
  std::vector<std::uint32_t> all_idx(neqs);

  // 1) A generic sample point, fixed seed: the learn must be reproducible.
  std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
  std::vector<std::uint64_t> learn_pt(nparams);
  for (unsigned i = 0; i < nparams; ++i)
    learn_pt[i] = rng() % learn_prime;
  GiNaC::exmap learn_subs;
  for (unsigned i = 0; i < nparams; ++i)
    learn_subs[GiNaC::ex(symbols[i])] = GiNaC::numeric(learn_pt[i]);
  const std::vector<std::vector<uint64_t>> learn_powers =
      make_powers(pool, learn_pt.data(), nparams, learn_prime);

  // 2) Evaluate the whole system at that point and eliminate it: masters + structure +
  // dependencies.
  std::vector<ResEq> learn_res_eqs;
  std::vector<std::uint32_t> learn_sys_idx;
  {
    for (unsigned i = 0; i < neqs; ++i)
      all_idx[i] = i;
    reduce_pool_for_prime(pool, learn_prime);
    if (!eval_pool_rows(pool, all_idx, learn_subs, learn_powers, learn_prime,
                        learn_res_eqs, learn_sys_idx)) {
      out.error = "coefficients were not reduced for the learn prime";
      return out;
    }
  }

  std::unordered_map<uint32_t, ResEq> learn_solmap;
  std::unordered_map<uint32_t, uint32_t> eq_index;
  std::unordered_map<uint32_t, std::vector<uint32_t>> fwd_deps, bwd_deps;
  {
    const auto t_learn0 = std::chrono::steady_clock::now();
    forward_eliminate(learn_res_eqs, learn_sys_idx, learn_prime, learn_solmap,
                      &eq_index, &fwd_deps);
    backward_eliminate(learn_solmap, learn_prime, &bwd_deps);
    const auto t_learn1 = std::chrono::steady_clock::now();
    out.learn_seconds = std::chrono::duration<double>(t_learn1 - t_learn0).count();
  }

  // Masters = union of the right-hand sides of the target reductions.
  std::set<unsigned> needed_masters;
  for (unsigned id : target_ids) {
    auto it = learn_solmap.find(id);
    if (it == learn_solmap.end())
      continue;
    const ResEq &red = it->second;
    for (std::size_t k = 1; k < red.size(); ++k)
      needed_masters.insert(red[k].first);
  }
  out.masters.assign(needed_masters.rbegin(), needed_masters.rend());
  out.nnindeps = static_cast<unsigned>(out.masters.size());

  for (unsigned id : target_ids)
    if (learn_solmap.find(id) != learn_solmap.end())
      out.dep_targets.push_back(id);

  for (unsigned j = 0; j < out.nnindeps; ++j)
    out.master_index[out.masters[j]] = j;

  // The reconstruction's item list: one entry per (target, master) pair whose coefficient
  // the reduction carries at the learn point.  A coefficient that is identically zero is
  // not an item -- reconstructing it would buy FireFly per-item interpolation state to
  // prove it is zero and then write an explicit zero into the table.  The right-hand
  // sides are descending by id, so each row's items come out descending as well, which is
  // the order evaluate_point's walk pairs against.
  out.items.begin.reserve(out.dep_targets.size() + 1);
  for (unsigned id : out.dep_targets) {
    out.items.begin.push_back(static_cast<std::uint32_t>(out.items.master.size()));
    const ResEq &red = learn_solmap.at(id);
    for (std::size_t k = 1; k < red.size(); ++k) {
      auto mit = out.master_index.find(red[k].first);
      if (mit == out.master_index.end())
        continue;   // cannot happen: the masters are the union of these right-hand sides
      out.items.master.push_back(red[k].first);
      out.items.col.push_back(mit->second);
    }
  }
  out.items.begin.push_back(static_cast<std::uint32_t>(out.items.master.size()));

  // Mark-and-sweep from the targets along the forward / backward dependency closure.
  auto sweep = [&](bool with_bwd) {
    std::unordered_set<uint32_t> s(out.dep_targets.begin(), out.dep_targets.end());
    std::vector<uint32_t> stack(out.dep_targets.begin(), out.dep_targets.end());
    while (!stack.empty()) {
      uint32_t w = stack.back();
      stack.pop_back();
      auto fit = fwd_deps.find(w);
      if (fit != fwd_deps.end())
        for (uint32_t u : fit->second)
          if (s.insert(u).second)
            stack.push_back(u);
      if (with_bwd) {
        auto bit = bwd_deps.find(w);
        if (bit != bwd_deps.end())
          for (uint32_t u : bit->second)
            if (s.insert(u).second)
              stack.push_back(u);
      }
    }
    return s;
  };
  std::unordered_set<uint32_t> needed =
      sweep(true); // fallback closure: forward and backward
  bool use_forward = false;
  {
    // The repair loop re-solves the subsystem at the learn sample point (same prime and
    // same symbolic values as learn).
    auto solve_at_learn = [&](const std::vector<uint32_t> &idx, SubSolve &sub) {
      std::vector<ResEq> res_eqs;
      std::vector<uint32_t> sys_idx;
      eval_pool_rows(pool, idx, learn_subs, learn_powers, learn_prime, res_eqs,
                     sys_idx);
      solve_res_eqs(res_eqs, sys_idx, learn_prime, sub);
    };
    GlobalSelection gsel = select_global_forward(
        eq_index, fwd_deps, bwd_deps, learn_solmap, out.dep_targets, out.master_index,
        solve_at_learn);
    if (gsel.fallback) {
      std::cerr << tag << " forward selection did not converge, falling back to the "
                   "legacy closure\n";
    } else {
      use_forward = true;
      // Logged before the moves: the decision line quotes the selection's own row count.
      std::cout << "\n";
      log_selection(std::string(tag) + " ", gsel);
      out.needed_eq_idx = std::move(gsel.needed_eq_idx);
      out.bwd_chain = std::move(gsel.bwd_chain);
    }
  }
  if (!use_forward) {
    out.needed_eq_idx.reserve(needed.size());
    for (uint32_t w : needed) {
      auto it = eq_index.find(w);
      if (it != eq_index.end())
        out.needed_eq_idx.push_back(it->second);
    }
    std::sort(out.needed_eq_idx.begin(), out.needed_eq_idx.end());
    // Lazy chain (targets plus the derived integrals in the backward dependency closure).
    std::unordered_set<uint32_t> chain_set(out.dep_targets.begin(),
                                           out.dep_targets.end());
    std::vector<uint32_t> stack(out.dep_targets.begin(), out.dep_targets.end());
    while (!stack.empty()) {
      uint32_t w = stack.back();
      stack.pop_back();
      auto bit = bwd_deps.find(w);
      if (bit != bwd_deps.end())
        for (uint32_t u : bit->second)
          if (chain_set.insert(u).second)
            stack.push_back(u);
    }
    out.bwd_chain.assign(chain_set.begin(), chain_set.end());
    std::sort(out.bwd_chain.begin(), out.bwd_chain.end());
  }

  out.selected_equations = static_cast<std::uint64_t>(learn_solmap.size());

  {
    LearnSummary ls;
    ls.independent_equations = out.selected_equations;
    ls.masters = out.nnindeps;
    ls.seconds = out.learn_seconds;
    std::cout << "\n" << format_learn_summary(ls, tag);
  }

  write_masters_global(masters_global_path, all_integrals, out.masters, tag);

  // Consolidated Reduction Statistics (as in the numeric path): "Final
  // equations" is what the mark-and-sweep keeps, i.e. what the reconstruction
  // actually runs on.
  {
    ReductionStats rs;
    rs.narrowing_success = narrowing_success;
    rs.narrowing_fail = narrowing_fail;
    rs.raw_equations = raw_equations_total;
    rs.global_equations = global_equations_count;
    rs.final_equations = out.needed_eq_idx.size();
    rs.masters = out.nnindeps;
    std::cout << "\n" << format_reduction_statistics(rs);
  }

  out.ok = true;
  return out;
}

}  // namespace secred
