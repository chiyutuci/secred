
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
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

// Run f over [0, n) on up to nthreads threads.
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

// Run f over [0, n) on up to nthreads threads, claiming indices dynamically:
// each free thread takes the next index, so a few heavy items no longer pin
// one worker while the others idle.
template <typename F>
void parallel_for_dynamic(unsigned nthreads, std::size_t n, F f) {
  if (nthreads <= 1 || n <= 1) {
    for (std::size_t i = 0; i < n; ++i)
      f(i);
    return;
  }
  std::atomic<std::size_t> next{0};
  const unsigned nt = static_cast<unsigned>(std::min<std::size_t>(nthreads, n));
  std::vector<std::thread> workers;
  workers.reserve(nt);
  for (unsigned t = 0; t < nt; ++t) {
    workers.emplace_back([&f, &next, n]() {
      for (;;) {
        const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= n)
          break;
        f(i);
      }
    });
  }
  for (auto &w : workers)
    w.join();
}


// Flattened target-to-master output layout.
struct ReductionItems {
  std::vector<std::uint32_t> master;
  std::vector<std::uint32_t> col;
  std::vector<std::uint32_t> begin;
  std::size_t size() const { return master.size(); }
};

// Reduces the selected symbolic system at one field point.
class SymbolicFFEvaluator {
public:

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

    retain_pool_rows(pool_, eq_idx_);
    mx_ = sym_max_exps(pool_, symbols_.size());
    for (const SymCoeff &c : pool_.coeffs)
      if (!c.precompiled())
        all_precompiled_ = false;

    build_fwd_plan(pool_.terms, pool_.eq_begin, eq_idx_, plan_);

    rows_.terms = pool_.terms.data();
    rows_.eq_begin = pool_.eq_begin.data();
    rows_.n_rows = pool_.n_equations();

    std::uint32_t max_id = 0;
    for (const auto &t : pool_.terms)
      max_id = std::max(max_id, t.first);
    for (std::uint32_t k : bwd_chain_)
      max_id = std::max(max_id, k);
    for (unsigned k : dep_targets_)
      max_id = std::max(max_id, k);
    store_ids_ = static_cast<std::size_t>(max_id) + 1;

    // The forward-elimination schedule (dag_) is built lazily from the first
    // probe whose coefficients are all nonzero: the schedule follows from the
    // coefficient values, so it needs one real elimination, and that probe
    // keeps its own elimination's store instead of paying for a separate
    // build pass at construction time.
  }

  struct Scratch {
    std::vector<std::vector<uint64_t>> powers;
    std::vector<uint64_t> values;
    FwdStore store;
    std::vector<uint64_t> acc;
    std::uint64_t zero_dens = 0;

    double eval_seconds = 0.0, fwd_seconds = 0.0, bwd_seconds = 0.0;

    FwdStats fwd_stats;

    unsigned fail_target = 0;
    unsigned fail_integral = 0;
  };

  enum class PointResult { Ok, NonMasterRhs, MissingItem, NotReduced };

  void reduce_for_prime(uint64_t p, unsigned nthreads) {
    parallel_for(nthreads, pool_.coeffs.size(),
                 [&](std::size_t i) { reduce_pool_coeff(pool_.coeffs[i], p); });
  }

  // Point the row table back at this object's own pool after a copy.
  void rebind_rows() {
    rows_.terms = pool_.terms.data();
    rows_.eq_begin = pool_.eq_begin.data();
    rows_.n_rows = pool_.n_equations();
  }

  PointResult evaluate_point(uint64_t p, const uint64_t *x, Scratch &scratch,
                             uint64_t *xout) const {

    const unsigned nvars = symbols_.size();
    scratch.powers.resize(nvars);
    for (unsigned i = 0; i < nvars; ++i)
      scratch.powers[i].resize(mx_[i] + 1, 1);
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

    scratch.store.reset(store_ids_);
    FwdStats point_stats;
    bool reduced = false;
    // A vanishing coefficient would bake a cancellation pattern that holds only at
    // this point, so the schedule is built only from a point whose values are all
    // nonzero.  Whichever thread gets there first builds it while the others keep
    // reducing dynamically; from then on every probe replays it.
    const bool no_zeros =
        std::find(scratch.values.begin(), scratch.values.end(), std::uint64_t(0)) ==
        scratch.values.end();
    FwdSchedule &sch = *fwd_schedule_;
    unsigned state = sch.state.load(std::memory_order_acquire);
    if (no_zeros && state == kDagUnbuilt &&
        sch.state.compare_exchange_strong(state, kDagBuilding,
                                          std::memory_order_acq_rel)) {
      reduced = build_fwd_dag(rows_, scratch.values.data(), scratch.values.size(),
                              p, plan_, sch.dag, scratch.store, &point_stats);
      sch.state.store(reduced ? kDagReady : kDagFailed, std::memory_order_release);
    } else if (no_zeros && state == kDagReady) {
      reduced = forward_eliminate_dag(rows_, scratch.values.data(), sch.dag, p,
                                      scratch.store, scratch.acc, nullptr,
                                      &point_stats);
    }
    if (!reduced) {
      scratch.store.reset(store_ids_);
      point_stats = FwdStats{};
      forward_eliminate_planned(
          rows_, scratch.values.data(), scratch.values.size(), plan_, p,
          scratch.store, nullptr, nullptr, nullptr, &point_stats);
    }
    scratch.fwd_stats.add(point_stats);
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

      std::size_t q = items_.begin[i];
      const std::size_t q_end = items_.begin[i + 1];
      for (std::uint32_t k = 1; k < red_size; ++k) {
        const uint32_t id = red_ids[k];
        if (master_index_.find(id) == master_index_.end()) {

          scratch.fail_target = dep_targets_[i];
          scratch.fail_integral = id;
          return PointResult::NonMasterRhs;
        }
        while (q < q_end && items_.master[q] > id)
          ++q;
        if (q < q_end && items_.master[q] == id) {
          xout[q] = red_vals[k];
          ++q;
        } else {

          scratch.fail_target = dep_targets_[i];
          scratch.fail_integral = id;
          return PointResult::MissingItem;
        }
      }
    }
    return PointResult::Ok;
  }

  unsigned n_outputs() const { return static_cast<unsigned>(items_.size()); }
  unsigned n_params() const { return static_cast<unsigned>(symbols_.size()); }

  std::size_t n_equations() const { return pool_.n_equations(); }

  std::size_t n_coefficient_terms() const { return pool_.n_terms(); }
  std::size_t n_distinct_coeffs() const { return pool_.coeffs.size(); }
  bool all_precompiled() const { return all_precompiled_; }

  bool has_fwd_dag() const {
    return fwd_schedule_->state.load(std::memory_order_acquire) == kDagReady;
  }
  std::size_t fwd_dag_rows() const { return fwd_schedule_->dag.rows.size(); }
  std::size_t fwd_dag_merges() const { return fwd_schedule_->dag.merges.size(); }
  // Debug hooks for the elimination benches.
  const FwdDag &fwd_dag() const { return fwd_schedule_->dag; }
  const FwdPlan &fwd_plan() const { return plan_; }
  const SymRowTable &row_table() const { return rows_; }
  std::size_t n_store_ids() const { return store_ids_; }

private:
  // state values.  The schedule is absent until one probe builds it, and a build
  // that cannot follow the point (bad table, zero-size store) fails for the whole
  // run rather than repeating the attempt.
  static constexpr unsigned kDagUnbuilt = 0, kDagBuilding = 1, kDagReady = 2,
                            kDagFailed = 3;

  // The schedule and its build state, shared by every copy of this evaluator:
  // the fflow backend keeps one copy per prime of a batch, and they all describe
  // the same row table, so one schedule serves them all -- building one per copy
  // would repeat the elimination and multiply the schedule's memory.
  struct FwdSchedule {
    FwdDag dag;
    std::atomic<unsigned> state{kDagUnbuilt};
  };

  SymCoeffPool pool_;

  SymRowTable rows_;
  std::vector<uint32_t> eq_idx_;
  std::vector<uint32_t> bwd_chain_;
  std::vector<unsigned> dep_targets_;
  std::unordered_map<unsigned, unsigned> master_index_;
  unsigned nnindeps_;
  ReductionItems items_;
  std::vector<GiNaC::ex> symbols_;
  std::vector<unsigned> mx_;
  bool all_precompiled_ = true;
  FwdPlan plan_;
  // Written by the one probe that builds the schedule, read by every probe (of
  // every copy) after the release store that publishes it.
  std::shared_ptr<FwdSchedule> fwd_schedule_ =
      std::make_shared<FwdSchedule>();
  std::size_t store_ids_ = 0;
};

// Evaluate the whole pool at one point and solve it with dependencies.
inline void solve_pool_planned(const SymCoeffPool &pool,
                               const std::vector<std::uint64_t> &values,
                               std::uint64_t p, SubSolve &out) {
  out.solmap.clear();
  out.eq_index.clear();
  out.fwd_deps.clear();
  out.bwd_deps.clear();

  const std::size_t nrows = pool.n_equations();
  std::vector<std::uint32_t> sys_idx(nrows);
  for (std::size_t i = 0; i < nrows; ++i)
    sys_idx[i] = static_cast<std::uint32_t>(i);
  FwdPlan plan;
  build_fwd_plan(pool.terms, pool.eq_begin, sys_idx, plan);
  SymRowTable rows;
  rows.terms = pool.terms.data();
  rows.eq_begin = pool.eq_begin.data();
  rows.n_rows = nrows;

  std::uint32_t max_id = 0;
  for (const auto &t : pool.terms)
    max_id = std::max(max_id, t.first);
  FwdStore store;
  store.reset(static_cast<std::size_t>(max_id) + 1);
  std::vector<std::uint32_t> row_src;
  forward_eliminate_planned(rows, values.data(), values.size(), plan, p, store,
                            nullptr, &out.fwd_deps, &row_src, nullptr);

  const std::size_t n_fwd_rows = store.rows();
  std::vector<std::uint32_t> keys;
  keys.reserve(n_fwd_rows);
  for (std::size_t r = 0; r < n_fwd_rows; ++r)
    keys.push_back(store.ids[store.begin[r]]);
  std::sort(keys.begin(), keys.end());
  backward_eliminate_planned(store, p, keys, nullptr, &out.bwd_deps);
  for (std::size_t r = 0; r < n_fwd_rows; ++r) {
    const std::uint32_t id = store.ids[store.begin[r]];
    const std::uint32_t *ri = store.row_ids(id);
    const std::uint64_t *rv = store.row_vals(id);
    const std::uint32_t n = store.row_size(id);
    ResEq re;
    re.reserve(n);
    for (std::uint32_t t = 0; t < n; ++t)
      re.emplace_back(ri[t], rv[t]);
    out.solmap.emplace(id, std::move(re));
  }
  for (std::size_t r = 0; r < n_fwd_rows; ++r)
    out.eq_index[store.ids[store.begin[r]]] = plan.sys_idx[row_src[r]];
}

// Solve already-evaluated residue equations.
inline void solve_res_eqs(std::vector<ResEq> &res_eqs, std::vector<uint32_t> &sys_idx,
                          uint64_t p, SubSolve &out) {
  out.solmap.clear();
  out.eq_index.clear();
  out.fwd_deps.clear();
  out.bwd_deps.clear();
  forward_eliminate(res_eqs, sys_idx, p, out.solmap, nullptr, nullptr);
  backward_eliminate(out.solmap, p, &out.bwd_deps);
}

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

// Result of the symbolic global learn.
struct SymbolicGlobalLearn {
  bool ok = false;
  std::string error;
  std::vector<unsigned> masters;
  std::uint64_t selected_equations = 0;
  unsigned nnindeps = 0;
  double learn_seconds = 0.0;
  std::vector<std::uint32_t> needed_eq_idx;
  std::vector<std::uint32_t> bwd_chain;
  std::vector<unsigned> dep_targets;
  std::unordered_map<unsigned, unsigned> master_index;
  ReductionItems items;
};

// Learn the global system symbolically at one prime.
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

  std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
  std::vector<std::uint64_t> learn_pt(nparams);
  for (unsigned i = 0; i < nparams; ++i)
    learn_pt[i] = rng() % learn_prime;
  GiNaC::exmap learn_subs;
  for (unsigned i = 0; i < nparams; ++i)
    learn_subs[GiNaC::ex(symbols[i])] = GiNaC::numeric(learn_pt[i]);
  const std::vector<std::vector<uint64_t>> learn_powers =
      make_powers(pool, learn_pt.data(), nparams, learn_prime);

  std::vector<std::uint64_t> learn_values;
  {
    reduce_pool_for_prime(pool, learn_prime);
    if (!eval_pool_point(pool, learn_subs, learn_powers, learn_prime, learn_values)) {
      out.error = "coefficients were not reduced for the learn prime";
      return out;
    }
  }

  SubSolve learn;
  {
    const auto t_learn0 = std::chrono::steady_clock::now();
    solve_pool_planned(pool, learn_values, learn_prime, learn);
    const auto t_learn1 = std::chrono::steady_clock::now();
    out.learn_seconds = std::chrono::duration<double>(t_learn1 - t_learn0).count();
  }
  std::unordered_map<uint32_t, ResEq> &learn_solmap = learn.solmap;
  std::unordered_map<uint32_t, uint32_t> &eq_index = learn.eq_index;
  std::unordered_map<uint32_t, std::vector<uint32_t>> &fwd_deps = learn.fwd_deps;
  std::unordered_map<uint32_t, std::vector<uint32_t>> &bwd_deps = learn.bwd_deps;

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

  out.items.begin.reserve(out.dep_targets.size() + 1);
  for (unsigned id : out.dep_targets) {
    out.items.begin.push_back(static_cast<std::uint32_t>(out.items.master.size()));
    const ResEq &red = learn_solmap.at(id);
    for (std::size_t k = 1; k < red.size(); ++k) {
      auto mit = out.master_index.find(red[k].first);
      if (mit == out.master_index.end())
        continue;
      out.items.master.push_back(red[k].first);
      out.items.col.push_back(mit->second);
    }
  }
  out.items.begin.push_back(static_cast<std::uint32_t>(out.items.master.size()));

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
      sweep(true);
  bool use_forward = false;
  {

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

}
