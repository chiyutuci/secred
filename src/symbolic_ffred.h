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

class SymbolicFFEvaluator {
public:
  SymbolicFFEvaluator(std::vector<SymEq> system, std::vector<uint32_t> eq_idx,
                      std::vector<uint32_t> bwd_chain,
                      std::vector<unsigned> dep_targets,
                      std::unordered_map<unsigned, unsigned> master_index,
                      unsigned nnindeps, std::vector<GiNaC::ex> symbols)
      : system_(std::move(system)), eq_idx_(std::move(eq_idx)),
        bwd_chain_(std::move(bwd_chain)), dep_targets_(std::move(dep_targets)),
        master_index_(std::move(master_index)), nnindeps_(nnindeps),
        symbols_(std::move(symbols)) {
    mx_ = sym_max_exps(system_, symbols_.size());
    for (const SymEq &eq : system_)
      for (const auto &pr : eq)
        if (!pr.second.precompiled)
          all_precompiled_ = false;
  }

  // Per-thread scratch for one sample point.  One per worker: nothing here is shared
  // between threads, and the coefficient tables (which are) live in the system.
  struct Scratch {
    std::vector<std::vector<uint64_t>> powers; // powers[i][k] = x_i^k at this point
    std::vector<ResEq> res_eqs;
    std::vector<uint32_t> sys_idx;
    std::unordered_map<uint32_t, ResEq> solmap;
    std::uint64_t zero_dens = 0;   // coefficients whose denominator vanished here
    // Set by a NonMasterRhs result, for the caller's message.
    unsigned fail_target = 0;
    unsigned fail_integral = 0;
  };

  enum class PointResult { Ok, NonMasterRhs, NotReduced };

  // Reduce every coefficient of the system mod p.  One prime's tables live in the
  // system at a time, so this runs between evaluation phases and never while a thread
  // is evaluating.
  void reduce_for_prime(uint64_t p, unsigned nthreads) {
    parallel_for(nthreads, system_.size(), [&](std::size_t i) {
      for (auto &pr : system_[i]) {
        if (!pr.second.precompiled)
          continue;
        reduce_sym_poly(pr.second.num, p);
        reduce_sym_poly(pr.second.den, p);
      }
    });
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
    fill_powers(scratch.powers, x, p);

    GiNaC::exmap subs;
    if (!all_precompiled_)
      for (unsigned i = 0; i < nvars; ++i)
        subs[symbols_[i]] = GiNaC::numeric(x[i]);

    if (!eval_sym_eqs_mod(system_, eq_idx_, subs, scratch.powers, p, scratch.res_eqs,
                          scratch.sys_idx, &scratch.zero_dens))
      return PointResult::NotReduced;

    scratch.solmap.clear();
    forward_eliminate(scratch.res_eqs, scratch.sys_idx, p, scratch.solmap, nullptr,
                      nullptr);
    backward_eliminate(scratch.solmap, p, nullptr, &bwd_chain_);

    const unsigned nndeps = static_cast<unsigned>(dep_targets_.size());
    std::fill(xout, xout + nndeps * nnindeps_, std::uint64_t(0));
    for (unsigned i = 0; i < nndeps; ++i) {
      auto it = scratch.solmap.find(dep_targets_[i]);
      if (it == scratch.solmap.end())
        continue;
      const ResEq &red = it->second;
      for (std::size_t k = 1; k < red.size(); ++k) {
        auto mit = master_index_.find(red[k].first);
        if (mit != master_index_.end()) {
          xout[i * nnindeps_ + mit->second] = red[k].second;
        } else {
          // A non-master on a right-hand side means the selection is stale at this
          // sample point and prime, so the output would miss a term.  The caller aborts
          // rather than emit a truncated reduction.
          scratch.fail_target = dep_targets_[i];
          scratch.fail_integral = red[k].first;
          return PointResult::NonMasterRhs;
        }
      }
    }
    return PointResult::Ok;
  }

  unsigned n_outputs() const {
    return static_cast<unsigned>(dep_targets_.size() * nnindeps_);
  }
  unsigned n_params() const { return static_cast<unsigned>(symbols_.size()); }
  std::size_t n_equations() const { return system_.size(); }
  bool all_precompiled() const { return all_precompiled_; }

private:
  std::vector<SymEq> system_;
  std::vector<uint32_t> eq_idx_;
  std::vector<uint32_t> bwd_chain_;
  std::vector<unsigned> dep_targets_;
  std::unordered_map<unsigned, unsigned> master_index_;
  unsigned nnindeps_;
  std::vector<GiNaC::ex> symbols_;
  std::vector<unsigned> mx_;   // highest power each symbol carries
  bool all_precompiled_ = true;
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
void write_masters_final(const std::string &path,
                         const std::vector<Integral> &all_integrals,
                         const std::vector<unsigned> &masters,
                         const char *tag = kFlowTag) {
  std::vector<Integral> basis;
  basis.reserve(masters.size());
  for (unsigned m : masters)
    basis.push_back(all_integrals[m]);
  if (!write_masters_file(path, basis)) {
    std::cerr << tag << " masters_final: cannot write " << path << "\n";
    return;
  }
  std::cout << "  " << tag << " masters_final: " << basis.size()
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
};

// The selection the two symbolic backends share, so that a difference between them is
// a difference between reconstruction engines and nothing else.
//
// Learn at one deterministic generic sample point (a fixed seed, no clock and no
// entropy, so the same input picks the same equations in every run and in both
// backends), forward/backward eliminate, take the masters as the union of the targets'
// right-hand sides, mark-and-sweep from the targets with the numeric sufficiency repair
// (global_select.h), write masters_final, and print the Learn Summary and Reduction
// Statistics blocks.
//
// `system` is mutated (its tables are reduced for the learn prime) and is meant to be
// moved into the evaluator afterwards; `all_integrals` is templated so this header does
// not have to pull in the sector machinery for one `push_back`.
template <typename Integral>
SymbolicGlobalLearn symbolic_global_learn(
    std::vector<SymEq> &system, const std::vector<unsigned> &target_ids,
    const std::vector<Integral> &all_integrals,
    const std::vector<GiNaC::possymbol> &symbols, std::uint64_t learn_prime,
    const std::string &masters_final_path, const char *tag,
    unsigned narrowing_success, unsigned narrowing_fail,
    unsigned raw_equations_total, std::size_t global_equations_count) {
  SymbolicGlobalLearn out;
  const unsigned nparams = symbols.size();
  const unsigned neqs = system.size();

  // 1) A generic sample point, fixed seed: the learn must be reproducible.
  std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
  std::vector<std::uint64_t> learn_pt(nparams);
  for (unsigned i = 0; i < nparams; ++i)
    learn_pt[i] = rng() % learn_prime;
  GiNaC::exmap learn_subs;
  for (unsigned i = 0; i < nparams; ++i)
    learn_subs[GiNaC::ex(symbols[i])] = GiNaC::numeric(learn_pt[i]);
  const std::vector<std::vector<uint64_t>> learn_powers =
      make_powers(system, learn_pt.data(), nparams, learn_prime);

  // 2) Evaluate the whole system at that point and eliminate it: masters + structure +
  // dependencies.
  std::vector<ResEq> learn_res_eqs;
  std::vector<std::uint32_t> learn_sys_idx;
  {
    std::vector<std::uint32_t> all_idx(neqs);
    for (unsigned i = 0; i < neqs; ++i)
      all_idx[i] = i;
    reduce_sym_system(system, learn_prime);
    if (!eval_sym_eqs_mod(system, all_idx, learn_subs, learn_powers, learn_prime,
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
      eval_sym_eqs_mod(system, idx, learn_subs, learn_powers, learn_prime, res_eqs,
                       sys_idx);
      solve_res_eqs(res_eqs, sys_idx, learn_prime, sub);
    };
    GlobalSelection gsel = select_global_forward(
        eq_index, fwd_deps, learn_solmap, out.dep_targets, out.master_index,
        solve_at_learn);
    if (gsel.fallback) {
      std::cerr << tag << " forward selection did not converge, falling back to the "
                   "legacy closure\n";
    } else {
      use_forward = true;
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

  write_masters_final(masters_final_path, all_integrals, out.masters, tag);

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
