// The evaluation half of the symbolic global stage, shared by both reconstruction
// backends: the coefficient tables reduced per prime, and one sample point turned into
// the flattened target x master coefficient matrix.  The two backends must differ only
// in their reconstruction engine and sample-point policy, so this is the one place a
// point is evaluated.
//
// Header-only so tests can drive the real evaluator.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ginac/ginac.h>

#include "ffred_solver.h"
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
    bool sized = false;
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
    const unsigned nvars = symbols_.size();
    if (!scratch.sized) {
      scratch.powers.assign(nvars, std::vector<uint64_t>(1, 1));
      for (unsigned i = 0; i < nvars; ++i)
        scratch.powers[i].resize(mx_[i] + 1);
      scratch.sized = true;
    }
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

}  // namespace secred
