// Equation selection for the global stage: forward closure plus numeric
// sufficiency repair.  The closure follows forward-elimination edges only; the
// repair then re-solves the selected subsystem numerically and feeds every
// non-master integral on a target's right-hand side back into the seed set until a
// fixed point, which recovers the equations a pure closure misses through hidden
// zeros.
//
// Only the equation set and the lazy back-substitution chain are decided here; the
// masters still come from the caller's unchanged full-system learn, so the master
// basis does not depend on the selection.
//
// Header-only so tests can drive the real repair loop: both paths in
// globalreduce.cpp call it, differing only in the re-solve callback they inject.

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ffred_solver.h"

struct GlobalSelection {
  bool ok = true;                      // repair converged
  bool fallback = false;               // stalled or malformed -> caller uses the legacy closure
  std::vector<uint32_t> needed_eq_idx; // ascending: raw equations sent to the reconstruction
  std::vector<uint32_t> bwd_chain;     // lazy back-substitution chain (consistent with needed_eq_idx)
  std::size_t n_fwd_only = 0;          // pivots in the pure forward closure (diagnostic)
  unsigned n_repair_rounds = 0;        // repair rounds
  std::size_t n_repair_added = 0;      // integrals the repair added back
};

// eq_index / fwd_deps come from the full-system learn (pivot -> raw equation that
// registered it / forward dependencies).  full_pivots is that learn's solmap, used
// for its key set; master_index maps a master to its output column; dep_targets are
// the reducible targets.  solve re-solves an equation subset at the learn prime /
// sample point (numeric path: solve_subset; symbolic path: evaluate, then eliminate).
inline GlobalSelection select_global_forward(
    const std::unordered_map<uint32_t, uint32_t> &eq_index,
    const std::unordered_map<uint32_t, std::vector<uint32_t>> &fwd_deps,
    const std::unordered_map<uint32_t, secred::ResEq> &full_pivots,
    const std::vector<unsigned> &dep_targets,
    const std::unordered_map<unsigned, unsigned> &master_index,
    const std::function<void(const std::vector<uint32_t> &,
                             secred::SubSolve &)> &solve,
    unsigned max_repair_rounds = 64) {
  GlobalSelection out;

  // Closure along forward elimination edges only (same graph as legacy sweep(true),
  // minus the backward edges).
  auto mark = [&](const std::unordered_set<uint32_t> &s) {
    std::unordered_set<uint32_t> needed(s.begin(), s.end());
    std::vector<uint32_t> stack(s.begin(), s.end());
    while (!stack.empty()) {
      uint32_t w = stack.back();
      stack.pop_back();
      auto fit = fwd_deps.find(w);
      if (fit != fwd_deps.end())
        for (uint32_t u : fit->second)
          if (needed.insert(u).second)
            stack.push_back(u);
    }
    return needed;
  };
  auto eq_idx_of = [&](const std::unordered_set<uint32_t> &needed) {
    std::vector<uint32_t> idx;
    idx.reserve(needed.size());
    for (uint32_t w : needed) {
      auto it = eq_index.find(w);
      if (it != eq_index.end())
        idx.push_back(it->second);
    }
    std::sort(idx.begin(), idx.end());
    idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
    return idx;
  };

  std::unordered_set<uint32_t> seeds(dep_targets.begin(), dep_targets.end());
  std::unordered_set<uint32_t> needed = mark(seeds);
  out.n_fwd_only = needed.size();
  out.needed_eq_idx = eq_idx_of(needed);

  for (unsigned round = 0; round < max_repair_rounds; ++round) {
    secred::SubSolve sub;
    solve(out.needed_eq_idx, sub);

    // Sufficiency test: every target must be solved, and its reduction may only
    // have masters on the right-hand side.  When that holds the sub-solve
    // reproduces the full reduction exactly (coefficients included), because the
    // difference of the two lies in the full row space while its right-hand side
    // spans master columns only, which are independent modulo that space.
    // Accumulated in a std::set so the selection does not depend on iteration order.
    std::set<uint32_t> missing;
    for (unsigned w : dep_targets) {
      auto it = sub.solmap.find(w);
      if (it == sub.solmap.end()) {
        missing.insert(w); // target not solved by the subset
        continue;
      }
      for (std::size_t k = 1; k < it->second.size(); ++k)
        if (!master_index.count(it->second[k].first))
          missing.insert(it->second[k].first);
    }

    if (missing.empty()) {
      // Converged: rebuild the chain from the selected subsystem, since the
      // full-system graph legacy uses is stale for a shrunken equation set.
      std::unordered_set<uint32_t> chain_set(dep_targets.begin(),
                                             dep_targets.end());
      std::vector<uint32_t> stack(dep_targets.begin(), dep_targets.end());
      while (!stack.empty()) {
        uint32_t w = stack.back();
        stack.pop_back();
        auto bit = sub.bwd_deps.find(w);
        if (bit != sub.bwd_deps.end())
          for (uint32_t u : bit->second)
            if (chain_set.insert(u).second)
              stack.push_back(u);
      }
      out.bwd_chain.assign(chain_set.begin(), chain_set.end());
      std::sort(out.bwd_chain.begin(), out.bwd_chain.end());
      return out;
    }

    // Add the missing integrals back as seeds; each enters the subsystem through
    // the equation that registers it.  An integral that is not a pivot of any
    // equation adds no equation, so this is not treated as fatal: only a round
    // that grows nothing at all counts as stuck.
    bool grew = false;
    std::size_t added = 0;
    for (uint32_t c : missing)
      if (full_pivots.count(c) && seeds.insert(c).second) {
        grew = true;
        ++added;
      }
    out.n_repair_added += added;
    if (!grew) {
      const uint32_t first = *missing.begin();
      std::cerr << "finiteflow global reduction: forward selection cannot be "
                   "repaired: target reductions still contain "
                << missing.size() << " non-master integral(s), e.g. " << first
                << (full_pivots.count(first)
                        ? " (already selected, but not solved by the subset)"
                        : " (not reducible in the full system)")
                << "; falling back to the legacy closure\n";
      out.ok = false;
      out.fallback = true;
      return out;
    }
    needed = mark(seeds);
    out.needed_eq_idx = eq_idx_of(needed);
    out.n_repair_rounds = round + 1;
  }

  out.ok = false;
  out.fallback = true; // repair round cap reached
  return out;
}
