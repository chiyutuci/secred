// Equation selection for the global stage: forward closure plus numeric
// sufficiency repair.  The closure follows forward-elimination edges only; the
// repair then re-solves the selected subsystem numerically and feeds every
// non-master integral on a target's right-hand side back into the seed set until a
// fixed point, which recovers the equations a pure closure misses through hidden
// zeros.
//
// The repair costs a full sub-solve per round, and the row set it can reach is bounded
// below by the forward closure it starts from and above by the legacy closure it is
// chasing: the whole prize is |legacy| - |fwd|.  Both closures are free to compute, the
// repair is not, so the repair only runs when that prize is worth having -- see
// kRepairMinGainPct.
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
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ffred_solver.h"

// The repair runs only when the legacy closure exceeds the forward one by more than this
// many percent; below it the legacy closure is returned as is.  Both closures solve the
// same problem -- the legacy one is provably sufficient, the repair only shrinks it -- so
// the trade is a per-round sub-solve (serial, one elimination over the selected rows)
// against a fraction of the reconstruction's per-point elimination (the same elimination,
// but once per probe).  Calibration (F234, bench/global_perf/select_bench.cpp): the legacy
// closure is 1.78 % larger than the forward one and the repair lands 0.96 % below it, so
// the repair spent 21.7 s of serial elimination to remove 5,259 rows from a 546,164-row
// selection -- 0.96 % of the row set, about 0.8 s of the reconstruction's 85 s at 64
// threads.  The ratio also has the property that a system where the repair cannot win now
// (small |legacy| - |fwd|) is one where it cannot win later either: it is a property of the
// family, not of the sample points.
constexpr unsigned kRepairMinGainPct = 105; // percent of the forward closure
// "Repair whatever the closures say": the value the per-sector stage passes, where the
// premise above does not hold.  A sector's sub-solve is milliseconds over a few hundred
// rows, and what the repair buys there is a *smaller spill* -- which the global stage, two
// orders of magnitude larger, then pays for -- so the sector stage always repairs and the
// guard stays where it was measured.
constexpr unsigned kRepairAlways = 0;

struct GlobalSelection {
  bool ok = true;                      // repair converged
  bool fallback = false;               // stalled or malformed -> caller uses the legacy closure
  bool skipped_repair = false;         // the legacy closure was taken: not worth repairing
  std::vector<uint32_t> needed_eq_idx; // ascending: raw equations sent to the reconstruction
  std::vector<uint32_t> bwd_chain;     // lazy back-substitution chain (consistent with needed_eq_idx)
  std::size_t n_fwd_rows = 0;          // equations in the pure forward closure (diagnostic)
  std::size_t n_legacy_rows = 0;       // equations in the forward-and-backward closure
  unsigned n_repair_rounds = 0;        // repair rounds
  std::size_t n_repair_added = 0;      // integrals the repair added back
};

// The guard's decision, printed whichever branch it took: a repair that succeeds says
// nothing, which in a run log is indistinguishable from a binary that predates the guard.
// The row counts are the same comparison the guard makes, so a run states its own premise.
// `prefix` is the caller's log tag.
inline void log_selection(const std::string &prefix, const GlobalSelection &sel) {
  const double over = sel.n_fwd_rows == 0
                          ? 0.0
                          : 100.0 * (static_cast<double>(sel.n_legacy_rows) -
                                     static_cast<double>(sel.n_fwd_rows)) /
                                static_cast<double>(sel.n_fwd_rows);
  std::cout << prefix << "selection: ";
  if (sel.skipped_repair)
    std::cout << "legacy closure without repair (" << sel.n_legacy_rows
              << " equations, " << over << " % over the forward closure)\n";
  else
    std::cout << "repaired (" << sel.n_repair_rounds << " round(s), +"
              << sel.n_repair_added << " integral(s); forward closure " << sel.n_fwd_rows
              << " equations, legacy closure " << sel.n_legacy_rows << " (" << over
              << " %), " << sel.needed_eq_idx.size() << " selected)\n";
}

// eq_index / fwd_deps / bwd_deps come from the full-system learn (pivot -> raw equation
// that registered it / forward dependencies / backward dependencies).  full_pivots is
// that learn's solmap, used for its key set; master_index maps a master to its output
// column; dep_targets are the reducible targets.  solve re-solves an equation subset at
// the learn prime / sample point (numeric path: solve_subset; symbolic path: evaluate,
// then eliminate).  `min_gain_pct` is the kRepairMinGainPct decision, in percent of the
// forward closure's row count, exposed so a test or a bench can drive the repair itself:
// 100 repairs whenever the legacy closure is strictly larger, 0 always repairs.
inline GlobalSelection select_global_forward(
    const std::unordered_map<uint32_t, uint32_t> &eq_index,
    const std::unordered_map<uint32_t, std::vector<uint32_t>> &fwd_deps,
    const std::unordered_map<uint32_t, std::vector<uint32_t>> &bwd_deps,
    const std::unordered_map<uint32_t, secred::ResEq> &full_pivots,
    const std::vector<unsigned> &dep_targets,
    const std::unordered_map<unsigned, unsigned> &master_index,
    const std::function<void(const std::vector<uint32_t> &,
                             secred::SubSolve &)> &solve,
    unsigned max_repair_rounds = 64, unsigned min_gain_pct = kRepairMinGainPct) {
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
  // The backward-only closure of the targets: the lazy chain the reconstruction
  // back-substitutes along, in both selections (a shrunken subsystem's chain is a subset
  // of this one, and on F234 the two came out the same size).
  auto bwd_chain_of = [&](std::unordered_set<uint32_t> &chain_set) {
    std::vector<uint32_t> stack(dep_targets.begin(), dep_targets.end());
    while (!stack.empty()) {
      const uint32_t w = stack.back();
      stack.pop_back();
      auto bit = bwd_deps.find(w);
      if (bit == bwd_deps.end())
        continue;
      for (uint32_t u : bit->second)
        if (chain_set.insert(u).second)
          stack.push_back(u);
    }
    out.bwd_chain.assign(chain_set.begin(), chain_set.end());
    std::sort(out.bwd_chain.begin(), out.bwd_chain.end());
  };

  std::unordered_set<uint32_t> seeds(dep_targets.begin(), dep_targets.end());
  std::unordered_set<uint32_t> needed = mark(seeds);
  out.needed_eq_idx = eq_idx_of(needed);
  out.n_fwd_rows = out.needed_eq_idx.size();

  // The legacy closure -- forward union backward, the provably sufficient one -- and the
  // decision whether repairing on top of the forward closure is worth a sub-solve.  Both
  // sides of the comparison are row counts: rows are what the reconstruction re-eliminates
  // per probe, and the integral counts differ from them by whatever pivots share a row.
  std::unordered_set<uint32_t> legacy(seeds);
  {
    std::vector<uint32_t> stack(seeds.begin(), seeds.end());
    while (!stack.empty()) {
      const uint32_t w = stack.back();
      stack.pop_back();
      for (const auto *deps : {&fwd_deps, &bwd_deps}) {
        auto it = deps->find(w);
        if (it == deps->end())
          continue;
        for (uint32_t u : it->second)
          if (legacy.insert(u).second)
            stack.push_back(u);
      }
    }
  }
  const std::vector<uint32_t> legacy_rows = eq_idx_of(legacy);
  out.n_legacy_rows = legacy_rows.size();
  // The two closures coincide whenever no backward edge leaves the forward set, and then
  // the prize is exactly zero -- the repair cannot go below the closure it starts from --
  // so equality is skipped at every threshold.
  if (legacy_rows.size() * 100 <= out.n_fwd_rows * min_gain_pct) {
    std::unordered_set<uint32_t> chain_set(dep_targets.begin(), dep_targets.end());
    bwd_chain_of(chain_set);
    out.needed_eq_idx = legacy_rows;
    out.skipped_repair = true;
    return out;
  }

  // The row set the last solve ran on.  A repair round can add an integral without
  // growing the closure (the added pivot's equation was already selected), and re-solving
  // an unchanged row set at the same prime reproduces the same result: the missing set
  // cannot shrink, so the loop would spend every remaining round on identical solves
  // before falling back.  Recognise that stall after the one wasted solve.
  std::vector<uint32_t> solved_idx;

  for (unsigned round = 0; round < max_repair_rounds; ++round) {
    if (round != 0 && out.needed_eq_idx == solved_idx) {
      std::cerr << "finiteflow global reduction: forward selection cannot be "
                   "repaired: the repair added "
                << out.n_repair_added
                << " integral(s) that register no new equation, so re-solving the "
                   "unchanged set would repeat the last result; falling back to the "
                   "legacy closure\n";
      out.ok = false;
      out.fallback = true;
      return out;
    }
    secred::SubSolve sub;
    solve(out.needed_eq_idx, sub);
    solved_idx = out.needed_eq_idx;

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
