
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

// Repair gain, in percent, above which the legacy closure is kept.
constexpr unsigned kRepairMinGainPct = 105;

constexpr unsigned kRepairAlways = 0;

// Forward-selected equation set with its backward chain.
struct GlobalSelection {
  bool ok = true;
  bool fallback = false;
  bool skipped_repair = false;
  std::vector<uint32_t> needed_eq_idx;
  std::vector<uint32_t> bwd_chain;
  std::size_t n_fwd_rows = 0;
  std::size_t n_legacy_rows = 0;
  unsigned n_repair_rounds = 0;
  std::size_t n_repair_added = 0;
};

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

// Select the global equations by forward closure, repairing missing pivots.
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

  if (legacy_rows.size() * 100 <= out.n_fwd_rows * min_gain_pct) {
    std::unordered_set<uint32_t> chain_set(dep_targets.begin(), dep_targets.end());
    bwd_chain_of(chain_set);
    out.needed_eq_idx = legacy_rows;
    out.skipped_repair = true;
    return out;
  }

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

    std::set<uint32_t> missing;
    for (unsigned w : dep_targets) {
      auto it = sub.solmap.find(w);
      if (it == sub.solmap.end()) {
        missing.insert(w);
        continue;
      }
      for (std::size_t k = 1; k < it->second.size(); ++k)
        if (!master_index.count(it->second[k].first))
          missing.insert(it->second[k].first);
    }

    if (missing.empty()) {

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
  out.fallback = true;
  return out;
}
