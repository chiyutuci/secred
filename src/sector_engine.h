
#pragma once

#include <algorithm>
#include <cstdint>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ffred_solver.h"
#include "sector.h"

namespace secred {

// Convert one solver row into a residue equation over sub-integrals.
inline ResEq res_eq_from_row(const unsigned *cols, const uint64_t *vals,
                             unsigned size, unsigned ncols, uint64_t p) {
  ResEq sorted;
  sorted.reserve(size);
  for (unsigned k = 0; k < size; ++k)
    sorted.emplace_back(ncols - 1 - cols[k], vals[k] % p);
  std::sort(sorted.begin(), sorted.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  ResEq out;
  out.reserve(sorted.size());
  for (const auto &t : sorted) {
    if (!out.empty() && out.back().first == t.first)
      out.back().second = (out.back().second + t.second) % p;
    else
      out.push_back(t);
  }

  out.erase(std::remove_if(out.begin(), out.end(),
                           [](const auto &t) { return t.second == 0; }),
            out.end());
  return out;
}

// Rows that mention an identifier neither solved nor a known master.
inline std::unordered_set<uint32_t> unreducible_subsector_rows(
    const std::vector<ResEq> &eqs, unsigned nsubint,
    const std::unordered_map<unsigned, unsigned> &master_index,
    const std::unordered_map<uint32_t, ResEq> &full_solmap) {
  std::unordered_set<uint32_t> rows;
  for (uint32_t r = 0; r < eqs.size(); ++r) {
    for (const auto &term : eqs[r]) {
      const unsigned id = term.first;
      if (id >= nsubint)
        continue;
      if (master_index.count(id) || full_solmap.count(id))
        continue;
      rows.insert(r);
      break;
    }
  }
  return rows;
}

// Eliminate a subset of residue equations, optionally recording dependencies.
inline void solve_res_subset(const std::vector<ResEq> &eqs,
                             const std::vector<uint32_t> &idx, uint64_t p,
                             SubSolve &out, bool want_fwd, bool want_bwd) {
  out.solmap.clear();
  out.eq_index.clear();
  out.fwd_deps.clear();
  out.bwd_deps.clear();

  std::vector<ResEq> sub;
  std::vector<uint32_t> sys_idx;
  sub.reserve(idx.size());
  sys_idx.reserve(idx.size());
  for (uint32_t i : idx)
    if (!eqs[i].empty()) {
      sub.push_back(eqs[i]);
      sys_idx.push_back(i);
    }

  forward_eliminate(sub, sys_idx, p, out.solmap,
                    want_fwd ? &out.eq_index : nullptr,
                    want_fwd ? &out.fwd_deps : nullptr);
  backward_eliminate(out.solmap, p, want_bwd ? &out.bwd_deps : nullptr);
}

// Masters of a learned sector: the non-targets its solutions depend on.
inline std::vector<unsigned> sector_masters(const SubSolve &learn,
                                            const std::vector<unsigned> &target_ids) {
  std::set<unsigned> m;
  for (unsigned id : target_ids) {
    auto it = learn.solmap.find(id);
    if (it == learn.solmap.end()) {
      m.insert(id);
      continue;
    }
    for (std::size_t k = 1; k < it->second.size(); ++k)
      m.insert(it->second[k].first);
  }
  return std::vector<unsigned>(m.rbegin(), m.rend());
}

// Seed integrals named by a symmetry relation.
inline std::vector<RawIntegral> symmetry_seed_masters(
    const std::vector<std::size_t> &needed_indep,
    const std::vector<RawIntegral> &seeds, unsigned nvars) {
  std::vector<RawIntegral> out;
  out.reserve(needed_indep.size());
  for (std::size_t id : needed_indep)
    if (id < nvars)
      out.push_back(seeds[nvars - 1 - id]);
  return out;
}

// Equation indices reachable from the seeds through recorded dependencies.
inline std::vector<uint32_t> legacy_closure(
    const std::vector<unsigned> &seeds,
    const std::unordered_map<uint32_t, std::vector<uint32_t>> &fwd_deps,
    const std::unordered_map<uint32_t, std::vector<uint32_t>> &bwd_deps,
    const std::unordered_map<uint32_t, uint32_t> &eq_index) {
  std::unordered_set<uint32_t> s(seeds.begin(), seeds.end());
  std::vector<uint32_t> stack(seeds.begin(), seeds.end());
  while (!stack.empty()) {
    uint32_t w = stack.back();
    stack.pop_back();
    for (const auto *deps : {&fwd_deps, &bwd_deps}) {
      auto it = deps->find(w);
      if (it == deps->end())
        continue;
      for (uint32_t u : it->second)
        if (s.insert(u).second)
          stack.push_back(u);
    }
  }

  std::vector<uint32_t> idx;
  idx.reserve(s.size());
  for (uint32_t w : s) {
    auto it = eq_index.find(w);
    if (it != eq_index.end())
      idx.push_back(it->second);
  }
  std::sort(idx.begin(), idx.end());
  idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
  return idx;
}

}
