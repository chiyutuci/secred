// Bridge between the per-sector stage and the ffred elimination engine: learn =
// forward_eliminate + backward_eliminate as in globalreduce.cpp, selection =
// select_global_forward (forward closure + numeric sufficiency repair) -- the same
// engine the global stage uses.
//
// Column -> id map: seeds col = nvars-1-weight -> id = nsubint + weight, subsector
// col = nvars+j -> id = nsubint-1-j, so in both cases a larger id is the deeper,
// more complex integral, matching the global stage (Eq sorted by descending id).
// The system is homogeneous, so it maps straight onto ResEq with no constant
// column, and alg.c already holds residues modulo the learn prime.

#pragma once

#include <algorithm>
#include <cstdint>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ffred_solver.h"

namespace secred {

// Sparse row (column numbers + coefficients already reduced mod the learn prime) ->
// ResEq: id = ncols-1-col, descending, duplicate columns merged mod p, zeros
// dropped -- the same invariant forward_eliminate maintains.
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
  // Merging can still leave zeros (same id cancelling); ResEq forbids them.
  out.erase(std::remove_if(out.begin(), out.end(),
                           [](const auto &t) { return t.second == 0; }),
            out.end());
  return out;
}

// Solve on the idx subset of eqs (shared by learn and the repair loop).  Identical to
// solve_subset except the input is already residues, saving an Eq -> ResEq conversion.
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

// Masters (descending id) = union over reducible targets of RHS(solmap[t]), plus every
// target this stage cannot reduce.  Same formula as the global stage: the reduction is
// already fully recursive, so the right-hand side is the master set.  Irreducible
// targets must be included -- they are the legacy "needed variables of non-pivot
// columns" and travel up through ctx.uptargets; dropping one loses the integral.
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
  return std::vector<unsigned>(m.rbegin(), m.rend()); // descending id = ascending column
}

// Legacy closure (the counterpart of fflow mark_and_sweep_eqs): BFS from the seeds along
// forward and backward dependency edges, returning ascending raw equation indices.  Used
// as the fallback when the selection stalls; nearly free, since learn already computed
// bwd_deps.
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

} // namespace secred
