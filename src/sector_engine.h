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
#include "sector.h"  // RawIntegral

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

// Kira 3's mandatory list (paper §3.2), applied to one sector's spill: the rows a
// target-driven selection drops because the integral they carry is not on any target's
// reduction path.  Subsector columns are eliminated before any seed column (every
// subsector id is lower), so a subsector integral still present in a raw row is one the
// sector never solved: either a master of this stage (the check passes it) or another
// sector's master, which the global stage can eliminate only if the rows carrying it are
// spilled.  Without those rows the spill is sufficient for the sector's own targets and
// incomplete for the global basis: a supersector's rows relating a subsector master are
// pruned here, and the master survives the global learn (44 masters where Kira has 43 on
// the Jpsi family).  Returned by index into eqs (the raw generated rows).
inline std::unordered_set<uint32_t> unreducible_subsector_rows(
    const std::vector<ResEq> &eqs, unsigned nsubint,
    const std::unordered_map<unsigned, unsigned> &master_index,
    const std::unordered_map<uint32_t, ResEq> &full_solmap) {
  std::unordered_set<uint32_t> rows;
  for (uint32_t r = 0; r < eqs.size(); ++r) {
    for (const auto &term : eqs[r]) {
      const unsigned id = term.first;
      if (id >= nsubint) // a seed of this sector, not a subsector column
        continue;
      if (master_index.count(id) || full_solmap.count(id))
        continue;
      rows.insert(r);
      break;
    }
  }
  return rows;
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

// Stage B's seed set: the sector's own masters, in the order `needed_indep` fixes.
// Only a relation carrying a master can cut a redundant master in the global stage,
// so those are the seeds the symmetry expansion walks -- the admitted-seed set the
// IBP system is built on would expand every seed a second time for relations the
// reduction never reads.  A needed independent variable at or above `nvars` is a
// subsector integral, not a seed of this sector: it travels up through
// ctx.uptargets, and it carries no relation here.  A master is `needed_indep[i] <
// nvars` and lives in `seeds` at rank `nvars - 1 - id`.
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
