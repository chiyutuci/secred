#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace secred {

// Coefficient as num/den (den > 0), parsed from the spill file.
struct Rat64 {
  int64_t num;
  int64_t den;
};

// An equation: (id, coefficient) in descending id.  ids are global dense ids
// (0 = simplest, N-1 = most complex); front() is the largest id.
using Eq = std::vector<std::pair<uint32_t, Rat64>>;

// The residue equation at one prime: (id, residue mod p), descending id.
using ResEq = std::vector<std::pair<uint32_t, uint64_t>>;

// Modular inverse of a modulo the prime p (gcd(a,p)=1), via __int128 extended Euclid.
uint64_t mod_inverse_u64(uint64_t a, uint64_t p);

// (num/den) mod p with den > 0, using 128-bit machine arithmetic.
uint64_t rat_mod_u64(int64_t num, int64_t den, uint64_t p);

// Reduce an Eq (num/den) to a ResEq (residues mod p), dropping zero coefficients.
void reduce_mod(const Eq &eq, uint64_t p, ResEq &out);

// Forward elimination (the eager form of ffred's SystemOfEqs::solve): equations are
// ordered by ascending front(), tied by sys_idx so the order is prime-independent;
// each equation repeatedly substitutes its highest reducible term (one solved in
// solmap), then normalises the remaining highest term as pivot (coefficient -> -1)
// and registers it.  eqs and sys_idx are parallel; eq_index / fwd_deps may be
// nullptr and otherwise record, per pivot, the registering equation and the pivots
// it substituted.
void forward_eliminate(std::vector<ResEq> &eqs,
                       const std::vector<uint32_t> &sys_idx, uint64_t p,
                       std::unordered_map<uint32_t, ResEq> &solmap,
                       std::unordered_map<uint32_t, uint32_t> *eq_index,
                       std::unordered_map<uint32_t, std::vector<uint32_t>> *fwd_deps);

// Back substitution (ffred's backward()): derived integrals may be solved only after
// earlier reductions, leaving reducible terms on their right-hand sides; substituting
// again in ascending id pushes every right-hand side onto masters.  bwd_deps may be
// nullptr.  A non-null only_keys restricts the work to those pivots (ascending order),
// which reconstruction uses to rebuild the lazy chain of the targets alone; keys absent
// from solmap are ignored, since a narrowed selection can leave bwd_chain dangling.
void backward_eliminate(std::unordered_map<uint32_t, ResEq> &solmap, uint64_t p,
                        std::unordered_map<uint32_t, std::vector<uint32_t>> *bwd_deps,
                        const std::vector<uint32_t> *only_keys = nullptr);

// Everything one subset solve produces: the solution map, the raw equation index of
// each pivot, and the forward / backward dependencies.  Full-system learn, the numeric
// sufficiency repair and the per-prime reconstruction all share this solve, which keeps
// the equations the selection picks and the equations reconstruction uses the same set.
struct SubSolve {
  std::unordered_map<uint32_t, ResEq> solmap;
  std::unordered_map<uint32_t, uint32_t> eq_index;
  std::unordered_map<uint32_t, std::vector<uint32_t>> fwd_deps, bwd_deps;
};

// reduce_mod + forward_eliminate + backward_eliminate over the eq_idx subset of system.
// want_fwd / want_bwd control whether eq_index+fwd_deps / bwd_deps are recorded;
// only_keys follows backward_eliminate (ascending; nullptr = every reduction).
void solve_subset(const std::vector<Eq> &system,
                  const std::vector<uint32_t> &eq_idx, uint64_t p,
                  SubSolve &out, bool want_fwd, bool want_bwd,
                  const std::vector<uint32_t> *only_keys = nullptr);

} // namespace secred
