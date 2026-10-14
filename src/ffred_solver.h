#pragma once

#include <cstdint>
#include <memory>
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

// The same inverse without a single division: binary extended Euclid over 64-bit words,
// carrying u = x1*a (mod p) and v = x2*a (mod p) alongside the remainders and halving both
// by the same rule, so the survivor's coefficient is the inverse.  a = 0 is not invertible
// and returns 0, exactly as mod_inverse_u64 does.
//
// The sample-point path normalises one pivot per row -- ~547k inverses per point at F234's
// size -- and that is where the 128-bit Euclid loop's ~50 iterations of __udivti3 put most
// of the forward elimination's constant factor.  Every intermediate here stays below
// 2p < 2^64 because p < 2^63, which is the same bound the rest of this file assumes.
// Preconditions: p odd and 0 <= a < p.
uint64_t mod_inverse_u64_fast(uint64_t a, uint64_t p);

// (num/den) mod p with den > 0, using 128-bit machine arithmetic.
uint64_t rat_mod_u64(int64_t num, int64_t den, uint64_t p);

// Reduce an Eq (num/den) to a ResEq (residues mod p), dropping zero coefficients.
//
// Every solver here -- this one and the planned sample-point path -- assumes p < 2^63:
// rat_mod_u64 reads p as an int64, and the merges add two residues before one conditional
// subtraction, which needs 2p < 2^64.  Every prime any backend produces is far below that
// (fflow's BIG_UINT_PRIMES top out at 9223372036854775783 ~ 2^63 - 25).
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

// ---------------------------------------------------------------------------
// The sample-point path: one plan, one flat store, Shoup products
// ---------------------------------------------------------------------------
//
// The learn, the numeric sufficiency repair and the sector stage solve a subset once, so
// forward_eliminate's unordered_map<uint32_t, ResEq> is fine for them.  Reconstruction
// solves the same subset at every sample point -- 547k pivots and ~20M terms per point at
// F234's size, 1149 points -- and there the map costs one heap row and one hash entry per
// pivot per point, plus a hash lookup per substituted term.  The three pieces below put
// that path on a flat store, a precomputed order and a division-free product.  They
// compute what forward_eliminate + backward_eliminate compute (tests/test_ffred.cpp pins
// the two against each other row by row), only faster.

// floor(c * 2^64 / p), the precomputed factor of shoup_mul: the merge multiplies one
// coefficient c into a whole right-hand side, so the division that `c*x % p` needs is
// done once per substitution instead of once per merged term.
uint64_t shoup_precomp(uint64_t c, uint64_t p);

// c*x mod p with the precomputed shoup_precomp(c, p), one 128-bit product and one
// conditional subtraction instead of a 128-bit division.  Valid for any x < 2^64 when
// c < p < 2^63, which every prime the reconstruction draws a sample point at satisfies.
uint64_t shoup_mul(uint64_t x, uint64_t c, uint64_t cp, uint64_t p);

// The forward-elimination order of one selected subset: the row order, structure only.
//
// It is what makes a point's elimination replayable: one order, shared by every prime and
// every point, so a point reproduces the pivot structure the learn recorded.
// forward_eliminate sorts by the same key (front id, row length, the id list, sys_idx) and
// never looks at a coefficient -- a residue in the key would order two structurally equal
// rows differently at different primes.  Building it once also removes the per-point
// O(n log n) row comparison: 547k rows compared ~20 times each against ~70 terms at
// F234's size.
//
// `terms` / `eq_begin` are a SymCoeffPool's term table (eq e owns
// [eq_begin[e], eq_begin[e+1]) with terms[k].first the integral id), `sys_idx[e]` the row's
// index in the reduced system -- the ordering's last tie-break, distinct per row.
//
// The plan also carries the structure it was built from (`front`, `len`, `sys_idx`), the
// pool's rows as they stand before any prime is chosen.  A sample point does not always
// hand the elimination those rows: a coefficient that vanishes there drops out of its row,
// which shortens the row and can move its front, and the reference -- which orders the rows
// it was handed -- then sorts that row somewhere else.  Keeping the structure lets the
// planned path check the rows it is given against the plan and fall back to the reference's
// own order when they differ (see forward_eliminate_planned).
struct FwdPlan {
  std::vector<uint32_t> order;   // row indices, ascending front id
  std::vector<uint32_t> front;   // per row: the pool row's leading id (0 if it has none)
  std::vector<uint32_t> len;     // per row: the pool row's term count
  std::vector<uint32_t> sys_idx; // per row: its index in the reduced system
};

void build_fwd_plan(const std::vector<std::pair<uint32_t, uint32_t>> &terms,
                    const std::vector<uint32_t> &eq_begin,
                    const std::vector<uint32_t> &sys_idx, FwdPlan &plan);

// The order forward_eliminate derives from a point's rows: the same key as the plan's, but
// read off the evaluated rows instead of the pool's structure.  eqs and sys_idx are
// parallel; rows that evaluated to zero sort last and are skipped by the elimination.
void fwd_order_from_rows(const std::vector<ResEq> &eqs,
                         const std::vector<uint32_t> &sys_idx,
                         std::vector<uint32_t> &order);

// A system's rows as the reconstruction holds them: `terms[k] = (integral id, coefficient
// slot)` and row e owns [eq_begin[e], eq_begin[e+1]); `values[slot]` is that coefficient's
// residue at one point.  This is the coefficient pool's own term table, which every worker
// shares, so a point walks the rows instead of materialising the system's numeric form:
// one slot array per thread instead of a heap row per equation (at F234's size, 8 B per
// term instead of 24 B of vector header plus 16 B per term, 574k headers included).
//
// Terms descend by integral id, as the spill reader wrote them, which is what the
// reference's merge assumes.  A term whose slot value is 0 at this point is dropped, the
// way reduce_mod drops a coefficient that vanishes mod p.
struct SymRowTable {
  const std::pair<uint32_t, uint32_t> *terms = nullptr; // (integral id, coefficient slot)
  const uint32_t *eq_begin = nullptr;                   // n_rows + 1 row boundaries
  std::size_t n_rows = 0;
  std::size_t begin(std::size_t e) const { return eq_begin[e]; }
  std::size_t row_size(std::size_t e) const { return eq_begin[e + 1] - eq_begin[e]; }
};

// One row of the planned elimination: ids and their values in two parallel arrays.
//
// Deliberately not a `std::vector` pair.  A point's merge runs ~19M times at F234's size
// and moves ~663M terms, and `push_back` pays a capacity test and a size rewrite on every
// one of them -- four memory operations to store a term that needs two.  Here the length
// and the capacity are separate, growth doubles and copies only the live terms (never
// value-initialising a tail nothing reads), and the merge writes through `ids`/`vals`
// directly and sets `n` once at the end.
//
// `ensure` preserves the first `n` terms, so a row can also be built term by term.
struct Row {
  std::unique_ptr<uint32_t[]> ids;
  std::unique_ptr<uint64_t[]> vals;
  std::size_t n = 0;   // terms held
  std::size_t cap = 0; // terms the arrays can hold

  void ensure(std::size_t need) {
    if (cap >= need)
      return;
    std::size_t nc = cap ? cap : 8;
    while (nc < need)
      nc *= 2;
    // `new T[n]` leaves the elements uninitialised for a trivial T, which is the point:
    // the merge overwrites all of [0, n) before anything reads it.
    std::unique_ptr<uint32_t[]> ni(new uint32_t[nc]);
    std::unique_ptr<uint64_t[]> nv(new uint64_t[nc]);
    for (std::size_t k = 0; k < n; ++k) {
      ni[k] = ids[k];
      nv[k] = vals[k];
    }
    cap = nc;
    ids = std::move(ni);
    vals = std::move(nv);
  }
  void clear() { n = 0; }
  bool empty() const { return n == 0; }
  void push(uint32_t id, uint64_t v) {
    ensure(n + 1);
    ids[n] = id;
    vals[n] = v;
    ++n;
  }
  void swap(Row &o) {
    ids.swap(o.ids);
    vals.swap(o.vals);
    std::swap(n, o.n);
    std::swap(cap, o.cap);
  }
};

// One point's solutions: rows in two flat arenas, membership in a bitset.
//
// Row r is ids[begin[r] .. begin[r]+len[r]) with ids[begin[r]] its pivot, descending, the
// pivot's coefficient normalised to -1: the same rows forward_eliminate stores, without a
// heap row per pivot.  "Is this id derived?" is one bit of a table covering the id space
// (160 kB at F234's 1.26M ids, so it stays in cache where an unordered_map's buckets do
// not) and the row index is only read once a bit says there is one.
struct FwdStore {
  std::vector<uint64_t> has;    // id -> bit
  std::vector<uint32_t> row_of; // id -> row (read only where the bit is set)
  std::vector<uint32_t> begin, len;
  std::vector<uint32_t> ids;
  std::vector<uint64_t> vals;
  // The order this point's elimination ran in: the plan's own vector where the point's
  // rows are the plan's rows, otherwise the one rebuilt from the evaluated rows
  // (`order_own`).  A pointer, because the plan's order is one vector shared by every prime
  // and every point and the copy would be 547k entries -- 2.3 MB -- per point.
  const std::vector<uint32_t> *order_ptr = nullptr;
  std::vector<uint32_t> order_own;
  const std::vector<uint32_t> &order() const { return *order_ptr; }
  // Reused merge buffers: one pending row and its merged replacement (`Row`, which is
  // where the parallel arrays and their reasons live).  Kept here so a point allocates
  // nothing after the first row.
  Row pend, merged;
  // Reused by the fallback order (a point where a coefficient vanished): each row's
  // evaluated length and leading id, so the order reads them instead of walking the rows.
  // Size of the id space, not of one row: kept here for the same reason as the buffers.
  std::vector<uint32_t> eval_len, eval_front;

  // Cover ids [0, n_ids) and forget every row.  `row_of` is not cleared: its entries are
  // only read where `has` says a row exists, and `has` is cleared here.
  void reset(std::size_t n_ids);
  bool has_row(uint32_t id) const { return (has[id >> 6] >> (id & 63)) & 1u; }
  void mark(uint32_t id, uint32_t row) {
    has[id >> 6] |= uint64_t(1) << (id & 63);
    row_of[id] = row;
  }
  std::size_t rows() const { return begin.size(); }
  const uint32_t *row_ids(uint32_t id) const { return ids.data() + begin[row_of[id]]; }
  const uint64_t *row_vals(uint32_t id) const { return vals.data() + begin[row_of[id]]; }
  uint32_t row_size(uint32_t id) const { return len[row_of[id]]; }
};

// forward_eliminate over `rows` at one point -- the plan's rows with this point's
// vanishing coefficients dropped, read straight out of the shared table -- into `store`:
// same pivots, same rows.  `values` holds the point's residue for every coefficient slot
// (`n_values` of them, the pool's distinct count); a zero value drops its term.
//
// The plan is checked first: a coefficient that vanishes here shortens its row and can
// move its front, and the reference -- which orders the rows it was handed -- would then
// sort that row somewhere else, so the order is rebuilt from the evaluated rows
// (`store.order()`).  Every slot occurs in some row, so "no value is zero" answers what a
// per-row length-and-front check would, in one scan of the slot array.
void forward_eliminate_planned(const SymRowTable &rows, const uint64_t *values,
                               std::size_t n_values, const FwdPlan &plan, uint64_t p,
                               FwdStore &store);

// backward_eliminate over `store`, restricted to `only_keys` (ascending; the lazy chain of
// the reconstruction).  A rewritten row is appended as a new row and the key's row index
// repointed: rows grow and shrink here, and the old row may still be read... it is not,
// because keys ascend and every substituted id is smaller, so no reader is left behind.
void backward_eliminate_planned(FwdStore &store, uint64_t p,
                                const std::vector<uint32_t> &only_keys);

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
