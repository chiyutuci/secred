#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace secred {

// A rational coefficient; den == 0 marks an interned big rational.
struct Rat64 {
  int64_t num;
  int64_t den;
};

// Intern a big rational and return its handle.
bool intern_big_rational(const char *b, const char *e, int64_t &handle);

std::string rat_text(const Rat64 &r);

// Numeric equation over rational coefficients.
using Eq = std::vector<std::pair<uint32_t, Rat64>>;

// Equation reduced modulo a prime.
using ResEq = std::vector<std::pair<uint32_t, uint64_t>>;

uint64_t mod_inverse_u64(uint64_t a, uint64_t p);

uint64_t mod_inverse_u64_fast(uint64_t a, uint64_t p);

uint64_t rat_mod_u64(int64_t num, int64_t den, uint64_t p);

void reduce_mod(const Eq &eq, uint64_t p, ResEq &out);

// Forward elimination, optionally recording equation and dependency maps.
void forward_eliminate(std::vector<ResEq> &eqs,
                       const std::vector<uint32_t> &sys_idx, uint64_t p,
                       std::unordered_map<uint32_t, ResEq> &solmap,
                       std::unordered_map<uint32_t, uint32_t> *eq_index,
                       std::unordered_map<uint32_t, std::vector<uint32_t>> *fwd_deps);

// Back substitution over the solved rows.
void backward_eliminate(std::unordered_map<uint32_t, ResEq> &solmap, uint64_t p,
                        std::unordered_map<uint32_t, std::vector<uint32_t>> *bwd_deps,
                        const std::vector<uint32_t> *only_keys = nullptr);

// Shoup's modular multiplication with a precomputed constant.
uint64_t shoup_precomp(uint64_t c, uint64_t p);

uint64_t shoup_mul(uint64_t x, uint64_t c, uint64_t cp, uint64_t p);

// Forward elimination order for one system.
struct FwdPlan {
  std::vector<uint32_t> order;
  std::vector<uint32_t> front;
  std::vector<uint32_t> len;
  std::vector<uint32_t> sys_idx;
};

void build_fwd_plan(const std::vector<std::pair<uint32_t, uint32_t>> &terms,
                    const std::vector<uint32_t> &eq_begin,
                    const std::vector<uint32_t> &sys_idx, FwdPlan &plan);

void fwd_order_from_rows(const std::vector<ResEq> &eqs,
                         const std::vector<uint32_t> &sys_idx,
                         std::vector<uint32_t> &order);

// Interned row table over shared term arrays.
struct SymRowTable {
  const std::pair<uint32_t, uint32_t> *terms = nullptr;
  const uint32_t *eq_begin = nullptr;
  std::size_t n_rows = 0;
  std::size_t begin(std::size_t e) const { return eq_begin[e]; }
  std::size_t row_size(std::size_t e) const { return eq_begin[e + 1] - eq_begin[e]; }
};

// Growable scratch row.
struct Row {
  std::unique_ptr<uint32_t[]> ids;
  std::unique_ptr<uint64_t[]> vals;
  std::size_t n = 0;
  std::size_t cap = 0;

  void ensure(std::size_t need) {
    if (cap >= need)
      return;
    std::size_t nc = cap ? cap : 8;
    while (nc < need)
      nc *= 2;

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

template <typename V> void purge_vector_tail(V &v) {
  using T = typename V::value_type;
  T *const base = v.data();
  if (base == nullptr || v.capacity() <= v.size())
    return;
  const long page = ::sysconf(_SC_PAGESIZE);
  if (page <= 0)
    return;
  const std::uintptr_t mask = static_cast<std::uintptr_t>(page) - 1;
  const std::uintptr_t first =
      (reinterpret_cast<std::uintptr_t>(base + v.size()) + mask) & ~mask;
  const std::uintptr_t last =
      reinterpret_cast<std::uintptr_t>(base + v.capacity()) & ~mask;
  if (first >= last)
    return;
  ::madvise(reinterpret_cast<void *>(first), static_cast<std::size_t>(last - first),
            MADV_DONTNEED);
}

// Forward-elimination store indexed by integral id.
struct FwdStore {
  std::vector<uint64_t> has;
  std::vector<uint32_t> row_of;
  std::vector<uint32_t> begin, len;
  std::vector<uint32_t> ids;
  std::vector<uint64_t> vals;

  const std::vector<uint32_t> *order_ptr = nullptr;
  std::vector<uint32_t> order_own;
  const std::vector<uint32_t> &order() const { return *order_ptr; }

  Row pend, merged;

  std::vector<uint32_t> eval_len, eval_front;

  // Fill sizes measured from a probe point at the same plan.  reset reserves them
  // (grow-only), so the first fill starts at the capacity it ends at instead of
  // doubling through it.  Zero means no hint, which leaves reset as it was.
  std::size_t fill_terms = 0;
  std::size_t fill_rows = 0;

  void set_fill_reserve(std::size_t terms, std::size_t rows) {
    fill_terms = terms;
    fill_rows = rows;
  }

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

// Forward-elimination counters.  `moved` counts every term written by a merge;
// `pivot_terms` counts the terms of the pivot rows those merges read.  The gap
// between the two is the copy work a merge spends re-writing the pending row.
struct FwdStats {
  std::uint64_t rows = 0;
  std::uint64_t gathered = 0;
  std::uint64_t merges = 0;
  std::uint64_t moved = 0;
  std::uint64_t pivot_terms = 0;
  std::uint64_t pivots = 0;
  void add(const FwdStats &o) {
    rows += o.rows;
    gathered += o.gathered;
    merges += o.merges;
    moved += o.moved;
    pivot_terms += o.pivot_terms;
    pivots += o.pivots;
  }
};

// A structural forward-elimination schedule.  Which rows become pivots, and in
// which order they are applied, follows from the coefficient values: equations
// that are exact linear combinations of the others cancel the same terms at
// every point, while a support-only (all-ones) build would keep them.  The
// schedule is therefore built from one elimination at an evaluated point of a
// fixed prime, and a replay at any other point reproduces that elimination as
// long as the cancellation pattern agrees.  Per plan row the schedule records
// the pivot rows applied and the accumulator slot each of their terms lands in;
// replaying it turns the row's terms into an accumulator update instead of a
// full merge, so a term is multiplied only when a pivot row actually
// contributes to it.
struct FwdDagMerge {
  uint32_t pivot_row;  // store row of the pivot row being applied
  uint32_t front_slot; // row-local slot holding that row's coefficient
};
// The merge's term slots follow the previous merge's in FwdDag::merge_slots, and
// its term count is the pivot row's length minus one (the front is excluded).
// A replay walks the merges in order with a running cursor, so neither the
// offset nor the count has to be stored: the store row that a merge names was
// written by this schedule's own build at exactly that length.

// Sentinels for the schedule.  Integral id 0 is a legal pivot, so "no pivot" is a
// separate value rather than 0.
inline constexpr uint32_t kFwdDagNoPivot = 0xffffffffu;
inline constexpr uint32_t kFwdDagNoSlot = 0xffffffffu;

// A terminal term of a row: its accumulator slot and the integral id it holds.
// The id is what the store row records; the slot is where the replay reads it.
struct FwdDagTerminal {
  uint32_t slot;
  uint32_t id;
};

struct FwdDagRow {
  uint32_t first_term = 0;   // offset into the row table of this row's terms
  uint32_t n_terms = 0;
  uint32_t n_slots = 0;      // accumulator entries (row-local slot count)
  uint32_t first_merge = 0, n_merges = 0;
  uint32_t first_terminal = 0, n_terminal = 0; // terminals, descending id
  uint32_t pivot = kFwdDagNoPivot; // the row's pivot id
  uint32_t store_row = 0;    // its store row index
};

struct FwdDag {
  std::vector<FwdDagRow> rows;     // one per row, in plan order
  std::vector<FwdDagMerge> merges;
  std::vector<uint32_t> merge_slots; // per-merge pivot term -> row-local slot
  std::vector<uint32_t> term_slots; // per row term -> row-local slot
  std::vector<FwdDagTerminal> terminals; // per-row terminal terms, descending id
  uint32_t max_slots = 0;
  bool empty() const { return rows.empty(); }

  // Bytes held by the schedule's vectors (at capacity) plus the structure itself.
  // The numeric driver shares one schedule across its workers, so this is also the
  // shared part it reports alongside the schedule's row and merge counts.
  std::size_t memory_bytes() const;
};

// Build the schedule by eliminating the row table at the coefficient values
// of one field point.  `values` holds one entry per coefficient slot, and `p`
// is the prime they were reduced for.  The elimination's own store goes into
// `store` (reset by the caller), so the point that builds the schedule is
// reduced like any other and no separate pass is needed.  Returns false when
// the table cannot carry a schedule; `dag` is then empty.  `abort` stops the
// build between rows, also returning false with an empty `dag`; the store is
// then partly filled and the caller must reset it before the next use.
bool build_fwd_dag(const SymRowTable &rows, const uint64_t *values,
                   std::size_t n_values, uint64_t p, const FwdPlan &plan,
                   FwdDag &dag, FwdStore &store, FwdStats *stats = nullptr,
                   const std::atomic<bool> *abort = nullptr);

// Replay the schedule at one point.  `values` must hold no zero (a vanishing
// coefficient makes the schedule structural only up to cancellation); returns
// false when the elimination departs from the schedule -- a cancelled pivot, a
// terminal that vanished, or a term left behind -- and the caller must then
// reset the store and fall back to forward_eliminate_planned.
bool forward_eliminate_dag(const SymRowTable &rows, const uint64_t *values,
                           const FwdDag &dag, uint64_t p, FwdStore &store,
                           std::vector<uint64_t> &acc,
                           const std::atomic<bool> *abort = nullptr,
                           FwdStats *stats = nullptr);

// Forward elimination driven by a prebuilt plan.
bool forward_eliminate_planned(const SymRowTable &rows, const uint64_t *values,
                               std::size_t n_values, const FwdPlan &plan, uint64_t p,
                               FwdStore &store,
                               const std::atomic<bool> *abort = nullptr,
                               std::unordered_map<uint32_t, std::vector<uint32_t>>
                                   *fwd_deps = nullptr,
                               std::vector<uint32_t> *row_src = nullptr,
                               FwdStats *stats = nullptr);

// Back substitution over a planned store.
bool backward_eliminate_planned(FwdStore &store, uint64_t p,
                                const std::vector<uint32_t> &only_keys,
                                const std::atomic<bool> *abort = nullptr,
                                std::unordered_map<uint32_t, std::vector<uint32_t>>
                                    *bwd_deps = nullptr);

// A system solution with its solve bookkeeping.
struct SubSolve {
  std::unordered_map<uint32_t, ResEq> solmap;
  std::unordered_map<uint32_t, uint32_t> eq_index;
  std::unordered_map<uint32_t, std::vector<uint32_t>> fwd_deps, bwd_deps;
};

// Solve a subset of a numeric system at one prime.
void solve_subset(const std::vector<Eq> &system,
                  const std::vector<uint32_t> &eq_idx, uint64_t p,
                  SubSolve &out, bool want_fwd, bool want_bwd,
                  const std::vector<uint32_t> *only_keys = nullptr);

}
