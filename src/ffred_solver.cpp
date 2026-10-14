#include "ffred_solver.h"

#include <algorithm>

namespace secred {

uint64_t mod_inverse_u64(uint64_t a, uint64_t p) {
  __int128 r0 = a, r1 = p;
  __int128 s0 = 1, s1 = 0;
  while (r1 != 0) {
    __int128 q = r0 / r1;
    __int128 r = r0 - q * r1;
    r0 = r1;
    r1 = r;
    __int128 s = s0 - q * s1;
    s0 = s1;
    s1 = s;
  }
  if (s0 < 0)
    s0 += p;
  return static_cast<uint64_t>(s0);
}

uint64_t rat_mod_u64(int64_t num, int64_t den, uint64_t p) {
  __int128 n = num % static_cast<int64_t>(p);
  if (n < 0)
    n += p;
  uint64_t d = static_cast<uint64_t>(den % static_cast<int64_t>(p));
  if (d == 0)
    return 0; // p | den (cannot happen: denominators are far below p)
  uint64_t inv = mod_inverse_u64(d, p);
  return static_cast<uint64_t>((n * static_cast<__int128>(inv)) % p);
}

void reduce_mod(const Eq &eq, uint64_t p, ResEq &out) {
  out.clear();
  out.reserve(eq.size());
  for (const auto &t : eq) {
    uint64_t v = rat_mod_u64(t.second.num, t.second.den, p);
    if (v)
      out.emplace_back(t.first, v);
  }
}

namespace {

// The elimination order's key, over any row accessor: ascending front id (the row's most
// complex integral), then row length, then the id list, and only then sys_idx.  No
// coefficient ever enters it, which is what makes the order prime-independent -- and one
// definition serves all three callers (the reference over ResEq rows, build_fwd_plan over
// the pool's term table, fwd_order_from_rows over a point's rows), so they cannot drift.
// Accessors: len(i) -> row length, front(i) -> leading id, id_at(i, k) -> the k-th id;
// front is only called for rows len says are not empty.
template <class Front, class Len, class IdAt>
bool fwd_order_less(uint32_t i, uint32_t j, const Front &front, const Len &len,
                    const IdAt &id_at, const std::vector<uint32_t> &sys_idx) {
  const std::size_t li = len(i), lj = len(j);
  if (li == 0)
    return false; // an empty row sorts last: the elimination skips it
  if (lj == 0)
    return true;
  if (front(i) != front(j))
    return front(i) < front(j);
  if (li != lj)
    return li < lj;
  for (std::size_t k = 0; k < li; ++k)
    if (id_at(i, k) != id_at(j, k))
      return id_at(i, k) < id_at(j, k);
  return sys_idx[i] < sys_idx[j];
}

template <class Front, class Len, class IdAt>
void sort_rows(std::vector<uint32_t> &order, const Front &front, const Len &len,
               const IdAt &id_at, const std::vector<uint32_t> &sys_idx) {
  order.resize(sys_idx.size());
  for (std::size_t i = 0; i < order.size(); ++i)
    order[i] = static_cast<uint32_t>(i);
  std::sort(order.begin(), order.end(), [&](uint32_t i, uint32_t j) {
    return fwd_order_less(i, j, front, len, id_at, sys_idx);
  });
}

} // namespace

void fwd_order_from_rows(const std::vector<ResEq> &eqs,
                         const std::vector<uint32_t> &sys_idx,
                         std::vector<uint32_t> &order) {
  sort_rows(order, [&](uint32_t i) { return eqs[i].front().first; },
            [&](uint32_t i) { return eqs[i].size(); },
            [&](uint32_t i, std::size_t k) { return eqs[i][k].first; }, sys_idx);
}

namespace {

// fwd_order_from_rows over rows that are never materialised: the same key, read out of the
// shared table and this point's slot values.  One pass records each evaluated row's length
// and leading id -- the two the sort compares first, and the two that decide all but a
// handful of the comparisons -- so only a pair of rows that tie on both has to walk its
// id lists, which is why the id lists can stay implicit.
void fwd_order_from_values(const SymRowTable &rows, const uint64_t *values,
                           const std::vector<uint32_t> &sys_idx,
                           std::vector<uint32_t> &eval_len,
                           std::vector<uint32_t> &eval_front,
                           std::vector<uint32_t> &order) {
  const std::size_t n = rows.n_rows;
  eval_len.assign(n, 0);
  eval_front.assign(n, 0);
  for (std::size_t e = 0; e < n; ++e) {
    for (std::size_t t = rows.begin(e), te = rows.begin(e + 1); t < te; ++t) {
      if (!values[rows.terms[t].second])
        continue;
      if (eval_len[e] == 0)
        eval_front[e] = rows.terms[t].first;
      ++eval_len[e];
    }
  }
  // The k-th surviving id of row e, k < eval_len[e] (the sort only asks for ids of rows it
  // has already seen a non-zero length for).
  auto id_at = [&](uint32_t e, std::size_t k) {
    for (std::size_t t = rows.begin(e), te = rows.begin(e + 1); t < te; ++t)
      if (values[rows.terms[t].second] && k-- == 0)
        return rows.terms[t].first;
    return uint32_t(0);
  };
  sort_rows(order, [&](uint32_t e) { return eval_front[e]; },
            [&](uint32_t e) { return static_cast<std::size_t>(eval_len[e]); }, id_at,
            sys_idx);
}

} // namespace

void forward_eliminate(std::vector<ResEq> &eqs,
                       const std::vector<uint32_t> &sys_idx, uint64_t p,
                       std::unordered_map<uint32_t, ResEq> &solmap,
                       std::unordered_map<uint32_t, uint32_t> *eq_index,
                       std::unordered_map<uint32_t, std::vector<uint32_t>> *fwd_deps) {
  // Ascending front(): equations for simpler integrals first.  Ties break on the
  // equation's *structure* -- its length and id list -- and finally on sys_idx, which is
  // prime-independent, so the elimination order recorded by learn replays identically at
  // every prime.  The coefficients must not enter it: they change with the prime, and two
  // rows of equal structure would then be ordered differently at different primes (and
  // ordered differently from build_fwd_plan, which reproduces this order without ever
  // looking at a residue).
  std::vector<uint32_t> order;
  fwd_order_from_rows(eqs, sys_idx, order);

  std::vector<std::pair<uint32_t, uint64_t>> pending, merged; // reused buffers
  for (uint32_t k : order) {
    const ResEq &eq = eqs[k];
    if (eq.empty())
      continue;

    // pending starts as the raw terms (descending id, unique).
    pending.assign(eq.begin(), eq.end());

    // Substitute the highest reducible term repeatedly.  Each substitution merges
    // sol[w]'s right-hand side into pending with an ordered merge, keeping pending
    // descending with unique ids and avoiding both a per-equation std::map and the
    // quadratic cost of appending and re-sorting.
    std::vector<uint32_t> deps;
    while (true) {
      std::size_t idx = pending.size();
      for (std::size_t i = 0; i < pending.size(); ++i) {
        if (solmap.find(pending[i].first) != solmap.end()) {
          idx = i;
          break;
        }
      }
      if (idx == pending.size())
        break;
      uint32_t w = pending[idx].first;
      uint64_t c = pending[idx].second;
      deps.push_back(w);
      const ResEq &sol = solmap.at(w); // sol[0] = (w, -1), sol[1..] = tail (descending)

      // Merge pending (minus w at idx) with c * sol[1..].
      merged.clear();
      merged.reserve(pending.size() + sol.size());
      std::size_t i = 0, j = 1;
      while (i < pending.size() && j < sol.size()) {
        if (i == idx) {
          ++i;
          continue;
        }
        uint32_t a = pending[i].first, b = sol[j].first;
        if (a > b) {
          merged.push_back(pending[i++]);
        } else if (a < b) {
          merged.emplace_back(b,
                              static_cast<uint64_t>((c * static_cast<__int128>(sol[j].second)) % p));
          ++j;
        } else {
          uint64_t v = (pending[i].second +
                        static_cast<uint64_t>((c * static_cast<__int128>(sol[j].second)) % p)) % p;
          if (v)
            merged.emplace_back(a, v);
          ++i;
          ++j;
        }
      }
      while (i < pending.size()) {
        if (i != idx)
          merged.push_back(pending[i]);
        ++i;
      }
      while (j < sol.size()) {
        merged.emplace_back(sol[j].first,
                            static_cast<uint64_t>((c * static_cast<__int128>(sol[j].second)) % p));
        ++j;
      }
      pending.swap(merged);
    }

    if (pending.empty())
      continue; // reduced to 0 = 0

    // Normalise the pivot on the remaining highest term: coefficient -> -1.
    uint32_t pivot = pending.front().first;
    uint64_t pivot_coeff = pending.front().second;
    uint64_t neg_inv = (p - mod_inverse_u64(pivot_coeff, p)) % p;

    ResEq result;
    result.reserve(pending.size());
    for (const auto &t : pending)
      result.emplace_back(t.first,
                          static_cast<uint64_t>((t.second * static_cast<__int128>(neg_inv)) % p));
    solmap.emplace(pivot, std::move(result));
    if (eq_index)
      (*eq_index)[pivot] = sys_idx[k];
    if (fwd_deps)
      (*fwd_deps)[pivot] = std::move(deps);
  }
}

void backward_eliminate(std::unordered_map<uint32_t, ResEq> &solmap, uint64_t p,
                        std::unordered_map<uint32_t, std::vector<uint32_t>> *bwd_deps,
                        const std::vector<uint32_t> *only_keys) {
  std::vector<uint32_t> keys;
  if (only_keys) {
    keys.assign(only_keys->begin(), only_keys->end()); // already ascending
  } else {
    keys.reserve(solmap.size());
    for (const auto &kv : solmap)
      keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end()); // ascending: the lowest ids substitute first
  }

  // One reused buffer instead of a std::map per equation (the old bottleneck).
  std::vector<std::pair<uint32_t, uint64_t>> masters;

  for (uint32_t k : keys) {
    // find, not operator[]: a key in only_keys need not be a pivot of this subset
    // (a narrowed selection can leave bwd_chain stale).  operator[] would insert an
    // empty entry, and a later solmap.find(id) != end() would then misread a genuine
    // master as a derived integral and silently drop its coefficient.
    auto kit = solmap.find(k);
    if (kit == solmap.end() || kit->second.size() <= 1)
      continue; // no right-hand side (degenerate k = 0)
    ResEq &red = kit->second;

    // red[1..] is descending with unique ids.  One pass: reducible terms (derived
    // integrals) are substituted, masters kept as they are, and every master term is
    // collected in masters for a final sorted merge.
    masters.clear();
    std::vector<uint32_t> deps;
    for (size_t i = 1; i < red.size(); ++i) {
      uint32_t id = red[i].first;
      uint64_t c = red[i].second;
      auto it = solmap.find(id);
      if (it == solmap.end()) {
        masters.emplace_back(id, c); // master: keep
      } else {
        deps.push_back(id); // derived: substitute
        const ResEq &sol = it->second; // sol[0]=(id,-1), sol[1..]=masters
        for (size_t j = 1; j < sol.size(); ++j)
          masters.emplace_back(sol[j].first,
                               static_cast<uint64_t>((c * static_cast<__int128>(sol[j].second)) % p));
      }
    }

    // Sort descending and merge duplicate ids (coefficients summed mod p, zeros dropped).
    std::sort(masters.begin(), masters.end(),
              [](const std::pair<uint32_t, uint64_t> &a,
                 const std::pair<uint32_t, uint64_t> &b) { return a.first > b.first; });

    red.clear();
    red.emplace_back(k, p - 1); // (k, -1)
    for (size_t i = 0; i < masters.size();) {
      uint32_t id = masters[i].first;
      uint64_t c = masters[i].second;
      size_t j = i + 1;
      while (j < masters.size() && masters[j].first == id) {
        c = (c + masters[j].second) % p;
        ++j;
      }
      if (c != 0)
        red.emplace_back(id, c);
      i = j;
    }

    if (bwd_deps && !deps.empty())
      (*bwd_deps)[k] = std::move(deps);
  }
}

void solve_subset(const std::vector<Eq> &system,
                  const std::vector<uint32_t> &eq_idx, uint64_t p,
                  SubSolve &out, bool want_fwd, bool want_bwd,
                  const std::vector<uint32_t> *only_keys) {
  out.solmap.clear();
  out.eq_index.clear();
  out.fwd_deps.clear();
  out.bwd_deps.clear();

  std::vector<ResEq> res_eqs;
  std::vector<uint32_t> sys_idx;
  res_eqs.reserve(eq_idx.size());
  sys_idx.reserve(eq_idx.size());
  for (uint32_t i : eq_idx) {
    ResEq re;
    reduce_mod(system[i], p, re);
    if (!re.empty()) {
      res_eqs.push_back(std::move(re));
      sys_idx.push_back(i);
    }
  }

  forward_eliminate(res_eqs, sys_idx, p, out.solmap,
                    want_fwd ? &out.eq_index : nullptr,
                    want_fwd ? &out.fwd_deps : nullptr);
  backward_eliminate(out.solmap, p, want_bwd ? &out.bwd_deps : nullptr,
                     only_keys);
}

// ---------------------------------------------------------------------------
// The sample-point path (see the header)
// ---------------------------------------------------------------------------

uint64_t shoup_precomp(uint64_t c, uint64_t p) {
  return static_cast<uint64_t>((static_cast<unsigned __int128>(c) << 64) / p);
}

uint64_t shoup_mul(uint64_t x, uint64_t c, uint64_t cp, uint64_t p) {
  // q = floor(cp * x / 2^64) is floor(c*x/p) or one less: cp = floor(c*2^64/p) is at most
  // c*2^64/p, so q <= floor(c*x/p); and cp > c*2^64/p - 1, so the shortfall
  // cp*x/2^64 <= c*x/p - x/2^64 < floor(c*x/p) + 1.  Hence c*x - q*p is the true residue
  // plus at most one p, and fits in 64 bits as long as 2p does.
  const uint64_t q =
      static_cast<uint64_t>((static_cast<unsigned __int128>(cp) * x) >> 64);
  uint64_t r = c * x - q * p; // mod 2^64: the true value is in [0, 2p) < 2^64
  if (r >= p)
    r -= p;
  return r;
}

void build_fwd_plan(const std::vector<std::pair<uint32_t, uint32_t>> &terms,
                    const std::vector<uint32_t> &eq_begin,
                    const std::vector<uint32_t> &sys_idx, FwdPlan &plan) {
  const std::size_t n = eq_begin.empty() ? 0 : eq_begin.size() - 1;
  auto front = [&](std::size_t e) { return terms[eq_begin[e]].first; };
  auto length = [&](std::size_t e) {
    return static_cast<std::size_t>(eq_begin[e + 1] - eq_begin[e]);
  };
  // forward_eliminate's order, computed from the structure alone: front id, row length,
  // the id list, then sys_idx.  Rows that agree on all four tie there too, and the
  // elimination then replays in the same order at every prime.  An empty row sorts last
  // and is skipped by the elimination.
  sort_rows(plan.order, front, length,
            [&](std::size_t e, std::size_t k) { return terms[eq_begin[e] + k].first; },
            sys_idx);
  // The structure itself, so a point can check its rows against the plan.
  plan.front.resize(n);
  plan.len.resize(n);
  for (std::size_t e = 0; e < n; ++e) {
    plan.len[e] = static_cast<uint32_t>(length(e));
    plan.front[e] = plan.len[e] ? front(e) : 0u;
  }
  plan.sys_idx = sys_idx;
}

void FwdStore::reset(std::size_t n_ids) {
  if (has.size() < (n_ids + 63) / 64)
    has.resize((n_ids + 63) / 64, 0);
  else
    std::fill(has.begin(), has.begin() + (n_ids + 63) / 64, 0);
  row_of.resize(n_ids);
  begin.clear();
  len.clear();
  ids.clear();
  vals.clear();
}

void forward_eliminate_planned(const SymRowTable &rows, const uint64_t *values,
                               std::size_t n_values, const FwdPlan &plan, uint64_t p,
                               FwdStore &store) {
  // The rows handed in are the pool's rows minus whatever vanished at this point (a
  // numerator or a denominator that is 0 mod p), and a dropped coefficient moves its row
  // in the reference's order: the row is shorter, and if the front term went, its front
  // moved with it.  The plan was built from the pool's structure, so check the point
  // against it: every slot occurs in some row, so a single zero among the slot values is
  // exactly what makes some evaluated row differ from the pool row the plan holds.  Where
  // they differ, order the rows the way the reference does, from the evaluated rows.
  // Skipping this would eliminate such a point in a different order than the reference,
  // which the chain-restricted back substitution turns into a NonMasterRhs abort blaming
  // a stale selection that is in fact fine.
  const bool plan_matches =
      plan.len.size() == rows.n_rows &&
      std::find(values, values + n_values, uint64_t(0)) == values + n_values;
  if (plan_matches)
    store.order = plan.order;
  else
    fwd_order_from_values(rows, values, plan.sys_idx, store.eval_len, store.eval_front,
                          store.order);

  std::vector<std::pair<uint32_t, uint64_t>> &pending = store.pending;
  std::vector<std::pair<uint32_t, uint64_t>> &merged = store.merged;
  for (uint32_t k : store.order) {
    // This point's row k, gathered on demand: the row's terms whose coefficient did not
    // vanish here.  Descending id, as the merge below assumes.
    pending.clear();
    for (std::size_t t = rows.begin(k), te = rows.begin(k + 1); t < te; ++t) {
      const uint64_t v = values[rows.terms[t].second];
      if (v)
        pending.emplace_back(rows.terms[t].first, v);
    }
    if (pending.empty())
      continue; // reduced to 0 = 0

    // Substitute the highest reducible term repeatedly.  Each substitution merges one
    // solved right-hand side into pending with an ordered merge, keeping pending
    // descending with unique ids.
    std::size_t scan = 0; // [0, scan) came through the last merge unchanged
    while (true) {
      std::size_t idx = pending.size();
      for (std::size_t i = scan; i < pending.size(); ++i) {
        if (store.has_row(pending[i].first)) {
          idx = i;
          break;
        }
      }
      if (idx == pending.size())
        break;

      const uint32_t w = pending[idx].first;
      const uint64_t c = pending[idx].second;
      const uint32_t row = store.row_of[w];
      const uint32_t rb = store.begin[row], rl = store.len[row];
      const uint32_t *rids = store.ids.data() + rb;
      const uint64_t *rvals = store.vals.data() + rb;
      const uint64_t cp = shoup_precomp(c, p);

      // Merge pending (minus w at idx) with c * sol[1..].  The solved row's terms are all
      // below its pivot w, and pending's are descending, so everything taken from the
      // solved row lands at or after idx: [0, idx) survives untouched and the next scan
      // can start there.
      merged.clear();
      merged.reserve(pending.size() + rl);
      std::size_t i = 0, j = 1;
      while (i < pending.size() && j < rl) {
        if (i == idx) {
          ++i;
          continue;
        }
        const uint32_t a = pending[i].first, b = rids[j];
        if (a > b) {
          merged.push_back(pending[i++]);
        } else if (a < b) {
          merged.emplace_back(b, shoup_mul(rvals[j], c, cp, p));
          ++j;
        } else {
          uint64_t v = pending[i].second + shoup_mul(rvals[j], c, cp, p);
          if (v >= p)
            v -= p;
          if (v)
            merged.emplace_back(a, v);
          ++i;
          ++j;
        }
      }
      while (i < pending.size()) {
        if (i != idx)
          merged.push_back(pending[i]);
        ++i;
      }
      while (j < rl) {
        merged.emplace_back(rids[j], shoup_mul(rvals[j], c, cp, p));
        ++j;
      }
      pending.swap(merged);
      scan = idx;
    }

    if (pending.empty())
      continue;

    // Normalise the pivot on the remaining highest term: coefficient -> -1.
    const uint32_t pivot = pending.front().first;
    const uint64_t neg_inv = (p - mod_inverse_u64(pending.front().second, p)) % p;
    const uint64_t np = shoup_precomp(neg_inv, p);
    const uint32_t row = static_cast<uint32_t>(store.begin.size());
    store.begin.push_back(static_cast<uint32_t>(store.ids.size()));
    store.len.push_back(static_cast<uint32_t>(pending.size()));
    for (const auto &t : pending) {
      store.ids.push_back(t.first);
      store.vals.push_back(shoup_mul(t.second, neg_inv, np, p));
    }
    store.mark(pivot, row);
  }
}

void backward_eliminate_planned(FwdStore &store, uint64_t p,
                                const std::vector<uint32_t> &only_keys) {
  std::vector<std::pair<uint32_t, uint64_t>> masters;
  std::vector<std::pair<uint32_t, uint64_t>> out;
  for (uint32_t k : only_keys) {
    if (!store.has_row(k))
      continue;
    if (store.row_size(k) <= 1)
      continue; // no right-hand side (degenerate k = 0)

    const uint32_t row = store.row_of[k];
    const uint32_t rb = store.begin[row], rl = store.len[row];
    masters.clear();
    for (uint32_t i = 1; i < rl; ++i) {
      const uint32_t id = store.ids[rb + i];
      const uint64_t c = store.vals[rb + i];
      if (!store.has_row(id)) {
        masters.emplace_back(id, c); // master: keep
      } else {
        // Derived: substitute one level deep.  Keys ascend and the row of a smaller id
        // has already been rewritten, so its right-hand side carries masters only.
        const uint32_t sb = store.begin[store.row_of[id]];
        const uint32_t sl = store.len[store.row_of[id]];
        const uint64_t cp = shoup_precomp(c, p);
        for (uint32_t j = 1; j < sl; ++j)
          masters.emplace_back(store.ids[sb + j],
                               shoup_mul(store.vals[sb + j], c, cp, p));
      }
    }

    // Sort descending and merge duplicate ids (summed mod p, zeros dropped).
    std::sort(masters.begin(), masters.end(),
              [](const std::pair<uint32_t, uint64_t> &a,
                 const std::pair<uint32_t, uint64_t> &b) { return a.first > b.first; });
    out.clear();
    out.emplace_back(k, p - 1); // (k, -1)
    for (std::size_t i = 0; i < masters.size();) {
      const uint32_t id = masters[i].first;
      uint64_t c = masters[i].second;
      std::size_t j = i + 1;
      while (j < masters.size() && masters[j].first == id) {
        c += masters[j].second;
        if (c >= p)
          c -= p;
        ++j;
      }
      if (c)
        out.emplace_back(id, c);
      i = j;
    }

    // A rewritten row is a new row: it can be longer or shorter than the one it replaces,
    // and the solve is already done, so nothing else reads the old stretch.
    const uint32_t nr = static_cast<uint32_t>(store.begin.size());
    store.begin.push_back(static_cast<uint32_t>(store.ids.size()));
    store.len.push_back(static_cast<uint32_t>(out.size()));
    for (const auto &t : out) {
      store.ids.push_back(t.first);
      store.vals.push_back(t.second);
    }
    store.row_of[k] = nr;
  }
}

} // namespace secred
