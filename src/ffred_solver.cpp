#include "ffred_solver.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <mutex>

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

uint64_t mod_inverse_u64_fast(uint64_t a, uint64_t p) {
  if (a == 0)
    return 0;
  uint64_t u = a, v = p, x1 = 1, x2 = 0;
  while (u != 1 && v != 1) {

    while ((u & 1) == 0) {
      u >>= 1;
      x1 = (x1 & 1) ? (x1 + p) >> 1 : x1 >> 1;
    }
    while ((v & 1) == 0) {
      v >>= 1;
      x2 = (x2 & 1) ? (x2 + p) >> 1 : x2 >> 1;
    }

    if (u >= v) {
      u -= v;
      x1 = (x1 >= x2) ? x1 - x2 : x1 + p - x2;
    } else {
      v -= u;
      x2 = (x2 >= x1) ? x2 - x1 : x2 + p - x1;
    }
  }
  return u == 1 ? x1 : x2;
}

namespace {

struct BigRationalTable {
  std::mutex mutex;
  std::vector<std::string> texts{std::string()};
  std::unordered_map<std::string, int64_t> handles;
};

// Global table of interned big rationals.
BigRationalTable &big_rationals() {
  static BigRationalTable table;
  return table;
}

uint64_t decimal_mod(const char *b, const char *e, uint64_t p) {
  uint64_t v = 0;
  for (; b != e; ++b)
    v = static_cast<uint64_t>(
        (static_cast<__uint128_t>(v) * 10u + static_cast<unsigned>(*b - '0')) % p);
  return v;
}

uint64_t big_part_mod(const char *b, const char *e, uint64_t p) {
  bool neg = false;
  if (b != e && (*b == '+' || *b == '-')) {
    neg = *b == '-';
    ++b;
  }
  const uint64_t v = decimal_mod(b, e, p);
  return neg && v != 0 ? p - v : v;
}

}

bool intern_big_rational(const char *b, const char *e, int64_t &handle) {

  const char *p = b;
  if (p != e && (*p == '+' || *p == '-'))
    ++p;
  const char *num_begin = p;
  while (p != e && *p >= '0' && *p <= '9')
    ++p;
  if (p == num_begin)
    return false;
  if (p != e) {
    if (*p != '/')
      return false;
    ++p;
    if (p != e && (*p == '+' || *p == '-'))
      ++p;
    const char *den_begin = p;
    bool nonzero = false;
    while (p != e && *p >= '0' && *p <= '9') {
      nonzero = nonzero || *p != '0';
      ++p;
    }
    if (p == den_begin || !nonzero)
      return false;
  }
  if (p != e)
    return false;

  std::string text(b, e);
  BigRationalTable &table = big_rationals();
  std::lock_guard<std::mutex> lock(table.mutex);
  const auto it = table.handles.find(text);
  if (it != table.handles.end()) {
    handle = it->second;
    return true;
  }
  handle = static_cast<int64_t>(table.texts.size());
  table.texts.push_back(std::move(text));
  table.handles.emplace(table.texts.back(), handle);
  return true;
}

std::string rat_text(const Rat64 &r) {
  if (r.den != 0)
    return std::to_string(r.num) + "/" + std::to_string(r.den);
  BigRationalTable &table = big_rationals();
  std::lock_guard<std::mutex> lock(table.mutex);
  if (r.num <= 0 || static_cast<std::size_t>(r.num) >= table.texts.size())
    return "<invalid big coefficient>";
  return table.texts[static_cast<std::size_t>(r.num)];
}

uint64_t rat_mod_u64(int64_t num, int64_t den, uint64_t p) {
  if (den == 0) {

    std::string text;
    {
      BigRationalTable &table = big_rationals();
      std::lock_guard<std::mutex> lock(table.mutex);
      if (num <= 0 || static_cast<std::size_t>(num) >= table.texts.size())
        return 0;
      text = table.texts[static_cast<std::size_t>(num)];
    }
    const char *b = text.data();
    const char *e = b + text.size();
    const char *slash = static_cast<const char *>(std::memchr(b, '/', text.size()));
    const uint64_t n = big_part_mod(b, slash ? slash : e, p);
    const uint64_t d = slash ? big_part_mod(slash + 1, e, p) : 1;
    if (d == 0)
      return 0;
    return static_cast<uint64_t>(
        (static_cast<__uint128_t>(n) * mod_inverse_u64(d, p)) % p);
  }
  __int128 n = num % static_cast<int64_t>(p);
  if (n < 0)
    n += p;
  uint64_t d = static_cast<uint64_t>(den % static_cast<int64_t>(p));
  if (d == 0)
    return 0;
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

template <class Front, class Len, class IdAt>
bool fwd_order_less(uint32_t i, uint32_t j, const Front &front, const Len &len,
                    const IdAt &id_at, const std::vector<uint32_t> &score,
                    const std::vector<uint32_t> &sys_idx) {
  const std::size_t li = len(i), lj = len(j);
  if (li == 0)
    return false;
  if (lj == 0)
    return true;
  if (front(i) != front(j))
    return front(i) < front(j);

  if (score[i] != score[j])
    return score[i] < score[j];
  if (li != lj)
    return li < lj;
  for (std::size_t k = 0; k < li; ++k)
    if (id_at(i, k) != id_at(j, k))
      return id_at(i, k) < id_at(j, k);
  return sys_idx[i] < sys_idx[j];
}

template <class Len, class IdAt>
void front_id_scores(std::size_t n, const Len &len, const IdAt &id_at,
                     std::vector<uint32_t> &score) {
  std::vector<uint32_t> fronts;
  fronts.reserve(n);
  for (std::size_t e = 0; e < n; ++e)
    if (len(e))
      fronts.push_back(id_at(e, 0));
  std::sort(fronts.begin(), fronts.end());
  fronts.erase(std::unique(fronts.begin(), fronts.end()), fronts.end());
  score.assign(n, 0);
  for (std::size_t e = 0; e < n; ++e)
    for (std::size_t k = 0, l = len(e); k < l; ++k)
      score[e] += std::binary_search(fronts.begin(), fronts.end(), id_at(e, k));
}

template <class Front, class Len, class IdAt>
void sort_rows(std::vector<uint32_t> &order, const Front &front, const Len &len,
               const IdAt &id_at, const std::vector<uint32_t> &sys_idx) {
  std::vector<uint32_t> score;
  front_id_scores(sys_idx.size(), len, id_at, score);
  order.resize(sys_idx.size());
  for (std::size_t i = 0; i < order.size(); ++i)
    order[i] = static_cast<uint32_t>(i);
  std::sort(order.begin(), order.end(), [&](uint32_t i, uint32_t j) {
    return fwd_order_less(i, j, front, len, id_at, score, sys_idx);
  });
}

}

void fwd_order_from_rows(const std::vector<ResEq> &eqs,
                         const std::vector<uint32_t> &sys_idx,
                         std::vector<uint32_t> &order) {
  sort_rows(order, [&](uint32_t i) { return eqs[i].front().first; },
            [&](uint32_t i) { return eqs[i].size(); },
            [&](uint32_t i, std::size_t k) { return eqs[i][k].first; }, sys_idx);
}

namespace {

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

}

void forward_eliminate(std::vector<ResEq> &eqs,
                       const std::vector<uint32_t> &sys_idx, uint64_t p,
                       std::unordered_map<uint32_t, ResEq> &solmap,
                       std::unordered_map<uint32_t, uint32_t> *eq_index,
                       std::unordered_map<uint32_t, std::vector<uint32_t>> *fwd_deps) {

  std::vector<uint32_t> order;
  fwd_order_from_rows(eqs, sys_idx, order);

  std::vector<std::pair<uint32_t, uint64_t>> pending, merged;
  for (uint32_t k : order) {
    const ResEq &eq = eqs[k];
    if (eq.empty())
      continue;

    pending.assign(eq.begin(), eq.end());

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
      const ResEq &sol = solmap.at(w);

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
      continue;

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
    keys.assign(only_keys->begin(), only_keys->end());
  } else {
    keys.reserve(solmap.size());
    for (const auto &kv : solmap)
      keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
  }

  std::vector<std::pair<uint32_t, uint64_t>> masters;

  for (uint32_t k : keys) {

    auto kit = solmap.find(k);
    if (kit == solmap.end() || kit->second.size() <= 1)
      continue;
    ResEq &red = kit->second;

    masters.clear();
    std::vector<uint32_t> deps;
    for (size_t i = 1; i < red.size(); ++i) {
      uint32_t id = red[i].first;
      uint64_t c = red[i].second;
      auto it = solmap.find(id);
      if (it == solmap.end()) {
        masters.emplace_back(id, c);
      } else {
        deps.push_back(id);
        const ResEq &sol = it->second;
        for (size_t j = 1; j < sol.size(); ++j)
          masters.emplace_back(sol[j].first,
                               static_cast<uint64_t>((c * static_cast<__int128>(sol[j].second)) % p));
      }
    }

    std::sort(masters.begin(), masters.end(),
              [](const std::pair<uint32_t, uint64_t> &a,
                 const std::pair<uint32_t, uint64_t> &b) { return a.first > b.first; });

    red.clear();
    red.emplace_back(k, p - 1);
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

uint64_t shoup_precomp(uint64_t c, uint64_t p) {
  return static_cast<uint64_t>((static_cast<unsigned __int128>(c) << 64) / p);
}

uint64_t shoup_mul(uint64_t x, uint64_t c, uint64_t cp, uint64_t p) {

  const uint64_t q =
      static_cast<uint64_t>((static_cast<unsigned __int128>(cp) * x) >> 64);
  uint64_t r = c * x - q * p;
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

  sort_rows(plan.order, front, length,
            [&](std::size_t e, std::size_t k) { return terms[eq_begin[e] + k].first; },
            sys_idx);

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

  purge_vector_tail(begin);
  purge_vector_tail(len);
  purge_vector_tail(ids);
  purge_vector_tail(vals);
  begin.clear();
  len.clear();
  ids.clear();
  vals.clear();
  // reserve only grows, so a worker whose arena already outgrew the probe keeps it.
  if (fill_rows != 0) {
    begin.reserve(fill_rows);
    len.reserve(fill_rows);
  }
  if (fill_terms != 0) {
    ids.reserve(fill_terms);
    vals.reserve(fill_terms);
  }
  order_ptr = nullptr;
}

bool forward_eliminate_planned(const SymRowTable &rows, const uint64_t *values,
                               std::size_t n_values, const FwdPlan &plan, uint64_t p,
                               FwdStore &store, const std::atomic<bool> *abort,
                               std::unordered_map<uint32_t, std::vector<uint32_t>>
                                   *fwd_deps,
                               std::vector<uint32_t> *row_src, FwdStats *stats) {
  if (stats)
    *stats = FwdStats{};

  const bool plan_matches =
      plan.len.size() == rows.n_rows &&
      std::find(values, values + n_values, uint64_t(0)) == values + n_values;
  if (plan_matches) {
    store.order_ptr = &plan.order;
  } else {
    fwd_order_from_values(rows, values, plan.sys_idx, store.eval_len, store.eval_front,
                          store.order_own);
    store.order_ptr = &store.order_own;
  }

  Row &pend = store.pend;
  Row &merged = store.merged;
  std::vector<uint32_t> deps;
  for (uint32_t k : *store.order_ptr) {
    if (abort && abort->load(std::memory_order_relaxed))
      return false;
    if (stats)
      ++stats->rows;

    deps.clear();
    pend.clear();
    for (std::size_t t = rows.begin(k), te = rows.begin(k + 1); t < te; ++t) {
      const uint64_t v = values[rows.terms[t].second];
      if (v)
        pend.push(rows.terms[t].first, v);
    }
    if (pend.empty())
      continue;
    if (stats)
      stats->gathered += pend.n;

    std::size_t scan = 0;
    while (true) {
      if (abort && abort->load(std::memory_order_relaxed))
        return false;
      std::size_t idx = pend.n;
      for (std::size_t i = scan; i < pend.n; ++i) {
        if (store.has_row(pend.ids[i])) {
          idx = i;
          break;
        }
      }
      if (idx == pend.n)
        break;

      const uint32_t w = pend.ids[idx];
      const uint64_t c = pend.vals[idx];
      if (fwd_deps)
        deps.push_back(w);
      const uint32_t row = store.row_of[w];
      const uint32_t rb = store.begin[row], rl = store.len[row];
      const uint32_t *rids = store.ids.data() + rb;
      const uint64_t *rvals = store.vals.data() + rb;
      const uint64_t cp = shoup_precomp(c, p);

      const std::size_t n = pend.n;
      merged.ensure(n + rl);
      uint32_t *oid = merged.ids.get();
      uint64_t *oval = merged.vals.get();
      std::memcpy(oid, pend.ids.get(), idx * sizeof(uint32_t));
      std::memcpy(oval, pend.vals.get(), idx * sizeof(uint64_t));
      oid += idx;
      oval += idx;
      std::size_t i = idx + 1, j = 1;
      while (i < n && j < rl) {
        const uint32_t a = pend.ids[i], b = rids[j];
        if (a > b) {
          *oid++ = a;
          *oval++ = pend.vals[i];
          ++i;
        } else if (a < b) {
          *oid++ = b;
          *oval++ = shoup_mul(rvals[j], c, cp, p);
          ++j;
        } else {
          uint64_t v = pend.vals[i] + shoup_mul(rvals[j], c, cp, p);
          if (v >= p)
            v -= p;
          if (v) {
            *oid++ = a;
            *oval++ = v;
          }
          ++i;
          ++j;
        }
      }
      while (i < n) {
        *oid++ = pend.ids[i];
        *oval++ = pend.vals[i];
        ++i;
      }
      while (j < rl) {
        *oid++ = rids[j];
        *oval++ = shoup_mul(rvals[j], c, cp, p);
        ++j;
      }
      merged.n = static_cast<std::size_t>(oid - merged.ids.get());
      if (stats) {
        ++stats->merges;
        stats->moved += merged.n;
        stats->pivot_terms += rl;
      }
      merged.swap(pend);
      scan = idx;
    }

    if (pend.empty())
      continue;

    const uint32_t pivot = pend.ids[0];
    const uint64_t neg_inv = (p - mod_inverse_u64_fast(pend.vals[0], p)) % p;
    const uint64_t np = shoup_precomp(neg_inv, p);
    const uint32_t row = static_cast<uint32_t>(store.begin.size());
    store.begin.push_back(static_cast<uint32_t>(store.ids.size()));
    store.len.push_back(static_cast<uint32_t>(pend.n));

    store.ids.insert(store.ids.end(), pend.ids.get(), pend.ids.get() + pend.n);
    for (std::size_t t = 0; t < pend.n; ++t)
      store.vals.push_back(shoup_mul(pend.vals[t], neg_inv, np, p));
    store.mark(pivot, row);
    if (stats)
      ++stats->pivots;
    if (fwd_deps)
      (*fwd_deps)[pivot] = std::move(deps);
    if (row_src)
      row_src->push_back(k);
  }
  return true;
}

namespace {


inline bool dag_has(const std::vector<uint64_t> &has, uint32_t id) {
  return (has[id >> 6] >> (id & 63)) & 1u;
}

} // namespace

bool build_fwd_dag(const SymRowTable &rows, const uint64_t *values,
                   std::size_t n_values, uint64_t p, const FwdPlan &plan,
                   FwdDag &dag, FwdStore &store, FwdStats *stats) {
  dag = FwdDag{};
  const std::size_t n_rows = rows.n_rows;
  if (p < 2 || n_rows == 0 || plan.order.size() != n_rows ||
      rows.eq_begin[n_rows] == 0)
    return false;

  uint32_t max_id = 0;
  const std::size_t n_terms = rows.eq_begin[n_rows];
  for (std::size_t t = 0; t < n_terms; ++t) {
    if (rows.terms[t].second >= n_values)
      return false;
    max_id = std::max(max_id, rows.terms[t].first);
  }
  if (max_id == 0)
    return false;
  // The store carries this elimination's own rows, so it must cover every id.
  if (static_cast<std::size_t>(max_id) >= store.row_of.size() ||
      (static_cast<std::size_t>(max_id) >> 6) >= store.has.size())
    return false;

  std::vector<uint32_t> slot_of(max_id + 1, kFwdDagNoSlot);
  std::vector<uint32_t> touched;

  std::vector<uint32_t> pend, merged;
  std::vector<uint64_t> pend_vals, merged_vals;

  dag.rows.resize(n_rows);
  dag.term_slots.assign(n_terms, kFwdDagNoSlot);
  for (std::size_t r = 0; r < n_rows; ++r) {
    const std::size_t e = plan.order[r];
    if (stats)
      ++stats->rows;
    FwdDagRow &dr = dag.rows[r];
    const std::size_t t0 = rows.begin(e), t1 = rows.begin(e + 1);
    dr.first_term = static_cast<uint32_t>(t0);
    dr.n_terms = static_cast<uint32_t>(t1 - t0);
    dr.first_slot = static_cast<uint32_t>(dag.slots.size());

    uint32_t nslots = 0;
    auto slot = [&](uint32_t id) {
      uint32_t s = slot_of[id];
      if (s == kFwdDagNoSlot) {
        s = nslots++;
        slot_of[id] = s;
        dag.slots.push_back(id);
        touched.push_back(id);
      }
      return s;
    };

    pend.clear();
    pend_vals.clear();
    for (std::size_t t = t0; t < t1; ++t) {
      const uint32_t id = rows.terms[t].first;
      const uint64_t v = values[rows.terms[t].second];
      dag.term_slots[t] = slot(id);
      if (v) {
        pend.push_back(id);
        pend_vals.push_back(v);
      }
    }
    if (stats)
      stats->gathered += t1 - t0;

    dr.first_merge = static_cast<uint32_t>(dag.merges.size());
    std::size_t scan = 0;
    while (true) {
      std::size_t idx = pend.size();
      for (std::size_t i = scan; i < pend.size(); ++i) {
        if (store.has_row(pend[i])) {
          idx = i;
          break;
        }
      }
      if (idx == pend.size())
        break;

      const uint32_t w = pend[idx];
      const uint64_t c = pend_vals[idx];
      const uint32_t row = store.row_of[w];
      const uint32_t rb = store.begin[row], rl = store.len[row];
      const uint32_t *rids = store.ids.data() + rb;
      const uint64_t *rvals = store.vals.data() + rb;
      const uint64_t cp = shoup_precomp(c, p);

      merged.clear();
      merged_vals.clear();
      merged.insert(merged.end(), pend.begin(), pend.begin() + idx);
      merged_vals.insert(merged_vals.end(), pend_vals.begin(),
                         pend_vals.begin() + idx);
      std::size_t i = idx + 1, j = 1;
      while (i < pend.size() && j < rl) {
        const uint32_t a = pend[i], b = rids[j];
        if (a > b) {
          merged.push_back(a);
          merged_vals.push_back(pend_vals[i]);
          ++i;
        } else if (a < b) {
          slot(b);
          merged.push_back(b);
          merged_vals.push_back(shoup_mul(rvals[j], c, cp, p));
          ++j;
        } else {
          uint64_t v = pend_vals[i] + shoup_mul(rvals[j], c, cp, p);
          if (v >= p)
            v -= p;
          if (v) {
            merged.push_back(a);
            merged_vals.push_back(v);
          }
          ++i;
          ++j;
        }
      }
      while (i < pend.size()) {
        merged.push_back(pend[i]);
        merged_vals.push_back(pend_vals[i]);
        ++i;
      }
      while (j < rl) {
        slot(rids[j]);
        merged.push_back(rids[j]);
        merged_vals.push_back(shoup_mul(rvals[j], c, cp, p));
        ++j;
      }
      pend.swap(merged);
      pend_vals.swap(merged_vals);

      FwdDagMerge mg;
      mg.pivot_row = row;
      mg.front_slot = slot_of[w];
      mg.first_slot = static_cast<uint32_t>(dag.merge_slots.size());
      mg.n_slots = rl - 1;
      for (j = 1; j < rl; ++j)
        dag.merge_slots.push_back(slot_of[rids[j]]);
      dag.merges.push_back(mg);
      if (stats) {
        ++stats->merges;
        stats->moved += rl;
        stats->pivot_terms += rl;
      }
      scan = idx;
    }
    dr.n_merges = static_cast<uint32_t>(dag.merges.size()) - dr.first_merge;
    dr.n_slots = nslots;

    dr.first_terminal = static_cast<uint32_t>(dag.terminals.size());
    dr.n_terminal = static_cast<uint32_t>(pend.size());
    if (!pend.empty()) {
      dr.pivot = pend[0];
      dr.store_row = static_cast<uint32_t>(store.begin.size());
      for (uint32_t id : pend)
        dag.terminals.push_back(slot_of[id]);
      store.begin.push_back(static_cast<uint32_t>(store.ids.size()));
      store.len.push_back(static_cast<uint32_t>(pend.size()));
      store.ids.insert(store.ids.end(), pend.begin(), pend.end());

      const uint64_t neg_inv = (p - mod_inverse_u64_fast(pend_vals[0], p)) % p;
      const uint64_t np = shoup_precomp(neg_inv, p);
      for (std::size_t t = 0; t < pend.size(); ++t)
        store.vals.push_back(shoup_mul(pend_vals[t], neg_inv, np, p));

      store.mark(dr.pivot, dr.store_row);
      if (stats)
        ++stats->pivots;
    }

    for (uint32_t id : touched)
      slot_of[id] = kFwdDagNoSlot;
    touched.clear();
    dag.max_slots = std::max(dag.max_slots, nslots);
  }
  return true;
}

bool forward_eliminate_dag(const SymRowTable &rows, const uint64_t *values,
                           const FwdDag &dag, uint64_t p, FwdStore &store,
                           std::vector<uint64_t> &acc,
                           const std::atomic<bool> *abort, FwdStats *stats) {
  if (stats)
    *stats = FwdStats{};
  if (dag.empty())
    return false;

  acc.resize(dag.max_slots);
  store.order_ptr = nullptr;
  const uint32_t *slots = dag.slots.data();
  const uint32_t *merge_slots = dag.merge_slots.data();
  const uint32_t *term_slots = dag.term_slots.data();
  for (std::size_t r = 0; r < dag.rows.size(); ++r) {
    if (abort && abort->load(std::memory_order_relaxed))
      return false;
    const FwdDagRow &dr = dag.rows[r];
    if (stats)
      ++stats->rows;

    std::fill(acc.begin(), acc.begin() + dr.n_slots, uint64_t(0));
    const auto *terms = rows.terms + dr.first_term;
    for (uint32_t t = 0; t < dr.n_terms; ++t)
      acc[term_slots[dr.first_term + t]] = values[terms[t].second];
    if (stats)
      stats->gathered += dr.n_terms;

    for (uint32_t mi = 0; mi < dr.n_merges; ++mi) {
      const FwdDagMerge &mg = dag.merges[dr.first_merge + mi];
      const uint64_t c = acc[mg.front_slot];
      if (!c)
        continue;
      const uint64_t cp = shoup_precomp(c, p);
      const uint32_t sb = store.begin[mg.pivot_row];
      const uint64_t *rv = store.vals.data() + sb;
      const uint32_t *sl = merge_slots + mg.first_slot;
      acc[mg.front_slot] = 0; // the pivot is consumed
      for (uint32_t j = 1; j <= mg.n_slots; ++j) {
        uint64_t &a = acc[sl[j - 1]];
        a += shoup_mul(rv[j], c, cp, p);
        if (a >= p)
          a -= p;
      }
      if (stats) {
        ++stats->merges;
        stats->moved += mg.n_slots + 1;
        stats->pivot_terms += mg.n_slots + 1;
      }
    }

    // A departure shows up as a term the elimination should have consumed but
    // did not, or as a terminal that vanished.  Either way the row's support
    // no longer matches the schedule, so the caller must redo the point.
    const uint32_t *ts = dag.terminals.data() + dr.first_terminal;
    for (uint32_t j = 0; j < dr.n_terminal; ++j)
      if (!acc[ts[j]])
        return false;

    if (dr.pivot != kFwdDagNoPivot) {
      const uint64_t neg_inv = (p - mod_inverse_u64_fast(acc[ts[0]], p)) % p;
      const uint64_t np = shoup_precomp(neg_inv, p);
      assert(dr.store_row == store.begin.size());
      store.begin.push_back(static_cast<uint32_t>(store.ids.size()));
      store.len.push_back(dr.n_terminal);
      for (uint32_t j = 0; j < dr.n_terminal; ++j) {
        store.ids.push_back(slots[dr.first_slot + ts[j]]);
        store.vals.push_back(shoup_mul(acc[ts[j]], neg_inv, np, p));
      }
      store.mark(dr.pivot, dr.store_row);
      if (stats)
        ++stats->pivots;
    }

    for (uint32_t j = 0; j < dr.n_terminal; ++j)
      acc[ts[j]] = 0;
    for (uint32_t s = 0; s < dr.n_slots; ++s)
      if (acc[s])
        return false;
  }
  return true;
}

bool backward_eliminate_planned(FwdStore &store, uint64_t p,
                                const std::vector<uint32_t> &only_keys,
                                const std::atomic<bool> *abort,
                                std::unordered_map<uint32_t, std::vector<uint32_t>>
                                    *bwd_deps) {
  std::vector<std::pair<uint32_t, uint64_t>> masters;
  std::vector<std::pair<uint32_t, uint64_t>> out;
  std::vector<uint32_t> deps;
  for (uint32_t k : only_keys) {
    if (abort && abort->load(std::memory_order_relaxed))
      return false;
    if (!store.has_row(k))
      continue;
    if (store.row_size(k) <= 1)
      continue;

    const uint32_t row = store.row_of[k];
    const uint32_t rb = store.begin[row], rl = store.len[row];
    masters.clear();
    deps.clear();
    for (uint32_t i = 1; i < rl; ++i) {
      const uint32_t id = store.ids[rb + i];
      const uint64_t c = store.vals[rb + i];
      if (!store.has_row(id)) {
        masters.emplace_back(id, c);
      } else {

        if (bwd_deps)
          deps.push_back(id);
        const uint32_t sb = store.begin[store.row_of[id]];
        const uint32_t sl = store.len[store.row_of[id]];
        const uint64_t cp = shoup_precomp(c, p);
        for (uint32_t j = 1; j < sl; ++j)
          masters.emplace_back(store.ids[sb + j],
                               shoup_mul(store.vals[sb + j], c, cp, p));
      }
    }

    std::sort(masters.begin(), masters.end(),
              [](const std::pair<uint32_t, uint64_t> &a,
                 const std::pair<uint32_t, uint64_t> &b) { return a.first > b.first; });
    out.clear();
    out.emplace_back(k, p - 1);
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

    const uint32_t nr = static_cast<uint32_t>(store.begin.size());
    store.begin.push_back(static_cast<uint32_t>(store.ids.size()));
    store.len.push_back(static_cast<uint32_t>(out.size()));
    for (const auto &t : out) {
      store.ids.push_back(t.first);
      store.vals.push_back(t.second);
    }
    store.row_of[k] = nr;
    if (bwd_deps && !deps.empty())
      (*bwd_deps)[k] = std::move(deps);
  }
  return true;
}

}
