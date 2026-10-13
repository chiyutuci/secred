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

void forward_eliminate(std::vector<ResEq> &eqs,
                       const std::vector<uint32_t> &sys_idx, uint64_t p,
                       std::unordered_map<uint32_t, ResEq> &solmap,
                       std::unordered_map<uint32_t, uint32_t> *eq_index,
                       std::unordered_map<uint32_t, std::vector<uint32_t>> *fwd_deps) {
  // Ascending front(): equations for simpler integrals first.  Ties break on the
  // equation content and finally on sys_idx, which is prime-independent, so the
  // elimination order recorded by learn replays identically at every prime.
  const std::size_t n = eqs.size();
  std::vector<uint32_t> order(n);
  for (std::size_t i = 0; i < n; ++i)
    order[i] = static_cast<uint32_t>(i);
  std::sort(order.begin(), order.end(), [&](uint32_t i, uint32_t j) {
    const ResEq &a = eqs[i];
    const ResEq &b = eqs[j];
    if (a.empty())
      return false;
    if (b.empty())
      return true;
    if (a.front().first != b.front().first)
      return a.front().first < b.front().first;
    if (a.size() != b.size())
      return a.size() < b.size();
    for (std::size_t k = 0; k < a.size(); ++k) {
      if (a[k].first != b[k].first)
        return a[k].first < b[k].first;
      if (a[k].second != b[k].second)
        return a[k].second < b[k].second;
    }
    return sys_idx[i] < sys_idx[j];
  });

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

} // namespace secred
