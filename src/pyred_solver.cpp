#include "pyred_solver.h"

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
    return 0; // p | den（输入系数分母都远小于 p，不会发生）
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
  // 按 front（最大 id）升序：更简积分（更小 id）的方程先处理。同 front 时按方程
  // 内容字典序，最终用 sys_idx（原始方程下标，素数无关）tie-break，保证跨素数
  // 确定性，使 learn 记录的消元顺序可在每个素数上重放。
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

  std::vector<std::pair<uint32_t, uint64_t>> pending, merged; // 复用缓冲
  for (uint32_t k : order) {
    const ResEq &eq = eqs[k];
    if (eq.empty())
      continue;

    // pending = 原始项（已按 id 降序、id 唯一）。
    pending.assign(eq.begin(), eq.end());

    // 反复代入最高可约项；每次代入用「有序归并」把 sol[w] 的 RHS 并进 pending，
    // 保持 pending 按 id 降序且 id 唯一（合并重复 id，系数相加 mod p，删零）。
    // 这样既避免每方程 std::map，又避免「append 后一次性排序」的二次爆炸。
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
      const ResEq &sol = solmap.at(w); // sol[0] = (w, -1), sol[1..] = tail（降序、唯一）

      // 归并 pending（去掉 idx 处的 w）与 c*sol[1..]（降序、唯一）。
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
      continue; // 约化为 0 = 0 的冗余方程

    // 对剩余最高项主元归一：系数 -> -1。
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
    keys.assign(only_keys->begin(), only_keys->end()); // 已升序
  } else {
    keys.reserve(solmap.size());
    for (const auto &kv : solmap)
      keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end()); // 升序：低 id 的解先代入
  }

  // 复用临时缓冲，避免每方程分配 std::map（旧实现的性能瓶颈）。
  std::vector<std::pair<uint32_t, uint64_t>> masters;

  for (uint32_t k : keys) {
    ResEq &red = solmap[k];
    if (red.size() <= 1)
      continue; // 无 RHS（k = 0 的退化情形）

    // red[1..] 已按 id 降序且 id 唯一。一次遍历：可约项（派生积分）代入其解，
    // 主积分直接保留；所有主积分项收集进 masters，最后排序归并。
    masters.clear();
    std::vector<uint32_t> deps;
    for (size_t i = 1; i < red.size(); ++i) {
      uint32_t id = red[i].first;
      uint64_t c = red[i].second;
      auto it = solmap.find(id);
      if (it == solmap.end()) {
        masters.emplace_back(id, c); // 主积分，保留
      } else {
        deps.push_back(id); // 派生积分，代入
        const ResEq &sol = it->second; // sol[0]=(id,-1), sol[1..]=masters
        for (size_t j = 1; j < sol.size(); ++j)
          masters.emplace_back(sol[j].first,
                               static_cast<uint64_t>((c * static_cast<__int128>(sol[j].second)) % p));
      }
    }

    // 按 id 降序排序并合并重复 id（系数相加 mod p，删零）。
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

} // namespace secred
