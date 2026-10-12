#include "test_runner.h"
#include "pyred_solver.h"

#include <cstdint>
#include <initializer_list>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using secred::Eq;
using secred::Rat64;
using secred::ResEq;

constexpr uint64_t P = 1000000007ULL;

Eq eq(std::initializer_list<std::pair<uint32_t, std::pair<int64_t, int64_t>>> terms) {
  Eq e;
  for (const auto &t : terms)
    e.push_back({t.first, Rat64{t.second.first, t.second.second}});
  return e;
}

void full_solve(const std::vector<Eq> &system, uint64_t p,
                std::unordered_map<uint32_t, ResEq> &solmap,
                std::unordered_map<uint32_t, uint32_t> &eq_index,
                std::unordered_map<uint32_t, std::vector<uint32_t>> &fwd_deps,
                std::unordered_map<uint32_t, std::vector<uint32_t>> &bwd_deps) {
  std::vector<ResEq> res_eqs;
  std::vector<uint32_t> sys_idx;
  for (std::size_t i = 0; i < system.size(); ++i) {
    ResEq re;
    secred::reduce_mod(system[i], p, re);
    if (!re.empty()) {
      res_eqs.push_back(std::move(re));
      sys_idx.push_back(static_cast<uint32_t>(i));
    }
  }
  secred::forward_eliminate(res_eqs, sys_idx, p, solmap, &eq_index, &fwd_deps);
  secred::backward_eliminate(solmap, p, &bwd_deps);
}

std::unordered_set<uint32_t> mark_sweep(
    const std::vector<uint32_t> &targets,
    const std::unordered_map<uint32_t, std::vector<uint32_t>> &fwd_deps,
    const std::unordered_map<uint32_t, std::vector<uint32_t>> &bwd_deps) {
  std::unordered_set<uint32_t> needed(targets.begin(), targets.end());
  std::vector<uint32_t> stack(targets.begin(), targets.end());
  while (!stack.empty()) {
    uint32_t w = stack.back();
    stack.pop_back();
    auto fit = fwd_deps.find(w);
    if (fit != fwd_deps.end())
      for (uint32_t u : fit->second)
        if (needed.insert(u).second)
          stack.push_back(u);
    auto bit = bwd_deps.find(w);
    if (bit != bwd_deps.end())
      for (uint32_t u : bit->second)
        if (needed.insert(u).second)
          stack.push_back(u);
  }
  return needed;
}

} // namespace

// 简单链：I_1 = I_0，I_2 = I_1 + I_0，I_3 = I_2 + I_0（I_3 与目标无关）。
// 目标 I_2 → I_2 = 2 I_0；mark-and-sweep 应只选 {I_2, I_1} 的方程，排除 I_3。
TEST(pyred_forward_backward_basic) {
  std::vector<Eq> system;
  system.push_back(eq({{1, {1, 1}}, {0, {-1, 1}}}));
  system.push_back(eq({{2, {1, 1}}, {1, {-1, 1}}, {0, {-1, 1}}}));
  system.push_back(eq({{3, {1, 1}}, {2, {-1, 1}}, {0, {-1, 1}}}));

  std::unordered_map<uint32_t, ResEq> solmap;
  std::unordered_map<uint32_t, uint32_t> eq_index;
  std::unordered_map<uint32_t, std::vector<uint32_t>> fwd_deps, bwd_deps;
  full_solve(system, P, solmap, eq_index, fwd_deps, bwd_deps);

  // I_2 = 2 I_0
  REQUIRE(solmap.find(2) != solmap.end());
  const ResEq &r2 = solmap.at(2);
  REQUIRE_EQ(r2.size(), 2u);
  REQUIRE_EQ(r2[0].first, 2u);
  REQUIRE_EQ(r2[0].second, P - 1);
  REQUIRE_EQ(r2[1].first, 0u);
  REQUIRE_EQ(r2[1].second, 2u);

  // mark-and-sweep 从 {2} 出发 → {2, 1}，排除 3。
  auto needed = mark_sweep({2}, fwd_deps, bwd_deps);
  REQUIRE(needed.count(2));
  REQUIRE(needed.count(1));
  REQUIRE(!needed.count(3));
  REQUIRE_EQ(needed.size(), 2u);
}

// 派生积分（对应 box 的 [1,2,1,2]）：I_125 无自身方程，作为「重复 front=127」方程的
// 主元晚注册，出现在 I_127 的 RHS 上，必须靠后向代入把它推掉。
//   I_126 = I_0
//   I_127 = 2 I_125 + 2 I_0           （E_b，注册 127）
//   I_127 = I_125 + I_126 + I_1       （E_a，代入 127 后注册 125）
//   I_128 = I_127 + I_0               （无关，不应被选中）
// 目标 I_127 → I_127 = 2 I_1。
TEST(pyred_forward_backward_derived_integral) {
  std::vector<Eq> system;
  system.push_back(eq({{126, {1, 1}}, {0, {-1, 1}}}));
  system.push_back(eq({{127, {1, 1}}, {125, {-2, 1}}, {126, {-1, 1}}, {0, {-1, 1}}}));
  system.push_back(eq({{127, {1, 1}}, {125, {-1, 1}}, {126, {-1, 1}}, {1, {-1, 1}}}));
  system.push_back(eq({{128, {1, 1}}, {127, {-1, 1}}, {0, {-1, 1}}}));

  std::unordered_map<uint32_t, ResEq> solmap;
  std::unordered_map<uint32_t, uint32_t> eq_index;
  std::unordered_map<uint32_t, std::vector<uint32_t>> fwd_deps, bwd_deps;
  full_solve(system, P, solmap, eq_index, fwd_deps, bwd_deps);

  // I_127 = 2 I_1（后向代入把 I_125 推掉后）
  REQUIRE(solmap.find(127) != solmap.end());
  const ResEq &r127 = solmap.at(127);
  REQUIRE_EQ(r127.size(), 2u);
  REQUIRE_EQ(r127[0].first, 127u);
  REQUIRE_EQ(r127[0].second, P - 1);
  REQUIRE_EQ(r127[1].first, 1u);
  REQUIRE_EQ(r127[1].second, 2u);

  // 依赖记录：fwd_deps[125] = [127, 126]（E_a 代入 127 再 126），
  //           bwd_deps[127] = [125]（后向代入 I_125）。
  REQUIRE(fwd_deps.find(125) != fwd_deps.end());
  REQUIRE_EQ(fwd_deps[125].size(), 2u);
  REQUIRE_EQ(fwd_deps[125][0], 127u);
  REQUIRE_EQ(fwd_deps[125][1], 126u);
  REQUIRE(bwd_deps.find(127) != bwd_deps.end());
  REQUIRE_EQ(bwd_deps[127].size(), 1u);
  REQUIRE_EQ(bwd_deps[127][0], 125u);

  // mark-and-sweep 从 {127} → {127, 126, 125}，排除 128。
  auto needed = mark_sweep({127}, fwd_deps, bwd_deps);
  REQUIRE(needed.count(127));
  REQUIRE(needed.count(126));
  REQUIRE(needed.count(125));
  REQUIRE(!needed.count(128));
  REQUIRE_EQ(needed.size(), 3u);

  // 精简重建（只选所需方程）应与完整结果一致。
  std::vector<uint32_t> needed_eq_idx;
  for (uint32_t w : needed)
    needed_eq_idx.push_back(eq_index.at(w));
  std::vector<ResEq> trim_eqs;
  std::vector<uint32_t> trim_idx;
  for (uint32_t i : needed_eq_idx) {
    ResEq re;
    secred::reduce_mod(system[i], P, re);
    if (!re.empty()) {
      trim_eqs.push_back(std::move(re));
      trim_idx.push_back(i);
    }
  }
  std::unordered_map<uint32_t, ResEq> trim_solmap;
  secred::forward_eliminate(trim_eqs, trim_idx, P, trim_solmap, nullptr, nullptr);
  secred::backward_eliminate(trim_solmap, P, nullptr);

  REQUIRE(trim_solmap.find(127) != trim_solmap.end());
  const ResEq &t127 = trim_solmap.at(127);
  REQUIRE_EQ(t127.size(), 2u);
  REQUIRE_EQ(t127[0].first, 127u);
  REQUIRE_EQ(t127[0].second, P - 1);
  REQUIRE_EQ(t127[1].first, 1u);
  REQUIRE_EQ(t127[1].second, 2u);
}
