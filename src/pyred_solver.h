#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace secred {

// 系数：分子/分母（den>0），从 spill 文件解析得到。
struct Rat64 {
  int64_t num;
  int64_t den;
};

// 一个方程：降序的 (id, 系数) 稠密向量。id 是全局 dense id（0=最简，N-1=最复杂），
// front() = 最大 id = 最复杂项。
using Eq = std::vector<std::pair<uint32_t, Rat64>>;

// 一个素数下的残基方程：降序的 (id, residue mod p)。
using ResEq = std::vector<std::pair<uint32_t, uint64_t>>;

// a 关于素数 p 的模逆（gcd(a,p)=1），用 __int128 扩欧。
uint64_t mod_inverse_u64(uint64_t a, uint64_t p);

// (num/den) mod p，den>0，128 位机器算术。
uint64_t rat_mod_u64(int64_t num, int64_t den, uint64_t p);

// 把 Eq（num/den）约化成 ResEq（mod p 残基），丢弃零系数。
void reduce_mod(const Eq &eq, uint64_t p, ResEq &out);

// 前向消元（pyred `SystemOfEqs::solve` 的 eager 版）：
//   按 front() 升序排方程（最终 tie-break 用 sys_idx，素数无关）；对每个方程，
//   反复代入「最高可约项」（其 id 在 solmap 中有解），再对剩余最高项主元归一
//   （系数 -> -1），注册进 solmap。
// eqs 与 sys_idx 平行：sys_idx[i] = eqs[i] 对应的原始系统方程下标。
// eq_index / fwd_deps 可为 nullptr（重建时不需要）。非空时记录：
//   eq_index[pivot] = 注册该 pivot 的原始方程下标；
//   fwd_deps[pivot] = 该方程代入过的 pivot 序列（依赖，供 mark-and-sweep）。
void forward_eliminate(std::vector<ResEq> &eqs,
                       const std::vector<uint32_t> &sys_idx, uint64_t p,
                       std::unordered_map<uint32_t, ResEq> &solmap,
                       std::unordered_map<uint32_t, uint32_t> *eq_index,
                       std::unordered_map<uint32_t, std::vector<uint32_t>> *fwd_deps);

// 后向代入（pyred `backward()`）：前向消元后，某些「派生积分」的约化式出现得晚，
// 导致早先约化式的 RHS 仍含可约项。按 id 升序重新代入每个约化式 RHS 的可约项，
// 把 RHS 彻底推到主积分。bwd_deps 可为 nullptr；非空时记录每个约化式代入过的 pivot。
// only_keys 非空时只处理这些 pivot（须升序），用于重建时惰性回代（只还原目标的约化链，
// 而非全部约化式）。
void backward_eliminate(std::unordered_map<uint32_t, ResEq> &solmap, uint64_t p,
                        std::unordered_map<uint32_t, std::vector<uint32_t>> *bwd_deps,
                        const std::vector<uint32_t> *only_keys = nullptr);

} // namespace secred
