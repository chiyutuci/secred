// 进程内全局约化实现:纯数值与带符号两条 FiniteFlow 路径。
#include "globalreduce.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <gmp.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fflow/alg_mp_reconstruction.hh>
#include <fflow/common.hh>
#include <fflow/gcd.hh>
#include <fflow/graph.hh>
#include <fflow/mp_common.hh>
#include <fflow/mp_gcd.hh>
#include <fflow/numeric_solver.hh>
#include <fflow/primes.hh>

#include <cln/cln.h>
#include <ginac/ginac.h>

#include <memory>
#include <random>

#include "pyred_solver.h"

// ---------------------------------------------------------------------------
// 纯数值 FiniteFlow 全局约化(无符号、无 Kira/FireFly、无 dum 哑变量)
// ---------------------------------------------------------------------------

namespace {
namespace ff = fflow;

// 把系数串("num" 或 "num/den")解析成 int64 num/den(den>0)。输入系数是 IBP
// 系数,都是小整数/简单分数,一定能塞进 int64;解析失败(溢出)返回 false。
bool parse_rat64(const std::string &s, int64_t &num, int64_t &den) {
  std::size_t slash = s.find('/');
  try {
    if (slash == std::string::npos) {
      num = std::stoll(s);
      den = 1;
    } else {
      num = std::stoll(s.substr(0, slash));
      den = std::stoll(s.substr(slash + 1));
      if (den < 0) {
        num = -num;
        den = -den;
      }
      if (den == 0)
        return false;
    }
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

// 简单并行 for:将 [0,n) 静态分块给 nthreads 个线程。
template <typename F>
void parallel_for(unsigned nthreads, std::size_t n, F f) {
  if (nthreads <= 1 || n <= 1) {
    for (std::size_t i = 0; i < n; ++i)
      f(i);
    return;
  }
  if (nthreads > n)
    nthreads = static_cast<unsigned>(n);
  std::vector<std::thread> threads;
  threads.reserve(nthreads);
  for (unsigned t = 0; t < nthreads; ++t) {
    std::size_t beg = n * t / nthreads;
    std::size_t end = n * (t + 1) / nthreads;
    threads.emplace_back([&f, beg, end]() {
      for (std::size_t i = beg; i < end; ++i)
        f(i);
    });
  }
  for (auto &th : threads)
    th.join();
}

// 带进度日志 + 多线程的有理数重建:逐素数用 pyred 前向消元得到目标系数,
// 再并行逐系数做 rat_rec / 中国剩余合并。返回 true 表示收敛、false 表示失败。
bool reconstruct_pyred(const std::vector<secred::Eq> &system,
                       const std::vector<uint32_t> &eq_idx,
                       const std::vector<uint32_t> &bwd_chain,
                       const std::vector<unsigned> &dep_targets,
                       const std::vector<unsigned> &masters,
                       const std::unordered_map<unsigned, unsigned> &master_index,
                       unsigned nthreads, unsigned max_primes,
                       ff::MPRational res[], double *elim_seconds) {
  const unsigned nndeps = dep_targets.size();
  const unsigned nnindeps = masters.size();
  const unsigned nparsout = nndeps * nnindeps;

  std::unique_ptr<ff::UInt[]> xout(new ff::UInt[nparsout]);
  std::unique_ptr<ff::MPInt[]> zout(new ff::MPInt[nparsout]);
  ff::MPInt totmod, c1, c2;

  std::size_t prime_i = 0;

  for (std::size_t ip = 0; ip < max_primes + 1; ++ip) {
    ff::Mod mod(ff::BIG_UINT_PRIMES[prime_i]);
    const uint64_t p = mod.n();

    // 本素数下只在 mark-and-sweep 筛出的「所需方程」上做 pyred 前向消元 + 提取目标系数。
    auto t_elim0 = std::chrono::steady_clock::now();
    std::vector<secred::ResEq> res_eqs;
    std::vector<uint32_t> sys_idx;
    res_eqs.reserve(eq_idx.size());
    sys_idx.reserve(eq_idx.size());
    for (uint32_t i : eq_idx) {
      secred::ResEq re;
      secred::reduce_mod(system[i], p, re);
      if (!re.empty()) {
        res_eqs.push_back(std::move(re));
        sys_idx.push_back(i);
      }
    }
    std::unordered_map<uint32_t, secred::ResEq> solmap;
    auto t_fwd0 = std::chrono::steady_clock::now();
    secred::forward_eliminate(res_eqs, sys_idx, p, solmap, nullptr, nullptr);
    auto t_bwd0 = std::chrono::steady_clock::now();
    secred::backward_eliminate(solmap, p, nullptr, &bwd_chain);
    auto t_bwd1 = std::chrono::steady_clock::now();
    double fwd_s = std::chrono::duration<double>(t_bwd0 - t_fwd0).count();
    double bwd_s = std::chrono::duration<double>(t_bwd1 - t_bwd0).count();

    std::fill(xout.get(), xout.get() + nparsout, ff::UInt(0));
    for (unsigned i = 0; i < nndeps; ++i) {
      auto it = solmap.find(dep_targets[i]);
      if (it == solmap.end())
        continue;
      const secred::ResEq &red = it->second;
      for (size_t k = 1; k < red.size(); ++k) {
        auto mit = master_index.find(red[k].first);
        if (mit != master_index.end())
          xout[i * nnindeps + mit->second] = red[k].second;
      }
    }
    auto t_elim1 = std::chrono::steady_clock::now();
    double elim_s = std::chrono::duration<double>(t_elim1 - t_elim0).count();
    if (elim_seconds)
      *elim_seconds += elim_s;

    // CRT（与 algorithm_reconstruct_numeric 一致）。
    auto t_rest0 = std::chrono::steady_clock::now();
    bool converged = false;
    bool failed = false;
    if (ip == 0) {
      totmod = mod.n();
      parallel_for(nthreads, nparsout, [&](unsigned j) {
        if (xout[j]) {
          zout[j] = xout[j];
          res[j] = ff::rat_rec(xout[j], mod.n());
        } else {
          zout[j] = ff::UInt(0);
          res[j] = ff::Int(0);
        }
      });
    } else {
      ff::MPInt this_mod(mod.n());
      std::atomic<bool> failed_at{false};
      std::atomic<bool> consistent{true};
      parallel_for(nthreads, nparsout, [&](unsigned j) {
        if (!consistent.load(std::memory_order_relaxed))
          return;
        ff::MPInt xmp;
        if (!ff::rat_mod(res[j], this_mod, xmp)) {
          failed_at.store(true, std::memory_order_relaxed);
          consistent.store(false, std::memory_order_relaxed);
          return;
        }
        if (xout[j] != xmp.to_uint())
          consistent.store(false, std::memory_order_relaxed);
      });
      failed = failed_at.load();
      if (!failed && consistent.load())
        converged = true;
      else if (!failed && ip == max_primes)
        return false;
      else if (!failed) {
        ff::chinese_remainder_coeffs(totmod, mod.n(), c1, c2, totmod);
        parallel_for(nthreads, nparsout, [&](unsigned j) {
          ff::chinese_remainder_from_coeffs(zout[j], xout[j], c1, c2, totmod,
                                            zout[j]);
          ff::rat_rec(zout[j], totmod, res[j]);
        });
      }
    }
    auto t_rest1 = std::chrono::steady_clock::now();
    double rest_s = std::chrono::duration<double>(t_rest1 - t_rest0).count();

    std::cout << "[global finiteflow] reconstruct: prime " << ip + 1 << "/"
              << max_primes << " evaluated (elimination "
              << format_duration(elim_s) << " [fwd " << format_duration(fwd_s)
              << " + bwd " << format_duration(bwd_s) << "], CRT/rat_rec "
              << format_duration(rest_s) << ")\n"
              << std::flush;

    if (failed)
      return false;
    if (converged) {
      std::cout << "[global finiteflow] reconstruct: converged after "
                << ip + 1 << " prime(s)\n";
      return true;
    }

    prime_i = (prime_i + 1) % ff::BIG_UINT_PRIMES_SIZE;
  }

  return false;
}

} // namespace

GlobalReductionResult reduce_global_finiteflow(
    const std::string &raw_equations_path,
    const std::vector<RawIntegral> &all_integrals,
    const std::unordered_map<RawIntegral, unsigned> &integral_id,
    const std::vector<RawIntegral> &targets,
    unsigned narrowing_success, unsigned narrowing_fail,
    unsigned raw_equations_total, std::size_t global_equations_count,
    unsigned nthreads) {
  GlobalReductionResult res;
  namespace ff = fflow;

  const unsigned nvars = all_integrals.size();
  const auto t0 = std::chrono::steady_clock::now();
  std::cout << "[global finiteflow] reading spill file, building system...\n";

  // 1) Read the spill file → vector<Eq>（降序 (id, num/den) 稠密向量）。
  std::vector<secred::Eq> system;
  bool parse_failed = false;
  {
    std::ifstream in(raw_equations_path);
    std::string line;
    std::vector<std::pair<unsigned, std::string>> eq;
    auto flush = [&]() {
      if (eq.empty())
        return;
      std::sort(eq.begin(), eq.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });
      secred::Eq e;
      e.reserve(eq.size());
      for (auto &[id, s] : eq) {
        int64_t n, d;
        if (!parse_rat64(s, n, d)) {
          std::cerr << "finiteflow global reduction: coefficient does not fit "
                       "in int64: "
                    << s << "\n";
          parse_failed = true;
          return;
        }
        e.push_back({id, secred::Rat64{n, d}});
      }
      system.push_back(std::move(e));
      eq.clear();
    };
    while (std::getline(in, line)) {
      if (line.empty()) {
        flush();
        continue;
      }
      auto tab = line.find('\t');
      if (tab == std::string::npos)
        continue;
      RawIntegral integ = RawIntegral::from_text(line.substr(0, tab));
      auto it = integral_id.find(integ);
      if (it == integral_id.end())
        continue;
      eq.emplace_back(it->second, line.substr(tab + 1));
    }
    flush();
  }

  if (parse_failed) {
    res.ok = false;
    return res;
  }

  const unsigned neqs = system.size();
  if (neqs == 0) {
    std::cerr << "finiteflow global reduction: empty system\n";
    return res;
  }

  const auto t1 = std::chrono::steady_clock::now();
  std::cout << "[global finiteflow] system built: " << neqs << " equations x "
            << nvars << " variables ("
            << format_duration(std::chrono::duration<double>(t1 - t0).count())
            << ")\n";

  // 2) Target ids.
  std::vector<unsigned> target_ids;
  target_ids.reserve(targets.size());
  for (const auto &t : targets) {
    auto it = integral_id.find(t);
    if (it != integral_id.end())
      target_ids.push_back(it->second);
  }
  if (target_ids.empty()) {
    std::cerr << "finiteflow global reduction: no targets\n";
    return res;
  }

  // 3) Learn（一次）：在学习素数上做前向消元 → 主积分 + 结构 + 依赖（供 mark-and-sweep）。
  const uint64_t learn_prime = ff::BIG_UINT_PRIMES[0];
  std::unordered_map<uint32_t, secred::ResEq> learn_solmap;
  std::unordered_map<uint32_t, uint32_t> eq_index; // pivot -> 原始方程下标
  std::unordered_map<uint32_t, std::vector<uint32_t>> fwd_deps, bwd_deps;
  double learn_s = 0.0;
  {
    auto t_learn0 = std::chrono::steady_clock::now();
    std::vector<secred::ResEq> res_eqs;
    std::vector<uint32_t> sys_idx;
    res_eqs.reserve(neqs);
    sys_idx.reserve(neqs);
    for (std::size_t i = 0; i < system.size(); ++i) {
      secred::ResEq re;
      secred::reduce_mod(system[i], learn_prime, re);
      if (!re.empty()) {
        res_eqs.push_back(std::move(re));
        sys_idx.push_back(static_cast<uint32_t>(i));
      }
    }
    secred::forward_eliminate(res_eqs, sys_idx, learn_prime, learn_solmap,
                              &eq_index, &fwd_deps);
    secred::backward_eliminate(learn_solmap, learn_prime, &bwd_deps);
    auto t_learn1 = std::chrono::steady_clock::now();
    learn_s = std::chrono::duration<double>(t_learn1 - t_learn0).count();
  }

  // 主积分 = 各目标约化式右侧积分的并集（约化已递归到底，右侧即主积分）。
  // 不能取「0..nvars-1 中未被解出的 id」：那会把不出现在任何约化式右侧的
  // 孤立变量也算作主积分。
  std::set<unsigned> needed_masters;
  for (unsigned id : target_ids) {
    auto it = learn_solmap.find(id);
    if (it == learn_solmap.end())
      continue; // 目标本身是主积分（无约化式）
    const secred::ResEq &red = it->second;
    for (size_t k = 1; k < red.size(); ++k)
      needed_masters.insert(red[k].first);
  }
  // 主积分按 id 降序（最复杂在前），与原 FiniteFlow 路径 needed_indepvars() 的
  // 顺序一致（升序列 = 降序 id）。
  std::vector<unsigned> masters(needed_masters.rbegin(), needed_masters.rend());
  const unsigned nnindeps = masters.size();

  // 可约目标（在 learn_solmap 中），按用户目标顺序。
  std::vector<unsigned> dep_targets;
  for (unsigned id : target_ids)
    if (learn_solmap.find(id) != learn_solmap.end())
      dep_targets.push_back(id);
  const unsigned nndeps = dep_targets.size();

  std::unordered_map<unsigned, unsigned> master_index; // id -> index j
  for (unsigned j = 0; j < nnindeps; ++j)
    master_index[masters[j]] = j;

  // Mark-and-sweep：从目标出发，沿前向/后向依赖闭包，找出重建所需的原始方程下标。
  // （pyred 用 insertions 数据库 + m_content 做同样的事；此处直接用记录好的依赖。）
  auto sweep = [&](bool with_bwd) {
    std::unordered_set<uint32_t> s(dep_targets.begin(), dep_targets.end());
    std::vector<uint32_t> stack(dep_targets.begin(), dep_targets.end());
    while (!stack.empty()) {
      uint32_t w = stack.back();
      stack.pop_back();
      auto fit = fwd_deps.find(w);
      if (fit != fwd_deps.end())
        for (uint32_t u : fit->second)
          if (s.insert(u).second)
            stack.push_back(u);
      if (with_bwd) {
        auto bit = bwd_deps.find(w);
        if (bit != bwd_deps.end())
          for (uint32_t u : bit->second)
            if (s.insert(u).second)
              stack.push_back(u);
      }
    }
    return s;
  };
  std::unordered_set<uint32_t> needed = sweep(true);
  const std::unordered_set<uint32_t> needed_fwd = sweep(false);
  std::size_t n_bwd_pivots = 0, n_bwd_entries = 0;
  for (const auto &kv : bwd_deps) {
    ++n_bwd_pivots;
    n_bwd_entries += kv.second.size();
  }
  std::size_t n_derived = 0; // 派生积分：注册该 pivot 的方程原始 front > pivot
  for (const auto &kv : eq_index) {
    uint32_t w = kv.first, i = kv.second;
    if (i < system.size() && !system[i].empty() && system[i].front().first > w)
      ++n_derived;
  }

  // 小系统诊断：打印完整约化过程，便于与 Kira 对比发现问题。
  if (neqs <= 1000) {
    std::cerr << "\n===== RAW equations (" << neqs << ") =====\n";
    for (std::size_t i = 0; i < system.size(); ++i) {
      std::cerr << "eq[" << i << "] front=" << system[i].front().first
                << " :";
      for (const auto &t : system[i])
        std::cerr << " " << t.first << "(" << t.second.num << "/" << t.second.den
                  << ")";
      std::cerr << "\n";
    }
    std::cerr << "\n===== forward reductions =====\n";
    for (const auto &kv : eq_index) {
      uint32_t w = kv.first, i = kv.second;
      std::cerr << "pivot " << w << " (eq " << i << ", raw front "
                << system[i].front().first << "):";
      if (fwd_deps.count(w))
        for (uint32_t u : fwd_deps.at(w))
          std::cerr << " dep" << u;
      std::cerr << "\n";
    }
    std::cerr << "\n===== backward deps =====\n";
    for (const auto &kv : bwd_deps) {
      std::cerr << "pivot " << kv.first << " bwd:";
      for (uint32_t u : kv.second)
        std::cerr << " " << u;
      std::cerr << "\n";
    }
    std::cerr << "\n===== needed (fwd-only " << needed_fwd.size()
              << ", bwd-only " << (needed.size() - needed_fwd.size())
              << ") =====\n";
    std::vector<uint32_t> bwd_only;
    for (uint32_t w : needed)
      if (!needed_fwd.count(w))
        bwd_only.push_back(w);
    std::sort(bwd_only.begin(), bwd_only.end());
    for (uint32_t w : bwd_only) {
      std::cerr << "bwd-only pivot " << w << " :";
      auto it = learn_solmap.find(w);
      if (it != learn_solmap.end())
        for (std::size_t k = 1; k < it->second.size(); ++k)
          std::cerr << " " << it->second[k].first << ":" << it->second[k].second;
      std::cerr << "  [ref-by:";
      for (const auto &kv : bwd_deps)
        for (uint32_t u : kv.second)
          if (u == w)
            std::cerr << " " << kv.first;
      std::cerr << "]\n";
    }
  }
  std::vector<uint32_t> needed_eq_idx;
  needed_eq_idx.reserve(needed.size());
  for (uint32_t w : needed) {
    auto it = eq_index.find(w);
    if (it != eq_index.end())
      needed_eq_idx.push_back(it->second);
  }
  std::sort(needed_eq_idx.begin(), needed_eq_idx.end());

  res.selected_equations = static_cast<std::uint64_t>(learn_solmap.size());
  res.raw_equations = neqs;

  std::cout << "[global finiteflow] learn done: " << res.selected_equations
            << " independent equations, " << nnindeps << " masters ("
            << format_duration(learn_s) << ")\n"
            << "[global finiteflow] mark-and-sweep: " << needed_eq_idx.size()
            << " needed equations of " << neqs << " (" << std::fixed
            << std::setprecision(1)
            << (neqs ? 100.0 * needed_eq_idx.size() / neqs : 0.0)
            << " %; fwd-only " << needed_fwd.size() << "; bwd " << n_bwd_pivots
            << " pivots / " << n_bwd_entries << " deps; derived " << n_derived
            << ")\n";

  // Consolidated Reduction Statistics.
  {
    unsigned total = narrowing_success + narrowing_fail;
    double ratio = total ? 100.0 * static_cast<double>(narrowing_success) / total
                         : 0.0;
    std::cout << "\n----------------- Reduction Statistics ------------------\n"
              << "  Probe narrowing:  " << narrowing_success << "/" << narrowing_fail
              << "/" << total << " (" << std::fixed << std::setprecision(1)
              << ratio << " %)\n"
              << "  Raw equations:    " << raw_equations_total << "\n"
              << "  Global equations: " << global_equations_count << "\n"
              << "  Final equations:  " << res.selected_equations << "\n"
              << "  Masters:          " << nnindeps << "\n"
              << "---------------------------------------------------------\n";
  }

  // 惰性回代链：目标 + 沿 bwd 依赖闭包的派生积分。重建时只还原这条链（而非全部约化式），
  // 与 Kira 的 backsubs（~0.1s）一致。
  std::vector<uint32_t> bwd_chain;
  {
    std::unordered_set<uint32_t> chain_set(dep_targets.begin(), dep_targets.end());
    std::vector<uint32_t> stack(dep_targets.begin(), dep_targets.end());
    while (!stack.empty()) {
      uint32_t w = stack.back();
      stack.pop_back();
      auto bit = bwd_deps.find(w);
      if (bit != bwd_deps.end())
        for (uint32_t u : bit->second)
          if (chain_set.insert(u).second)
            stack.push_back(u);
    }
    bwd_chain.assign(chain_set.begin(), chain_set.end());
    std::sort(bwd_chain.begin(), bwd_chain.end());
  }

  // 4) Reconstruct：逐素数前向消元 + CRT。
  const unsigned max_primes = 200;
  std::unique_ptr<ff::MPRational[]> resmp(
      new ff::MPRational[(std::size_t)nndeps * nnindeps]);
  double elim_seconds = 0.0;
  std::cout << "[global finiteflow] reconstruct (rational reconstruction, up to "
            << max_primes << " primes, " << nthreads << " threads)...\n";
  const auto t2 = std::chrono::steady_clock::now();
  bool ok = reconstruct_pyred(system, needed_eq_idx, bwd_chain, dep_targets, masters,
                              master_index, nthreads, max_primes, resmp.get(),
                              &elim_seconds);
  const auto t3 = std::chrono::steady_clock::now();
  if (!ok) {
    std::cerr << "finiteflow global reduction: reconstruct failed\n";
    return res;
  }
  std::cout << "[global finiteflow] reconstruct done ("
            << format_duration(std::chrono::duration<double>(t3 - t2).count())
            << ")\n";

  // 5) Assemble the output（主积分 + 各目标约化）。
  for (unsigned j = 0; j < nnindeps; ++j)
    res.masters.push_back(all_integrals[masters[j]]);

  std::unordered_map<unsigned, unsigned> dep_index; // id -> dep index
  for (unsigned i = 0; i < nndeps; ++i)
    dep_index[dep_targets[i]] = i;

  for (const auto &t : targets) {
    res.reductions.emplace_back(t, std::string("-1"));
    auto it = integral_id.find(t);
    if (it == integral_id.end())
      continue;
    auto dit = dep_index.find(it->second);
    if (dit == dep_index.end())
      continue; // 目标本身是主积分 → 无约化项
    unsigned i = dit->second;
    for (unsigned j = 0; j < nnindeps; ++j) {
      ff::MPRational c = resmp[(std::size_t)i * nnindeps + j];
      if (c.sign() == 0)
        continue;
      std::ostringstream oss;
      oss << c;
      res.reductions.emplace_back(all_integrals[masters[j]], oss.str());
    }
  }

  std::cout << "[timing] global learn: " << format_duration(learn_s)
            << ", reconstruction: "
            << format_duration(std::chrono::duration<double>(t3 - t2).count())
            << "\n";

  res.ok = true;
  return res;
}

// ---------------------------------------------------------------------------
// 符号 FiniteFlow 全局约化（系数为 GiNaC 多项式，消元用 pyred，重建用
// FiniteFlow 的多变量有理函数重建）。
// ---------------------------------------------------------------------------

namespace {

// 符号系数：symbols 上的多项式（有理系数）。消元时在采样点代入符号值得到
// 有理数，再模 p。
struct SymCoeff {
  GiNaC::ex poly;
};
using SymEq = std::vector<std::pair<uint32_t, SymCoeff>>;

// cl_I 模 p 归约到 uint64（结果 ∈ [0,p)）。
uint64_t cl_I_mod_u64(const cln::cl_I &x, uint64_t p) {
  cln::cl_I r = cln::mod(x, cln::cl_I(p));
  if (cln::minusp(r))
    r += cln::cl_I(p);
  return static_cast<uint64_t>(cln::cl_I_to_ulong(r));
}

// 在采样点（subs）求符号系数并模 p 归约。系数是多项式，代入整数采样值后是
// 有理数 numeric；numer()/denom() 给整数分子分母。den ≡ 0 (mod p) 时返回 0
// （与数值路径 rat_mod_u64 一致：标准 IBP 系数分母远小于素数，不会发生）。
uint64_t eval_sym_coeff_mod(const SymCoeff &c, const GiNaC::exmap &subs,
                            uint64_t p) {
  GiNaC::ex val = c.poly.subs(subs);
  if (!is_a<GiNaC::numeric>(val))
    return 0;
  const GiNaC::numeric &nv = GiNaC::ex_to<GiNaC::numeric>(val);
  cln::cl_I num = cln::the<cln::cl_I>(nv.numer().to_cl_N());
  cln::cl_I den = cln::the<cln::cl_I>(nv.denom().to_cl_N());
  uint64_t n = cl_I_mod_u64(num, p);
  uint64_t d = cl_I_mod_u64(den, p);
  if (d == 0)
    return 0;
  uint64_t inv = secred::mod_inverse_u64(d, p);
  return static_cast<uint64_t>((static_cast<__int128>(n) * inv) % p);
}

// 在采样点（subs）把一组符号方程模 p 归约成 ResEq 系统。
void eval_sym_eqs_mod(const std::vector<SymEq> &system,
                      const std::vector<uint32_t> &eq_idx,
                      const GiNaC::exmap &subs, uint64_t p,
                      std::vector<secred::ResEq> &res_eqs,
                      std::vector<uint32_t> &sys_idx) {
  res_eqs.clear();
  sys_idx.clear();
  res_eqs.reserve(eq_idx.size());
  sys_idx.reserve(eq_idx.size());
  for (uint32_t i : eq_idx) {
    const SymEq &eq = system[i];
    secred::ResEq re;
    re.reserve(eq.size());
    for (const auto &[id, c] : eq) {
      uint64_t v = eval_sym_coeff_mod(c, subs, p);
      if (v)
        re.emplace_back(id, v);
    }
    if (!re.empty()) {
      res_eqs.push_back(std::move(re));
      sys_idx.push_back(i);
    }
  }
}

// 供 FiniteFlow 重建驱动的自定义算法：采样点代入符号值 → 模 p 系数 → pyred
// 前向+回代 → 输出目标×主积分系数。
class SymbolicPyRedAlgorithm : public fflow::Algorithm {
public:
  SymbolicPyRedAlgorithm(std::vector<SymEq> system,
                         std::vector<uint32_t> eq_idx,
                         std::vector<uint32_t> bwd_chain,
                         std::vector<unsigned> dep_targets,
                         std::unordered_map<unsigned, unsigned> master_index,
                         unsigned nnindeps, std::vector<GiNaC::ex> symbols)
      : system_(std::move(system)), eq_idx_(std::move(eq_idx)),
        bwd_chain_(std::move(bwd_chain)), dep_targets_(std::move(dep_targets)),
        master_index_(std::move(master_index)), nnindeps_(nnindeps),
        symbols_(std::move(symbols)) {
    nparsin.push_back(static_cast<unsigned>(symbols_.size()));
    nparsout = static_cast<unsigned>(dep_targets_.size() * nnindeps_);
  }

  fflow::Ret evaluate(fflow::Context *, fflow::AlgInput xin[], fflow::Mod mod,
                      fflow::AlgorithmData *,
                      fflow::UInt xout[]) const override {
    const uint64_t p = mod.n();
    const fflow::UInt *x = xin[0];
    const unsigned nvars = symbols_.size();

    // 每采样点构建一次替换表 symbol -> 整数采样值。
    GiNaC::exmap subs;
    for (unsigned i = 0; i < nvars; ++i)
      subs[symbols_[i]] = GiNaC::numeric(x[i]);

    eval_sym_eqs_mod(system_, eq_idx_, subs, p, res_eqs_, sys_idx_);

    solmap_.clear();
    secred::forward_eliminate(res_eqs_, sys_idx_, p, solmap_, nullptr, nullptr);
    secred::backward_eliminate(solmap_, p, nullptr, &bwd_chain_);

    const unsigned nndeps = dep_targets_.size();
    std::fill(xout, xout + nndeps * nnindeps_, fflow::UInt(0));
    for (unsigned i = 0; i < nndeps; ++i) {
      auto it = solmap_.find(dep_targets_[i]);
      if (it == solmap_.end())
        continue;
      const secred::ResEq &red = it->second;
      for (size_t k = 1; k < red.size(); ++k) {
        auto mit = master_index_.find(red[k].first);
        if (mit != master_index_.end())
          xout[i * nnindeps_ + mit->second] = red[k].second;
      }
    }
    return fflow::SUCCESS;
  }

private:
  std::vector<SymEq> system_;
  std::vector<uint32_t> eq_idx_;
  std::vector<uint32_t> bwd_chain_;
  std::vector<unsigned> dep_targets_;
  std::unordered_map<unsigned, unsigned> master_index_;
  unsigned nnindeps_;
  std::vector<GiNaC::ex> symbols_;

  // 单线程复用缓冲（重建在单线程内逐个采样点求值）。
  mutable std::vector<secred::ResEq> res_eqs_;
  mutable std::vector<uint32_t> sys_idx_;
  mutable std::unordered_map<uint32_t, secred::ResEq> solmap_;
};

} // namespace

GlobalReductionResult reduce_global_finiteflow_symbolic(
    const std::string &raw_equations_path,
    const std::vector<RawIntegral> &all_integrals,
    const std::unordered_map<RawIntegral, unsigned> &integral_id,
    const std::vector<RawIntegral> &targets,
    const std::vector<GiNaC::possymbol> &symbols,
    unsigned narrowing_success, unsigned narrowing_fail,
    unsigned raw_equations_total, std::size_t global_equations_count,
    unsigned nthreads) {
  GlobalReductionResult res;
  namespace ff = fflow;

  (void)nthreads; // 重建目前单线程（可后续按输出系数并行化）

  const unsigned nvars = all_integrals.size();
  const unsigned nparams = symbols.size();
  const auto t0 = std::chrono::steady_clock::now();
  std::cout << "[global finiteflow] reading spill file, building symbolic "
               "system...\n";

  // 符号表（name -> symbol），解析系数串。
  GiNaC::symtab symtab;
  for (const auto &s : symbols)
    symtab[s.get_name()] = s;
  GiNaC::parser parser(symtab);

  // 1) 读 spill 文件 → vector<SymEq>（降序 (id, 符号多项式)）。
  std::vector<SymEq> system;
  bool parse_failed = false;
  {
    std::ifstream in(raw_equations_path);
    std::string line;
    std::vector<std::pair<unsigned, GiNaC::ex>> eq;
    auto flush = [&]() {
      if (eq.empty())
        return;
      std::sort(eq.begin(), eq.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });
      SymEq e;
      e.reserve(eq.size());
      for (auto &[id, ex] : eq) {
        ex = ex.expand();
        if (ex.is_zero())
          continue;
        e.push_back({id, SymCoeff{std::move(ex)}});
      }
      system.push_back(std::move(e));
      eq.clear();
    };
    while (std::getline(in, line)) {
      if (line.empty()) {
        flush();
        continue;
      }
      auto tab = line.find('\t');
      if (tab == std::string::npos)
        continue;
      RawIntegral integ = RawIntegral::from_text(line.substr(0, tab));
      auto it = integral_id.find(integ);
      if (it == integral_id.end())
        continue;
      try {
        eq.emplace_back(it->second, parser(line.substr(tab + 1)));
      } catch (const std::exception &ex) {
        std::cerr << "finiteflow symbolic global reduction: cannot parse "
                     "coefficient '"
                  << line.substr(tab + 1) << "': " << ex.what() << "\n";
        parse_failed = true;
        break;
      }
    }
    flush();
  }
  if (parse_failed) {
    res.ok = false;
    return res;
  }
  const unsigned neqs = system.size();
  if (neqs == 0) {
    std::cerr << "finiteflow symbolic global reduction: empty system\n";
    return res;
  }
  const auto t1 = std::chrono::steady_clock::now();
  std::cout << "[global finiteflow] symbolic system built: " << neqs
            << " equations x " << nvars << " variables, " << nparams
            << " parameters ("
            << format_duration(std::chrono::duration<double>(t1 - t0).count())
            << ")\n";

  // 2) Target ids.
  std::vector<unsigned> target_ids;
  target_ids.reserve(targets.size());
  for (const auto &t : targets) {
    auto it = integral_id.find(t);
    if (it != integral_id.end())
      target_ids.push_back(it->second);
  }
  if (target_ids.empty()) {
    std::cerr << "finiteflow symbolic global reduction: no targets\n";
    return res;
  }

  // 3) Learn：在一个通用采样点、学习素数上做前向+回代 → 主积分 + 结构 + 依赖。
  const uint64_t learn_prime = ff::BIG_UINT_PRIMES[0];
  std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
  std::vector<ff::UInt> learn_pt(nparams);
  for (unsigned i = 0; i < nparams; ++i)
    learn_pt[i] = rng() % learn_prime;
  GiNaC::exmap learn_subs;
  for (unsigned i = 0; i < nparams; ++i)
    learn_subs[GiNaC::ex(symbols[i])] = GiNaC::numeric(learn_pt[i]);

  std::vector<secred::ResEq> learn_res_eqs;
  std::vector<uint32_t> learn_sys_idx;
  {
    std::vector<uint32_t> all_idx(neqs);
    for (unsigned i = 0; i < neqs; ++i)
      all_idx[i] = i;
    eval_sym_eqs_mod(system, all_idx, learn_subs, learn_prime, learn_res_eqs,
                     learn_sys_idx);
  }

  std::unordered_map<uint32_t, secred::ResEq> learn_solmap;
  std::unordered_map<uint32_t, uint32_t> eq_index;
  std::unordered_map<uint32_t, std::vector<uint32_t>> fwd_deps, bwd_deps;
  double learn_s = 0.0;
  {
    auto t_learn0 = std::chrono::steady_clock::now();
    secred::forward_eliminate(learn_res_eqs, learn_sys_idx, learn_prime,
                              learn_solmap, &eq_index, &fwd_deps);
    secred::backward_eliminate(learn_solmap, learn_prime, &bwd_deps);
    auto t_learn1 = std::chrono::steady_clock::now();
    learn_s = std::chrono::duration<double>(t_learn1 - t_learn0).count();
  }

  // 主积分 = 各目标约化式右侧积分并集。
  std::set<unsigned> needed_masters;
  for (unsigned id : target_ids) {
    auto it = learn_solmap.find(id);
    if (it == learn_solmap.end())
      continue;
    const secred::ResEq &red = it->second;
    for (size_t k = 1; k < red.size(); ++k)
      needed_masters.insert(red[k].first);
  }
  std::vector<unsigned> masters(needed_masters.rbegin(), needed_masters.rend());
  const unsigned nnindeps = masters.size();

  std::vector<unsigned> dep_targets;
  for (unsigned id : target_ids)
    if (learn_solmap.find(id) != learn_solmap.end())
      dep_targets.push_back(id);
  const unsigned nndeps = dep_targets.size();

  std::unordered_map<unsigned, unsigned> master_index;
  for (unsigned j = 0; j < nnindeps; ++j)
    master_index[masters[j]] = j;

  // Mark-and-sweep：从目标出发沿前向/后向依赖闭包找所需原始方程。
  auto sweep = [&](bool with_bwd) {
    std::unordered_set<uint32_t> s(dep_targets.begin(), dep_targets.end());
    std::vector<uint32_t> stack(dep_targets.begin(), dep_targets.end());
    while (!stack.empty()) {
      uint32_t w = stack.back();
      stack.pop_back();
      auto fit = fwd_deps.find(w);
      if (fit != fwd_deps.end())
        for (uint32_t u : fit->second)
          if (s.insert(u).second)
            stack.push_back(u);
      if (with_bwd) {
        auto bit = bwd_deps.find(w);
        if (bit != bwd_deps.end())
          for (uint32_t u : bit->second)
            if (s.insert(u).second)
              stack.push_back(u);
      }
    }
    return s;
  };
  std::unordered_set<uint32_t> needed = sweep(true);
  std::vector<uint32_t> needed_eq_idx;
  needed_eq_idx.reserve(needed.size());
  for (uint32_t w : needed) {
    auto it = eq_index.find(w);
    if (it != eq_index.end())
      needed_eq_idx.push_back(it->second);
  }
  std::sort(needed_eq_idx.begin(), needed_eq_idx.end());

  res.selected_equations = static_cast<std::uint64_t>(learn_solmap.size());
  res.raw_equations = neqs;

  std::cout << "[global finiteflow] learn done: " << res.selected_equations
            << " independent equations, " << nnindeps << " masters ("
            << format_duration(learn_s) << "); mark-and-sweep kept "
            << needed_eq_idx.size() << " of " << neqs << " equations\n";

  // Consolidated Reduction Statistics（与数值路径一致）。
  {
    unsigned total = narrowing_success + narrowing_fail;
    double ratio = total ? 100.0 * static_cast<double>(narrowing_success) / total
                         : 0.0;
    std::cout << "\n----------------- Reduction Statistics ------------------\n"
              << "  Probe narrowing:  " << narrowing_success << "/" << narrowing_fail
              << "/" << total << " (" << std::fixed << std::setprecision(1)
              << ratio << " %)\n"
              << "  Raw equations:    " << raw_equations_total << "\n"
              << "  Global equations: " << global_equations_count << "\n"
              << "  Final equations:  " << res.selected_equations << "\n"
              << "  Masters:          " << nnindeps << "\n"
              << "---------------------------------------------------------\n";
  }

  // 惰性回代链（目标 + 沿 bwd 依赖闭包的派生积分）。
  std::vector<uint32_t> bwd_chain;
  {
    std::unordered_set<uint32_t> chain_set(dep_targets.begin(),
                                           dep_targets.end());
    std::vector<uint32_t> stack(dep_targets.begin(), dep_targets.end());
    while (!stack.empty()) {
      uint32_t w = stack.back();
      stack.pop_back();
      auto bit = bwd_deps.find(w);
      if (bit != bwd_deps.end())
        for (uint32_t u : bit->second)
          if (chain_set.insert(u).second)
            stack.push_back(u);
    }
    bwd_chain.assign(chain_set.begin(), chain_set.end());
    std::sort(bwd_chain.begin(), bwd_chain.end());
  }

  // 4) 构造算法 + FiniteFlow 多变量有理函数重建。
  std::vector<GiNaC::ex> symbol_ex;
  symbol_ex.reserve(nparams);
  for (const auto &s : symbols)
    symbol_ex.push_back(GiNaC::ex(s));

  SymbolicPyRedAlgorithm alg(std::move(system), std::move(needed_eq_idx),
                             std::move(bwd_chain), dep_targets,
                             std::move(master_index), nnindeps,
                             std::move(symbol_ex));

  const unsigned nparsout = alg.nparsout;
  std::cout << "[global finiteflow] reconstruct (rational-function, "
            << nparsout << " outputs = " << nndeps << " targets x " << nnindeps
            << " masters)...\n";

  ff::ReconstructionOptions opt;
  opt.max_primes = 5; // 系数重建 CRT 素数上限

  // 默认 shift（同 FiniteFlow interface 的 set_default_shift），避免多变量
  // 有理函数在无 shift 的默认采样点触发 BAD_SHIFT。
  std::unique_ptr<ff::UInt[]> shift(new ff::UInt[nparams]);
  {
    ff::Mod smod(ff::BIG_UINT_PRIMES[opt.start_mod]);
    for (unsigned i = 0; i < nparams; ++i)
      shift[i] = ff::sample_uint(ff::OFFSET_4, i, smod);
  }

  std::unique_ptr<unsigned[]> numdeg(new unsigned[nparsout]);
  std::unique_ptr<unsigned[]> dendeg(new unsigned[nparsout]);
  std::unique_ptr<ff::RatFunVarDegrees[]> vdegs(
      new ff::RatFunVarDegrees[nparsout]);
  for (unsigned i = 0; i < nparsout; ++i)
    vdegs[i].resize(nparams);

  ff::Ret ret = ff::algorithm_get_degrees(alg, nullptr, nullptr, shift.get(),
                                          opt, numdeg.get(), dendeg.get());
  if (ret != ff::SUCCESS) {
    std::cerr << "finiteflow symbolic: degree estimation failed (ret " << ret
              << ")\n";
    return res;
  }
  for (unsigned var = 0; var < nparams; ++var) {
    ret = ff::algorithm_get_var_degrees(alg, nullptr, nullptr, var, shift.get(),
                                        opt, vdegs.get());
    if (ret != ff::SUCCESS) {
      std::cerr << "finiteflow symbolic: var-degree estimation failed (ret "
                << ret << ")\n";
      return res;
    }
  }

  ff::GenerateSamplePoints sfun(nparams);
  ff::algorithm_generate_sample_points(sfun, nparams, nparsout, shift.get(),
                                       opt, vdegs.get(), numdeg.get(),
                                       dendeg.get());
  ff::SamplePointsVector sample_points;
  sfun.append_to_vector(sample_points);
  ff::sort_by_mod(sample_points.data(),
                  sample_points.data() + sample_points.size(), nparams);
  std::cout << "[global finiteflow] " << sample_points.size()
            << " sample points to evaluate\n";

  ff::UIntCache cache;
  cache.init(nparams + 1);
  const auto t2 = std::chrono::steady_clock::now();
  ff::evaluate_and_cache_samples(alg, nullptr, nullptr, sample_points.data(),
                                 sample_points.data() + sample_points.size(),
                                 cache);
  const auto t3 = std::chrono::steady_clock::now();
  std::cout << "[global finiteflow] evaluated " << cache.size()
            << " sample points ("
            << format_duration(std::chrono::duration<double>(t3 - t2).count())
            << ")\n";

  std::unique_ptr<ff::MPReconstructedRatFun[]> recf(
      new ff::MPReconstructedRatFun[nparsout]);
  for (unsigned idx = 0; idx < nparsout; ++idx) {
    ret = ff::algorithm_reconstruct(cache, nparams, nparsout, idx, shift.get(),
                                    opt, vdegs.get(), numdeg.get(), dendeg.get(),
                                    recf[idx]);
    if (ret != ff::SUCCESS) {
      std::cerr << "finiteflow symbolic: reconstruction failed for output "
                << idx << " (ret " << ret << ")\n";
      return res;
    }
  }

  // 5) 组装输出。
  std::vector<std::string> varnames(nparams);
  for (unsigned i = 0; i < nparams; ++i)
    varnames[i] = symbols[i].get_name();

  for (unsigned j = 0; j < nnindeps; ++j)
    res.masters.push_back(all_integrals[masters[j]]);

  std::unordered_map<unsigned, unsigned> dep_index;
  for (unsigned i = 0; i < nndeps; ++i)
    dep_index[dep_targets[i]] = i;

  for (const auto &t : targets) {
    res.reductions.emplace_back(t, std::string("-1"));
    auto it = integral_id.find(t);
    if (it == integral_id.end())
      continue;
    auto dit = dep_index.find(it->second);
    if (dit == dep_index.end())
      continue; // 目标本身是主积分
    unsigned i = dit->second;
    for (unsigned j = 0; j < nnindeps; ++j) {
      const ff::MPReconstructedRatFun &r = recf[(std::size_t)i * nnindeps + j];
      if (r.numerator().size() == 0)
        continue; // 零系数
      res.reductions.emplace_back(all_integrals[masters[j]],
                                  r.to_str(varnames.data()));
    }
  }

  std::cout << "[timing] symbolic global learn: " << format_duration(learn_s)
            << ", reconstruction: "
            << format_duration(std::chrono::duration<double>(t3 - t2).count())
            << "\n";

  res.ok = true;
  return res;
}
