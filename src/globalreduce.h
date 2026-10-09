// 进程内全局约化:纯数值与带符号两条 FiniteFlow 路径。
// 两者都先用自带的模算术高斯消元(pyred_solver)选出独立方程,
// 再交给 FiniteFlow 做系数重建;不依赖任何外部工具。
#pragma once

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ginac/ginac.h>

#include "sector.h" // RawIntegral, GlobalEquation

// 将秒数格式化为 "XhYminZs" / "XminYs" / "Xs" 的紧凑时长字符串。
inline std::string format_duration(double seconds) {
  using ull = unsigned long long;
  ull h = static_cast<ull>(seconds) / 3600;
  ull m = (static_cast<ull>(seconds) % 3600) / 60;
  double s = seconds - static_cast<double>(h) * 3600.0 -
             static_cast<double>(m) * 60.0;
  std::ostringstream oss;
  if (h)
    oss << h << "h" << m << "min" << static_cast<ull>(s) << "s";
  else if (m)
    oss << m << "min" << static_cast<ull>(s) << "s";
  else
    oss << std::fixed << std::setprecision(1) << s << "s";
  return oss.str();
}

struct GlobalReductionResult {
  std::vector<RawIntegral> masters;              // 主积分
  std::vector<std::pair<RawIntegral, std::string>> reductions; // 目标约化(符号系数串,每项一行)
  std::uint64_t raw_equations = 0;               // 送入全局系统的方程数
  std::uint64_t selected_equations = 0;          // select 选中的方程数
  bool ok = false;
};

// 纯数值(无符号)进程内全局约化:直接用 FiniteFlow 的稀疏求解器消元 +
// algorithm_reconstruct_numeric 做有理数重建,无需 Kira 与 dum 哑变量。
// 输入 raw_equations_path 为落盘方程文件(<RawIntegral>\t<coeff> 每行、空行分
// 隔方程);all_integrals / integral_id 为全局积分排序与编号;targets 为目标积分。
// 末尾四个统计量用于在 learn 之后、重建之前打印统一的 Reduction Statistics
// 汇总(探测趟收窄 + learn 后方程数 + 最终约化统计)。
GlobalReductionResult reduce_global_finiteflow(
    const std::string &raw_equations_path,
    const std::vector<RawIntegral> &all_integrals,
    const std::unordered_map<RawIntegral, unsigned> &integral_id,
    const std::vector<RawIntegral> &targets,
    unsigned narrowing_success, unsigned narrowing_fail,
    unsigned raw_equations_total, std::size_t global_equations_count,
    unsigned nthreads = 64);

// 符号(带符号参数)进程内全局约化:系数是 symbols 中的 GiNaC 多项式,消元用
// pyred(与纯数值路径同一套优化),系数重建用 FiniteFlow 的多变量有理函数
// 重建(algorithm_reconstruct)。symbols 为有理函数的变量(d + 运动学不变量),
// 顺序即 FiniteFlow 重建的 nparsin 变量顺序。
GlobalReductionResult reduce_global_finiteflow_symbolic(
    const std::string &raw_equations_path,
    const std::vector<RawIntegral> &all_integrals,
    const std::unordered_map<RawIntegral, unsigned> &integral_id,
    const std::vector<RawIntegral> &targets,
    const std::vector<GiNaC::possymbol> &symbols,
    unsigned narrowing_success, unsigned narrowing_fail,
    unsigned raw_equations_total, std::size_t global_equations_count,
    unsigned nthreads = 64);
