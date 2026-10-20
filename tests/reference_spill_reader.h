// The global stage's spill readers as they stood before sub-project B, kept as the
// differential reference: the scanner's tests (tests/test_spill_reader.cpp) read the
// same file with both and compare the systems structurally, and the read bench
// (bench/global_perf/read_bench.cpp) times both on the real spill.
//
// Do not "fix" anything in here. It is a record of the old behaviour, and the point of
// the comparison is that the new reader reproduces it.
#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <ginac/ginac.h>

#include "ffred_solver.h"
#include "sector.h"
#include "sym_coeff.h"

namespace ref {

using secred::Eq;
using ::RawIntegral;   // sector.h declares RawIntegral at global scope
using secred::Rat64;
using secred::SymEq;

// globalreduce.cpp:52-72
inline bool parse_rat64(const std::string &s, int64_t &num, int64_t &den) {
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

// globalreduce.cpp:293-335
inline bool legacy_numeric(const std::string &path,
                           const std::unordered_map<RawIntegral, unsigned> &integral_id,
                           std::vector<Eq> &system, std::uint64_t &terms) {
  std::ifstream in(path);
  std::string line;
  std::vector<std::pair<unsigned, std::string>> eq;
  bool ok = true;
  auto flush = [&]() {
    if (eq.empty())
      return;
    std::sort(eq.begin(), eq.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });
    Eq e;
    e.reserve(eq.size());
    for (auto &[id, s] : eq) {
      int64_t n, d;
      if (!parse_rat64(s, n, d)) {
        std::cerr << "coefficient does not fit in int64: " << s << "\n";
        ok = false;
        return;
      }
      e.push_back({id, Rat64{n, d}});
    }
    terms += e.size();
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
  return ok;
}

// globalreduce.cpp:866-911
inline bool legacy_symbolic(const std::string &path,
                            const std::unordered_map<RawIntegral, unsigned> &integral_id,
                            const std::vector<GiNaC::possymbol> &symbols,
                            std::vector<SymEq> &system, std::uint64_t &terms) {
  GiNaC::symtab symtab;
  std::unordered_map<std::string, unsigned> sym_index;
  for (unsigned i = 0; i < symbols.size(); ++i) {
    symtab[symbols[i].get_name()] = symbols[i];
    sym_index[symbols[i].get_name()] = i;
  }
  GiNaC::parser parser(symtab);

  std::ifstream in(path);
  std::string line;
  std::vector<std::pair<unsigned, GiNaC::ex>> eq;
  bool ok = true;
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
      e.push_back({id, secred::make_sym_coeff(ex, sym_index)});
    }
    terms += e.size();
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
      std::cerr << "cannot parse coefficient '" << line.substr(tab + 1)
                << "': " << ex.what() << "\n";
      ok = false;
      break;
    }
  }
  flush();
  return ok;
}

}  // namespace ref
