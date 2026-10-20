// One tiny symbolic system for the FireFly tests: 4 integrals, 3 equations, one master
// (id 0) and one target (id 3), whose reduction is the rational function
// 2(d-2)/s + (d-4)/(d*s) - 3 in (d, s).
#pragma once

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "sector.h"
#include "symbolic_ffred.h"

namespace tiny {

inline std::vector<RawIntegral> integrals() {
  return {RawIntegral(std::vector<int>{0, 0}), RawIntegral(std::vector<int>{1, 0}),
          RawIntegral(std::vector<int>{0, 1}), RawIntegral(std::vector<int>{1, 1})};
}

inline std::unordered_map<RawIntegral, unsigned> integral_ids() {
  std::unordered_map<RawIntegral, unsigned> ids;
  const auto all = integrals();
  for (unsigned i = 0; i < all.size(); ++i)
    ids.emplace(all[i], i);
  return ids;
}

inline std::vector<GiNaC::possymbol> symbols() {
  return {GiNaC::possymbol("d"), GiNaC::possymbol("s")};
}

// The spill the two backends read: the text is exactly what print_row_poly writes.
inline std::string write_spill(const std::string &path) {
  std::ofstream out(path, std::ios::trunc);
  out << "[1, 1]\t1\n[1, 0]\t(-2+d)/d\n[0, 1]\t1/d\n[0, 0]\t3\n\n"
      << "[0, 1]\t1\n[0, 0]\t(d-4)/s\n\n"
      << "[1, 0]\t1\n[0, 0]\t(2*d)/s\n\n";
  out.close();
  return path;
}

// The same system with s = d + 2: one symbol, which drives the FiniteFlow
// symbolic path's univariate loop when the tests call it directly (no symbol
// count selects that engine now).  The reduction of [1, 1] is the two-symbol
// one with the same substitution.
inline std::vector<GiNaC::possymbol> univariate_symbols() {
  return {GiNaC::possymbol("d")};
}

inline std::string write_univariate_spill(const std::string &path) {
  std::ofstream out(path, std::ios::trunc);
  out << "[1, 1]\t1\n[1, 0]\t(-2+d)/d\n[0, 1]\t1/d\n[0, 0]\t3\n\n"
      << "[0, 1]\t1\n[0, 0]\t(d-4)/(d+2)\n\n"
      << "[1, 0]\t1\n[0, 0]\t(2*d)/(d+2)\n\n";
  out.close();
  return path;
}

// The same substitution, but the target equation carries no 1/d: the reduction
// is 2(d-2)/(d+2) + (d-4)/(d+2) - 3 = -14/(d+2), whose denominator has a
// non-zero constant term, so the univariate linear reconstruction applies at
// every prime instead of falling back to Thiele.
inline std::string write_univariate_polynomial_spill(const std::string &path) {
  std::ofstream out(path, std::ios::trunc);
  out << "[1, 1]\t1\n[1, 0]\t(-2+d)/d\n[0, 1]\t1\n[0, 0]\t3\n\n"
      << "[0, 1]\t1\n[0, 0]\t(d-4)/(d+2)\n\n"
      << "[1, 0]\t1\n[0, 0]\t(2*d)/(d+2)\n\n";
  out.close();
  return path;
}

// The polynomial spill with one coefficient scaled past 2^63: a single
// reconstruction prime cannot determine it, so the univariate loop runs at
// least to prime 2, where the remaining primes are sampled in a batch.
inline std::string write_univariate_big_coeff_spill(const std::string &path) {
  std::ofstream out(path, std::ios::trunc);
  out << "[1, 1]\t1\n[1, 0]\t(-2+d)/d\n[0, 1]\t1\n[0, 0]\t3\n\n"
      << "[0, 1]\t1\n[0, 0]\t(d-4)/(d+2)\n\n"
      << "[1, 0]\t1\n[0, 0]\t(2*d*1180591620717411303427)/(d+2)\n\n";
  out.close();
  return path;
}

}  // namespace tiny
