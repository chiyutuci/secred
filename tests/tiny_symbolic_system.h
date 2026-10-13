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

}  // namespace tiny
