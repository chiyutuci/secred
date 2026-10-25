#pragma once

#include <algorithm>
#include <iostream>
#include <map>

#include "arith/ffint.h"
#include "arith/numeric.h"
#include "arith/symbolic.h"
#include "utils.h"

#include "fflow/alg_functions.hh"
#include "ginac/ginac.h"

using namespace fflow;

// A sparse equation over finite-field coefficients.
class EquationFF {
public:
  unsigned operator[](unsigned i) const { return _eq[i].first; }

  unsigned &operator[](unsigned i) { return _eq[i].first; }

  [[nodiscard]] ffint coeff(unsigned i) const { return _eq[i].second; }

  void insert(unsigned integral, const ffint &coeff) {
    _eq.emplace_back(integral, coeff);
  }

  [[nodiscard]] unsigned size() const { return _eq.size(); }

  [[nodiscard]] unsigned first_integral() const { return _eq[0].first; }

  ffint first_coeff() { return _eq[0].second; }

  [[nodiscard]] bool empty() const { return _eq.empty(); }

  void sort() {
    std::sort(_eq.begin(), _eq.end(),
              [](const std::pair<unsigned, ffint> &i,
                 const std::pair<unsigned, ffint> &j) {
                return i.first < j.first;
              });
  }

  friend std::ostream &operator<<(std::ostream &out, const EquationFF &eq) {
    out << eq._eq;
    return out;
  }

private:
  std::vector<std::pair<unsigned, ffint>> _eq;
};

// A sparse equation over polynomial coefficients.
class EquationM {
public:
  unsigned operator[](unsigned i) const { return _eq[i].first; }

  unsigned &operator[](unsigned i) { return _eq[i].first; }

  [[nodiscard]] mpoly coeff(unsigned i) const { return _eq[i].second; }

  void insert(unsigned integral, const mpoly &coeff) {
    _eq.emplace_back(integral, coeff);
  }

  [[nodiscard]] unsigned size() const { return _eq.size(); }

  [[nodiscard]] bool empty() const { return _eq.empty(); }

  void sort() {
    std::sort(
        _eq.begin(), _eq.end(),
        [](const std::pair<unsigned, mpoly> &i,
           const std::pair<unsigned, mpoly> &j) { return i.first < j.first; });
  }

  friend std::ostream &operator<<(std::ostream &out, const EquationM &eq) {
    out << eq._eq;
    return out;
  }

private:
  std::vector<std::pair<unsigned, mpoly>> _eq;
};
