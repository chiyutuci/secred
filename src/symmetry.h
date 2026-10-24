#pragma once

#include <algorithm>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "ginac/ginac.h"

#include "sector.h"

std::string kira_to_ginac(const std::string &s);

// One verified sector symmetry: source, target and index mapping.
struct PropPrescription {
  unsigned from = 0;
  unsigned to = 0;

  std::vector<int> ing;
  GiNaC::matrix f;
  GiNaC::matrix c;

  bool dot_only = false;

  std::vector<std::vector<rational>> f_num;
  std::vector<rational> c_num;

  bool has_numeric_rows = false;
  void to_numeric();

  bool try_to_numeric();

  bool rows_ok(const std::vector<std::vector<std::string>> &rows, unsigned nprops,
               GiNaC::symtab &st) const;
};

// One term of an expanded prescription.
template <typename Coeff>
struct SymTerm {
  std::vector<int> powers;
  Coeff coeff;
};

// Sum terms with equal keys in place.
template <typename Key, typename Coeff>
void flat_merge(std::vector<std::pair<Key, Coeff>> &v) {
  std::stable_sort(v.begin(), v.end(),
                   [](const std::pair<Key, Coeff> &a, const std::pair<Key, Coeff> &b) {
                     return a.first < b.first;
                   });
  const auto same = [](const Key &a, const Key &b) { return !(a < b) && !(b < a); };
  std::size_t out = 0;
  for (std::size_t i = 0; i < v.size();) {
    std::size_t j = i + 1;
    while (j < v.size() && same(v[i].first, v[j].first)) {
      v[i].second += v[j].second;
      ++j;
    }
    if (!v[i].second.is_zero())
      v[out++] = std::move(v[i]);
    i = j;
  }
  v.resize(out);
}

// Expand a prescription applied to one seed.
template <typename Coeff>
bool expand_prescription(const PropPrescription &p,
                         const std::vector<std::vector<Coeff>> &f,
                         const std::vector<Coeff> &c, const RawIntegral &seed,
                         const std::function<bool(unsigned)> &trivial,
                         std::vector<SymTerm<Coeff>> &out);

extern template bool expand_prescription<GiNaC::ex>(
    const PropPrescription &, const std::vector<std::vector<GiNaC::ex>> &,
    const std::vector<GiNaC::ex> &, const RawIntegral &,
    const std::function<bool(unsigned)> &, std::vector<SymTerm<GiNaC::ex>> &);
extern template bool expand_prescription<rational>(
    const PropPrescription &, const std::vector<std::vector<rational>> &,
    const std::vector<rational> &, const RawIntegral &,
    const std::function<bool(unsigned)> &, std::vector<SymTerm<rational>> &);

// Sector symmetries derived from the propagator list.
class SymmetryTable {
public:

  struct Inputs {

    GiNaC::ex G;
    const std::vector<GiNaC::ex> &propagators;
    const std::vector<GiNaC::ex> &propMomenta;
    const std::vector<GiNaC::possymbol> &internals;
    const std::vector<GiNaC::possymbol> &externals;

    const GiNaC::lst &spsRules;
    const GiNaC::lst &spsFromProps;
    const std::vector<GiNaC::symbol> &indexSymbols;
    const std::vector<GiNaC::symbol> &propSymbols;
    unsigned nprops = 0;

    bool numeric = false;
  };

  static SymmetryTable find(const Inputs &in, const std::vector<bool> &nonTrivial,
                            unsigned top);

  static std::optional<PropPrescription>
  verify(unsigned from, unsigned to, const std::vector<int> &ing, const Inputs &in);

  static PropPrescription dot_only_entry(unsigned from, unsigned to,
                                         const std::vector<int> &ing,
                                         const Inputs &in);

  const std::vector<std::string> &canonical() const { return _canonical; }

  unsigned nrows(unsigned sector) const;

  const std::vector<std::vector<int>> &permutations(unsigned sector) const;

  unsigned representative(unsigned sector) const;

  const std::vector<PropPrescription> &bySource(unsigned sector) const;

  void restrictTo(const std::set<unsigned> &reduced);

  void dropCross();

  void try_to_numeric_all();

private:
  std::vector<std::string> _canonical;
  std::vector<std::vector<std::vector<int>>> _perms;
  std::vector<unsigned> _representative;
  std::vector<std::vector<PropPrescription>> _bySource;
};
