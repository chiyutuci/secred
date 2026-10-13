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

// Kira spells m^2 as m2 in the coefficient strings it emits; this rewrites those
// the way a family's own symbols spell them, so a golden coefficient parses.
std::string kira_to_ginac(const std::string &s);

// One accepted symmetry of a sector: the partner sector, the line
// correspondence, and -- when one realises it -- the momentum map, in the
// family's own symbols.  `f` and `c` are what Kira writes to
// sectormappings/<fam>/{relations,symmetries}: with the family's own D symbols
// on both sides,
//     D_k = sum_j f(k,j) D'_j + c_k .
// A prescription out of S is applied to a seed a of S by expanding
// I(a) = prod_k D_k^{-a_k} through those rows (spec 5.1).
struct PropPrescription {
  unsigned from = 0;  // the sector this maps out of
  unsigned to = 0;    // its image sector
  // Family line k of `from` -> line of `to`; -1 for the lines the
  // correspondence does not cover (always the lines outside `from`).
  std::vector<int> ing;
  GiNaC::matrix f;  // nprops x nprops
  GiNaC::matrix c;  // nprops x 1
  // Kira's symDOTS: no momentum map realises this correspondence, so the
  // relation holds only on seeds with no numerator outside `from` (spec 4.6,
  // 5.1).
  bool dot_only = false;

  // The same entries as exact rationals, for the numeric path.  Filled once,
  // before any thread starts; throws when an entry is not a number (spec 5.1).
  std::vector<std::vector<rational>> f_num;
  std::vector<rational> c_num;
  // True when f_num/c_num hold this prescription's rows.  Filled by to_numeric()
  // (throwing) or try_to_numeric() (not); the sector stage reads it to choose
  // between the rational expansion and the GiNaC one, and never reads f_num when
  // it is false (a failed conversion may have filled part of it) -- spec 5.1.
  bool has_numeric_rows = false;
  void to_numeric();
  // Fill the rational rows when every entry is a number, and report whether it
  // worked, without throwing: a symbolic row leaves the flag false and the
  // sector stage keeps the GiNaC expansion for that prescription.
  bool try_to_numeric();

  // Compare the stored rows with Kira-printed cells (nprops coefficients then
  // the constant, one row per family line) as expressions, after the m2
  // rewrite.  Used by the golden comparison of tests/test_symmetry.cpp.
  bool rows_ok(const std::vector<std::vector<std::string>> &rows, unsigned nprops,
               GiNaC::symtab &st) const;
};

// One term of an expanded relation: the image's index vector and its
// coefficient.
template <typename Coeff>
struct SymTerm {
  std::vector<int> powers;
  Coeff coeff;
};

// Sum equal keys of a flat term list in place and drop the terms that cancel to
// zero.  The callers append in arrival order and need the container's own order
// (sorted by key) for the emitted text to be unchanged, so the sort is stable
// and the merge is one linear pass: this replaces a tree insertion per term
// (spec 5.3) and is deterministic because the key order is total.  Only
// `operator<` is required of the key, and `+=`, `is_zero()` of the coefficient.
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

// Spec 5.1.  Expands I(a) = prod_k D_k^{-a_k} through a prescription's rows and
// returns the image terms of  I(a) = sum_r coeff_r I(a'_r)  -- the seed itself
// is not among them; the caller adds it with coefficient -1 (Kira's convention,
// relations.cpp:558-578).
//   * a dot_only prescription applies only to seeds with no numerator outside
//     `from` (returns false otherwise);
//   * a denominator line (a_k > 0) must have a single-term row with no constant
//     (throws std::runtime_error otherwise -- Kira's "linear combination of
//     propagators in the denominator", relations.cpp:514-521);
//   * a numerator line (a_k < 0) expands its row as a polynomial, and every
//     D'_j power it contributes is *subtracted* from the running target powers;
//     a power falling from 1 to 0 clears the line from the image's sector, and
//     the term is dropped when that lowered sector is trivial
//     (relations.cpp:525-545);
//   * terms are merged by index vector and zero coefficients are dropped.
// Defined once in symmetry.cpp and instantiated there for GiNaC::ex (symbolic
// families) and rational (numeric families).
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

// The sector symmetries of one family: the canonical form of each non-trivial
// sector (spec 4.4), the classes those forms define (4.5), and the relations
// emitted from them (4.5-4.6).  Built by Family::init_reduce before any thread
// starts, pruned at the end of Reduce::prepare_sectors, read-only after.
class SymmetryTable {
public:
  // Everything the detector reads off a family, handed over as plain references
  // so the module never sees a Family and the tests can drive it with a
  // hand-built set.
  struct Inputs {
    // _uPoly + _fPoly.  By value: at the call site this is the sum of two
    // members, so a reference member would dangle.
    GiNaC::ex G;
    const std::vector<GiNaC::ex> &propagators;       // _propagators
    const std::vector<GiNaC::ex> &propMomenta;       // _propMomenta
    const std::vector<GiNaC::possymbol> &internals;  // _internals
    const std::vector<GiNaC::possymbol> &externals;  // _externals
    // _spsRules (the family's external scalar products, p_i.p_j = ...).  Needed
    // because the momentum map's shift M^-1 N p is linear in the external
    // momenta, so squaring it produces p.p products that the stored
    // propagators no longer contain; _compute_symanzik reduces them the same
    // way (family.cpp:1212).
    const GiNaC::lst &spsRules;
    const GiNaC::lst &spsFromProps;                  // _spsFromProps
    const std::vector<GiNaC::symbol> &indexSymbols;  // _symIndices (a_1 ... a_n)
    const std::vector<GiNaC::symbol> &propSymbols;   // _symProps (D_1 ... D_n)
    unsigned nprops = 0;
    // True for a family with no symbols left after `_symbols`: its dimension,
    // masses and invariants are GiNaC numbers, so every coefficient converts to
    // `rational` (and no `GiNaC::ex` arithmetic is needed at reduction time).
    bool numeric = false;
  };

  // Sector-restricted canonical form, grouping and emission (spec 4.4-4.6).
  // Called from Family::init_reduce between _search_trivial_sectors and
  // prepare_sectors.  `nonTrivial` is Reduce::_sectors: index = sector id,
  // true = non-trivial.
  static SymmetryTable find(const Inputs &in, const std::vector<bool> &nonTrivial,
                            unsigned top);

  // One correspondence, verified and turned into a prescription -- the seam
  // between the class pairing and the mathematics (spec 4.3, 4.6).  `ing` is
  // source line -> target line, read off two canonical permutations.  Empty
  // when no momentum map realises it; the caller then emits dot_only_entry().
  static std::optional<PropPrescription>
  verify(unsigned from, unsigned to, const std::vector<int> &ing, const Inputs &in);

  // Kira's faked rows for an entry no map realises (spec 4.6): a covered line
  // gets the unit vector at ing[k], an uncovered one the unit vector at the
  // lowest-numbered line outside `to`, and every constant is zero.
  static PropPrescription dot_only_entry(unsigned from, unsigned to,
                                         const std::vector<int> &ing,
                                         const Inputs &in);

  // Canonical key per sector id; empty for a sector that was not considered.
  const std::vector<std::string> &canonical() const { return _canonical; }
  // Number of canonical permutations of a sector (Kira's nrows); 0 if the
  // sector was not considered.
  unsigned nrows(unsigned sector) const;
  // The sector's canonical permutations, each a list of global line numbers.
  const std::vector<std::vector<int>> &permutations(unsigned sector) const;
  // rep(rep(S)) == rep(S); rep(S) == S exactly when S has no cross
  // prescription, which is the test the stage-B scheduler edge uses.
  unsigned representative(unsigned sector) const;
  // Every prescription out of a sector -- its one cross entry, or its self
  // entries, never both.  Empty for a sector that emits nothing.
  const std::vector<PropPrescription> &bySource(unsigned sector) const;
  // Drop every prescription with an endpoint outside `reduced` (spec 4.7); a
  // source left with none reports itself.
  void restrictTo(const std::set<unsigned> &reduced);
  // Drop every cross prescription, keeping the self ones (spec 5.5, `layer`).
  void dropCross();
  // Spec 5.1: give every prescription its rational rows where they exist, so a
  // symbolic family's sector stage can expand with `rational` instead of GiNaC.
  // One-off work, called from Family::init_reduce before any thread starts.
  void try_to_numeric_all();
  // Diagnostic dump of the table (spec 4.1).
  void dump(const std::string &path) const;

private:
  std::vector<std::string> _canonical;                   // per sector id
  std::vector<std::vector<std::vector<int>>> _perms;     // per sector id
  std::vector<unsigned> _representative;                 // per sector id
  std::vector<std::vector<PropPrescription>> _bySource;  // per sector id
};
