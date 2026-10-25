#include "family_model.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "prime_subs.h"
#include "utils.h"

// Collect the distinct free symbols of an expression, in the order they first
// appear.  Only symbols are collected: constants such as Pi are not parameters.
static void collect_symbol_names(const GiNaC::ex &e,
                                 std::vector<std::string> &names) {
  if (GiNaC::is_a<GiNaC::symbol>(e)) {
    const std::string name = GiNaC::ex_to<GiNaC::symbol>(e).get_name();
    if (std::find(names.begin(), names.end(), name) == names.end())
      names.push_back(name);
    return;
  }
  for (std::size_t i = 0; i < e.nops(); ++i)
    collect_symbol_names(e.op(i), names);
}

namespace {

// Fraction-free (Bareiss) rank over the integers.  The matrices here have at
// most one row per monomial of U+F and at most 17 columns (16 propagators plus
// the right-hand side); every intermediate entry is a minor of the input, so
// 128-bit arithmetic holds the products with a wide margin.
int integer_rank(std::vector<std::vector<__int128_t>> m) {
  if (m.empty())
    return 0;
  const std::size_t rows = m.size();
  const std::size_t cols = m.front().size();
  std::size_t r = 0;
  __int128_t prev = 1;
  for (std::size_t c = 0; c < cols && r < rows; ++c) {
    std::size_t pivot = r;
    while (pivot < rows && m[pivot][c] == 0)
      ++pivot;
    if (pivot == rows)
      continue;
    std::swap(m[r], m[pivot]);
    for (std::size_t i = r + 1; i < rows; ++i) {
      for (std::size_t j = c + 1; j < cols; ++j)
        m[i][j] = (m[i][j] * m[r][c] - m[i][c] * m[r][j]) / prev;
      m[i][c] = 0;
    }
    prev = m[r][c];
    ++r;
  }
  return static_cast<int>(r);
}

// Is `k . e == 1` solvable over Q for every exponent vector e in `rows`?
// That is rank(A) == rank([A|1]).  With no rows the system is empty (GiNaC's
// lsolve returns the free solution), hence consistent.
bool equations_consistent(const std::vector<std::vector<unsigned>> &rows,
                          unsigned ncols) {
  std::vector<std::vector<__int128_t>> a;
  std::vector<std::vector<__int128_t>> aug;
  a.reserve(rows.size());
  aug.reserve(rows.size());
  for (const std::vector<unsigned> &e : rows) {
    std::vector<__int128_t> ar(ncols, 0);
    std::vector<__int128_t> gr(ncols + 1, 0);
    for (unsigned i = 0; i < ncols; ++i) {
      ar[i] = e[i];
      gr[i] = e[i];
    }
    gr[ncols] = 1;
    a.push_back(std::move(ar));
    aug.push_back(std::move(gr));
  }
  return integer_rank(std::move(a)) == integer_rank(std::move(aug));
}

} // namespace

// Read the family node into the model.
void FamilyModel::parse(const YAML::Node &family_config, GiNaC::symtab &st) {

  _name = family_config["name"].as<std::string>();

  // The dimension may be a rational number or an expression in one symbol
  // (d = 4-2*eps).  The expression's symbol becomes a symbolic parameter, like
  // an invariant: the numeric path substitutes a prime for it and the symbolic
  // path reconstructs the dependence.  More than one symbol has no meaning.
  if (family_config["dimension"]) {
    const std::string text = family_config["dimension"].as<std::string>();
    GiNaC::parser parser;
    const GiNaC::ex parsed = parser(text);
    std::vector<std::string> names;
    collect_symbol_names(parsed, names);
    if (names.empty()) {
      if (!GiNaC::is_a<GiNaC::numeric>(parsed) ||
          !parsed.info(GiNaC::info_flags::rational))
        throw std::runtime_error(
            "dimension must be a rational number or an expression in one "
            "symbol: " + text);
      _dimension = parsed;
    } else if (names.size() == 1) {
      if (st.contains(names[0]))
        throw std::runtime_error("symbol " + names[0] +
                                 " defined more than once");
      _symbols.emplace_back(names[0]);
      st[names[0]] = _symbols.back();
      // Re-read through the table so the expression uses the registered
      // possymbol, not the plain symbol the default parser made.
      GiNaC::parser reader(st);
      _dimension = reader(text);
    } else {
      std::string list;
      for (std::size_t i = 0; i < names.size(); ++i) {
        if (i)
          list += ", ";
        list += names[i];
      }
      throw std::runtime_error(
          "dimension must be a rational number or an expression in one "
          "symbol: " + text + " (symbols: " + list + ")");
    }
  } else {
    _dimension = GiNaC::possymbol("d");
    _symbols.emplace_back("d");
    _dimension = _symbols.back();
    st["d"] = _dimension;
  }

  for (const auto &sym :
       family_config["internals"].as<std::vector<std::string>>()) {
    if (st.contains(sym))
      throw std::runtime_error("symbol " + sym + " defined more than once");
    _internals.emplace_back(sym);
    st[sym] = _internals.back();
  }

  for (const auto &sym :
       family_config["externals"].as<std::vector<std::string>>()) {
    if (st.contains(sym))
      throw std::runtime_error("symbol " + sym + " defined more than once");
    _externals.emplace_back(sym);
    st[sym] = _externals.back();
  }

  if (family_config["invariants"] && !family_config["invariants"].IsNull()) {
    std::string one;
    if (family_config["invar_one"] && !family_config["invar_one"].IsNull())
      one = family_config["invar_one"].as<std::string>();
    for (const auto &inv :
         family_config["invariants"]
             .as<std::vector<std::pair<std::string, unsigned>>>()) {
      if (st.contains(inv.first))
        throw std::runtime_error("symbol " + inv.first +
                                 " defined more than once");
      _invariants.emplace_back(inv.first, inv.second);
      st[inv.first] = _invariants.back().first;
      if (inv.first != one)
        _symbols.emplace_back(_invariants.back().first);
    }
  }

  // The numeric IBP generator substitutes a prime for each symbol by position,
  // and it has kNumPrimeSubs primes.  A symbol past that limit would survive
  // into the coefficient text and be parsed as a number, so the config is
  // refused here, with the names, before any reduction starts.
  if (_symbols.size() > kNumPrimeSubs) {
    std::string names;
    for (std::size_t i = 0; i < _symbols.size(); ++i) {
      if (i)
        names += ", ";
      names += _symbols[i].get_name();
    }
    throw std::runtime_error(
        "family has " + std::to_string(_symbols.size()) +
        " symbolic parameters but at most " + std::to_string(kNumPrimeSubs) +
        " are supported: " + names);
  }

  if (family_config["invar_one"] && !family_config["invar_one"].IsNull()) {
    auto one = family_config["invar_one"].as<std::string>();
    if (!st.contains(one))
      throw std::runtime_error("the invariant set to one is not valid");
    _one.append(st[one] == 1);
  }

  GiNaC::parser reader(st);

  _nints = _internals.size();
  _nexts = _externals.size();
  _nprops = _nints * _nexts + _nints * (_nints + 1) / 2;
  unsigned nsps = _nexts * (_nexts + 1) / 2;

  if (family_config["sps_rules"].size() != nsps)
    throw std::runtime_error(
        "number of scalar product rules does not match the topology");
  for (const auto &sp :
       family_config["sps_rules"]
           .as<std::vector<std::vector<std::string>>>()) {
    if (sp.size() != 3 || !st.contains(sp[0]) || !st.contains(sp[1]))
      throw std::runtime_error(
          "scalar product rules contains invalid expressions");
    _spsRules.append(reader(sp[0]) * reader(sp[1]) ==
                     reader(sp[2]).subs(_one));
  }

  if (family_config["propagators"].size() != _nprops)
    throw std::runtime_error(
        "number of propagators does not match the topology");
  for (const auto &prop :
       family_config["propagators"]
           .as<std::vector<std::pair<std::string, std::string>>>()) {

    _propMomenta.emplace_back(reader(prop.first));
    _propagators.emplace_back(
        (pow(reader(prop.first), 2) - pow(reader(prop.second), 2))
            .expand()
            .subs(_spsRules, GiNaC::subs_options::algebraic)
            .subs(_one)
            .expand());
  }

  _symIndices = generate_symbols("a", _nprops);
  _symProps = generate_symbols("D", _nprops);
}

// Solve for the scalar-product substitution rules.
void FamilyModel::_compute_sps() {
  GiNaC::matrix propSpMat(_nprops, _nprops);
  GiNaC::matrix propSpConst(_nprops, 1);

  GiNaC::lst allZero;
  for (unsigned i = 0; i < _nints; ++i) {
    for (unsigned j = i; j < _nints; ++j)
      allZero.append(_internals[i] * _internals[j] == 0);
    for (unsigned j = 0; j < _nexts; ++j)
      allZero.append(_internals[i] * _externals[j] == 0);
  }

  for (unsigned s = 0; s < _nprops; ++s) {
    propSpConst(s, 0) =
        _propagators[s].subs(allZero, GiNaC::subs_options::algebraic);

    unsigned index = 0;
    for (unsigned i = 0; i < _nints; ++i) {
      GiNaC::ex coeff = _propagators[s].diff(_internals[i]);
      propSpMat(s, index) = coeff.diff(_internals[i]) / 2;
      ++index;
      for (unsigned j = i + 1; j < _nints; ++j) {
        propSpMat(s, index) = coeff.diff(_internals[j]);
        ++index;
      }
      for (unsigned j = 0; j < _nexts; ++j) {
        propSpMat(s, index) = coeff.diff(_externals[j]);
        ++index;
      }
    }
  }

  if (propSpMat.rank() < _nprops)
    throw std::runtime_error("the set of propagators is incomplete");

  GiNaC::matrix spPropVec(_nprops, 1);
  for (unsigned i = 0; i < _nprops; ++i)
    spPropVec(i, 0) = _symProps[i] - propSpConst(i, 0);
  GiNaC::matrix propSpInv = propSpMat.inverse();
  spPropVec = propSpInv.mul(spPropVec);

  unsigned index = 0;
  for (unsigned i = 0; i < _nints; ++i) {
    for (unsigned j = i; j < _nints; ++j) {
      _spsFromProps.append(_internals[i] * _internals[j] ==
                           spPropVec(index, 0));
      ++index;
    }
    for (unsigned j = 0; j < _nexts; ++j) {
      _spsFromProps.append(_internals[i] * _externals[j] ==
                           spPropVec(index, 0));
      ++index;
    }
  }
}

// Compute the Symanzik polynomials.
void FamilyModel::_compute_symanzik() {
  GiNaC::ex schwinger;
  for (unsigned i = 0; i < _nprops; ++i)
    schwinger += -_symIndices[i] * _propagators[i];

  GiNaC::lst allZero;
  for (const auto &ex : _internals)
    allZero.append(ex == 0);
  GiNaC::ex J = -schwinger.subs(allZero, GiNaC::subs_options::algebraic);

  GiNaC::matrix M(_nints, _nints);

  GiNaC::matrix V(_nints, 1);

  for (unsigned i = 0; i < _nints; ++i) {
    GiNaC::ex di = schwinger.diff(_internals[i]);
    for (unsigned j = i; j < _nints; ++j) {
      M(i, j) = di.diff(_internals[j]) / 2;
      if (i != j)
        M(j, i) = M(i, j);
    }
    V(i, 0) = -di.subs(allZero, GiNaC::subs_options::algebraic) / 2;
  }

  _uPoly = M.determinant().expand();
  _fPoly = (_uPoly * J +
            V.transpose().mul(M.inverse().mul_scalar(_uPoly)).mul(V)(0, 0))
               .expand()
               .subs(_spsRules, GiNaC::subs_options::algebraic)
               .expand();
}

// The per-candidate GiNaC classifier.  The support classifier below covers
// every input whose monomials are at most quadratic in each propagator and
// whose propagator count fits the exact rank arithmetic; this stays as the
// fallback for anything else.
std::vector<bool> FamilyModel::nontrivial_sectors_ginac(unsigned top) const {
  GiNaC::ex gPoly = _uPoly + _fPoly;
  GiNaC::ex gDiff;

  std::vector<GiNaC::symbol> kvec = generate_symbols("k", _nprops);
  GiNaC::lst klst;
  for (const auto &sym : kvec)
    klst.append(sym);

  for (unsigned i = 0; i < _nprops; ++i)
    gDiff += kvec[i] * _symIndices[i] * gPoly.diff(_symIndices[i]);
  gDiff = (gDiff - gPoly).expand();

  std::vector<bool> nontrivial(top + 1, false);
  std::vector<unsigned> trivials;
  for (unsigned sector = top; sector >= (1 << _nints) - 1; --sector) {

    if ((sector & top) == sector) {

      if (std::find_if(trivials.begin(), trivials.end(),
                       [sector](unsigned trivial) {
                         return (sector & trivial) == sector;
                       }) != trivials.end())
        continue;

      GiNaC::lst zeros;
      for (unsigned i = 0; i < _nprops; ++i)
        if ((sector & (1 << i)) == 0)
          zeros.append(_symIndices[i] == 0);
      GiNaC::ex gSector = gDiff.subs(zeros);

      std::map<std::vector<unsigned>, GiNaC::ex> coeffs;
      for (unsigned item = 0; item < gSector.nops(); ++item) {
        GiNaC::ex term = gSector.op(item);
        std::vector<unsigned> indices(_nprops, 0);
        for (unsigned i = 0; i < _nprops; ++i) {
          if (sector & (1 << i)) {
            if (term.coeff(_symIndices[i], 2) != 0) {
              term = term.coeff(_symIndices[i], 2);
              indices[i] = 2;
            } else if (term.coeff(_symIndices[i], 1) != 0) {
              term = term.coeff(_symIndices[i], 1);
              indices[i] = 1;
            }
          }
        }
        coeffs[indices] += term;
      }
      GiNaC::lst eqs;
      for (const auto &item : coeffs) {
        GiNaC::ex coeff = item.second.expand();
        if (coeff != 0)
          eqs.append(coeff == 0);
      }

      if (nops(GiNaC::lsolve(eqs, klst)) == 0)
        nontrivial[sector] = true;
      else
        trivials.push_back(sector);
    }
  }
  return nontrivial;
}

// Sectors worth reducing under the top-sector classification.
//
// Write g = U + F = sum_e c_e x^e.  Then
//   gDiff = sum_i k_i x_i d_i g - g = sum_e c_e x^e (sum_i e_i k_i - 1),
// so zeroing the lines outside a sector keeps exactly the monomials of g whose
// support lies inside it, and each gives c_e (sum_i e_i k_i - 1) = 0.  c_e is
// a nonzero coefficient as an expanded polynomial, hence a unit in the
// coefficient field, and the equation is sum_i e_i k_i = 1.  The
// classification is therefore an exact integer rank question on the exponent
// vectors: a sector is trivial iff k . e = 1 is consistent for every kept e.
// GiNaC is used once, to read the monomial support; the per-candidate
// substitution, coefficient map and lsolve disappear.
std::vector<bool> FamilyModel::nontrivial_sectors(unsigned top) const {
  if (_nprops > 16)
    return nontrivial_sectors_ginac(top);

  const GiNaC::ex g = (_uPoly + _fPoly).expand();
  // g = 0 has no monomials, so every sector is trivial -- the empty system
  // lsolve used to return a free solution for.
  if (g == 0)
    return std::vector<bool>(top + 1, false);
  std::vector<std::vector<unsigned>> support;
  std::vector<unsigned> masks;
  const std::size_t nterms = GiNaC::is_a<GiNaC::add>(g) ? g.nops() : 1;
  support.reserve(nterms);
  masks.reserve(nterms);
  for (std::size_t item = 0; item < nterms; ++item) {
    const GiNaC::ex &term = GiNaC::is_a<GiNaC::add>(g) ? g.op(item) : g;
    std::vector<unsigned> e(_nprops, 0);
    unsigned mask = 0;
    for (unsigned i = 0; i < _nprops; ++i) {
      const int degree = GiNaC::degree(term, _symIndices[i]);
      // The equation form above assumes the quadratic Symanzik structure the
      // old coefficient extraction handled; leave anything else to it.
      if (degree < 0 || degree > 2)
        return nontrivial_sectors_ginac(top);
      e[i] = static_cast<unsigned>(degree);
      if (degree != 0)
        mask |= 1u << i;
    }
    support.push_back(std::move(e));
    masks.push_back(mask);
  }

  std::vector<bool> nontrivial(top + 1, false);
  std::vector<unsigned> trivials;
  for (unsigned sector = top; sector >= (1 << _nints) - 1; --sector) {
    if ((sector & top) != sector)
      continue;
    if (std::find_if(trivials.begin(), trivials.end(),
                     [sector](unsigned trivial) {
                       return (sector & trivial) == sector;
                     }) != trivials.end())
      continue;
    std::vector<std::vector<unsigned>> rows;
    for (std::size_t t = 0; t < support.size(); ++t)
      if ((masks[t] & ~sector) == 0)
        rows.push_back(support[t]);
    if (equations_consistent(rows, _nprops))
      trivials.push_back(sector);
    else
      nontrivial[sector] = true;
  }
  return nontrivial;
}

// Assemble the symmetry search inputs.
SymmetryTable::Inputs FamilyModel::symmetry_inputs() const {
  return SymmetryTable::Inputs{_uPoly + _fPoly,
                               _propagators,
                               _propMomenta,
                               _internals,
                               _externals,
                               _spsRules,
                               _spsFromProps,
                               _symIndices,
                               _symProps,
                               _nprops,
                               _symbols.empty()};
}
