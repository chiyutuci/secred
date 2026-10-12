#include "symmetry.h"

#include <algorithm>
#include <bit>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace {

std::string show(const GiNaC::ex &e) {
  std::ostringstream oss;
  oss << e;
  return oss.str();
}

// q_k = a_k . l + b_k: the linear data of a momentum flow, read off a
// propagator's momentum by differentiation, the way _compute_sps reads the
// scalar products off _propagators.
struct Flow {
  std::vector<GiNaC::ex> a;
  GiNaC::ex b;
};

Flow read_flow(const GiNaC::ex &q, const std::vector<GiNaC::possymbol> &internals) {
  Flow fl;
  GiNaC::ex b = q;
  for (const auto &l : internals) {
    fl.a.push_back(q.diff(l));
    b = b.subs(l == 0);
  }
  fl.b = b.expand();
  return fl;
}

GiNaC::matrix matrix_of(const std::vector<std::vector<GiNaC::ex>> &rows) {
  GiNaC::matrix m(rows.size(), rows.empty() ? 0 : rows[0].size());
  for (unsigned i = 0; i < rows.size(); ++i)
    for (unsigned j = 0; j < rows[i].size(); ++j)
      m(i, j) = rows[i][j];
  return m;
}

// v * m, with v a row vector.
std::vector<GiNaC::ex> row_times(const std::vector<GiNaC::ex> &v,
                                 const GiNaC::matrix &m) {
  std::vector<GiNaC::ex> out(m.cols(), 0);
  for (unsigned j = 0; j < m.cols(); ++j) {
    GiNaC::ex s = 0;
    for (unsigned i = 0; i < m.rows(); ++i)
      s += v[i] * m(i, j);
    out[j] = s.expand();
  }
  return out;
}

// m * v, with v a column vector.
std::vector<GiNaC::ex> times_col(const GiNaC::matrix &m,
                                 const std::vector<GiNaC::ex> &v) {
  std::vector<GiNaC::ex> out(m.rows(), 0);
  for (unsigned i = 0; i < m.rows(); ++i) {
    GiNaC::ex s = 0;
    for (unsigned j = 0; j < m.cols(); ++j)
      s += m(i, j) * v[j];
    out[i] = s.expand();
  }
  return out;
}

bool invertible(const GiNaC::matrix &m, GiNaC::matrix &inv) {
  if (m.rows() != m.cols() || m.rows() == 0)
    return false;
  if (m.determinant().expand().is_zero())
    return false;
  inv = m.inverse();
  return true;
}

// +1 / -1 when the expression is exactly that number, 0 otherwise.
int as_sign(const GiNaC::ex &e) {
  if (e == 1)
    return 1;
  if (e == -1)
    return -1;
  return 0;
}

// A numeric family's coefficients are exact numbers; the conversion to rational
// is the stringstream idiom _generate_ibp_ff_n already uses (family.cpp:993),
// and it is exact, so anything else here is a hard error rather than a silent
// approximation (spec 5.1).
rational to_rational(const GiNaC::ex &e) {
  if (!GiNaC::is_a<GiNaC::numeric>(e) ||
      !GiNaC::ex_to<GiNaC::numeric>(e).is_rational())
    throw std::runtime_error("symmetry: coefficient " + show(e) +
                             " is not a rational number");
  std::ostringstream oss;
  oss << e;
  return rational(oss.str());
}

}  // namespace

std::string kira_to_ginac(const std::string &s) {
  std::string t;
  for (std::size_t i = 0; i < s.size();) {
    if (s.compare(i, 2, "m2") == 0) {
      t += "(m^2)";
      i += 2;
    } else {
      t += s[i++];
    }
  }
  return t;
}

void PropPrescription::to_numeric() {
  f_num.assign(f.rows(), std::vector<rational>(f.cols()));
  for (unsigned i = 0; i < f.rows(); ++i)
    for (unsigned j = 0; j < f.cols(); ++j)
      f_num[i][j] = to_rational(f(i, j));
  c_num.assign(c.rows(), rational());
  for (unsigned i = 0; i < c.rows(); ++i)
    c_num[i] = to_rational(c(i, 0));
}

bool PropPrescription::rows_ok(const std::vector<std::vector<std::string>> &rows,
                               unsigned nprops, GiNaC::symtab &st) const {
  if (rows.size() != f.rows())
    return false;
  for (unsigned k = 0; k < f.rows(); ++k) {
    if (rows[k].size() != nprops + 1)
      return false;
    for (unsigned j = 0; j < nprops; ++j) {
      GiNaC::parser par(st);
      if (!(f(k, j) - par(kira_to_ginac(rows[k][j]))).expand().is_zero())
        return false;
    }
    GiNaC::parser par(st);
    if (!(c(k, 0) - par(kira_to_ginac(rows[k][nprops]))).expand().is_zero())
      return false;
  }
  return true;
}

// Spec 4.6.  A line correspondence sigma extends to a symmetry iff there are a
// matrix M, a vector N and signs eps_k with
//     q_k(l, p) = eps_k q'_{sigma(k)}(l', p),      l' = M l + N p .
// For a subset K of the source's in-sector lines whose a-matrices are invertible
// on both sides the map is forced, M = A'_K^{-1} E A_K with E = diag(eps_K);
// each remaining line then fixes its own eps by one division, and every
// remaining line's p-equation is checked against the solved map.  Kira reaches
// the same decision by enumerating 2^loops sign patterns over an invertible
// subset drawn from the in-sector lines only (combProps(j, nn),
// trivial_sym.cpp:1401-1420), which is why a correspondence covering fewer lines
// than loops has no map.
std::optional<PropPrescription>
SymmetryTable::verify(unsigned from, unsigned to, const std::vector<int> &ing,
                      const Inputs &in) {
  const unsigned n = in.nprops, nl = in.internals.size();
  std::vector<unsigned> src, dst;
  for (unsigned k = 0; k < n; ++k)
    if (ing[k] >= 0) {
      src.push_back(k);
      dst.push_back(unsigned(ing[k]));
    }
  if (nl == 0 || src.size() < nl)
    return std::nullopt;

  std::vector<Flow> fl(n);
  for (unsigned k = 0; k < n; ++k)
    fl[k] = read_flow(in.propMomenta[k], in.internals);

  const unsigned m = src.size();
  const auto row_of = [&](unsigned k) { return fl[k].a; };

  // K: the first subset, in Kira's own ascending order, with both a-matrices
  // invertible -- Kira's testProps check.
  std::vector<unsigned> K;
  GiNaC::matrix Ai(nl, nl), Api(nl, nl);
  {
    std::vector<unsigned> c(nl);
    for (unsigned i = 0; i < nl; ++i)
      c[i] = i;
    while (true) {
      std::vector<std::vector<GiNaC::ex>> ra, rb;
      for (unsigned i : c) {
        ra.push_back(row_of(src[i]));
        rb.push_back(row_of(dst[i]));
      }
      GiNaC::matrix inva, invb;
      if (invertible(matrix_of(ra), inva) && invertible(matrix_of(rb), invb)) {
        K = c;
        Ai = inva;
        Api = invb;
        break;
      }
      int i = int(nl) - 1;
      while (i >= 0 && c[i] == m - nl + unsigned(i))
        --i;
      if (i < 0)
        return std::nullopt;
      ++c[i];
      for (unsigned j = unsigned(i) + 1; j < nl; ++j)
        c[j] = c[j - 1] + 1;
    }
  }

  std::vector<std::vector<GiNaC::ex>> AK, ApK;
  std::vector<GiNaC::ex> BK, BpK;
  for (unsigned i : K) {
    AK.push_back(row_of(src[i]));
    ApK.push_back(row_of(dst[i]));
    BK.push_back(fl[src[i]].b);
    BpK.push_back(fl[dst[i]].b);
  }
  const GiNaC::matrix AKm = matrix_of(AK), ApKm = matrix_of(ApK);

  const auto in_K = [&](unsigned i) {
    return std::find(K.begin(), K.end(), i) != K.end();
  };

  // Sign patterns, all-plus first (Kira's gi = 0), over the K lines only.
  for (unsigned branch = 0; branch < (1u << nl); ++branch) {
    std::vector<int> eps(nl, 0), eps_full(m, 0);
    for (unsigned r = 0; r < nl; ++r) {
      eps[r] = (branch >> r) & 1 ? -1 : 1;
      eps_full[K[r]] = eps[r];
    }
    GiNaC::matrix E(nl, nl);
    for (unsigned r = 0; r < nl; ++r)
      E(r, r) = eps[r];
    const GiNaC::matrix M = Api.mul(E).mul(AKm);

    bool ok = true;
    std::vector<std::vector<GiNaC::ex>> cvec(m), dvec(m);
    for (unsigned i = 0; i < m && ok; ++i) {
      if (in_K(i))
        continue;
      cvec[i] = row_times(row_of(dst[i]), Api);  // a'_{sigma(i)} A'_K^{-1}
      dvec[i] = row_times(row_of(src[i]), Ai);   // a_i A_K^{-1}
      int e = 0;
      for (unsigned r = 0; r < nl && ok; ++r) {
        if (dvec[i][r].is_zero())
          continue;
        const int v = as_sign((cvec[i][r] * eps[r] / dvec[i][r]).expand());
        if (v == 0)
          ok = false;
        else if (e == 0)
          e = v;
        else if (e != v)
          ok = false;
      }
      eps_full[i] = e;  // 0 = unconstrained here; pinned by the p-equation below
    }
    if (!ok)
      continue;

    // A'_K N + B'_K = E B_K
    std::vector<GiNaC::ex> rhs(nl);
    for (unsigned r = 0; r < nl; ++r)
      rhs[r] = eps[r] * BK[r] - BpK[r];
    const std::vector<GiNaC::ex> N = times_col(Api, rhs);

    // Every remaining line's p-equation: b_i = eps_i (a'_{sigma(i)} N + b'_{sigma(i)}).
    for (unsigned i = 0; i < m && ok; ++i) {
      if (in_K(i))
        continue;
      GiNaC::ex s = fl[dst[i]].b;
      const std::vector<GiNaC::ex> &ad = fl[dst[i]].a;
      for (unsigned r = 0; r < nl; ++r)
        s += ad[r] * N[r];
      s = s.expand();
      if (eps_full[i] == 0) {
        if ((fl[src[i]].b - s).expand().is_zero())
          eps_full[i] = 1;
        else if ((fl[src[i]].b + s).expand().is_zero())
          eps_full[i] = -1;
        else
          ok = false;
      } else if (!(fl[src[i]].b - eps_full[i] * s).expand().is_zero()) {
        ok = false;
      }
    }
    if (!ok)
      continue;

    // The rows: D_k rewritten over the target's D'_j, by substituting
    //     l_i -> sum_j (M^-1)_{i j} l'_j - (M^-1 N p)_i
    // and reducing every l'.l' and l'.p through the family's own rules -- the
    // idiom of _generate_ibp.  In-sector lines come out as single terms
    // (D_k = D'_{sigma(k)}), out-of-sector lines as general combinations.
    const GiNaC::matrix Minv = M.inverse();
    const std::vector<GiNaC::ex> shift = times_col(Minv, N);
    GiNaC::lst subst;
    for (unsigned i = 0; i < nl; ++i) {
      GiNaC::ex l = -shift[i];
      for (unsigned j = 0; j < nl; ++j)
        l += Minv(i, j) * in.internals[j];
      subst.append(in.internals[i] == l);
    }

    PropPrescription p;
    p.from = from;
    p.to = to;
    p.ing = ing;
    p.dot_only = false;
    p.f = GiNaC::matrix(n, n);
    p.c = GiNaC::matrix(n, 1);
    for (unsigned k = 0; k < n; ++k) {
      GiNaC::ex d = in.propagators[k]
                        .subs(subst, GiNaC::subs_options::algebraic)
                        .expand()
                        .subs(in.spsRules, GiNaC::subs_options::algebraic)
                        .expand()
                        .subs(in.spsFromProps, GiNaC::subs_options::algebraic)
                        .expand();
      for (unsigned j = 0; j < n; ++j) {
        p.f(k, j) = d.diff(in.propSymbols[j]).expand();
        d = d.subs(in.propSymbols[j] == 0);
      }
      p.c(k, 0) = d.expand();
    }
    if (in.numeric)
      p.to_numeric();
    return p;
  }
  return std::nullopt;
}

PropPrescription SymmetryTable::dot_only_entry(unsigned from, unsigned to,
                                               const std::vector<int> &ing,
                                               const Inputs &in) {
  const unsigned n = in.nprops;
  PropPrescription p;
  p.from = from;
  p.to = to;
  p.ing = ing;
  p.dot_only = true;
  p.f = GiNaC::matrix(n, n);
  p.c = GiNaC::matrix(n, 1);
  unsigned hole = 0;
  while (hole < n && (to & (1u << hole)))
    ++hole;
  for (unsigned k = 0; k < n; ++k) {
    const unsigned j = ing[k] >= 0 ? unsigned(ing[k]) : hole;
    for (unsigned l = 0; l < n; ++l)
      p.f(k, l) = (l == j) ? GiNaC::ex(1) : GiNaC::ex(0);
    p.c(k, 0) = 0;
  }
  if (in.numeric)
    p.to_numeric();
  return p;
}

// ---- filled in by later tasks ----------------------------------------------

SymmetryTable SymmetryTable::find(const Inputs &, const std::vector<bool> &,
                                  unsigned) {
  throw std::runtime_error("SymmetryTable::find not implemented yet");
}

unsigned SymmetryTable::nrows(unsigned sector) const {
  return sector < _perms.size() ? unsigned(_perms[sector].size()) : 0;
}

const std::vector<std::vector<int>> &
SymmetryTable::permutations(unsigned sector) const {
  static const std::vector<std::vector<int>> empty;
  return sector < _perms.size() ? _perms[sector] : empty;
}

unsigned SymmetryTable::representative(unsigned sector) const {
  return sector < _representative.size() ? _representative[sector] : sector;
}

const std::vector<PropPrescription> &
SymmetryTable::bySource(unsigned sector) const {
  static const std::vector<PropPrescription> empty;
  return sector < _bySource.size() ? _bySource[sector] : empty;
}

void SymmetryTable::restrictTo(const std::set<unsigned> &) {}

void SymmetryTable::dropCross() {}

void SymmetryTable::dump(const std::string &) const {}
