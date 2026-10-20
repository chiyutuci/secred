#include "symmetry.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
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

int as_sign(const GiNaC::ex &e) {
  if (e == 1)
    return 1;
  if (e == -1)
    return -1;
  return 0;
}

rational to_rational(const GiNaC::ex &e) {
  if (!GiNaC::is_a<GiNaC::numeric>(e) ||
      !GiNaC::ex_to<GiNaC::numeric>(e).is_rational())
    throw std::runtime_error("symmetry: coefficient " + show(e) +
                             " is not a rational number");
  std::ostringstream oss;
  oss << e;
  return rational(oss.str());
}

}

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
  has_numeric_rows = true;
}

bool PropPrescription::try_to_numeric() {
  try {
    to_numeric();
  } catch (const std::exception &) {
    has_numeric_rows = false;
  }
  return has_numeric_rows;
}

void SymmetryTable::try_to_numeric_all() {
  for (std::vector<PropPrescription> &per_source : _bySource)
    for (PropPrescription &p : per_source)
      p.try_to_numeric();
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
      cvec[i] = row_times(row_of(dst[i]), Api);
      dvec[i] = row_times(row_of(src[i]), Ai);
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
      eps_full[i] = e;
    }
    if (!ok)
      continue;

    std::vector<GiNaC::ex> rhs(nl);
    for (unsigned r = 0; r < nl; ++r)
      rhs[r] = eps[r] * BK[r] - BpK[r];
    const std::vector<GiNaC::ex> N = times_col(Api, rhs);

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

namespace {

struct Row {
  std::vector<unsigned> e;
  GiNaC::ex coef;
  unsigned power_sum = 0;
};

std::vector<Row> restricted_rows(const SymmetryTable::Inputs &in, unsigned sector) {
  const unsigned n = in.nprops;
  GiNaC::lst zeros;
  for (unsigned i = 0; i < n; ++i)
    if (!(sector & (1u << i)))
      zeros.append(in.indexSymbols[i] == 0);
  const GiNaC::ex g = in.G.subs(zeros).expand();

  std::vector<unsigned> lines;
  for (unsigned i = 0; i < n; ++i)
    if (sector & (1u << i))
      lines.push_back(i);

  std::map<std::vector<unsigned>, GiNaC::ex> merged;
  for (unsigned it = 0; it < g.nops(); ++it) {
    GiNaC::ex term = g.op(it);
    std::vector<unsigned> e(lines.size(), 0);
    GiNaC::ex coef = term;
    for (unsigned j = 0; j < lines.size(); ++j) {
      e[j] = term.degree(in.indexSymbols[lines[j]]);
      coef = coef.subs(in.indexSymbols[lines[j]] == 1);
    }
    merged[e] += coef;
  }
  std::vector<Row> rows;
  for (auto &[e, coef] : merged) {
    const GiNaC::ex c = coef.expand();
    if (c.is_zero())
      continue;
    Row r;
    r.e = e;
    r.coef = c;
    for (unsigned x : e)
      r.power_sum += x;
    rows.push_back(std::move(r));
  }
  return rows;
}

bool row_less(const Row &a, const Row &b) {
  if (a.power_sum != b.power_sum)
    return a.power_sum < b.power_sum;
  const std::string sa = show(a.coef), sb = show(b.coef);
  if (sa != sb)
    return sa < sb;
  return a.e < b.e;
}

std::string matrix_key(const std::vector<Row> &rows, unsigned ncols) {
  std::string out;
  for (const Row &r : rows) {
    out += show(r.coef);
    out += ' ';
  }
  for (unsigned c = 0; c < ncols; ++c) {
    for (const Row &r : rows) {
      out += std::to_string(r.e[c]);
      out += ' ';
    }
    out += '|';
  }
  return out;
}

std::string canonical_key(const SymmetryTable::Inputs &in, unsigned sector,
                          std::vector<std::vector<int>> &perms) {
  std::vector<Row> rows = restricted_rows(in, sector);
  const unsigned ncols = std::popcount(sector);
  std::sort(rows.begin(), rows.end(), row_less);

  std::vector<unsigned> lines;
  for (unsigned i = 0; i < in.nprops; ++i)
    if (sector & (1u << i))
      lines.push_back(i);

  struct Branch {
    std::vector<Row> rows;
    std::vector<unsigned> swaps;
  };
  std::vector<Branch> branches;
  branches.push_back({rows, {}});

  for (unsigned i = 0; i + 1 < ncols; ++i) {

    struct Cand {
      std::vector<unsigned> col;
      unsigned branch = 0, j = 0;
    };
    std::vector<Cand> cands;
    for (unsigned b = 0; b < branches.size(); ++b) {
      for (unsigned j = i; j < ncols; ++j) {
        std::vector<Row> tmp = branches[b].rows;
        const auto cmp = [i, j](const Row &x, const Row &y) {
          if (x.power_sum != y.power_sum)
            return x.power_sum < y.power_sum;
          const std::string sx = show(x.coef), sy = show(y.coef);
          if (sx != sy)
            return sx < sy;
          for (unsigned c = 0; c < i; ++c)
            if (x.e[c] != y.e[c])
              return x.e[c] < y.e[c];
          return x.e[j] < y.e[j];
        };
        std::stable_sort(tmp.begin(), tmp.end(), cmp);
        std::vector<unsigned> col;
        for (const Row &r : tmp)
          col.push_back(r.e[j]);
        cands.push_back({col, b, j});
      }
    }

    std::vector<unsigned> best;
    for (const Cand &c : cands)
      if (best.empty() || best < c.col)
        best = c.col;
    std::vector<Branch> next;
    for (const Cand &c : cands) {
      if (c.col != best)
        continue;
      Branch br = branches[c.branch];
      for (Row &r : br.rows)
        std::swap(r.e[i], r.e[c.j]);
      br.swaps.push_back(c.j);
      next.push_back(std::move(br));
    }
    branches = std::move(next);
  }

  perms.clear();
  for (Branch &br : branches) {
    std::vector<int> perm;
    for (unsigned k : lines)
      perm.push_back(int(k));
    for (unsigned c = 0; c < br.swaps.size(); ++c)
      std::swap(perm[c], perm[br.swaps[c]]);
    perms.push_back(perm);
  }
  if (branches.empty())
    return "";

  for (Branch &br : branches)
    std::sort(br.rows.begin(), br.rows.end(), row_less);
  return matrix_key(branches.front().rows, ncols);
}

}

SymmetryTable SymmetryTable::find(const Inputs &in, const std::vector<bool> &nonTrivial,
                                  unsigned top) {
  SymmetryTable t;
  const unsigned n = in.nprops;
  t._canonical.assign(1u << n, "");
  t._perms.assign(1u << n, {});
  t._representative.assign(1u << n, 0);
  t._bySource.assign(1u << n, {});
  for (unsigned s = 0; s < (1u << n); ++s)
    t._representative[s] = s;

  std::vector<unsigned> considered;
  for (unsigned s = 1; s <= top && s < nonTrivial.size(); ++s)
    if (nonTrivial[s])
      considered.push_back(s);

  for (unsigned s : considered)
    t._canonical[s] = canonical_key(in, s, t._perms[s]);

  std::map<std::string, std::vector<unsigned>> groups;
  for (unsigned s : considered)
    groups[t._canonical[s]].push_back(s);
  for (const auto &[key, g] : groups) {
    const unsigned rep = *std::min_element(g.begin(), g.end());
    for (unsigned s : g)
      t._representative[s] = rep;
  }

  for (unsigned s : considered) {
    const unsigned rep = t._representative[s];
    if (rep != s) {
      std::vector<int> ing(n, -1);
      const auto &p0 = t._perms[s].front(), &p1 = t._perms[rep].front();
      for (unsigned g = 0; g < p0.size(); ++g)
        ing[p0[g]] = p1[g];
      auto p = verify(s, rep, ing, in);
      t._bySource[s].push_back(p ? *p : dot_only_entry(s, rep, ing, in));
    } else if (t._perms[s].size() > 1) {
      for (unsigned g = 1; g < t._perms[s].size(); ++g) {
        std::vector<int> ing(n, -1);
        const auto &p0 = t._perms[s].front(), &p1 = t._perms[s][g];
        for (unsigned h = 0; h < p0.size(); ++h)
          ing[p0[h]] = p1[h];
        auto p = verify(s, s, ing, in);
        t._bySource[s].push_back(p ? *p : dot_only_entry(s, s, ing, in));
      }
    }
  }
  return t;
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

void SymmetryTable::restrictTo(const std::set<unsigned> &reduced) {
  for (unsigned s = 0; s < _bySource.size(); ++s) {
    if (_bySource[s].empty())
      continue;
    std::vector<PropPrescription> keep;
    for (PropPrescription &p : _bySource[s])
      if (reduced.count(p.from) && reduced.count(p.to))
        keep.push_back(std::move(p));
    _bySource[s] = std::move(keep);

    if (_bySource[s].empty())
      _representative[s] = s;
  }
}

void SymmetryTable::dropCross() {
  for (unsigned s = 0; s < _bySource.size(); ++s) {
    std::vector<PropPrescription> keep;
    for (PropPrescription &p : _bySource[s])
      if (p.from == p.to)
        keep.push_back(std::move(p));
    _bySource[s] = std::move(keep);
    if (_bySource[s].empty())
      _representative[s] = s;
  }
}

namespace {

template <typename Coeff>
Coeff ring_one();

template <>
GiNaC::ex ring_one<GiNaC::ex>() {
  return GiNaC::ex(1);
}

template <>
rational ring_one<rational>() {
  return rational(1);
}

template <typename Coeff>
Coeff ring_pow(const Coeff &x, unsigned e) {
  Coeff r = ring_one<Coeff>();
  for (unsigned i = 0; i < e; ++i)
    r = r * x;
  return r;
}

template <>
GiNaC::ex ring_pow<GiNaC::ex>(const GiNaC::ex &x, unsigned e) {
  return GiNaC::pow(x, e);
}

struct RowKey {
  std::array<std::int16_t, RawIntegral::MAX_PROPS> e{};
  std::uint32_t n = 0;

  bool operator<(const RowKey &o) const {
    if (n != o.n)
      return n < o.n;
    return std::lexicographical_compare(e.begin(), e.begin() + n, o.e.begin(),
                                        o.e.begin() + o.n);
  }
};

inline RowKey row_key(const std::vector<int> &powers) {
  RowKey k;
  k.n = static_cast<std::uint32_t>(powers.size());
  for (std::size_t i = 0; i < powers.size(); ++i)
    k.e[i] = static_cast<std::int16_t>(powers[i]);
  return k;
}

constexpr const char *kDenominatorRowError =
    "symmetry: a denominator line's row is not a single propagator (Kira: "
    "\"encountered a linear combination of propagators in the denominator\")";

}

template <typename Coeff>
bool expand_prescription(const PropPrescription &p, const std::vector<std::vector<Coeff>> &f,
                         const std::vector<Coeff> &c, const RawIntegral &seed,
                         const std::function<bool(unsigned)> &trivial,
                         std::vector<SymTerm<Coeff>> &out) {
  out.clear();

  const unsigned n = f.size();
  if (p.dot_only) {

    for (unsigned k = 0; k < seed.size(); ++k)
      if (!(p.from & (1u << k)) && seed[k] != 0)
        return false;
  }

  std::vector<int> image(n, 0);
  Coeff coeff = ring_one<Coeff>();
  for (unsigned k = 0; k < seed.size(); ++k) {
    const int e = seed[k];
    if (e <= 0)
      continue;
    unsigned nonzero = 0;
    int j = -1;
    for (unsigned l = 0; l < n; ++l)
      if (!f[k][l].is_zero()) {
        ++nonzero;
        j = int(l);
      }
    if (nonzero != 1 || !c[k].is_zero())
      throw std::runtime_error(kDenominatorRowError);

    image[unsigned(j)] += e;
    coeff = coeff / ring_pow(f[k][unsigned(j)], unsigned(e));
  }

  bool has_numerator = false;
  for (unsigned k = 0; k < seed.size(); ++k)
    if (seed[k] < 0) {
      has_numerator = true;
      break;
    }
  if (!has_numerator) {
    if (!coeff.is_zero())
      out.push_back(SymTerm<Coeff>{image, coeff});
    return true;
  }

  std::vector<std::pair<RowKey, Coeff>> acc;
  acc.emplace_back(row_key(image), coeff);

  for (unsigned k = 0; k < seed.size(); ++k) {
    if (seed[k] >= 0)
      continue;
    for (int factor = 0; factor < -seed[k]; ++factor) {
      std::vector<std::pair<RowKey, Coeff>> next;
      next.reserve(2 * acc.size());
      for (const auto &[key, cf] : acc) {
        if (!c[k].is_zero())
          next.emplace_back(key, cf * c[k]);
        for (unsigned j = 0; j < n; ++j) {
          if (f[k][j].is_zero())
            continue;
          RowKey v = key;
          if (v.e[j] == 1) {

            v.e[j] = 0;
            unsigned sector = 0;
            for (unsigned m = 0; m < n; ++m)
              if (v.e[m] > 0)
                sector |= 1u << m;
            if (trivial(sector))
              continue;
          } else {
            v.e[j] -= 1;
          }
          next.emplace_back(v, cf * f[k][j]);
        }
      }
      flat_merge(next);
      acc.swap(next);
    }
  }

  out.reserve(acc.size());
  for (const auto &[key, cf] : acc)

    if (!cf.is_zero()) {
      std::vector<int> powers(key.n);
      for (std::uint32_t i = 0; i < key.n; ++i)
        powers[i] = key.e[i];
      out.push_back(SymTerm<Coeff>{std::move(powers), cf});
    }
  return true;
}

template bool expand_prescription<GiNaC::ex>(const PropPrescription &,
                                             const std::vector<std::vector<GiNaC::ex>> &,
                                             const std::vector<GiNaC::ex> &, const RawIntegral &,
                                             const std::function<bool(unsigned)> &,
                                             std::vector<SymTerm<GiNaC::ex>> &);
template bool expand_prescription<rational>(const PropPrescription &,
                                            const std::vector<std::vector<rational>> &,
                                            const std::vector<rational> &, const RawIntegral &,
                                            const std::function<bool(unsigned)> &,
                                            std::vector<SymTerm<rational>> &);
