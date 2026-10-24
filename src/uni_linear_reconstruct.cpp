#include "uni_linear_reconstruct.h"

#include <algorithm>
#include <utility>

#include <fflow/function_cache.hh>
#include <fflow/univariate_reconstruction.hh>

namespace secred {

namespace {

// Points visited by FiniteFlow's univariate sampler: x_i = OFFSET_0_HASH +
// i*SAMPLING_STRIDE modulo p.  The sample cache is keyed on the unreduced
// start value, so the walk has to reproduce it exactly.
constexpr fflow::UInt kFirstPoint = fflow::OFFSET_0_HASH;
constexpr fflow::UInt kStride = fflow::SAMPLING_STRIDE;

}

unsigned uni_unknown_count(const UniLinearState &st, unsigned pivot) {
  unsigned n = 0;
  for (unsigned char k : st.known_num)
    n += k ? 0 : 1;
  for (unsigned j = 0; j < st.known_den.size(); ++j)
    if (j != pivot && !st.known_den[j])
      ++n;
  return n;
}

void uni_seed(UniLinearState &st, const fflow::SparseRationalFunction &f,
              unsigned nd, unsigned dd, unsigned pivot, fflow::UInt p) {
  const std::size_t nnum = nd + 1, nden = dd + 1;
  st.crt_num.assign(nnum, fflow::MPInt());
  st.crt_den.assign(nden, fflow::MPInt());
  st.rat_num.assign(nnum, fflow::MPRational());
  st.rat_den.assign(nden, fflow::MPRational());
  st.prev_num.assign(nnum, fflow::MPRational());
  st.prev_den.assign(nden, fflow::MPRational());
  st.prev_ok_num.assign(nnum, 0);
  st.prev_ok_den.assign(nden, 0);

  // A coefficient absent from the first prime is structurally zero.
  st.known_num.assign(nnum, 1);
  st.known_den.assign(nden, 1);
  for (const auto &m : f.numerator())
    if (m.degree() <= nd) {
      st.crt_num[m.degree()] = fflow::MPInt(m.coeff());
      st.known_num[m.degree()] = 0;
    }
  for (const auto &m : f.denominator())
    if (m.degree() <= dd) {
      st.crt_den[m.degree()] = fflow::MPInt(m.coeff());
      st.known_den[m.degree()] = 0;
    }

  // The trial function is normalized by ThieleURationalFunction::toStdRep:
  // the lowest present denominator coefficient is 1.
  st.crt_den[pivot] = fflow::MPInt(1);
  st.rat_den[pivot] = fflow::MPRational(1);
  st.known_den[pivot] = 1;
  st.mod = fflow::MPInt(p);
}

fflow::Ret uni_linear_step(UniLinearState &st, unsigned nd, unsigned dd,
                           unsigned pivot, fflow::Mod mod, unsigned n_checks,
                           unsigned n_singular,
                           fflow::SampledRatFun &sampledfun,
                           std::vector<fflow::UInt> &num,
                           std::vector<fflow::UInt> &den) {
  const fflow::MPInt mp(mod.n());
  fflow::MPInt res;

  // Recovered coefficients enter the equations as constants modulo this prime.
  std::vector<fflow::UInt> knum(nd + 1, 0), kden(dd + 1, 0);
  for (unsigned j = 0; j <= nd; ++j)
    if (st.known_num[j]) {
      if (!fflow::rat_mod(st.rat_num[j], mp, res))
        return fflow::FAILED;
      knum[j] = res.to_uint();
    }
  for (unsigned j = 0; j <= dd; ++j)
    if (j != pivot && st.known_den[j]) {
      if (!fflow::rat_mod(st.rat_den[j], mp, res))
        return fflow::FAILED;
      kden[j] = res.to_uint();
    }

  // Unknown columns: numerator coefficients first, then denominator ones.
  std::vector<std::pair<unsigned, unsigned>> cols;
  for (unsigned j = 0; j <= nd; ++j)
    if (!st.known_num[j])
      cols.push_back({0, j});
  for (unsigned j = 0; j <= dd; ++j)
    if (j != pivot && !st.known_den[j])
      cols.push_back({1, j});
  const unsigned n_unknowns = static_cast<unsigned>(cols.size());
  const unsigned n_rows = n_unknowns + n_checks;
  const unsigned top = std::max(nd, dd);

  // sum n_j x^j - f*sum d_k x^k = f*x^pivot, with the known terms moved to the
  // right-hand side.
  fflow::DynamicMatrix mat(n_rows, n_unknowns + 1);
  std::vector<fflow::UInt> xpow(top + 1);
  fflow::UInt x = kFirstPoint;
  for (unsigned i = 0; i < n_rows;) {
    fflow::UInt fv = fflow::FAILED;
    for (unsigned skipped = 0;; ++skipped) {
      fflow::UInt xa[1] = {x};
      fv = sampledfun.evaluate(xa, mod);
      if (fv != fflow::FAILED)
        break;
      if (skipped >= n_singular)
        return fflow::FAILED;
      x = fflow::add_mod(x, kStride, mod);
    }

    xpow[0] = 1;
    for (unsigned k = 1; k <= top; ++k)
      xpow[k] = fflow::mul_mod(xpow[k - 1], x, mod);

    fflow::UInt rhs = fflow::mul_mod(fv, xpow[pivot], mod);
    for (unsigned j = 0; j <= nd; ++j)
      if (st.known_num[j])
        rhs = fflow::sub_mod(rhs, fflow::mul_mod(knum[j], xpow[j], mod), mod);
    for (unsigned j = 0; j <= dd; ++j)
      if (j != pivot && st.known_den[j])
        rhs = fflow::add_mod(
            rhs, fflow::mul_mod(kden[j], fflow::mul_mod(xpow[j], fv, mod), mod),
            mod);
    mat(i, n_unknowns) = rhs;
    for (unsigned c = 0; c < n_unknowns; ++c) {
      const unsigned j = cols[c].second;
      fflow::UInt v = xpow[j];
      if (cols[c].first)
        v = fflow::neg_mod(fflow::mul_mod(v, fv, mod), mod);
      mat(i, c) = v;
    }
    x = fflow::add_mod(x, kStride, mod);
    ++i;
  }

  if (n_unknowns == 0) {
    // Nothing left to solve: every row is a consistency check on the
    // coefficients already recovered.
    for (unsigned i = 0; i < n_rows; ++i)
      if (mat(i, 0) != 0)
        return fflow::FAILED;
  } else {
    fflow::MatrixView mv = mat.getMatrixView();
    mv.toReducedRowEcholon(mod);
    if (mv.isImpossibleSystem() || !mv.isZeroDimSystem())
      return fflow::FAILED;
  }

  num.assign(nd + 1, 0);
  den.assign(dd + 1, 0);
  unsigned c = 0;
  for (unsigned j = 0; j <= nd; ++j)
    num[j] = st.known_num[j] ? knum[j] : mat(c++, n_unknowns);
  for (unsigned j = 0; j <= dd; ++j)
    if (j == pivot)
      den[j] = 1;
    else
      den[j] = st.known_den[j] ? kden[j] : mat(c++, n_unknowns);

  // Extend the CRT modulus, then promote the coefficients whose rational
  // reconstruction agrees at the previous and the current prime.
  fflow::MPInt c1, c2, mod12;
  fflow::chinese_remainder_coeffs(st.mod, mod.n(), c1, c2, mod12);
  auto merge = [&](std::vector<fflow::MPInt> &crt,
                   const std::vector<fflow::UInt> &vals,
                   std::vector<unsigned char> &known,
                   std::vector<fflow::MPRational> &rat,
                   std::vector<unsigned char> &prev_ok,
                   std::vector<fflow::MPRational> &prev) {
    for (std::size_t j = 0; j < crt.size(); ++j) {
      fflow::chinese_remainder_from_coeffs(crt[j], vals[j], c1, c2, mod12,
                                           crt[j]);
      if (known[j])
        continue;
      fflow::MPRational q;
      const bool ok = fflow::rat_rec(crt[j], mod12, q);
      if (ok && prev_ok[j] && q.cmp(prev[j]) == 0) {
        known[j] = 1;
        rat[j] = q;
        prev_ok[j] = 0;
      } else {
        prev_ok[j] = ok ? 1 : 0;
        if (ok)
          prev[j] = q;
      }
    }
  };
  merge(st.crt_num, num, st.known_num, st.rat_num, st.prev_ok_num, st.prev_num);
  merge(st.crt_den, den, st.known_den, st.rat_den, st.prev_ok_den, st.prev_den);
  st.mod = mod12;
  return fflow::SUCCESS;
}

}
