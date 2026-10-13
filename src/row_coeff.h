// Precompiled IBP row coefficients for the settling round.
//
// The settling round re-expresses each selected equation with its full
// coefficient and spills it to the raw equation file.  On a symbolic family that
// coefficient was rebuilt as a GiNaC expression per equation, expanded and
// printed under one process-wide lock, which serialised every sector (spec 1.1).
// A row's coefficients do not depend on the seed -- only the integer vector that
// scales them does -- so here each row is expanded once into integer monomials,
// and a seed is then a signed-integer polynomial sum plus one exact print, with
// no GiNaC and no lock.
//
// Header-only, like sym_coeff.h, so the tests drive the real precompiler.
//
// Invariants of a precompiled row (relied on by the evaluator and the printer):
//   * every coefficient shares the row's `q` and `mono`, so the sum is over one
//     common denominator and needs no per-seed lcm;
//   * every term's `vars` is sorted by symbol index and carries no zero exponent;
//   * `terms` empty means the zero polynomial.

#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cln/cln.h>
#include <ginac/ginac.h>

#include "sector.h"
#include "sym_coeff.h"

namespace secred {

// One monomial factor: symbol index (into the table's name list), exponent.
// `unsigned`, not SymTerm's uint8_t, because the row's monomial lcm is applied
// by subtracting each term's own denominator exponent, which must not wrap.
using RowExp = std::pair<std::uint8_t, unsigned>;
using RowVars = std::vector<RowExp>;

// One numerator term after a seed has been substituted into a row.
struct RowTerm {
  cln::cl_I coef;
  RowVars vars; // sorted by symbol index, no zero exponents
};

// A row coefficient's numerator over the row's common denominator.
using RowNum = std::vector<RowTerm>;

// One coefficient of an IBP row: value = num / (q * mono).
struct RowPoly {
  SymPoly num;
  std::uint32_t q = 1; // positive integer denominator, shared by the row
  RowVars mono;        // denominator monomial, shared by the row; empty == 1
};

// One IBP row: the nprops propagator coefficients and the constant, last.
struct RowCoeff {
  bool precompiled = false;
  std::vector<RowPoly> c;
};

// What an evaluation produced.
enum class RowEval {
  Ok,             // `out` holds the coefficient text
  Zero,           // the coefficient cancels: the caller emits nothing
  NotPrecompiled, // the row is not precompiled (or not built): use GiNaC
};

// ---------------------------------------------------------------------------
// Small monomial helpers.  The vectors are at most MAX_PROPS long, so the linear
// scans are what a map would cost several allocations to beat.
// ---------------------------------------------------------------------------

inline unsigned row_exp_of(const RowVars &v, std::uint8_t idx) {
  for (const RowExp &e : v)
    if (e.first == idx)
      return e.second;
  return 0;
}

// v += idx^exp, keeping v sorted by symbol index.
inline void row_add_exp(RowVars &v, std::uint8_t idx, unsigned exp) {
  if (exp == 0)
    return;
  for (RowExp &e : v)
    if (e.first == idx) {
      e.second += exp;
      return;
    }
  v.emplace_back(idx, exp);
  for (std::size_t i = v.size(); i > 1 && v[i - 1].first < v[i - 2].first; --i)
    std::swap(v[i - 1], v[i - 2]);
}

// v := exponent-wise max(v, idx^exp), keeping v sorted: the lcm of monomials.
inline void row_max_exp(RowVars &v, std::uint8_t idx, unsigned exp) {
  if (exp == 0)
    return;
  for (RowExp &e : v)
    if (e.first == idx) {
      e.second = std::max(e.second, exp);
      return;
    }
  v.emplace_back(idx, exp);
  std::sort(v.begin(), v.end(),
            [](const RowExp &a, const RowExp &b) { return a.first < b.first; });
}

inline void row_drop_zero_exps(RowVars &v) {
  v.erase(std::remove_if(v.begin(), v.end(),
                         [](const RowExp &e) { return e.second == 0; }),
          v.end());
}

inline bool row_vars_less(const RowVars &a, const RowVars &b) {
  return std::lexicographical_compare(
      a.begin(), a.end(), b.begin(), b.end(), [](const RowExp &x, const RowExp &y) {
        return x.first != y.first ? x.first < y.first : x.second < y.second;
      });
}

inline std::string cl_i_text(const cln::cl_I &x) {
  std::ostringstream oss;
  oss << x;
  return oss.str();
}

// Append `v1^e1*v2^e2...` for the non-zero exponents, in index order.  No
// leading separator: the caller has already written the coefficient and its '*'.
inline void row_print_vars(const RowVars &vars, const std::vector<std::string> &names,
                           std::string &out) {
  bool first = true;
  for (const RowExp &e : vars) {
    if (e.second == 0)
      continue;
    if (!first)
      out += '*';
    first = false;
    out += e.first < names.size() ? names[e.first]
                                  : "x" + std::to_string(unsigned(e.first));
    if (e.second != 1) {
      out += '^';
      out += std::to_string(e.second);
    }
  }
}

// Append `num / (q * mono)` in the spelling both consumers already accept
// (spec 4.2):
//   * when the monomial divides every term it is factored out and each term is
//     printed as an exact rational -- `-9/2`, `d - 5`, `3/2*d + 1/2`;
//   * otherwise the whole sum is parenthesised -- `(d - 2)/(2*m^2)`, `(1)/(d)`;
//   * a unit coefficient is omitted (`d`, `-d`), and an empty sum prints `0`.
// `names` is the table's symbol-index-to-name list.
inline void print_row_poly(const RowNum &num, std::uint32_t q, const RowVars &mono,
                           const std::vector<std::string> &names, std::string &out) {
  if (num.empty()) {
    out += '0';
    return;
  }
  bool factor_mono = true;
  for (const RowTerm &t : num)
    for (const RowExp &m : mono)
      if (row_exp_of(t.vars, m.first) < m.second) {
        factor_mono = false;
        break;
      }
  const bool paren = !factor_mono;
  const std::size_t begin = out.size();
  bool any = false;
  for (const RowTerm &t : num) {
    cln::cl_I a = t.coef;
    RowVars vars = t.vars;
    std::uint32_t den = q;
    if (paren) {
      den = 1; // the monomial and q are printed around the parentheses
    } else if (factor_mono) {
      for (const RowExp &m : mono)
        for (RowExp &e : vars)
          if (e.first == m.first)
            e.second -= m.second;
      row_drop_zero_exps(vars);
    }
    const bool neg = cln::minusp(a);
    if (neg)
      a = -a;
    if (any)
      out += neg ? " - " : " + ";
    else if (neg)
      out += '-';
    if (!paren && den != 1) {
      // One exact reduction keeps `-9/2` from printing as `-18/4`.
      const cln::cl_I g = cln::gcd(a, cln::cl_I(static_cast<unsigned long>(den)));
      if (g != cln::cl_I(1)) {
        a = cln::exquo(a, g);
        den = static_cast<std::uint32_t>(cln::cl_I_to_ulong(
            cln::exquo(cln::cl_I(static_cast<unsigned long>(den)), g)));
      }
    }
    if (!(a == cln::cl_I(1) && den == 1 && !vars.empty())) {
      out += cl_i_text(a);
      if (den != 1) {
        out += '/';
        out += std::to_string(den);
      }
      if (!vars.empty())
        out += '*';
    }
    row_print_vars(vars, names, out);
    any = true;
  }
  if (paren) {
    out.insert(begin, 1, '(');
    out += ")/(";
    if (q != 1) {
      out += std::to_string(q);
      out += '*';
    }
    row_print_vars(mono, names, out);
    out += ')';
  }
}

} // namespace secred
