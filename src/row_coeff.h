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
//     printed as an exact rational -- `-9/2`, `d-5`, `3/2*d+1/2`;
//   * otherwise the whole sum is parenthesised -- `(d-2)/(2*m^2)`, `(1)/(d)`;
//   * a unit coefficient is omitted (`d`, `-d`), and an empty sum prints `0`.
// `names` is the table's symbol-index-to-name list.
//
// No whitespace anywhere: the writer copies this text verbatim into the global
// system file, and its reader splits on whitespace, so a space inside a
// coefficient silently truncates it (spec 4.2, `emits_a_single_
// whitespace_free_token`).
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
      out += neg ? "-" : "+";
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

// Append `power * poly` to `acc`, each term's monomial unchanged: the seed's
// power is a plain integer scale, so a negative one just flips signs and the
// coefficient is a signed `cl_I`, never an unsigned count.
inline void row_add_scaled(RowNum &acc, const SymPoly &poly, int power) {
  if (power == 0 || poly.terms.empty())
    return;
  const cln::cl_I scale(static_cast<long>(power));
  acc.reserve(acc.size() + poly.terms.size());
  for (const SymTerm &t : poly.terms) {
    RowTerm rt;
    rt.coef = t.coef * scale;
    rt.vars.reserve(t.vars.size());
    for (const auto &ve : t.vars)
      rt.vars.emplace_back(ve.first, static_cast<unsigned>(ve.second));
    acc.push_back(std::move(rt));
  }
}

// Sum equal monomials and drop what cancels.  The sort is total, so the result
// is deterministic, and stable, so the surviving terms read in the order the
// row's coefficients contributed them.
inline void row_merge_terms(RowNum &v) {
  std::stable_sort(v.begin(), v.end(), [](const RowTerm &a, const RowTerm &b) {
    return row_vars_less(a.vars, b.vars);
  });
  std::size_t out = 0;
  for (std::size_t i = 0; i < v.size();) {
    std::size_t j = i + 1;
    while (j < v.size() && !row_vars_less(v[i].vars, v[j].vars) &&
           !row_vars_less(v[j].vars, v[i].vars)) {
      v[i].coef += v[j].coef;
      ++j;
    }
    if (!cln::zerop(v[i].coef)) {
      // Only when something was dropped earlier: `v[out] = std::move(v[out])`
      // is a self-move-assignment, and the standard library's containers leave
      // the moved-from object empty, which would silently delete the monomial.
      if (out != i)
        v[out] = std::move(v[i]);
      ++out;
    }
    i = j;
  }
  v.resize(out);
}

// A term's own exponent of `idx`, 0 when the term does not carry the symbol.
// `SymTerm::vars` is a vector of (uint8_t, uint8_t) pairs, so the RowVars helper
// above does not apply to it.
inline unsigned sym_term_exp_of(const std::vector<std::pair<std::uint8_t, std::uint8_t>> &v,
                                std::uint8_t idx) {
  for (const auto &ve : v)
    if (ve.first == idx)
      return ve.second;
  return 0;
}

// Precompile one row (`nprops` propagator coefficients and the constant, last).
// False when a coefficient is outside the representable shape or a denominator
// is not a positive integer times a monomial; the row then keeps the GiNaC path.
// On success every coefficient carries the row's common `q` and `mono`, so the
// evaluation is a signed integer sum with no per-seed lcm, and every term's
// `vars` is sorted by index with no zero exponents.
inline bool build_row_coeff(const std::vector<GiNaC::ex> &row,
                            const std::unordered_map<std::string, unsigned> &sym_index,
                            RowCoeff &out) {
  constexpr std::uint64_t kMaxDen = 1u << 20;
  out.precompiled = false;
  out.c.clear();
  out.c.reserve(row.size());

  std::uint64_t q_lcm = 1;
  RowVars mono;
  for (const GiNaC::ex &e : row) {
    const SymCoeff sc = make_sym_coeff(e.expand(), sym_index);
    if (!sc.precompiled)
      return false;
    RowPoly c;
    if (!sc.den.terms.empty()) {
      const SymTerm &dn = sc.den.terms[0];
      if (sc.den.terms.size() != 1 || !cln::plusp(dn.coef) ||
          dn.coef > cln::cl_I(static_cast<unsigned long>(kMaxDen)))
        return false; // a polynomial denominator, or not a small positive integer
      c.q = static_cast<std::uint32_t>(cln::cl_I_to_ulong(dn.coef));
      // `c.mono` is this coefficient's *own* denominator monomial and `mono` the
      // running lcm of all of them so far; the two are built separately because
      // the second pass lifts a term onto the lcm by the difference between the
      // lcm and `c.mono`, so a coefficient shallower than an earlier one has to
      // be lifted by its own depth, not by the depth the row has reached.
      for (const auto &ve : dn.vars)
        row_max_exp(c.mono, ve.first, static_cast<unsigned>(ve.second));
      for (const RowExp &ve : c.mono)
        row_max_exp(mono, ve.first, ve.second);
    }
    const std::uint64_t g = std::gcd(q_lcm, static_cast<std::uint64_t>(c.q));
    const std::uint64_t f = static_cast<std::uint64_t>(c.q) / g;
    if (f > kMaxDen / q_lcm)
      return false;
    q_lcm *= f;
    c.num = sc.num;
    out.c.push_back(std::move(c));
  }

  // Second pass: put every coefficient over the row's lcm (q_lcm, mono), which
  // means multiplying its numerator by `mono / own mono`.  A symbol the lcm
  // carries and the term does not is exactly where that factor is not 1, so the
  // term's monomial is rebuilt from the lcm rather than patched in place: a term
  // with no entry for the symbol must gain one (a constant term of a coefficient
  // whose own denominator is shallower is the common case), and one whose
  // exponent cancels to zero must lose it.
  for (RowPoly &c : out.c) {
    const cln::cl_I scale(static_cast<unsigned long>(q_lcm / c.q));
    SymPoly scaled;
    scaled.terms.reserve(c.num.terms.size());
    for (const SymTerm &t : c.num.terms) {
      SymTerm s;
      s.coef = t.coef * scale;
      s.vars.reserve(t.vars.size() + mono.size());
      for (const auto &ve : t.vars)
        if (row_exp_of(mono, ve.first) == 0)
          s.vars.push_back(ve); // untouched by the shift: not in the lcm
      for (const RowExp &me : mono) {
        const unsigned e = sym_term_exp_of(t.vars, me.first) + me.second -
                           row_exp_of(c.mono, me.first);
        if (e > 255)
          return false; // would not survive the term's uint8_t exponent
        if (e != 0)
          s.vars.emplace_back(me.first, static_cast<std::uint8_t>(e));
      }
      std::sort(s.vars.begin(), s.vars.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
      scaled.terms.push_back(std::move(s));
    }
    c.num = std::move(scaled);
    c.q = static_cast<std::uint32_t>(q_lcm);
    c.mono = mono;
  }
  out.precompiled = true;
  return true;
}

// Evaluate one row for `seed`: sum_p seed[p]*c[p] + c.back() over the row's
// common denominator.  `out` is cleared and then carries the coefficient text;
// `names` is the table's symbol-index-to-name list.  A row that was not
// precompiled, or a seed whose length is not the row's nprops, is reported as
// NotPrecompiled and the caller takes the GiNaC path.
inline RowEval eval_row_coeff(const RowCoeff &row, const RawIntegral &seed,
                              const std::vector<std::string> &names, std::string &out) {
  if (!row.precompiled || row.c.empty() || row.c.size() != seed.size() + 1)
    return RowEval::NotPrecompiled;
  RowNum acc;
  for (unsigned p = 0; p + 1 < row.c.size(); ++p)
    row_add_scaled(acc, row.c[p].num, seed[p]);
  row_add_scaled(acc, row.c.back().num, 1);
  row_merge_terms(acc);
  if (acc.empty()) {
    out = "0";
    return RowEval::Zero;
  }
  out.clear();
  print_row_poly(acc, row.c[0].q, row.c[0].mono, names, out);
  return RowEval::Ok;
}

// The precompiled rows of a family's IBP generators: one entry per generator,
// holding one RowCoeff per item, because an IBPProtoM bundles several global
// rows into one generator.
//
// A row is the same row in every sector -- its coefficients do not depend on
// the seed, and neither does the seed's sector -- so the table is the family's:
// `Family::init_reduce` builds it once with `build_all`, single-threaded and
// before any sector thread starts, and every sector reads it afterwards
// (spec 4.3).  That is what makes `add`'s GiNaC work happen once per generator
// per run instead of once per generator per sector, and it is why the sector's
// evaluation loop takes no lock: by the time it runs, the table is read-only.
struct RowTable {
  std::vector<std::string> names;                  // symbol index -> name
  std::unordered_map<std::string, unsigned> index; // name -> symbol index
  std::vector<std::vector<RowCoeff>> rows;         // [ibp_idx][item]
  unsigned fast_rows = 0;                          // item rows precompiled
  unsigned slow_rows = 0;                          // item rows on the GiNaC path

  bool contains(unsigned ibp_idx) const {
    return ibp_idx < rows.size() && !rows[ibp_idx].empty();
  }

  void add(unsigned ibp_idx, const IBPProtoM &items);

  // Build every generator, in index order: what the family-level table wants,
  // and exactly what adding them one at a time gives (the counters and the name
  // list are the same either way).  Idempotent, since `add` is.
  void build_all(const std::vector<IBPProtoM> &ibps) {
    for (unsigned i = 0; i < ibps.size(); ++i)
      add(i, ibps[i]);
  }

  RowEval eval(unsigned ibp_idx, unsigned item, const RawIntegral &seed,
               std::string &out) const {
    if (ibp_idx >= rows.size() || item >= rows[ibp_idx].size())
      return RowEval::NotPrecompiled;
    return eval_row_coeff(rows[ibp_idx][item], seed, names, out);
  }
};

// Collect one expression's symbols into the table's name list, in the order they
// are first seen.  GiNaC has no symbol-set query, so this walks the expression;
// the order is the spelling order and is deterministic per table.  Defined here,
// after RowTable, because its signature names the type.
inline void collect_row_symbols(const GiNaC::ex &e, RowTable &table) {
  if (GiNaC::is_a<GiNaC::symbol>(e)) {
    const std::string n = GiNaC::ex_to<GiNaC::symbol>(e).get_name();
    if (table.index.find(n) == table.index.end()) {
      table.index[n] = static_cast<unsigned>(table.names.size());
      table.names.push_back(n);
    }
    return;
  }
  for (std::size_t k = 0; k < e.nops(); ++k)
    collect_row_symbols(e.op(k), table);
}

inline void RowTable::add(unsigned ibp_idx, const IBPProtoM &items) {
  if (ibp_idx >= rows.size())
    rows.resize(ibp_idx + 1);
  if (!rows[ibp_idx].empty())
    return; // already built
  for (const auto &item : items)
    for (const GiNaC::ex &e : item.second)
      collect_row_symbols(e, *this);
  rows[ibp_idx].resize(items.size());
  for (unsigned i = 0; i < items.size(); ++i) {
    if (build_row_coeff(items[i].second, index, rows[ibp_idx][i]))
      ++fast_rows;
    else
      ++slow_rows;
  }
}

} // namespace secred
