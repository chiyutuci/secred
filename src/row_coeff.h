
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

// One monomial: variable index and exponent.
using RowExp = std::pair<std::uint8_t, unsigned>;
using RowVars = std::vector<RowExp>;

// One term of a row numerator.
struct RowTerm {
  cln::cl_I coef;
  RowVars vars;
};

using RowNum = std::vector<RowTerm>;

// A coefficient with a shared monomial and denominator.
struct RowPoly {
  SymPoly num;
  std::uint32_t q = 1;
  RowVars mono;
};

// Precompiled coefficients of one IBP row.
struct RowCoeff {
  bool precompiled = false;
  std::vector<RowPoly> c;
};

enum class RowEval {
  Ok,
  Zero,
  NotPrecompiled,
};

inline unsigned row_exp_of(const RowVars &v, std::uint8_t idx) {
  for (const RowExp &e : v)
    if (e.first == idx)
      return e.second;
  return 0;
}

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
      den = 1;
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

      if (out != i)
        v[out] = std::move(v[i]);
      ++out;
    }
    i = j;
  }
  v.resize(out);
}

inline unsigned sym_term_exp_of(const std::vector<std::pair<std::uint8_t, std::uint8_t>> &v,
                                std::uint8_t idx) {
  for (const auto &ve : v)
    if (ve.first == idx)
      return ve.second;
  return 0;
}

// Compile a row's GiNaC coefficients for fast evaluation.
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
    if (!sc.precompiled())
      return false;
    RowPoly c;
    if (!sc.den().terms.empty()) {
      const SymTerm &dn = sc.den().terms[0];
      if (sc.den().terms.size() != 1 || !cln::plusp(dn.coef) ||
          dn.coef > cln::cl_I(static_cast<unsigned long>(kMaxDen)))
        return false;
      c.q = static_cast<std::uint32_t>(cln::cl_I_to_ulong(dn.coef));

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
    c.num = sc.num();
    out.c.push_back(std::move(c));
  }

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
          s.vars.push_back(ve);
      for (const RowExp &me : mono) {
        const unsigned e = sym_term_exp_of(t.vars, me.first) + me.second -
                           row_exp_of(c.mono, me.first);
        if (e > 255)
          return false;
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

// Evaluate a row coefficient at one seed.
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

// Precompiled coefficient tables for every IBP row.
struct RowTable {
  std::vector<std::string> names;
  std::unordered_map<std::string, unsigned> index;
  std::vector<std::vector<RowCoeff>> rows;
  unsigned fast_rows = 0;
  unsigned slow_rows = 0;

  bool contains(unsigned ibp_idx) const {
    return ibp_idx < rows.size() && !rows[ibp_idx].empty();
  }

  void add(unsigned ibp_idx, const IBPProtoM &items);

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
    return;
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

}
