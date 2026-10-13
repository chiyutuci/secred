// Symbolic coefficients of the FiniteFlow global system, precompiled for evaluation.
//
// The rational-function reconstruction substitutes the same few hundred coefficients
// at every sample point, so the GiNaC expressions are expanded once into monomial
// tables.  A point then costs a handful of modular multiplications instead of a GiNaC
// `subs` tree walk, which measured ~800 ns even for a one-monomial coefficient; over
// the 23 million substitutions of a box.yaml reconstruction that was the entire cost
// of the stage (~26 s of 29 s).
//
// A coefficient the expander cannot represent (a non-integer or negative exponent, an
// operand that is not a symbol power, a coefficient that is not an integer) keeps its
// GiNaC form and goes through `subs`, so no input is silently mis-evaluated.
//
// Header-only so tests can drive the real precompiler and evaluator.

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cln/cln.h>
#include <ginac/ginac.h>

#include <fflow/common.hh>

#include "ffred_solver.h"

namespace secred {

struct SymTerm {
  cln::cl_I coef;
  // (symbol index, exponent) for the non-zero exponents only.
  std::vector<std::pair<uint8_t, uint8_t>> vars;
};

struct SymPoly {
  std::vector<SymTerm> terms;
  // `terms[i].coef` reduced mod `mod_p`, one entry per term.  Filled for a whole
  // system by `reduce_sym_system` *before* an evaluation phase and only read during
  // it, so the vector is never written while several threads are evaluating and one
  // reduced system can be shared by all of them.  `mod_p` remembers which prime the
  // values belong to, so a missed reduce is caught instead of answered with another
  // prime's residue.  A prime is never 0, and an unreduced polynomial is 0.
  std::vector<uint64_t> mod_vals;
  uint64_t mod_p = 0;
};

struct SymCoeff {
  SymPoly num;
  SymPoly den; // empty -> denominator is 1
  bool precompiled = false;
  GiNaC::ex slow_num; // fallback when !precompiled
  GiNaC::ex slow_den;
};

using SymEq = std::vector<std::pair<uint32_t, SymCoeff>>;

// cl_I mod p into uint64 (result in [0,p)).
inline uint64_t cl_I_mod_u64(const cln::cl_I &x, uint64_t p) {
  cln::cl_I r = cln::mod(x, cln::cl_I(p));
  if (cln::minusp(r))
    r += cln::cl_I(p);
  return static_cast<uint64_t>(cln::cl_I_to_ulong(r));
}

// a*b mod p for a, b < p.  p is a 63+-bit prime, so the product needs 128 bits; a
// plain uint64 multiply wraps and silently returns a wrong residue.
inline uint64_t mul_mod_u64(uint64_t a, uint64_t b, uint64_t p) {
  return static_cast<uint64_t>(static_cast<unsigned __int128>(a) * b % p);
}

// Expand one polynomial into a monomial table over the parameter symbols.  Returns
// false as soon as a term is out of the representable shape; the caller then keeps
// the GiNaC expression for that coefficient.  `e` must already be expanded.
inline bool build_sym_poly(const GiNaC::ex &e,
                           const std::unordered_map<std::string, unsigned> &sym_index,
                           SymPoly &out) {
  out.terms.clear();
  out.mod_vals.clear();
  out.mod_p = 0;
  auto one_term = [&](const GiNaC::ex &t) -> bool {
    SymTerm term;
    GiNaC::ex coef = GiNaC::ex(1);
    std::vector<GiNaC::ex> ops;
    if (GiNaC::is_a<GiNaC::mul>(t))
      for (std::size_t k = 0; k < t.nops(); ++k)
        ops.push_back(t.op(k));
    else
      ops.push_back(t);
    for (const GiNaC::ex &op : ops) {
      if (GiNaC::is_a<GiNaC::numeric>(op)) {
        coef = coef * op;
        continue;
      }
      GiNaC::ex base = op;
      unsigned exp = 1;
      if (GiNaC::is_a<GiNaC::power>(op)) {
        base = op.op(0);
        if (!GiNaC::is_a<GiNaC::numeric>(op.op(1)))
          return false;
        const GiNaC::numeric &ne = GiNaC::ex_to<GiNaC::numeric>(op.op(1));
        if (!ne.is_integer() || ne.is_negative() || ne.to_int() > 255)
          return false;
        exp = static_cast<unsigned>(ne.to_int());
      }
      if (!GiNaC::is_a<GiNaC::symbol>(base))
        return false;
      auto it = sym_index.find(GiNaC::ex_to<GiNaC::symbol>(base).get_name());
      if (it == sym_index.end() || it->second > 255)
        return false;
      term.vars.emplace_back(static_cast<uint8_t>(it->second),
                             static_cast<uint8_t>(exp));
    }
    if (!GiNaC::is_a<GiNaC::numeric>(coef))
      return false;
    const GiNaC::numeric &nc = GiNaC::ex_to<GiNaC::numeric>(coef);
    if (!nc.is_integer())
      return false;
    term.coef = cln::the<cln::cl_I>(nc.to_cl_N());
    out.terms.push_back(std::move(term));
    return true;
  };

  if (GiNaC::is_a<GiNaC::add>(e)) {
    for (std::size_t k = 0; k < e.nops(); ++k)
      if (!one_term(e.op(k)))
        return false;
  } else if (!e.is_zero()) {
    if (!one_term(e))
      return false;
  }
  return true;
}

// Reduce every coefficient of one polynomial mod p, so that evaluating it at p is a
// pure read.  Must be run for the whole system before it is evaluated at p.
inline void reduce_sym_poly(SymPoly &poly, uint64_t p) {
  poly.mod_vals.resize(poly.terms.size());
  for (std::size_t i = 0; i < poly.terms.size(); ++i)
    poly.mod_vals[i] = cl_I_mod_u64(poly.terms[i].coef, p);
  poly.mod_p = p;
}

// Reduce every precompiled coefficient of `system` mod p.  One prime covers all the
// sample points that share it, so this is the only place a coefficient is reduced:
// the 201-point ladder of a reconstruction then reuses the table instead of reducing
// the same coefficient again per point.  Coefficients that could not be precompiled
// carry no table; their GiNaC form is substituted instead.
inline void reduce_sym_system(std::vector<SymEq> &system, uint64_t p) {
  for (SymEq &eq : system)
    for (auto &pr : eq) {
      if (!pr.second.precompiled)
        continue;
      reduce_sym_poly(pr.second.num, p);
      reduce_sym_poly(pr.second.den, p);
    }
}

// Evaluate one precompiled polynomial.  `powers[i][k]` holds x_i^k, precomputed once
// per sample point because every coefficient shares the same powers.  False when the
// polynomial was not reduced for `p`, which would otherwise return another prime's
// residue; the caller turns that into a hard failure.
inline bool eval_sym_poly_mod(const SymPoly &poly,
                              const std::vector<std::vector<uint64_t>> &powers,
                              uint64_t p, uint64_t &out) {
  if (poly.mod_p != p || poly.mod_vals.size() != poly.terms.size())
    return false;
  unsigned __int128 acc = 0; // one reduction covers the whole sum
  for (std::size_t i = 0; i < poly.terms.size(); ++i) {
    uint64_t mon = poly.mod_vals[i];
    if (mon == 0)
      continue;
    for (const auto &ve : poly.terms[i].vars)
      mon = mul_mod_u64(mon, powers[ve.first][ve.second], p);
    acc += mon;
  }
  out = static_cast<uint64_t>(acc % p);
  return true;
}

// Precompile one coefficient.  `e` must already be expanded; the numerator and the
// denominator are precompiled separately so each evaluates to a plain integer.
inline SymCoeff
make_sym_coeff(const GiNaC::ex &e,
               const std::unordered_map<std::string, unsigned> &sym_index) {
  SymCoeff c;
  c.slow_num = e.numer();
  c.slow_den = e.denom();
  c.precompiled = build_sym_poly(c.slow_num, sym_index, c.num) &&
                  build_sym_poly(c.slow_den, sym_index, c.den);
  if (c.precompiled && c.slow_den.is_equal(GiNaC::ex(1)))
    c.den.terms.clear(); // no modular inverse needed for a trivial denominator
  return c;
}

// Largest exponent each symbol carries anywhere in the precompiled coefficients.
inline std::vector<unsigned> sym_max_exps(const std::vector<SymEq> &system,
                                          unsigned nvars) {
  std::vector<unsigned> mx(nvars, 0);
  for (const SymEq &eq : system)
    for (const auto &pr : eq)
      for (const SymPoly *poly : {&pr.second.num, &pr.second.den})
        for (const SymTerm &t : poly->terms)
          for (const auto &ve : t.vars)
            mx[ve.first] = std::max<unsigned>(mx[ve.first], ve.second);
  return mx;
}

// Refill x_i^k for every symbol i and every k the table is sized for.
inline void fill_powers(std::vector<std::vector<uint64_t>> &powers,
                        const fflow::UInt *x, uint64_t p) {
  for (std::size_t i = 0; i < powers.size(); ++i) {
    std::vector<uint64_t> &pw = powers[i];
    const uint64_t v = x[i] % p;
    for (std::size_t k = 1; k < pw.size(); ++k)
      pw[k] = mul_mod_u64(pw[k - 1], v, p);
  }
}

// Power table sized for `system` and filled at one sample point (used for the one-off
// evaluations that precede the reconstruction).
inline std::vector<std::vector<uint64_t>>
make_powers(const std::vector<SymEq> &system, const fflow::UInt *x,
            unsigned nvars, uint64_t p) {
  const std::vector<unsigned> mx = sym_max_exps(system, nvars);
  std::vector<std::vector<uint64_t>> powers(nvars, std::vector<uint64_t>(1, 1));
  for (unsigned i = 0; i < nvars; ++i)
    powers[i].resize(mx[i] + 1);
  fill_powers(powers, x, p);
  return powers;
}

// Evaluate a symbolic coefficient at the sample point and reduce mod p.  Numerator
// and denominator are evaluated separately, so both sides are plain integers.
// Writes 0 to `out` when the denominator vanishes mod p; standard IBP denominators
// are far below the prime, so this does not arise.  False means the precompiled
// tables were not reduced for `p` — see `reduce_sym_system`.
inline bool eval_sym_coeff_mod(
    const SymCoeff &c, const GiNaC::exmap &subs,
    const std::vector<std::vector<uint64_t>> &powers, uint64_t p, uint64_t &out) {
  if (c.precompiled) {
    uint64_t n = 0;
    if (!eval_sym_poly_mod(c.num, powers, p, n))
      return false;
    if (c.den.terms.empty()) {
      out = n;
      return true;
    }
    uint64_t d = 0;
    if (!eval_sym_poly_mod(c.den, powers, p, d))
      return false;
    if (d == 0) {
      out = 0;
      return true;
    }
    uint64_t inv = mod_inverse_u64(d, p);
    out = static_cast<uint64_t>((static_cast<__int128>(n) * inv) % p);
    return true;
  }

  GiNaC::ex nval = c.slow_num.subs(subs);
  if (!GiNaC::is_a<GiNaC::numeric>(nval)) {
    out = 0;
    return true;
  }
  const GiNaC::numeric &nv = GiNaC::ex_to<GiNaC::numeric>(nval);
  if (!nv.is_integer()) {
    out = 0;
    return true;
  }
  uint64_t n = cl_I_mod_u64(cln::the<cln::cl_I>(nv.to_cl_N()), p);

  uint64_t d = 1;
  if (!c.slow_den.is_equal(GiNaC::ex(1))) {
    GiNaC::ex dval = c.slow_den.subs(subs);
    if (!GiNaC::is_a<GiNaC::numeric>(dval)) {
      out = 0;
      return true;
    }
    const GiNaC::numeric &dv = GiNaC::ex_to<GiNaC::numeric>(dval);
    if (!dv.is_integer()) {
      out = 0;
      return true;
    }
    d = cl_I_mod_u64(cln::the<cln::cl_I>(dv.to_cl_N()), p);
  }
  if (d == 0) {
    out = 0;
    return true;
  }
  uint64_t inv = mod_inverse_u64(d, p);
  out = static_cast<uint64_t>((static_cast<__int128>(n) * inv) % p);
  return true;
}

} // namespace secred
