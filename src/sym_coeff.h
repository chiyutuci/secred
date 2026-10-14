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
// Writes 0 to `out` when the denominator vanishes mod p, setting `*zero_den` when the
// caller passes it: such a coefficient is indistinguishable from a true zero in the
// result, so the count is the only record that the point is not trustworthy; standard
// IBP denominators are far below the prime, so this does not arise.  False means the
// precompiled tables were not reduced for `p` — see `reduce_sym_system`.
inline bool eval_sym_coeff_mod(
    const SymCoeff &c, const GiNaC::exmap &subs,
    const std::vector<std::vector<uint64_t>> &powers, uint64_t p, uint64_t &out,
    bool *zero_den = nullptr) {
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
      if (zero_den)
        *zero_den = true;
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
    if (zero_den)
      *zero_den = true;
    return true;
  }
  uint64_t inv = mod_inverse_u64(d, p);
  out = static_cast<uint64_t>((static_cast<__int128>(n) * inv) % p);
  return true;
}

// Evaluate `system` over the `eq_idx` subset at one sample point and prime.  False when
// the coefficient tables have not been reduced for `p` (see reduce_sym_system); a
// coefficient whose denominator vanishes mod p evaluates to 0, and `zero_dens` counts
// those points (they are indistinguishable from a true zero in the result).
inline bool eval_sym_eqs_mod(const std::vector<SymEq> &system,
                             const std::vector<uint32_t> &eq_idx,
                             const GiNaC::exmap &subs,
                             const std::vector<std::vector<uint64_t>> &powers,
                             uint64_t p, std::vector<ResEq> &res_eqs,
                             std::vector<uint32_t> &sys_idx,
                             std::uint64_t *zero_dens = nullptr) {
  res_eqs.clear();
  sys_idx.clear();
  res_eqs.reserve(eq_idx.size());
  sys_idx.reserve(eq_idx.size());
  for (uint32_t i : eq_idx) {
    const SymEq &eq = system[i];
    ResEq re;
    re.reserve(eq.size());
    for (const auto &[id, c] : eq) {
      uint64_t v = 0;
      bool zd = false;
      if (!eval_sym_coeff_mod(c, subs, powers, p, v, &zd))
        return false;
      if (zd && zero_dens)
        *zero_dens += 1;
      if (v)
        re.emplace_back(id, v);
    }
    if (!re.empty()) {
      res_eqs.push_back(std::move(re));
      sys_idx.push_back(i);
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Coefficient interning
// ---------------------------------------------------------------------------
//
// A global system spells the same rational function out again and again: Kira's own log
// for F234 reads "Parsed 8040387 functions of which 1617 are distinct".  Evaluating every
// (integral, coefficient) entry of the reconstructed system at every sample point -- with
// a modular inverse each, ~11 million of them for F234 -- was the dominant cost of a
// point.  The pool evaluates each *distinct* function once per point and lets the entries
// read the answer out of the table, which is Kira's model.

// A term with more distinct symbols than this is not interned: a coefficient that wide is
// unlikely to repeat, so the table slot would only be paid for and never shared.  A missed
// intern costs one duplicate evaluation, never a wrong residue.
inline constexpr std::size_t kSymCoeffMaxVars = 16;

// The shape interning keys on: each term's (symbol, exponent) pairs sorted, then the terms
// sorted.  One function can be spelled several ways -- the strict spill grammar
// (spill_reader.cpp) and GiNaC's parser order a monomial's factors their own way -- and
// the key must see through that: a miss costs a duplicate evaluation, while a wrong match
// would hand an entry another function's residue.  The aligned `mod_vals` table is
// dropped, because it belongs to a term order that no longer holds; `eval_sym_poly_mod`
// refuses a table that was not reduced for the prime it is asked about, so the next
// `reduce_pool_for_prime` is what refills it.
inline bool sym_term_less(const SymTerm &a, const SymTerm &b) {
  if (a.vars != b.vars)
    return a.vars < b.vars;   // pairs compare by symbol index, then exponent
  return a.coef < b.coef;
}

inline bool sym_poly_internable(const SymPoly &poly) {
  for (const SymTerm &t : poly.terms)
    if (t.vars.size() > kSymCoeffMaxVars)
      return false;
  return true;
}

// Canonical copy of one polynomial into `out`.  `out` is reused across coefficients, so
// its buffers keep their capacity; only a coefficient that grows past the previous one
// allocates.
inline void sym_poly_canonical_into(const SymPoly &in, SymPoly &out) {
  out.terms.resize(in.terms.size());
  for (std::size_t i = 0; i < in.terms.size(); ++i) {
    out.terms[i].coef = in.terms[i].coef;
    out.terms[i].vars.assign(in.terms[i].vars.begin(), in.terms[i].vars.end());
    std::sort(out.terms[i].vars.begin(), out.terms[i].vars.end());
  }
  std::sort(out.terms.begin(), out.terms.end(), sym_term_less);
  out.mod_vals.clear();
  out.mod_p = 0;
}

// Two canonical polynomials of the same function are equal term by term.  `cln::equal_
// hashcode` is invariant under equal integers, so hashing may key on it.
inline bool sym_poly_equal(const SymPoly &a, const SymPoly &b) {
  if (a.terms.size() != b.terms.size())
    return false;
  for (std::size_t i = 0; i < a.terms.size(); ++i)
    if (a.terms[i].vars != b.terms[i].vars || !(a.terms[i].coef == b.terms[i].coef))
      return false;
  return true;
}

inline void sym_hash_mix(std::uint64_t &h, std::uint64_t v) {
  h ^= v;
  h *= 1099511628211ULL; // FNV-1a
}

inline std::uint64_t hash_sym_poly(const SymPoly &poly) {
  std::uint64_t h = 1469598103934665603ULL;
  sym_hash_mix(h, poly.terms.size());
  for (const SymTerm &t : poly.terms) {
    sym_hash_mix(h, cln::equal_hashcode(t.coef));
    sym_hash_mix(h, t.vars.size());
    for (const auto &ve : t.vars)
      sym_hash_mix(h, (static_cast<std::uint64_t>(ve.first) << 8) | ve.second);
  }
  return h;
}

// The interning key.  A coefficient the precompiler could not represent keeps its GiNaC
// form and is keyed on that instead (`spill_reader.cpp`'s fast path never sets
// `slow_num`/`slow_den`, so keying a precompiled coefficient on those would collapse every
// one of them into a single slot -- a wrong residue, not a missed optimisation).
struct SymCoeffHash {
  std::size_t operator()(const SymCoeff &c) const {
    std::uint64_t h = 1469598103934665603ULL;
    if (!c.precompiled) {
      sym_hash_mix(h, c.slow_num.gethash());
      sym_hash_mix(h, c.slow_den.gethash());
      return static_cast<std::size_t>(h);
    }
    sym_hash_mix(h, hash_sym_poly(c.num));
    sym_hash_mix(h, hash_sym_poly(c.den));
    return static_cast<std::size_t>(h);
  }
};

struct SymCoeffEq {
  bool operator()(const SymCoeff &a, const SymCoeff &b) const {
    if (a.precompiled != b.precompiled)
      return false;
    if (!a.precompiled)
      return a.slow_num.is_equal(b.slow_num) && a.slow_den.is_equal(b.slow_den);
    return sym_poly_equal(a.num, b.num) && sym_poly_equal(a.den, b.den);
  }
};

// The distinct coefficients of a selected system, with every equation term pointing at
// one of them.  Built once, before the reconstruction; one prime's tables live in
// `coeffs`, so a sample point evaluates `coeffs.size()` functions and the entries read the
// rest out of the table.  `terms` and `eq_begin` are read-only during evaluation, so one
// pool serves every thread.
struct SymCoeffPool {
  std::vector<SymCoeff> coeffs;                     // the distinct functions
  std::vector<std::pair<uint32_t, uint32_t>> terms;  // (integral id, index into coeffs)
  std::vector<uint32_t> eq_begin;                    // n_equations() + 1 row boundaries
  std::size_t n_equations() const { return eq_begin.empty() ? 0 : eq_begin.size() - 1; }
  std::size_t n_terms() const { return terms.size(); }
};

// The interning table itself, usable while a system is still being read.
//
// `build_sym_coeff_pool` fills a pool from a materialised system; the spill reader needs
// the same key and the same slot assignment while it parses, so that a row appended
// during the read is the row a second pass would have produced.  Both go through this
// object, and neither can drift from the other.
struct SymCoeffInterner {
  SymCoeffPool pool;
  std::unordered_map<SymCoeff, uint32_t, SymCoeffHash, SymCoeffEq> slot_of;
  SymCoeff probe;   // one reused key: filling it allocates nothing after the first terms

  // One equation's term table starts here.  Rows are appended in order, so a row is the
  // half-open range [terms[eq_begin[i]], terms[eq_begin[i + 1]]).
  void begin_row() { pool.eq_begin.push_back(static_cast<uint32_t>(pool.terms.size())); }

  void add_term(uint32_t id, const SymCoeff &c) { pool.terms.emplace_back(id, intern(c)); }

  // The pool's last row boundary.  Every `begin_row` needs one `finish` at the end of the
  // last row, which is what makes `eq_begin` have n_rows + 1 entries.
  void finish() {
    pool.eq_begin.push_back(static_cast<uint32_t>(pool.terms.size()));
  }

  // The slot that evaluates `c`.  A coefficient the precompiler could not represent, or
  // one too wide to be worth a table slot, is never interned: it takes its own slot, and
  // a repeat of it costs one duplicate evaluation rather than a wrong residue.
  std::uint32_t intern(const SymCoeff &c) {
    bool internable = true;
    if (c.precompiled) {
      internable = sym_poly_internable(c.num) && sym_poly_internable(c.den);
      if (internable) {
        sym_poly_canonical_into(c.num, probe.num);
        sym_poly_canonical_into(c.den, probe.den);
      }
    } else {
      // Keyed on the GiNaC form; the partial monomial tables are not a key.
      probe.num.terms.clear();
      probe.den.terms.clear();
    }
    probe.precompiled = c.precompiled;
    probe.slow_num = c.slow_num;
    probe.slow_den = c.slow_den;

    if (!internable) {
      pool.coeffs.push_back(c); // never interned: it gets its own slot
      return static_cast<std::uint32_t>(pool.coeffs.size() - 1);
    }
    const auto it = slot_of.find(probe);
    if (it != slot_of.end())
      return it->second;
    pool.coeffs.push_back(probe);
    const std::uint32_t slot = static_cast<std::uint32_t>(pool.coeffs.size() - 1);
    slot_of.emplace(pool.coeffs.back(), slot);
    return slot;
  }
};

// Canonicalise the selected coefficients of `system`, intern them, and fill the pool.
// Every equation keeps the system's term order (descending integral id, as the spill
// reader wrote it) and every term points at the one slot that evaluates it.
inline void build_sym_coeff_pool(const std::vector<SymEq> &system,
                                 const std::vector<uint32_t> &eq_idx,
                                 SymCoeffPool &pool) {
  SymCoeffInterner in;
  in.pool.eq_begin.reserve(eq_idx.size() + 1);
  for (uint32_t i : eq_idx) {
    in.begin_row();
    for (const auto &pr : system[i])
      in.add_term(pr.first, pr.second);
  }
  in.finish();
  pool = std::move(in.pool);
}

// Intern a whole materialised system -- the shape `read_spill_symbolic` produces.  The
// reader has its own interning sink (spill_reader.h) and never materialises one; this is
// for the callers that already hold a system.
inline void intern_sym_system(const std::vector<SymEq> &system, SymCoeffPool &pool) {
  SymCoeffInterner in;
  in.pool.eq_begin.reserve(system.size() + 1);
  for (const SymEq &eq : system) {
    in.begin_row();
    for (const auto &pr : eq)
      in.add_term(pr.first, pr.second);
  }
  in.finish();
  pool = std::move(in.pool);
}

// Keep only the rows `eq_idx` names, in that order, and release the rest: both the term
// table and the coefficient table shrink to what those rows touch, so a pool read as a
// whole system (every equation the spill held) becomes the pool of the selection the
// learn made, which is the only part the reconstruction evaluates.  A coefficient no
// remaining row uses is dropped, and the surviving slots are renumbered densely; the
// rows keep their order and their terms, so a row of `eq_idx[k]` is row `k` afterwards.
inline void retain_pool_rows(SymCoeffPool &pool, const std::vector<uint32_t> &eq_idx) {
  std::vector<std::pair<uint32_t, uint32_t>> terms;
  std::vector<uint32_t> eq_begin;
  std::vector<uint32_t> remap(pool.coeffs.size(), UINT32_MAX);
  std::vector<SymCoeff> coeffs;
  std::size_t n_terms = 0;
  for (uint32_t i : eq_idx)
    n_terms += pool.eq_begin[i + 1] - pool.eq_begin[i];
  terms.reserve(n_terms);
  eq_begin.reserve(eq_idx.size() + 1);
  for (uint32_t i : eq_idx) {
    eq_begin.push_back(static_cast<uint32_t>(terms.size()));
    for (std::size_t t = pool.eq_begin[i], te = pool.eq_begin[i + 1]; t < te; ++t) {
      uint32_t &slot = remap[pool.terms[t].second];
      if (slot == UINT32_MAX) {
        slot = static_cast<uint32_t>(coeffs.size());
        coeffs.push_back(std::move(pool.coeffs[pool.terms[t].second]));
      }
      terms.emplace_back(pool.terms[t].first, slot);
    }
  }
  eq_begin.push_back(static_cast<uint32_t>(terms.size()));
  pool.coeffs = std::move(coeffs);
  pool.terms = std::move(terms);
  pool.eq_begin = std::move(eq_begin);
}

// Reduce one pool coefficient mod p (a coefficient the precompiler could not represent
// carries no table; its GiNaC form is substituted at every point instead).
inline void reduce_pool_coeff(SymCoeff &c, uint64_t p) {
  if (!c.precompiled)
    return;
  reduce_sym_poly(c.num, p);
  reduce_sym_poly(c.den, p);
}

// Reduce every distinct coefficient mod p.  One prime covers all the sample points that
// share it, so this is the only place a pool coefficient is reduced.
inline void reduce_pool_for_prime(SymCoeffPool &pool, uint64_t p) {
  for (SymCoeff &c : pool.coeffs)
    reduce_pool_coeff(c, p);
}

// Largest exponent each symbol carries anywhere in the pool's precompiled coefficients.
inline std::vector<unsigned> sym_max_exps(const SymCoeffPool &pool, unsigned nvars) {
  std::vector<unsigned> mx(nvars, 0);
  for (const SymCoeff &c : pool.coeffs)
    for (const SymPoly *poly : {&c.num, &c.den})
      for (const SymTerm &t : poly->terms)
        for (const auto &ve : t.vars)
          mx[ve.first] = std::max<unsigned>(mx[ve.first], ve.second);
  return mx;
}

// Power table sized for a pool's distinct coefficients and filled at one sample point --
// the pool form of the overload above, and cheaper by the interning factor: the
// exponents are read off the distinct functions (1329 of them at F234's size), not off
// every entry of the system.
inline std::vector<std::vector<uint64_t>>
make_powers(const SymCoeffPool &pool, const fflow::UInt *x, unsigned nvars, uint64_t p) {
  const std::vector<unsigned> mx = sym_max_exps(pool, nvars);
  std::vector<std::vector<uint64_t>> powers(nvars, std::vector<uint64_t>(1, 1));
  for (unsigned i = 0; i < nvars; ++i)
    powers[i].resize(mx[i] + 1);
  fill_powers(powers, x, p);
  return powers;
}

// Evaluate one sample point: `values[i]` is the residue of `pool.coeffs[i]`, one entry per
// distinct function.  The equations are not evaluated: every entry reads its coefficient
// out of `values` through the pool's term table, which is what
// forward_eliminate_planned's SymRowTable walks row by row, so a point holds one slot array
// instead of a row per equation.  False when the tables were not reduced for `p` (see
// reduce_pool_for_prime): the caller must not evaluate against the wrong modulus.
inline bool eval_pool_point(const SymCoeffPool &pool, const GiNaC::exmap &subs,
                            const std::vector<std::vector<uint64_t>> &powers,
                            uint64_t p, std::vector<uint64_t> &values,
                            std::uint64_t *zero_dens = nullptr) {
  std::uint64_t zero_here = 0;
  values.resize(pool.coeffs.size());
  for (std::size_t i = 0; i < pool.coeffs.size(); ++i) {
    bool zero_den = false;
    if (!eval_sym_coeff_mod(pool.coeffs[i], subs, powers, p, values[i], &zero_den))
      return false;
    zero_here += zero_den ? 1 : 0;
  }
  // One count per distinct function, not per entry: the entries share the answer.
  if (zero_dens)
    *zero_dens += zero_here;
  return true;
}

// The pool's `eq_idx` rows at one sample point, as numeric equations: the same result
// `eval_sym_eqs_mod` produces from a materialised system, but with every distinct
// coefficient evaluated once and each entry read out of the slot table.  `sys_idx[k]` is
// the pool row `res_eqs[k]` came from, so a caller that indexes the system by the
// learn's indices keeps working; a row whose coefficients all vanished is dropped, the
// way the reference drops it.
inline bool eval_pool_rows(const SymCoeffPool &pool, const std::vector<uint32_t> &eq_idx,
                           const GiNaC::exmap &subs,
                           const std::vector<std::vector<uint64_t>> &powers, uint64_t p,
                           std::vector<ResEq> &res_eqs, std::vector<uint32_t> &sys_idx,
                           std::uint64_t *zero_dens = nullptr) {
  std::vector<uint64_t> values;
  if (!eval_pool_point(pool, subs, powers, p, values, zero_dens))
    return false;
  res_eqs.clear();
  sys_idx.clear();
  res_eqs.reserve(eq_idx.size());
  sys_idx.reserve(eq_idx.size());
  ResEq re; // hoisted: a dropped row's buffer is reused by the next row
  for (uint32_t i : eq_idx) {
    re.clear();
    const std::size_t t0 = pool.eq_begin[i], t1 = pool.eq_begin[i + 1];
    re.reserve(t1 - t0); // the row's size up front, not a doubling sequence
    for (std::size_t t = t0; t < t1; ++t) {
      const uint64_t v = values[pool.terms[t].second];
      if (v)
        re.emplace_back(pool.terms[t].first, v);
    }
    if (!re.empty()) {
      res_eqs.push_back(std::move(re));
      sys_idx.push_back(i);
    }
  }
  return true;
}

} // namespace secred
