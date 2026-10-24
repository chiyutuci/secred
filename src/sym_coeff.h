
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

// One monomial over symbol indices.
struct SymTerm {
  cln::cl_I coef;

  std::vector<std::pair<uint8_t, uint8_t>> vars;
};

// A polynomial over symbol indices.
struct SymPoly {
  std::vector<SymTerm> terms;

  std::vector<uint64_t> mod_vals;
  uint64_t mod_p = 0;
};

// A rational coefficient over symbol indices, with a GiNaC fallback.
struct SymCoeff {
  SymPoly num;
  SymPoly den;
  bool precompiled = false;
  GiNaC::ex slow_num;
  GiNaC::ex slow_den;
};

using SymEq = std::vector<std::pair<uint32_t, SymCoeff>>;

inline uint64_t cl_I_mod_u64(const cln::cl_I &x, uint64_t p) {
  cln::cl_I r = cln::mod(x, cln::cl_I(p));
  if (cln::minusp(r))
    r += cln::cl_I(p);
  return static_cast<uint64_t>(cln::cl_I_to_ulong(r));
}

inline uint64_t mul_mod_u64(uint64_t a, uint64_t b, uint64_t p) {
  return static_cast<uint64_t>(static_cast<unsigned __int128>(a) * b % p);
}

// Compile a GiNaC expression into a SymPoly.
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

inline void reduce_sym_poly(SymPoly &poly, uint64_t p) {
  poly.mod_vals.resize(poly.terms.size());
  for (std::size_t i = 0; i < poly.terms.size(); ++i)
    poly.mod_vals[i] = cl_I_mod_u64(poly.terms[i].coef, p);
  poly.mod_p = p;
}

// Reduce every precompiled polynomial modulo p.
inline void reduce_sym_system(std::vector<SymEq> &system, uint64_t p) {
  for (SymEq &eq : system)
    for (auto &pr : eq) {
      if (!pr.second.precompiled)
        continue;
      reduce_sym_poly(pr.second.num, p);
      reduce_sym_poly(pr.second.den, p);
    }
}

inline bool eval_sym_poly_mod(const SymPoly &poly,
                              const std::vector<std::vector<uint64_t>> &powers,
                              uint64_t p, uint64_t &out) {
  if (poly.mod_p != p || poly.mod_vals.size() != poly.terms.size())
    return false;
  unsigned __int128 acc = 0;
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

inline SymCoeff
make_sym_coeff(const GiNaC::ex &e,
               const std::unordered_map<std::string, unsigned> &sym_index) {
  SymCoeff c;
  c.slow_num = e.numer();
  c.slow_den = e.denom();
  c.precompiled = build_sym_poly(c.slow_num, sym_index, c.num) &&
                  build_sym_poly(c.slow_den, sym_index, c.den);
  if (c.precompiled && c.slow_den.is_equal(GiNaC::ex(1)))
    c.den.terms.clear();
  return c;
}

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

inline void fill_powers(std::vector<std::vector<uint64_t>> &powers,
                        const fflow::UInt *x, uint64_t p) {
  for (std::size_t i = 0; i < powers.size(); ++i) {
    std::vector<uint64_t> &pw = powers[i];
    const uint64_t v = x[i] % p;
    for (std::size_t k = 1; k < pw.size(); ++k)
      pw[k] = mul_mod_u64(pw[k - 1], v, p);
  }
}

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

// Evaluate every coefficient of a system at one field point.
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

inline constexpr std::size_t kSymCoeffMaxVars = 16;

inline bool sym_term_less(const SymTerm &a, const SymTerm &b) {
  if (a.vars != b.vars)
    return a.vars < b.vars;
  return a.coef < b.coef;
}

inline bool sym_poly_internable(const SymPoly &poly) {
  for (const SymTerm &t : poly.terms)
    if (t.vars.size() > kSymCoeffMaxVars)
      return false;
  return true;
}

// Canonical form used as the interning key.
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
  h *= 1099511628211ULL;
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

// A sparse system whose coefficients are interned slots.
struct SymCoeffPool {
  std::vector<SymCoeff> coeffs;
  std::vector<std::pair<uint32_t, uint32_t>> terms;
  std::vector<uint32_t> eq_begin;
  std::size_t n_equations() const { return eq_begin.empty() ? 0 : eq_begin.size() - 1; }
  std::size_t n_terms() const { return terms.size(); }
};

// Builds a SymCoeffPool, deduplicating equal coefficients.
struct SymCoeffInterner {
  SymCoeffPool pool;
  std::unordered_map<SymCoeff, uint32_t, SymCoeffHash, SymCoeffEq> slot_of;
  SymCoeff probe;

  void begin_row() { pool.eq_begin.push_back(static_cast<uint32_t>(pool.terms.size())); }

  void add_term(uint32_t id, const SymCoeff &c) { pool.terms.emplace_back(id, intern(c)); }

  void finish() {
    pool.eq_begin.push_back(static_cast<uint32_t>(pool.terms.size()));
  }

  std::uint32_t intern(const SymCoeff &c) {
    bool internable = true;
    if (c.precompiled) {
      internable = sym_poly_internable(c.num) && sym_poly_internable(c.den);
      if (internable) {
        sym_poly_canonical_into(c.num, probe.num);
        sym_poly_canonical_into(c.den, probe.den);
      }
    } else {

      probe.num.terms.clear();
      probe.den.terms.clear();
    }
    probe.precompiled = c.precompiled;
    probe.slow_num = c.slow_num;
    probe.slow_den = c.slow_den;

    if (!internable) {
      pool.coeffs.push_back(c);
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

// Intern the rows eq_idx of an existing system.
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

// Drop coefficients unreferenced by the retained rows.
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

inline void reduce_pool_coeff(SymCoeff &c, uint64_t p) {
  if (!c.precompiled)
    return;
  reduce_sym_poly(c.num, p);
  reduce_sym_poly(c.den, p);
}

inline void reduce_pool_for_prime(SymCoeffPool &pool, uint64_t p) {
  for (SymCoeff &c : pool.coeffs)
    reduce_pool_coeff(c, p);
}

inline std::vector<unsigned> sym_max_exps(const SymCoeffPool &pool, unsigned nvars) {
  std::vector<unsigned> mx(nvars, 0);
  for (const SymCoeff &c : pool.coeffs)
    for (const SymPoly *poly : {&c.num, &c.den})
      for (const SymTerm &t : poly->terms)
        for (const auto &ve : t.vars)
          mx[ve.first] = std::max<unsigned>(mx[ve.first], ve.second);
  return mx;
}

inline std::vector<std::vector<uint64_t>>
make_powers(const SymCoeffPool &pool, const fflow::UInt *x, unsigned nvars, uint64_t p) {
  const std::vector<unsigned> mx = sym_max_exps(pool, nvars);
  std::vector<std::vector<uint64_t>> powers(nvars, std::vector<uint64_t>(1, 1));
  for (unsigned i = 0; i < nvars; ++i)
    powers[i].resize(mx[i] + 1);
  fill_powers(powers, x, p);
  return powers;
}

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

  if (zero_dens)
    *zero_dens += zero_here;
  return true;
}

// Evaluate the retained rows at one field point.
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
  ResEq re;
  for (uint32_t i : eq_idx) {
    re.clear();
    const std::size_t t0 = pool.eq_begin[i], t1 = pool.eq_begin[i + 1];
    re.reserve(t1 - t0);
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

}
