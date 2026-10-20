#pragma once

#include <cstdint>
#include <iostream>
#include <string>

#include "flint/fmpq.h"
#include "flint/fmpz.h"

#include "numeric.h"

// Finite-field element: a residue modulo FiniteFlow's learn() prime
// (BIG_UINT_PRIMES[0]).  Coefficients on the numeric FiniteFlow path are
// reduced here instead of being carried as full arbitrary-precision rationals
// (fmpq / MPRational) — the source of FiniteFlow's memory cost relative to
// pyred's finite-field Coeff_int.  The modulus must equal FiniteFlow's
// learn() modulus (LearningOptions::prime_no == 0 -> BIG_UINT_PRIMES[0]) so
// the structural elimination sees the same field.
class ffint {
public:
  static constexpr std::uint64_t PRIME = 9223372036854775783ULL;

  ffint() : v_(0) {}
  explicit ffint(std::uint64_t x) : v_(x % PRIME) {}
  explicit ffint(int x) : v_(mod_si(x)) {}
  explicit ffint(const std::string &s) : ffint(rational(s)) {}
  explicit ffint(const rational &r) : v_(from_fmpq(r)) {}

  ffint operator+(const ffint &o) const { return ffint(addmod(v_, o.v_)); }
  ffint operator-(const ffint &o) const { return ffint(submod(v_, o.v_)); }
  ffint operator*(const ffint &o) const { return ffint(mulmod(v_, o.v_)); }
  ffint operator/(const ffint &o) const { return ffint(mulmod(v_, inv(o.v_))); }
  ffint operator-() const { return ffint(v_ == 0 ? 0 : PRIME - v_); }

  ffint &operator+=(const ffint &o) {
    v_ = addmod(v_, o.v_);
    return *this;
  }
  ffint &operator*=(const ffint &o) {
    v_ = mulmod(v_, o.v_);
    return *this;
  }

  bool operator==(const ffint &o) const { return v_ == o.v_; }
  bool operator!=(const ffint &o) const { return v_ != o.v_; }
  bool is_zero() const { return v_ == 0; }

  std::uint64_t value() const { return v_; }
  std::string str() const { return std::to_string(v_); }

  friend std::ostream &operator<<(std::ostream &out, const ffint &f) {
    out << f.str();
    return out;
  }

private:
  std::uint64_t v_;

  static std::uint64_t addmod(std::uint64_t a, std::uint64_t b) {
    std::uint64_t s = a + b; // a+b < 2*PRIME < 2^64, no overflow
    return s >= PRIME ? s - PRIME : s;
  }
  static std::uint64_t submod(std::uint64_t a, std::uint64_t b) {
    std::int64_t d = (std::int64_t)a - (std::int64_t)b;
    if (d < 0)
      d += (std::int64_t)PRIME;
    return (std::uint64_t)d;
  }
  static std::uint64_t mulmod(std::uint64_t a, std::uint64_t b) {
    return (std::uint64_t)((__uint128_t)a * b % PRIME);
  }
  static std::uint64_t powmod(std::uint64_t a, std::uint64_t e) {
    std::uint64_t r = 1;
    while (e) {
      if (e & 1)
        r = mulmod(r, a);
      a = mulmod(a, a);
      e >>= 1;
    }
    return r;
  }
  static std::uint64_t inv(std::uint64_t a) { return powmod(a, PRIME - 2); }

  static std::uint64_t mod_si(int x) {
    if (x >= 0)
      return (std::uint64_t)x;
    return PRIME - (std::uint64_t)(-(std::int64_t)x);
  }

  // Reduce a rational (n/d) to its residue n * d^{-1} mod PRIME.
  static std::uint64_t from_fmpq(const rational &r) {
    const fmpq_t &q = r.raw();
    std::uint64_t n = (std::uint64_t)fmpz_fdiv_ui(fmpq_numref(q), (ulong)PRIME);
    std::uint64_t d = (std::uint64_t)fmpz_fdiv_ui(fmpq_denref(q), (ulong)PRIME);
    return mulmod(n, inv(d));
  }
};
