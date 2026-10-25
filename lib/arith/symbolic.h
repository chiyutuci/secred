#pragma once

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "flint/fmpq_poly.h"
#include "flint/fmpz.h"

class mpoly {
public:
  mpoly() { fmpq_poly_init(_p); }

  mpoly(const mpoly &other) {
    fmpq_poly_init(_p);
    fmpq_poly_set(_p, other._p);
  }
  mpoly &operator=(const mpoly &other) {
    fmpq_poly_set(_p, other._p);
    return *this;
  }

  explicit mpoly(const std::string &s) {
    fmpq_poly_init(_p);
    if (fmpq_poly_set_str(_p, s.c_str()) != 0) {
      fmpq_poly_clear(_p);
      throw std::runtime_error("cannot parse polynomial: " + s);
    }
  }

  ~mpoly() { fmpq_poly_clear(_p); }

  void set_zero() {
    fmpq_poly_zero(_p);
  }

  bool is_zero() const { return fmpq_poly_is_zero(_p); }

  unsigned length() const { return fmpq_poly_length(_p); }

  slong denom() const {
    fmpz_t res;
    fmpz_init(res);
    fmpq_poly_get_denominator(res, _p);
    slong den = fmpz_get_si(res);
    fmpz_clear(res);
    return den;
  }

  std::pair<std::vector<unsigned>, std::vector<slong>> numer() const {
    std::pair<std::vector<unsigned>, std::vector<slong>> res;
    unsigned length = fmpq_poly_length(_p);
    fmpz_t coeff;
    fmpz_init(coeff);
    for (unsigned i = 0; i < length; ++i) {
      fmpq_poly_get_coeff_fmpz(coeff, _p, i);
      if (fmpz_is_zero(coeff) == 0) {
        slong num = fmpz_get_si(coeff);
        res.first.emplace_back(i);
        res.second.emplace_back(num);
      }
    }
    fmpz_clear(coeff);
    return res;
  }

  mpoly operator+(const mpoly &other) const {
    mpoly res;
    fmpq_poly_add(res._p, _p, other._p);
    return res;
  }

  mpoly &operator+=(const mpoly &other) {
    fmpq_poly_add(_p, _p, other._p);
    return *this;
  }

  mpoly operator*(const mpoly &other) const {
    mpoly res;
    fmpq_poly_mul(res._p, _p, other._p);
    return res;
  }

  mpoly operator*(int num) const {
    mpoly res;
    fmpq_poly_scalar_mul_si(res._p, _p, num);
    return res;
  }

  std::string str() const {
    char *cstr = fmpq_poly_get_str(_p);
    std::string s = cstr;
    free(cstr);
    cstr = nullptr;
    return s;
  }

  friend std::ostream &operator<<(std::ostream &out, const mpoly &poly) {
    out << poly.str();
    return out;
  }

private:
  fmpq_poly_t _p;
};