#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

#include "flint/fmpq.h"

class rational {
public:
  rational() { fmpq_init(_q); }

  rational(const rational &other) {
    fmpq_init(_q);
    fmpq_set(_q, other._q);
  }

  rational &operator=(const rational &other) {
    fmpq_set(_q, other._q);
    return *this;
  }

  explicit rational(int n) {
    fmpq_init(_q);
    fmpq_set_si(_q, n, 1);
  }

  explicit rational(const std::string &s) {
    fmpq_init(_q);
    if (fmpq_set_str(_q, s.c_str(), 10) != 0) {
      fmpq_clear(_q);
      throw std::runtime_error("cannot parse rational: " + s);
    }
  }

  // 底层 fmpq 访问(供多项式 GCD 约分等直接使用)
  const fmpq_t &raw() const { return _q; }

  static rational from_raw(const fmpq_t q) {
    rational r;
    fmpq_set(r._q, q);
    return r;
  }

  ~rational() { fmpq_clear(_q); }

  void set_zero() {
    fmpq_zero(_q);
  }
  bool is_zero() const { return fmpq_is_zero(_q); }

  rational operator+(const rational &other) const {
    rational res;
    fmpq_add(res._q, _q, other._q);
    return res;
  }

  rational operator*(const rational &other) const {
    rational res;
    fmpq_mul(res._q, _q, other._q);
    return res;
  }

  rational operator-() const {
    rational res;
    fmpq_neg(res._q, _q);
    return res;
  }

  rational operator-(const rational &other) const {
    rational res;
    fmpq_sub(res._q, _q, other._q);
    return res;
  }

  rational operator/(const rational &other) const {
    rational res;
    fmpq_div(res._q, _q, other._q);
    return res;
  }

  bool operator==(const rational &other) const {
    return fmpq_equal(_q, other._q) != 0;
  }

  bool operator!=(const rational &other) const {
    return !(*this == other);
  }

  rational &operator+=(const rational &other) {
    fmpq_add(_q, _q, other._q);
    return *this;
  }

  std::string str() const {
    char *cstr = fmpq_get_str(NULL, 10, _q);
    std::string s = cstr;
    free(cstr);
    cstr = nullptr;
    return s;
  }

  friend std::ostream &operator<<(std::ostream &out, const rational &rat) {
    out << rat.str();
    return out;
  }

private:
  fmpq_t _q;
};