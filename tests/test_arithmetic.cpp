#include "test_runner.h"
#include "arith/ffint.h"
#include "arith/numeric.h"
#include "arith/symbolic.h"

// ---------------------------------------------------------------------------
// rational — arbitrary-precision rational number (FLINT fmpq_t)
// ---------------------------------------------------------------------------

TEST(rational_default_zero_after_set_zero) {
  rational r;
  r.set_zero();
  REQUIRE(r.is_zero());
}

TEST(rational_construct_from_int_zero) {
  rational r(0);
  REQUIRE(r.is_zero());
}

TEST(rational_construct_from_int_nonzero) {
  rational r(5);
  REQUIRE(!r.is_zero());
}

TEST(rational_construct_from_string) {
  rational r("3/4");
  REQUIRE(!r.is_zero());
}

TEST(rational_rejects_a_malformed_string) {
  // fmpq_set_str reports failure; a silent zero would hide a coefficient the
  // reduction cannot represent.
  bool threw = false;
  try {
    rational r("not a number");
  } catch (const std::runtime_error &e) {
    threw = true;
    REQUIRE(std::string(e.what()).find("not a number") != std::string::npos);
  }
  REQUIRE(threw);
}

TEST(rational_rejects_a_partially_parsed_string) {
  // Only a whole numerator and denominator parse; the rest is not ignored.
  bool threw = false;
  try {
    rational r("1/2/3");
  } catch (const std::runtime_error &e) {
    threw = true;
    REQUIRE(std::string(e.what()).find("1/2/3") != std::string::npos);
  }
  REQUIRE(threw);
}

TEST(rational_str_integer) {
  rational r(7);
  REQUIRE(r.str() == "7");
}

TEST(rational_str_fraction) {
  rational r("2/3");
  REQUIRE(r.str() == "2/3");
}

TEST(rational_str_zero) {
  rational r(0);
  REQUIRE(r.str() == "0");
}

TEST(rational_str_negative) {
  rational r(-3);
  REQUIRE(r.str() == "-3");
}

TEST(rational_addition_integers) {
  rational a(3), b(4);
  rational c = a + b;
  REQUIRE(c.str() == "7");
}

TEST(rational_addition_fractions) {
  rational a("1/2"), b("1/3");
  rational c = a + b;
  REQUIRE(c.str() == "5/6");
}

TEST(rational_addition_with_zero) {
  rational a(5), b;
  b.set_zero();
  rational c = a + b;
  REQUIRE(c.str() == "5");
}

TEST(rational_multiplication_integers) {
  rational a(3), b(4);
  rational c = a * b;
  REQUIRE(c.str() == "12");
}

TEST(rational_multiplication_fractions) {
  rational a("2/3"), b("3/4");
  rational c = a * b;
  REQUIRE(c.str() == "1/2");
}

TEST(rational_multiplication_by_zero) {
  rational a(5), b(0);
  rational c = a * b;
  REQUIRE(c.is_zero());
}

TEST(rational_plus_equals) {
  rational a(3), b(2);
  a += b;
  REQUIRE(a.str() == "5");
}

TEST(rational_plus_equals_fraction) {
  rational a("1/4"), b("3/4");
  a += b;
  REQUIRE(a.str() == "1");
}

TEST(rational_copy_constructor) {
  rational a(42);
  rational b(a);
  REQUIRE(b.str() == "42");
  // modifying b should not affect a (they have separate fmpq_t)
  b.set_zero();
  REQUIRE(a.str() == "42");
}

TEST(rational_assignment) {
  rational a(10);
  rational b;
  b.set_zero();
  b = a;
  REQUIRE(b.str() == "10");
}

TEST(rational_negative_number) {
  rational a(-7);
  rational b(3);
  rational c = a + b;
  REQUIRE(c.str() == "-4");
}

TEST(rational_sum_of_many) {
  // 1 + 2 + 3 + 4 = 10
  rational acc(0);
  for (int i = 1; i <= 4; ++i)
    acc += rational(i);
  REQUIRE(acc.str() == "10");
}

// ---------------------------------------------------------------------------
// mpoly — univariate polynomial with rational coefficients (FLINT fmpq_poly_t)
// ---------------------------------------------------------------------------

TEST(mpoly_set_zero_is_zero) {
  mpoly p;
  p.set_zero();
  REQUIRE(p.is_zero());
}

TEST(mpoly_nonzero_from_string) {
  // fmpq_poly_set_str expects FLINT format: "3  1 2 3" (length then coeffs)
  // constant polynomial "1": "1  1"
  mpoly p("1  1");
  REQUIRE(!p.is_zero());
}

TEST(mpoly_rejects_a_malformed_string) {
  bool threw = false;
  try {
    mpoly p("not a polynomial");
  } catch (const std::runtime_error &e) {
    threw = true;
    REQUIRE(std::string(e.what()).find("not a polynomial") !=
            std::string::npos);
  }
  REQUIRE(threw);
}

TEST(mpoly_addition_zeros) {
  mpoly a, b;
  a.set_zero();
  b.set_zero();
  mpoly c = a + b;
  REQUIRE(c.is_zero());
}

TEST(mpoly_plus_equals) {
  mpoly a("1  1"), b("1  2");  // constant 1 + constant 2
  a += b;
  // result should be constant 3
  REQUIRE(!a.is_zero());
  REQUIRE(a.length() == 1);
}

TEST(mpoly_multiplication_by_zero) {
  mpoly a("1  5"), b;
  b.set_zero();
  mpoly c = a * b;
  REQUIRE(c.is_zero());
}

TEST(mpoly_length_constant) {
  mpoly p("1  7");  // constant polynomial: degree 0, 1 term
  REQUIRE(p.length() == 1);
}

TEST(mpoly_length_linear) {
  // "2  0 1" represents 0 + 1*x (degree-1, 2 coefficients)
  mpoly p("2  0 1");
  REQUIRE(p.length() == 2);
}

TEST(mpoly_copy_constructor) {
  mpoly a("1  5");
  mpoly b(a);
  REQUIRE(!b.is_zero());
  // Modifying b should not affect a
  b.set_zero();
  REQUIRE(!a.is_zero());
}

TEST(mpoly_scalar_multiply_int) {
  mpoly a("1  3");  // constant 3
  mpoly b = a * 4;
  // 3 * 4 = 12
  REQUIRE(!b.is_zero());
}

// ---------------------------------------------------------------------------
// ffint — finite-field residue (mod FiniteFlow's learn() prime)
// ---------------------------------------------------------------------------

TEST(ffint_default_is_zero) {
  ffint f;
  REQUIRE(f.is_zero());
}

TEST(ffint_construct_from_int) {
  ffint f(5);
  REQUIRE(!f.is_zero());
  REQUIRE(f.str() == "5");
}

TEST(ffint_construct_from_negative_int) {
  ffint f(-3);
  REQUIRE(f.value() == ffint::PRIME - 3);
}

TEST(ffint_construct_from_string) {
  ffint f("5");
  REQUIRE(f.str() == "5");
}

TEST(ffint_rejects_a_malformed_string) {
  // The finite-field element delegates to rational, so the refusal reaches
  // the caller with the original text.
  bool threw = false;
  try {
    ffint f("not a number");
  } catch (const std::runtime_error &e) {
    threw = true;
    REQUIRE(std::string(e.what()).find("not a number") != std::string::npos);
  }
  REQUIRE(threw);
}

TEST(ffint_addition) {
  ffint a(2), b(3);
  REQUIRE((a + b).str() == "5");
}

TEST(ffint_addition_wraps) {
  ffint a(ffint::PRIME - 1), b(1);
  REQUIRE((a + b).is_zero()); // (P-1) + 1 ≡ 0
}

TEST(ffint_subtraction) {
  ffint a(5), b(3);
  REQUIRE((a - b).str() == "2");
}

TEST(ffint_subtraction_wraps) {
  ffint a(2), b(3);
  REQUIRE((a - b).value() == ffint::PRIME - 1); // 2 - 3 ≡ -1
}

TEST(ffint_multiplication) {
  ffint a(3), b(4);
  REQUIRE((a * b).str() == "12");
}

TEST(ffint_multiplication_large) {
  ffint a(ffint::PRIME - 1), b(ffint::PRIME - 1);
  REQUIRE((a * b).str() == "1"); // (-1) * (-1) = 1
}

TEST(ffint_division_is_inverse) {
  ffint a(3), b(2);
  ffint q = a / b;
  REQUIRE((q * b).str() == "3");
}

TEST(ffint_reduce_integer) {
  ffint f = ffint(rational("7"));
  REQUIRE(f.str() == "7");
}

TEST(ffint_reduce_rational) {
  // 3/2 mod PRIME == 3 * 2^{-1} mod PRIME
  ffint f = ffint(rational("3/2"));
  ffint expected = ffint(3) / ffint(2);
  REQUIRE(f == expected);
}

TEST(ffint_negation) {
  ffint a(3);
  REQUIRE((-a).value() == ffint::PRIME - 3);
}

TEST(ffint_plus_equals) {
  ffint a(3), b(2);
  a += b;
  REQUIRE(a.str() == "5");
}
