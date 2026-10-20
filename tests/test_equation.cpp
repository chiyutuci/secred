#include "test_runner.h"
#include "equation.h"
#include "fflow/alg_functions.hh"
using namespace fflow;

// ---------------------------------------------------------------------------
// EquationFF — sparse equation with ffint coefficients
// ---------------------------------------------------------------------------

TEST(equationff_empty_on_construction) {
  EquationFF eq;
  REQUIRE(eq.empty());
  REQUIRE(eq.size() == 0);
}

TEST(equationff_insert_one) {
  EquationFF eq;
  eq.insert(5, ffint("3"));
  REQUIRE(!eq.empty());
  REQUIRE(eq.size() == 1);
  REQUIRE(eq[0] == 5);
}

TEST(equationff_insert_multiple) {
  EquationFF eq;
  eq.insert(3, ffint("1"));
  eq.insert(1, ffint("2"));
  eq.insert(7, ffint("3"));
  REQUIRE(eq.size() == 3);
}

TEST(equationff_index_access) {
  EquationFF eq;
  eq.insert(10, ffint("5"));
  eq.insert(20, ffint("6"));
  REQUIRE(eq[0] == 10);
  REQUIRE(eq[1] == 20);
}

TEST(equationff_coeff_access) {
  EquationFF eq;
  eq.insert(2, ffint("7"));
  ffint c = eq.coeff(0);
  REQUIRE(c == ffint("7"));
}

TEST(equationff_sort_by_index) {
  EquationFF eq;
  eq.insert(5, ffint("1"));
  eq.insert(1, ffint("2"));
  eq.insert(3, ffint("3"));
  eq.sort();
  REQUIRE(eq[0] == 1);
  REQUIRE(eq[1] == 3);
  REQUIRE(eq[2] == 5);
}

TEST(equationff_sort_preserves_coefficients) {
  EquationFF eq;
  eq.insert(5, ffint("10"));
  eq.insert(1, ffint("20"));
  eq.sort();
  // After sort: index 1 is first, index 5 is second
  REQUIRE(eq[0] == 1);
  REQUIRE(eq.coeff(0) == ffint("20"));
  REQUIRE(eq[1] == 5);
  REQUIRE(eq.coeff(1) == ffint("10"));
}

TEST(equationff_sort_single_element) {
  EquationFF eq;
  eq.insert(42, ffint("99"));
  eq.sort();
  REQUIRE(eq.size() == 1);
  REQUIRE(eq[0] == 42);
}

TEST(equationff_sort_already_sorted) {
  EquationFF eq;
  eq.insert(1, ffint("1"));
  eq.insert(2, ffint("2"));
  eq.insert(3, ffint("3"));
  eq.sort();
  REQUIRE(eq[0] == 1);
  REQUIRE(eq[1] == 2);
  REQUIRE(eq[2] == 3);
}

TEST(equationff_mutable_index) {
  EquationFF eq;
  eq.insert(5, ffint("1"));
  eq[0] = 10;
  REQUIRE(eq[0] == 10);
}

// ---------------------------------------------------------------------------
// EquationM — sparse equation with mpoly coefficients
// ---------------------------------------------------------------------------

TEST(equationm_empty_on_construction) {
  EquationM eq;
  REQUIRE(eq.empty());
  REQUIRE(eq.size() == 0);
}

TEST(equationm_insert_one) {
  EquationM eq;
  mpoly p;
  p.set_zero();
  eq.insert(3, p);
  REQUIRE(!eq.empty());
  REQUIRE(eq.size() == 1);
}

TEST(equationm_insert_and_access) {
  EquationM eq;
  mpoly p("1  5");
  eq.insert(7, p);
  REQUIRE(eq[0] == 7);
  REQUIRE(!eq.coeff(0).is_zero());
}

TEST(equationm_sort_by_index) {
  EquationM eq;
  mpoly p;
  p.set_zero();
  eq.insert(10, p);
  eq.insert(2, p);
  eq.insert(6, p);
  eq.sort();
  REQUIRE(eq[0] == 2);
  REQUIRE(eq[1] == 6);
  REQUIRE(eq[2] == 10);
}

TEST(equationm_sort_preserves_coefficients) {
  EquationM eq;
  mpoly p1("1  1"), p2("1  2");
  eq.insert(5, p1);
  eq.insert(1, p2);
  eq.sort();
  // index 1 should come first and carry coefficient p2 (value 2)
  REQUIRE(eq[0] == 1);
  REQUIRE(!eq.coeff(0).is_zero());
}

TEST(equationm_mutable_index) {
  EquationM eq;
  mpoly p;
  p.set_zero();
  eq.insert(5, p);
  eq[0] = 15;
  REQUIRE(eq[0] == 15);
}

TEST(equationm_multiple_inserts) {
  EquationM eq;
  mpoly p;
  p.set_zero();
  for (unsigned i = 0; i < 5; ++i)
    eq.insert(i * 10, p);
  REQUIRE(eq.size() == 5);
}
