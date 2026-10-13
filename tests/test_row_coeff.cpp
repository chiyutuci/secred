// The precompiled row coefficients of the settling round (src/row_coeff.h) and
// the flat term list helper it shares with symmetry.h.  The printer is checked
// by parsing its output back with GiNaC and comparing expressions, which is what
// spec 3.2 makes the contract: the global stage reads the text, so the text must
// mean exactly what the coefficient is.
#include "test_runner.h"

#include "row_coeff.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ginac/ginac.h>

// Box's four symbols, in the index order these tests give them.  Plain symbols,
// not the family's possymbols: the printer is driven by the table's name list.
static const std::vector<std::string> kBoxNames = {"d", "s", "t", "m"};
static const GiNaC::symbol d("d"), s("s"), t("t"), m("m");

static std::string ex_text(const GiNaC::ex &e) {
  std::ostringstream oss;
  oss << e;
  return oss.str();
}

// d = 0, s = 1, t = 2, m = 3: the same indices kBoxNames spells.
static secred::RowTerm term(long coef,
                            std::initializer_list<std::pair<unsigned, unsigned>> vars = {}) {
  secred::RowTerm out;
  out.coef = cln::cl_I(coef);
  for (const auto &ve : vars)
    out.vars.emplace_back(static_cast<std::uint8_t>(ve.first), ve.second);
  return out;
}

static std::string printed(const secred::RowNum &num, std::uint32_t q,
                           std::initializer_list<std::pair<unsigned, unsigned>> mono = {}) {
  secred::RowVars den;
  for (const auto &me : mono)
    den.emplace_back(static_cast<std::uint8_t>(me.first), me.second);
  std::string out;
  secred::print_row_poly(num, q, den, kBoxNames, out);
  return out;
}

// The expectation is built as a GiNaC expression, never as a string, so a typo
// in the corpus cannot hide behind the parser.
static void require_prints_to(const secred::RowNum &num, std::uint32_t q,
                              std::initializer_list<std::pair<unsigned, unsigned>> mono,
                              const GiNaC::ex &want) {
  const std::string got = printed(num, q, mono);
  // GiNaC 1.8 does not intern symbols: a fresh `GiNaC::symbol("d")` is a
  // *different* symbol from the `d` above, and `3*d - 3*d'` does not cancel.
  // The parser has to be handed these very objects, as test_symmetry.cpp does
  // with Family::symtab.
  GiNaC::symtab st;
  st["d"] = d;
  st["s"] = s;
  st["t"] = t;
  st["m"] = m;
  GiNaC::parser par(st);
  GiNaC::ex back;
  bool parsed = true;
  std::string why;
  try {
    back = par(got);
  } catch (const std::exception &e) {
    parsed = false;
    why = e.what();
  }
  REQUIRE_MSG(parsed, "the printer emitted '" + got + "', which GiNaC cannot parse: " + why);
  REQUIRE_MSG((back - want).expand().is_zero(),
              "printed '" + got + "' parses to '" + ex_text(back) + "', expected '" +
                  ex_text(want) + "'");
}

// Every shape the settling round emits on a symbolic family: plain rationals,
// negatives, unit coefficients, powers, sums, the denominator monomial that
// cancels and the one that does not.
TEST(row_coeff_printer_round_trips_through_the_ginac_parser) {
  require_prints_to({term(-9)}, 2, {}, GiNaC::ex(-9) / 2);
  require_prints_to({term(1)}, 1, {}, GiNaC::ex(1));
  require_prints_to({term(-1)}, 1, {}, GiNaC::ex(-1));
  require_prints_to({}, 1, {}, GiNaC::ex(0));
  require_prints_to({term(3, {{0, 1}})}, 1, {}, 3 * d);
  require_prints_to({term(1, {{0, 1}})}, 1, {}, d);
  require_prints_to({term(-1, {{0, 1}})}, 1, {}, -d);
  require_prints_to({term(1, {{0, 3}})}, 1, {}, GiNaC::pow(d, 3));
  require_prints_to({term(-1, {{0, 1}}), term(4)}, 1, {}, 4 - d);
  require_prints_to({term(1, {{0, 1}}), term(-2)}, 2, {}, (d - 2) / 2);
  require_prints_to({term(3, {{0, 1}}), term(1)}, 2, {}, (3 * d + 1) / 2);
  require_prints_to({term(2, {{0, 1}, {1, 1}, {3, 1}})}, 1, {}, 2 * d * s * m);
  // A monomial denominator that divides every term is factored out.
  require_prints_to({term(-9, {{3, 2}})}, 2, {{3, 2}}, GiNaC::ex(-9) / 2);
  require_prints_to({term(1, {{0, 1}, {3, 2}}), term(-2, {{3, 4}})}, 1, {{3, 2}},
                    d - 2 * m * m);
  // One that does not: the whole sum is parenthesised.
  require_prints_to({term(1, {{0, 1}}), term(-2)}, 2, {{3, 2}}, (d - 2) / (2 * m * m));
  require_prints_to({term(1)}, 1, {{0, 1}}, 1 / d);
  require_prints_to({term(1)}, 2, {{0, 1}}, 1 / (2 * d));
  require_prints_to({term(2, {{1, 1}}), term(-1, {{2, 1}})}, 1, {{3, 1}},
                    (2 * s - t) / m);
}

// The trap the parenthesised form exists for: a term whose own monomial is
// cancelled by the row's lcm must print without a dangling separator.
TEST(row_coeff_printer_drops_a_fully_cancelled_monomial) {
  require_prints_to({term(-9, {{3, 2}}), term(1, {{0, 1}, {3, 2}})}, 2, {{3, 2}},
                    d / 2 - GiNaC::ex(9) / 2);
}

// A coefficient that has to fall back to a plain sum of a symbol and a number
// with a negative leading term, the shape `-5+d`.
TEST(row_coeff_printer_prints_a_signed_sum) {
  require_prints_to({term(1, {{0, 1}}), term(-5)}, 1, {}, d - 5);
  require_prints_to({term(-1, {{0, 1}}), term(-5)}, 1, {}, -d - 5);
}
