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

// A parser over these very symbols: GiNaC 1.8 does not intern symbols, so one
// built from fresh `GiNaC::symbol("d")` objects would return expressions that do
// not cancel against the file-scope `d` above.
static GiNaC::parser box_parser() {
  GiNaC::symtab st;
  st["d"] = d;
  st["s"] = s;
  st["t"] = t;
  st["m"] = m;
  return GiNaC::parser(st);
}

// The expectation is built as a GiNaC expression, never as a string, so a typo
// in the corpus cannot hide behind the parser.
static void require_prints_to(const secred::RowNum &num, std::uint32_t q,
                              std::initializer_list<std::pair<unsigned, unsigned>> mono,
                              const GiNaC::ex &want) {
  const std::string got = printed(num, q, mono);
  GiNaC::parser par = box_parser();
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

// ---------------------------------------------------------------------------
// The precompiler and the evaluator (Task 2).
// ---------------------------------------------------------------------------

// One generator's rows: each item is an index delta (ignored here, its length is
// what the evaluator reads the seed against) and the row's coefficients.
// RawIntegral and IBPProtoM are global-namespace types (sector.h); only the
// names this plan adds to row_coeff.h live in namespace secred.
static IBPProtoM rows_of(const std::vector<std::vector<GiNaC::ex>> &rows) {
  IBPProtoM items;
  for (const std::vector<GiNaC::ex> &row : rows)
    items.emplace_back(RawIntegral(std::vector<int>(row.size() - 1, 0)), row);
  return items;
}

// The evaluation is the settling round's arithmetic -- sum_p row[p]*seed[p] +
// row.back(), over the row's common denominator -- so it is compared against
// GiNaC's own computation, on rows that mix integers, fractions and symbols and
// on seeds that scale them positively, negatively and not at all.
TEST(row_coeff_evaluation_matches_ginac) {
  const std::vector<GiNaC::ex> row = {-5 + d, GiNaC::ex(-9) / 2, 3 * d + 1,
                                      (d - 2) / (2 * m * m)};
  secred::RowTable table;
  table.add(0, rows_of({row}));
  REQUIRE(table.rows[0][0].precompiled);

  GiNaC::parser par = box_parser();

  const int seeds[9][3] = {{0, 0, 0},  {1, 0, 0},  {0, 1, 0}, {2, 1, 0}, {1, 1, 1},
                           {-1, 1, 0}, {2, 0, 1},  {-2, -1, 0}, {3, 2, 1}};
  for (const auto &sv : seeds) {
    const RawIntegral seed(std::vector<int>(sv, sv + 3));
    std::string got;
    const secred::RowEval st_r = table.eval(0, 0, seed, got);
    GiNaC::ex want;
    for (unsigned p = 0; p < 3; ++p)
      want += row[p] * seed[p];
    want += row.back();
    want = want.expand();
    if (st_r == secred::RowEval::Zero) {
      REQUIRE(want.is_zero());
      continue;
    }
    REQUIRE(st_r == secred::RowEval::Ok);
    REQUIRE_MSG((par(got) - want).expand().is_zero(),
                "seed evaluated to '" + got + "', expected '" + ex_text(want) + "'");
  }
}

// The shapes the precompiler must refuse: a symbolic denominator that is not a
// monomial, a non-integer coefficient, a half-integer exponent -- and one that
// is representable per coefficient but whose row lcm would overflow the term
// exponents, which must fall back rather than wrap.
TEST(row_coeff_falls_back_on_shapes_it_cannot_represent) {
  secred::RowTable table;
  table.add(0, rows_of({{1 / (1 + d), GiNaC::ex(2), GiNaC::ex(0)}}));
  table.add(1, rows_of({{GiNaC::sqrt(GiNaC::ex(2)), GiNaC::ex(2), GiNaC::ex(0)}}));
  table.add(2, rows_of({{GiNaC::pow(d, GiNaC::ex(1) / 2), GiNaC::ex(2), GiNaC::ex(0)}}));
  // `d^255` alone is representable, and `1/d^200` alone is representable, but
  // over their row's lcm the first needs `d^455`, past the term's uint8_t
  // exponent.  The row must fall back, not wrap.
  table.add(3, rows_of({{GiNaC::pow(d, 255), GiNaC::ex(1) / GiNaC::pow(d, 200),
                         GiNaC::ex(0)}}));
  for (unsigned i = 0; i < 4; ++i) {
    REQUIRE(!table.rows[i][0].precompiled);
    std::string got;
    REQUIRE(table.eval(i, 0, RawIntegral(std::vector<int>({1, 1})), got) ==
            secred::RowEval::NotPrecompiled);
  }
  REQUIRE_EQ(table.fast_rows, 0u);
  REQUIRE_EQ(table.slow_rows, 4u);
}

// The row's common denominator is the lcm of its coefficients' own monomials, so
// a coefficient whose denominator is shallower is *lifted*: a term that lacks
// the symbol altogether -- a constant term is the plainest case -- has to gain
// an entry, or the coefficient silently comes out too small by that power.
// `1/m^2` beside `d/m^4` must stay `1/m^2`, not become `1/m^4`.
TEST(row_coeff_lifts_a_term_that_lacks_a_symbol_of_the_row_lcm) {
  const std::vector<GiNaC::ex> row = {GiNaC::ex(1) / GiNaC::pow(m, 2),
                                      d / GiNaC::pow(m, 4), GiNaC::ex(0)};
  secred::RowTable table;
  table.add(0, rows_of({row}));
  REQUIRE(table.rows[0][0].precompiled);

  GiNaC::parser par = box_parser();
  const int seeds[4][2] = {{1, 0}, {0, 1}, {1, 1}, {-1, 2}};
  for (const auto &sv : seeds) {
    const RawIntegral seed(std::vector<int>(sv, sv + 2));
    std::string got;
    REQUIRE(table.eval(0, 0, seed, got) == secred::RowEval::Ok);
    const GiNaC::ex want = (row[0] * seed[0] + row[1] * seed[1] + row.back()).expand();
    REQUIRE_MSG((par(got) - want).expand().is_zero(),
                "seed evaluated to '" + got + "', expected '" + ex_text(want) + "'");
  }
}

// A coefficient that cancels exactly is reported as such, not printed: the
// global system must not gain a column from a phantom integral.
TEST(row_coeff_reports_a_vanishing_coefficient) {
  secred::RowTable table;
  table.add(0, rows_of({{d, -d, GiNaC::ex(0), GiNaC::ex(0)}}));
  REQUIRE(table.rows[0][0].precompiled);
  std::string got;
  REQUIRE(table.eval(0, 0, RawIntegral(std::vector<int>({1, 1, 0})), got) ==
          secred::RowEval::Zero);
  REQUIRE(table.eval(0, 0, RawIntegral(std::vector<int>({1, 0, 0})), got) ==
          secred::RowEval::Ok);
  REQUIRE(!got.empty() && got != "0");
}

// The lazy build: a generator's rows are precompiled the first time one of its
// equations is used, all of its items together, and the counters say how many.
TEST(row_table_builds_one_generator_at_a_time_and_counts) {
  secred::RowTable table;
  REQUIRE(!table.contains(0));
  table.add(0, rows_of({{d + 1, GiNaC::ex(2), GiNaC::ex(0)},
                        {1 / (1 + m), GiNaC::ex(2), GiNaC::ex(0)}}));
  REQUIRE(table.contains(0));
  REQUIRE(!table.contains(1));
  REQUIRE_EQ(table.fast_rows, 1u);
  REQUIRE_EQ(table.slow_rows, 1u);
  REQUIRE_EQ(table.names.size(), 2u);
  REQUIRE_EQ(table.index.at("d"), 0u);
  REQUIRE_EQ(table.index.at("m"), 1u);
  REQUIRE(table.rows[0][0].precompiled);
  REQUIRE(!table.rows[0][1].precompiled);
  // An item index that was never added is a miss, not a crash.
  std::string got;
  REQUIRE(table.eval(0, 2, RawIntegral(std::vector<int>({1, 1})), got) ==
          secred::RowEval::NotPrecompiled);
  REQUIRE(table.eval(1, 0, RawIntegral(std::vector<int>({1, 1})), got) ==
          secred::RowEval::NotPrecompiled);
}

// A seed whose length does not match the row cannot be evaluated: the row is
// only valid for the nprops its coefficients were built for.
TEST(row_coeff_refuses_a_seed_of_the_wrong_length) {
  secred::RowTable table;
  table.add(0, rows_of({{d, GiNaC::ex(1), GiNaC::ex(1)}}));
  std::string got;
  REQUIRE(table.eval(0, 0, RawIntegral(std::vector<int>({1, 1, 1})), got) ==
          secred::RowEval::NotPrecompiled);
}

// The flat term list: equal monomials sum, cancellations and zero terms go.
TEST(row_coeff_merges_equal_monomials_and_drops_zeros) {
  secred::RowNum terms = {term(1, {{0, 1}}), term(2, {{1, 1}}), term(-1, {{0, 1}}),
                          term(3)};
  secred::row_merge_terms(terms);
  REQUIRE_EQ(terms.size(), 2u);
  REQUIRE(terms[0].vars.empty());  // the constant sorts first
  REQUIRE(terms[0].coef == cln::cl_I(3));
  REQUIRE_EQ(terms[1].vars.size(), 1u);
  REQUIRE_EQ(unsigned(terms[1].vars[0].first), 1u);
  REQUIRE(terms[1].coef == cln::cl_I(2));
  // Nothing merged and nothing dropped: every survivor stays where it already
  // is, so the compaction must not move a term onto itself -- and the standard
  // library leaves a self-move-assigned vector empty, which would read as the
  // monomial having vanished.
  secred::RowNum compact = {term(5, {{0, 1}}), term(2, {{1, 2}})};
  secred::row_merge_terms(compact);
  REQUIRE_EQ(compact.size(), 2u);
  REQUIRE_EQ(compact[0].vars.size(), 1u);
  REQUIRE_EQ(unsigned(compact[0].vars[0].second), 1u);
  REQUIRE_EQ(compact[1].vars.size(), 1u);
  REQUIRE_EQ(unsigned(compact[1].vars[0].second), 2u);
}
