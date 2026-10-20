// The scanner over the global stage's spill file: boundaries, the strict parsers and
// their fallbacks, and (Task 2-3) the symbolic grammar.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "reference_spill_reader.h"
#include "row_coeff.h"
#include "spill_reader.h"
#include "sym_coeff.h"
#include "test_runner.h"

using secred::Eq;
using ::RawIntegral;   // sector.h declares RawIntegral at global scope
using secred::SpillReadStats;

namespace {

std::string write_spill(const std::string &name, const std::string &text) {
  const std::string path = "/tmp/secred_test_spill_" + name;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << text;
  out.close();
  return path;
}

std::unordered_map<RawIntegral, unsigned> ids_of(
    const std::vector<std::vector<int>> &list) {
  std::unordered_map<RawIntegral, unsigned> ids;
  for (unsigned i = 0; i < list.size(); ++i)
    ids.emplace(RawIntegral(list[i]), i);
  return ids;
}

// One term line: "[a, b]\t<coeff>".
std::string term_line(const std::vector<int> &integ, const std::string &coeff) {
  std::string s = "[";
  for (std::size_t i = 0; i < integ.size(); ++i) {
    if (i)
      s += ", ";
    s += std::to_string(integ[i]);
  }
  return s + "]\t" + coeff + "\n";
}

}  // namespace

TEST(spill_numeric_reads_equations_and_sorts_ids_descending) {
  const std::string p = write_spill("basic",
      term_line({1, 1}, "3") + term_line({0, 1}, "1/2") + "\n" +
      term_line({1, 0}, "-2") + "\n");
  const auto ids = ids_of({{0, 0}, {1, 0}, {0, 1}, {1, 1}});  // ids 0..3
  std::vector<Eq> system;
  SpillReadStats stats;
  std::string error;
  REQUIRE(secred::read_spill_numeric(p, ids, system, stats, error));
  REQUIRE_EQ(system.size(), std::size_t(2));
  REQUIRE_EQ(system[0].size(), std::size_t(2));
  REQUIRE_EQ(system[0][0].first, 3u);          // [1,1] first: descending id
  REQUIRE_EQ(system[0][1].first, 2u);          // [0,1]
  REQUIRE_EQ(system[0][0].second.num, 3);
  REQUIRE_EQ(system[0][0].second.den, 1);
  REQUIRE_EQ(system[0][1].second.num, 1);
  REQUIRE_EQ(system[0][1].second.den, 2);
  REQUIRE_EQ(system[1].size(), std::size_t(1));
  REQUIRE_EQ(system[1][0].first, 1u);
  REQUIRE_EQ(system[1][0].second.num, -2);
  REQUIRE_EQ(stats.equations, std::uint64_t(2));
  REQUIRE_EQ(stats.terms, std::uint64_t(3));
  REQUIRE_EQ(stats.fallbacks, std::uint64_t(0));
  REQUIRE_EQ(stats.skipped_unknown, std::uint64_t(0));
}

TEST(spill_numeric_skips_unknown_integrals_and_tapless_lines) {
  const std::string p = write_spill("unknown",
      term_line({1, 1}, "3") + "junk without a tab\n" + term_line({2, 2}, "5") + "\n");
  const auto ids = ids_of({{0, 0}, {1, 1}});
  std::vector<Eq> system;
  SpillReadStats stats;
  std::string error;
  REQUIRE(secred::read_spill_numeric(p, ids, system, stats, error));
  REQUIRE_EQ(system.size(), std::size_t(1));
  REQUIRE_EQ(system[0].size(), std::size_t(1));
  REQUIRE_EQ(stats.skipped_unknown, std::uint64_t(1));
}

// Boundary case: the final equation has no blank line after it (an interrupted run).
// Both readers flush it, exactly as the getline readers did.
TEST(spill_numeric_flushes_a_final_unterminated_equation) {
  const std::string p = write_spill("unterminated", term_line({1, 1}, "3"));
  const auto ids = ids_of({{0, 0}, {1, 1}});
  std::vector<Eq> system;
  SpillReadStats stats;
  std::string error;
  REQUIRE(secred::read_spill_numeric(p, ids, system, stats, error));
  REQUIRE_EQ(system.size(), std::size_t(1));
}

// Boundary case: blank lines at EOF and a CRLF line contribute nothing.
TEST(spill_numeric_ignores_empty_tail_and_crlf) {
  const std::string p = write_spill("crlf",
      term_line({1, 1}, "3") + "\n\n\r\n\n");
  const auto ids = ids_of({{0, 0}, {1, 1}});
  std::vector<Eq> system;
  SpillReadStats stats;
  std::string error;
  REQUIRE(secred::read_spill_numeric(p, ids, system, stats, error));
  REQUIRE_EQ(system.size(), std::size_t(1));
  REQUIRE_EQ(system[0].size(), std::size_t(1));
}

TEST(spill_numeric_empty_file_is_an_empty_system) {
  const std::string p = write_spill("empty", "");
  const auto ids = ids_of({{0, 0}});
  std::vector<Eq> system;
  SpillReadStats stats;
  std::string error;
  REQUIRE(secred::read_spill_numeric(p, ids, system, stats, error));
  REQUIRE_EQ(system.size(), std::size_t(0));
}

TEST(spill_numeric_missing_file_reports_an_error) {
  std::vector<Eq> system;
  SpillReadStats stats;
  std::string error;
  REQUIRE(!secred::read_spill_numeric("/tmp/secred_test_spill_does_not_exist", {},
                                      system, stats, error));
  REQUIRE(!error.empty());
}

// A coefficient beyond int64 used to be a hard failure; a pure-numeric family's rational
// parameters (s, t, the masses) are arbitrary-precision FLINT values, so the reader now
// interns the spelling and tags the term {handle, 0}.  2^64 = 2p + 50 at fflow's largest
// prime, so the residue below is exact arithmetic rather than a copy of the reader.
TEST(spill_numeric_reads_a_coefficient_beyond_int64) {
  const std::string p = write_spill("big",
      term_line({1, 1}, "18446744073709551616") + "\n");
  const auto ids = ids_of({{0, 0}, {1, 1}});
  std::vector<Eq> system;
  SpillReadStats stats;
  std::string error;
  REQUIRE(secred::read_spill_numeric(p, ids, system, stats, error));
  REQUIRE_EQ(system.size(), std::size_t(1));
  REQUIRE_EQ(system[0].size(), std::size_t(1));
  const secred::Rat64 &c = system[0][0].second;
  REQUIRE_EQ(c.den, int64_t(0)); // the big tag, num is an intern handle
  REQUIRE(secred::rat_text(c) == "18446744073709551616");
  REQUIRE_EQ(secred::rat_mod_u64(c.num, c.den, 9223372036854775783ull),
             std::uint64_t(50));
  // Counted as a fallback: the strict grammar refused it, and it is one occurrence.
  REQUIRE_EQ(stats.fallbacks, std::uint64_t(1));
  REQUIRE_EQ(stats.terms, std::uint64_t(1));
}

// A 20-digit coefficient whose uint64 wrap-around lands back *inside* int64 (2^64 + 1
// wraps to 1): a digit loop that accumulates first and checks after would accept the
// wrapped value silently.  The value is interned, and reduces to 51 -- not to 1.
TEST(spill_numeric_beyond_int64_does_not_wrap) {
  const std::string p = write_spill("wrap",
      term_line({1, 1}, "18446744073709551617") + "\n");
  const auto ids = ids_of({{0, 0}, {1, 1}});
  std::vector<Eq> system;
  SpillReadStats stats;
  std::string error;
  REQUIRE(secred::read_spill_numeric(p, ids, system, stats, error));
  REQUIRE_EQ(system.size(), std::size_t(1));
  const secred::Rat64 &c = system[0][0].second;
  REQUIRE_EQ(c.den, int64_t(0));
  REQUIRE(secred::rat_text(c) == "18446744073709551617");
  REQUIRE_EQ(secred::rat_mod_u64(c.num, c.den, 9223372036854775783ull),
             std::uint64_t(51));
}

// --- the integral fallback (Review Focus 3) -------------------------------------

TEST(spill_integral_parser_matches_from_text_on_odd_spellings) {
  const std::string p = write_spill("odd",
      "[1,2]\t3\n" + std::string("[ 1, 2 ]\t3\n") +
      std::string("[1, 2]junk\t3\n") + term_line({-3, 4}, "3") + "\n");
  const auto ids = ids_of({{1, 2}, {-3, 4}});
  std::vector<Eq> system;
  SpillReadStats stats;
  std::string error;
  REQUIRE(secred::read_spill_numeric(p, ids, system, stats, error));
  REQUIRE_EQ(system.size(), std::size_t(1));
  // Three spellings of [1,2] the strict parser refuses and from_text accepts, plus
  // one [-3, 4] it reads itself.
  REQUIRE_EQ(system[0].size(), std::size_t(4));
  REQUIRE_EQ(stats.skipped_unknown, std::uint64_t(0));
}

TEST(spill_integral_beyond_max_props_falls_back_without_corruption) {
  std::vector<int> big(33, 1);
  const std::string p = write_spill("toomany", term_line(big, "3") + "\n");
  // from_text builds the same RawIntegral the strict parser refused, so the term is
  // kept and nothing is written past the 32-index storage.
  const auto ids = ids_of({big});
  std::vector<Eq> system;
  SpillReadStats stats;
  std::string error;
  REQUIRE(secred::read_spill_numeric(p, ids, system, stats, error));
  REQUIRE_EQ(system.size(), std::size_t(1));
}

// Spec §7's `+`-prefixed and `-0` cases: both readers must read them alike.  `-0` is
// inside the strict grammar (a sign and digits) and gives 0; `+` is not, so that one
// falls back and is counted -- the value is the reference's either way.
TEST(spill_numeric_signed_and_zero_coefficients) {
  const std::string p = write_spill("signed",
      term_line({1, 1}, "+5") + term_line({0, 1}, "-0") + "\n");
  const auto ids = ids_of({{0, 0}, {1, 0}, {0, 1}, {1, 1}});
  std::vector<Eq> system;
  SpillReadStats stats;
  std::string error;
  REQUIRE(secred::read_spill_numeric(p, ids, system, stats, error));
  REQUIRE_EQ(system.size(), std::size_t(1));
  REQUIRE_EQ(system[0][0].second.num, int64_t(5));   // [1,1], id 3, first
  REQUIRE_EQ(system[0][0].second.den, int64_t(1));
  REQUIRE_EQ(system[0][1].second.num, int64_t(0));   // [0,1], id 2
  REQUIRE_EQ(stats.fallbacks, std::uint64_t(1));
}

// --- the differential reference (spec §7) ---------------------------------------

// The scanner against the reader it replaces, on one file that exercises every branch
// at once: a sort that matters, an unknown integral, a fallback coefficient, a
// trailing-junk line, and no blank line at EOF.  The two systems must agree term for
// term -- ids in order and exact num/den -- on everything the old reader accepted; the
// 20-digit coefficient it stopped at is now interned instead (see
// spill_numeric_reads_a_coefficient_beyond_int64), so ours carries one equation more.
TEST(spill_numeric_matches_the_reference_reader) {
  const std::string p = write_spill("refdiff",
      term_line({1, 1}, "3") + term_line({0, 1}, "1/2") + term_line({2, 2}, "7") + "\n" +
      term_line({1, 0}, "-2") + "\n" +
      term_line({0, 0}, "4/6") + term_line({0, 1}, "0000000000000000001") + "\n" +
      "not a term line\n" + "\n" +
      term_line({1, 1}, "99999999999999999999") + "\n");   // beyond int64
  const auto ids = ids_of({{0, 0}, {1, 0}, {0, 1}, {1, 1}});

  std::vector<Eq> theirs;
  std::uint64_t their_terms = 0;
  REQUIRE(!ref::legacy_numeric(p, ids, theirs, their_terms));   // stops at the overflow

  std::vector<Eq> ours;
  SpillReadStats stats;
  std::string error;
  REQUIRE(secred::read_spill_numeric(p, ids, ours, stats, error));
  REQUIRE_EQ(ours.size(), theirs.size() + 1);

  for (std::size_t i = 0; i < theirs.size(); ++i) {
    REQUIRE_EQ(ours[i].size(), theirs[i].size());
    for (std::size_t j = 0; j < ours[i].size(); ++j) {
      REQUIRE_EQ(ours[i][j].first, theirs[i][j].first);
      REQUIRE_EQ(ours[i][j].second.num, theirs[i][j].second.num);
      REQUIRE_EQ(ours[i][j].second.den, theirs[i][j].second.den);
    }
  }
  // `4/6` stays 4/6 in both: the readers normalise only the sign of the denominator,
  // and the solver reduces mod p later.
  REQUIRE_EQ(ours[2][1].second.num, int64_t(4));
  REQUIRE_EQ(ours[2][1].second.den, int64_t(6));
  REQUIRE_EQ(stats.skipped_unknown, std::uint64_t(1));
  // The equation the old reader could not finish: one term, tagged as a big rational.
  REQUIRE_EQ(ours.back().size(), std::size_t(1));
  REQUIRE_EQ(ours.back()[0].second.den, int64_t(0));
  REQUIRE(secred::rat_text(ours.back()[0].second) == "99999999999999999999");
}

// --- the symbolic coefficient grammar (Review Focus 1) --------------------------

namespace {

// A small builder so an expectation reads like the polynomial it is: pol({{3,{{0,1}}},
// {1,{}}}) is 3*d+1, and pol({{2,{}}}) is 2.
secred::SymPoly pol(std::vector<std::pair<long, std::vector<std::pair<int, int>>>> terms) {
  secred::SymPoly p;
  for (const auto &t : terms) {
    secred::SymTerm term;
    term.coef = cln::cl_I(t.first);
    for (const auto &v : t.second)
      term.vars.emplace_back(static_cast<std::uint8_t>(v.first),
                             static_cast<std::uint8_t>(v.second));
    p.terms.push_back(std::move(term));
  }
  return p;
}

// Term-order-insensitive equality: the scanner builds its terms in printed order and
// GiNaC builds them in its own, and neither order is part of the contract.  The same
// goes for the variable list inside a term, which is a set of (index, exponent)
// pairs -- `row_print_vars` emits it in stored order, the scanner in read order.
std::string term_key(const secred::SymTerm &t) {
  std::vector<std::string> vs;
  for (const auto &v : t.vars)
    vs.push_back(std::to_string(int(v.first)) + "^" + std::to_string(int(v.second)));
  std::sort(vs.begin(), vs.end());
  std::ostringstream os;
  os << t.coef << '|';
  for (const auto &v : vs)
    os << v << '*';
  return os.str();
}

bool sym_poly_equal(const secred::SymPoly &a, const secred::SymPoly &b) {
  std::vector<std::string> ka, kb;
  for (const auto &t : a.terms)
    ka.push_back(term_key(t));
  for (const auto &t : b.terms)
    kb.push_back(term_key(t));
  std::sort(ka.begin(), ka.end());
  std::sort(kb.begin(), kb.end());
  return ka == kb;
}

// Round-tripping through the *printer* is the grammar's definition: whatever
// `print_row_poly` writes for (num, q, mono), the scanner must read back as (want_num,
// want_den).  Every call site states both, so the expectation is written out rather
// than derived by a second implementation of the same arithmetic.
bool round_trip(const secred::RowNum &num, std::uint32_t q, const secred::RowVars &mono,
                const std::vector<std::string> &names,
                const std::unordered_map<std::string, unsigned> &sym_index,
                const secred::SymPoly &want_num, const secred::SymPoly &want_den) {
  std::string text;
  secred::print_row_poly(num, q, mono, names, text);
  secred::SymCoeff got;
  if (!secred::parse_spill_coeff(text.data(), text.data() + text.size(), sym_index, got))
    return false;
  return got.precompiled && ::sym_poly_equal(got.num, want_num) &&
         ::sym_poly_equal(got.den, want_den);
}

}  // namespace

TEST(spill_coeff_factored_forms_round_trip) {
  const std::vector<std::string> names = {"d", "s", "t", "m"};
  const std::unordered_map<std::string, unsigned> sym_index = {
      {"d", 0}, {"s", 1}, {"t", 2}, {"m", 3}};
  // 3/2*d+1/2: the printer reduces each term's denominator by its own gcd, so the
  // coefficient's denominator is their lcm (2 here) and the numerator is 3d+1.
  REQUIRE(round_trip(secred::RowNum{{cln::cl_I(3), {{0, 1}}}, {cln::cl_I(1), {}}}, 2, {},
                     names, sym_index, pol({{3, {{0, 1}}}, {1, {}}}), pol({{2, {}}})));
  // -9/2*d^2+3/2
  REQUIRE(round_trip(secred::RowNum{{cln::cl_I(-9), {{0, 2}}}, {cln::cl_I(3), {}}}, 2, {},
                     names, sym_index, pol({{-9, {{0, 2}}}, {3, {}}}), pol({{2, {}}})));
  // d (a unit coefficient is omitted), -d, -3*d, with no denominator at all
  REQUIRE(round_trip(secred::RowNum{{cln::cl_I(1), {{0, 1}}}}, 1, {}, names, sym_index,
                     pol({{1, {{0, 1}}}}), pol({})));
  REQUIRE(round_trip(secred::RowNum{{cln::cl_I(-1), {{0, 1}}}}, 1, {}, names, sym_index,
                     pol({{-1, {{0, 1}}}}), pol({})));
  REQUIRE(round_trip(secred::RowNum{{cln::cl_I(-3), {{0, 1}}}}, 1, {}, names, sym_index,
                     pol({{-3, {{0, 1}}}}), pol({})));
  // a constant, and a bare 1
  REQUIRE(round_trip(secred::RowNum{{cln::cl_I(7), {}}}, 1, {}, names, sym_index,
                     pol({{7, {}}}), pol({})));
  REQUIRE(round_trip(secred::RowNum{{cln::cl_I(1), {}}}, 1, {}, names, sym_index,
                     pol({{1, {}}}), pol({})));
  // d^12 (a multi-digit exponent) and d^2*s^3
  REQUIRE(round_trip(secred::RowNum{{cln::cl_I(1), {{0, 12}}}}, 1, {}, names, sym_index,
                     pol({{1, {{0, 12}}}}), pol({})));
  REQUIRE(round_trip(secred::RowNum{{cln::cl_I(1), {{0, 2}, {1, 3}}}}, 1, {}, names,
                     sym_index, pol({{1, {{0, 2}, {1, 3}}}}), pol({})));
  // the empty sum prints 0 (the caller drops such a coefficient, as GiNaC's is_zero did)
  secred::SymCoeff zero;
  const std::string ztext = "0";
  REQUIRE(secred::parse_spill_coeff(ztext.data(), ztext.data() + 1, sym_index, zero));
  REQUIRE(zero.precompiled);
  REQUIRE(zero.num.terms.empty());
}

TEST(spill_coeff_paren_forms_round_trip) {
  const std::vector<std::string> names = {"d", "s", "t", "m"};
  const std::unordered_map<std::string, unsigned> sym_index = {
      {"d", 0}, {"s", 1}, {"t", 2}, {"m", 3}};
  // (d-2)/(2*m^2): the monomial does not divide every term, so the printer
  // parenthesises the whole sum and prints q and the monomial outside.  The
  // denominator is their product, one polynomial term -- GiNaC's denom() of the same
  // text (probed with the reference reader), not two.
  REQUIRE(round_trip(secred::RowNum{{cln::cl_I(1), {{0, 1}}}, {cln::cl_I(-2), {}}}, 2,
                     {{3, 2}}, names, sym_index, pol({{1, {{0, 1}}}, {-2, {}}}),
                     pol({{2, {{3, 2}}}})));
  // (1)/(d)
  REQUIRE(round_trip(secred::RowNum{{cln::cl_I(1), {}}}, 1, {{0, 1}}, names, sym_index,
                     pol({{1, {}}}), pol({{1, {{0, 1}}}})));
  // (2*d+4)/(2*m^2): numerator and denominator share the content 2, so the scanned
  // coefficient is (d+2)/m^2 -- the same normalisation make_sym_coeff gives the text.
  REQUIRE(round_trip(secred::RowNum{{cln::cl_I(2), {{0, 1}}}, {cln::cl_I(4), {}}}, 2,
                     {{3, 2}}, names, sym_index, pol({{1, {{0, 1}}}, {2, {}}}),
                     pol({{1, {{3, 2}}}})));
}

// Review Focus 1: a symbol name that is a prefix of another must not be mis-read.
TEST(spill_coeff_prefers_the_longest_symbol_name) {
  const std::vector<std::string> names = {"d", "dd", "d1"};
  const std::unordered_map<std::string, unsigned> sym_index = {
      {"d", 0}, {"dd", 1}, {"d1", 2}, {"x3", 3}};
  // dd^2+d*d1 -- `dd` and `d1` must each be one symbol, not `d` followed by a letter
  REQUIRE(round_trip(secred::RowNum{{cln::cl_I(1), {{1, 2}}}, {cln::cl_I(1), {{0, 1}, {2, 1}}}},
                     1, {}, names, sym_index, pol({{1, {{1, 2}}}, {1, {{0, 1}, {2, 1}}}}),
                     pol({})));
  // symbol 3 has no entry in `names`, so `row_print_vars` invents `x3`: the scanner
  // reads whatever the printer wrote, looked up in `sym_index`
  REQUIRE(round_trip(secred::RowNum{{cln::cl_I(1), {{3, 1}, {2, 1}}}}, 1, {}, names,
                     sym_index, pol({{1, {{3, 1}, {2, 1}}}}), pol({})));
}

TEST(spill_coeff_refuses_text_outside_the_grammar) {
  const std::unordered_map<std::string, unsigned> sym_index = {{"d", 0}, {"s", 1}};
  const char *bad[] = {"",      "d/",     "/2",     "(d-2)/(",  "d^0",   "d^256",
                       "2**d",   "d + 1",  "d+1-",   "x9",       "1/(d)", "d+1-d",
                       "^2",     "(d)/(1", "d^^2",   "d-",       "-",     "(", ")"};
  for (const char *t : bad) {
    secred::SymCoeff c;
    const std::size_t n = std::strlen(t);
    REQUIRE_MSG(!secred::parse_spill_coeff(t, t + n, sym_index, c), t);
  }
  // `d*d` is not a spelling the printer emits (it writes d^2), so the strict parser
  // refuses it rather than building a term GiNaC would have combined.
  const char *dup = "d*d";
  secred::SymCoeff c;
  REQUIRE(!secred::parse_spill_coeff(dup, dup + 3, sym_index, c));
}

TEST(spill_symbolic_reads_polynomials_and_drops_zero_terms) {
  const std::string p = write_spill("sym",
      term_line({1, 1}, "3/2*d+1/2") + term_line({0, 1}, "(d-2)/(2*m^2)") + "\n" +
      term_line({1, 0}, "0") + "\n");
  const auto ids = ids_of({{0, 0}, {1, 0}, {0, 1}, {1, 1}});
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d"), GiNaC::possymbol("s"),
                                                 GiNaC::possymbol("t"), GiNaC::possymbol("m")};
  std::vector<secred::SymEq> system;
  secred::SpillReadStats stats;
  std::string error;
  REQUIRE(secred::read_spill_symbolic(p, ids, symbols, system, stats, error));
  REQUIRE_EQ(system.size(), std::size_t(2));
  REQUIRE_EQ(system[0].size(), std::size_t(2));
  REQUIRE_EQ(system[0][0].first, 3u);
  REQUIRE(system[0][0].second.precompiled);
  REQUIRE_EQ(stats.fallbacks, std::uint64_t(0));
  REQUIRE_EQ(system[1].size(), std::size_t(0));   // `0` is dropped, as GiNaC did
  REQUIRE_EQ(stats.equations, std::uint64_t(2));
}

// The slow path must stay reachable: a coefficient outside the strict grammar is read
// by GiNaC, exactly as before.
TEST(spill_symbolic_falls_back_to_ginac_and_counts_it) {
  const std::string p = write_spill("symfallback",
      term_line({1, 1}, "(d-2)*(d+1)/(2*m^2)") + "\n");
  const auto ids = ids_of({{0, 0}, {1, 1}});
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d"), GiNaC::possymbol("s"),
                                                 GiNaC::possymbol("t"), GiNaC::possymbol("m")};
  std::vector<secred::SymEq> system;
  secred::SpillReadStats stats;
  std::string error;
  REQUIRE(secred::read_spill_symbolic(p, ids, symbols, system, stats, error));
  REQUIRE_EQ(system.size(), std::size_t(1));
  REQUIRE_EQ(system[0].size(), std::size_t(1));
  REQUIRE_EQ(stats.fallbacks, std::uint64_t(1));
  REQUIRE(::sym_poly_equal(system[0][0].second.num,
                         secred::SymPoly{{{cln::cl_I(1), {{0, 2}}},
                                          {cln::cl_I(-1), {{0, 1}}},
                                          {cln::cl_I(-2), {}}}}));
}

TEST(spill_symbolic_unparsable_coefficient_is_an_error) {
  const std::string p = write_spill("symbad", term_line({1, 1}, "d +") + "\n");
  const auto ids = ids_of({{0, 0}, {1, 1}});
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d")};
  std::vector<secred::SymEq> system;
  secred::SpillReadStats stats;
  std::string error;
  REQUIRE(!secred::read_spill_symbolic(p, ids, symbols, system, stats, error));
  REQUIRE(error.find("cannot parse coefficient") != std::string::npos);
}

// The scanner against the reader it replaces (spec §7): every coefficient of a mixed
// file must come out equal as a rational function -- the same numerator and denominator
// polynomials -- with the same ids, in the same order.
TEST(spill_symbolic_matches_the_reference_reader) {
  const std::string p = write_spill("symdiff",
      term_line({1, 1}, "3/2*d+1/2") + term_line({0, 1}, "(d-2)/(2*m^2)") +
          term_line({2, 2}, "7") + "\n" +
      term_line({1, 0}, "d") + "\n" +
      term_line({0, 0}, "(d-2)*(d+1)/(2*m^2)") + "\n" +   // outside the fast grammar
      term_line({1, 0}, "0") + term_line({0, 0}, "1/3*s-2/5") + "\n" + "\n");
  const auto ids = ids_of({{0, 0}, {1, 0}, {0, 1}, {1, 1}});
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d"), GiNaC::possymbol("s"),
                                                 GiNaC::possymbol("t"), GiNaC::possymbol("m")};

  std::vector<secred::SymEq> theirs;
  std::uint64_t their_terms = 0;
  REQUIRE(ref::legacy_symbolic(p, ids, symbols, theirs, their_terms));

  std::vector<secred::SymEq> ours;
  secred::SpillReadStats stats;
  std::string error;
  REQUIRE(secred::read_spill_symbolic(p, ids, symbols, ours, stats, error));

  REQUIRE_EQ(ours.size(), theirs.size());
  for (std::size_t i = 0; i < ours.size(); ++i) {
    REQUIRE_EQ(ours[i].size(), theirs[i].size());
    for (std::size_t j = 0; j < ours[i].size(); ++j) {
      REQUIRE_EQ(ours[i][j].first, theirs[i][j].first);
      REQUIRE(::sym_poly_equal(ours[i][j].second.num, theirs[i][j].second.num));
      REQUIRE(::sym_poly_equal(ours[i][j].second.den, theirs[i][j].second.den));
    }
  }
  REQUIRE_EQ(stats.fallbacks, std::uint64_t(1));
  REQUIRE_EQ(stats.terms, their_terms);
}

// The production reader against the reader it is built from: interning during the scan
// must assign exactly the rows and the slots a scan followed by `intern_sym_system`
// assigns, over a file with a zero term (dropped), a fallback coefficient, a re-spelled
// repeat and an equation whose coefficient all vanish.  Anything less and a pool read
// during the scan would not be the pool the learn and the evaluator expect.
TEST(spill_interned_matches_the_materialising_reader_term_for_term) {
  const std::string p = write_spill("symintern",
      term_line({1, 1}, "3/2*d+1/2") + term_line({0, 1}, "(d-2)/(2*m^2)") +
          term_line({2, 2}, "7") + "\n" +
      term_line({1, 0}, "d") + term_line({0, 0}, "3/2*d+1/2") + "\n" +
      term_line({0, 0}, "(d-2)*(d+1)/(2*m^2)") + "\n" +   // outside the fast grammar
      term_line({1, 0}, "0") + "\n" +                     // a zero term leaves a row behind
      term_line({0, 0}, "1/3*s-2/5") + term_line({1, 0}, "(d-2)/(2*m^2)") + "\n" + "\n");
  const auto ids = ids_of({{0, 0}, {1, 0}, {0, 1}, {1, 1}});
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d"), GiNaC::possymbol("s"),
                                                 GiNaC::possymbol("t"), GiNaC::possymbol("m")};

  std::vector<secred::SymEq> system;
  secred::SpillReadStats materialising;
  std::string error;
  REQUIRE(secred::read_spill_symbolic(p, ids, symbols, system, materialising, error));
  secred::SymCoeffPool reference;
  secred::intern_sym_system(system, reference);

  secred::SymCoeffPool pool;
  secred::SpillReadStats interned;
  REQUIRE(secred::read_spill_interned(p, ids, symbols, pool, interned, error));

  REQUIRE(pool.terms == reference.terms);
  REQUIRE(pool.eq_begin == reference.eq_begin);
  REQUIRE_EQ(pool.coeffs.size(), reference.coeffs.size());
  REQUIRE_EQ(interned.equations, materialising.equations);
  REQUIRE_EQ(interned.terms, materialising.terms);
  REQUIRE_EQ(interned.fallbacks, materialising.fallbacks);
  REQUIRE_EQ(interned.skipped_unknown, materialising.skipped_unknown);
  // The repeated `3/2*d+1/2` and `(d-2)/(2*m^2)` intern to the slots their first
  // occurrences took, which is what makes the whole system cost 8 B per term.
  REQUIRE_EQ(pool.n_equations(), std::size_t(5));
  REQUIRE_EQ(pool.n_terms(), std::size_t(7));
  REQUIRE_EQ(pool.coeffs.size(), std::size_t(5));   // seven entries, five functions
  REQUIRE_EQ(pool.terms[0].second, pool.terms[3].second);
  REQUIRE_EQ(pool.terms[1].second, pool.terms[5].second);
}

// The interned reader errors out on the same input the materialising one does.
TEST(spill_interned_reports_an_unparsable_coefficient) {
  const std::string p = write_spill("syminternbad", term_line({1, 1}, "d +") + "\n");
  const auto ids = ids_of({{0, 0}, {1, 1}});
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d")};
  secred::SymCoeffPool pool;
  secred::SpillReadStats stats;
  std::string error;
  REQUIRE(!secred::read_spill_interned(p, ids, symbols, pool, stats, error));
  REQUIRE(error.find("cannot parse coefficient") != std::string::npos);
}

// --- the binary spill (Task 5) --------------------------------------------------

namespace {

// The same system through the production writer, so what the reader is tested against
// is the file a run actually leaves behind.
std::string write_binary_spill(const std::string &name,
                               const std::vector<GlobalEquation> &eqs) {
  const std::string path = "/tmp/secred_test_spill_" + name + ".bin";
  ReductionContext ctx;
  ctx.rawEquationsPath = path;
  ctx.openSpill();
  ctx.appendRawEquations(eqs);
  std::vector<RawIntegral> integrals;
  ctx.finishSpill(integrals);
  return path;
}

// The same system in the text format, which the reader still reads: what
// `appendRawEquation` wrote before the binary frame.  The two files must read back
// term for term -- that is the whole claim of the format change.
std::string write_text_spill(const std::string &name,
                             const std::vector<GlobalEquation> &eqs) {
  std::string text;
  for (const GlobalEquation &eq : eqs) {
    for (const auto &[integ, coeff] : eq) {
      std::vector<int> v(integ.size());
      for (unsigned i = 0; i < integ.size(); ++i)
        v[i] = integ[i];
      text += term_line(v, coeff);
    }
    text += "\n";
  }
  return write_spill(name, text);
}

// A rational system with every branch of the numeric reader in it: an integral the id
// map knows and two it does not (one of them an equation's only term, which leaves no
// equation behind), an unreduced fraction, a zero coefficient -- which this reader
// keeps, as the text one does -- and a coefficient repeated across equations, which
// the binary tables intern.
std::vector<GlobalEquation> numeric_fixture() {
  std::vector<GlobalEquation> eqs(4);
  eqs[0].emplace_back(RawIntegral(std::vector<int>{1, 1}), "3");
  eqs[0].emplace_back(RawIntegral(std::vector<int>{0, 1}), "1/2");
  eqs[0].emplace_back(RawIntegral(std::vector<int>{2, 2}), "7");
  eqs[1].emplace_back(RawIntegral(std::vector<int>{7, 7}), "-2");  // unknown
  eqs[1].emplace_back(RawIntegral(std::vector<int>{1, 0}), "4/6");
  eqs[2].emplace_back(RawIntegral(std::vector<int>{0, 0}), "0");
  eqs[2].emplace_back(RawIntegral(std::vector<int>{0, 1}), "1/2");
  eqs[3].emplace_back(RawIntegral(std::vector<int>{5, 5}), "9");  // unknown, alone
  return eqs;
}

// A symbolic system: the strict grammar's spellings, a coefficient the grammar
// refuses (the GiNaC fallback, which the binary reader parses once per table entry),
// a re-spelled repeat that must intern to the first spelling's slot, and a zero.
std::vector<GlobalEquation> symbolic_fixture() {
  std::vector<GlobalEquation> eqs(5);
  eqs[0].emplace_back(RawIntegral(std::vector<int>{1, 1}), "3/2*d+1/2");
  eqs[0].emplace_back(RawIntegral(std::vector<int>{0, 1}), "(d-2)/(2*m^2)");
  eqs[0].emplace_back(RawIntegral(std::vector<int>{2, 2}), "7");
  eqs[1].emplace_back(RawIntegral(std::vector<int>{1, 0}), "d");
  eqs[1].emplace_back(RawIntegral(std::vector<int>{0, 0}), "3/2*d+1/2");
  eqs[2].emplace_back(RawIntegral(std::vector<int>{0, 0}), "(d-2)*(d+1)/(2*m^2)");
  eqs[3].emplace_back(RawIntegral(std::vector<int>{1, 0}), "0");  // a zero term
  eqs[4].emplace_back(RawIntegral(std::vector<int>{0, 0}), "1/3*s-2/5");
  eqs[4].emplace_back(RawIntegral(std::vector<int>{1, 0}), "(d-2)/(2*m^2)");
  return eqs;
}

}  // namespace

TEST(spill_binary_numeric_matches_the_text_reader) {
  const std::vector<GlobalEquation> eqs = numeric_fixture();
  const auto ids = ids_of({{0, 0}, {1, 0}, {0, 1}, {1, 1}, {2, 2}});
  const std::string txt = write_text_spill("numdiff", eqs);
  const std::string bin = write_binary_spill("numdiff", eqs);

  std::vector<Eq> theirs;
  SpillReadStats tstats;
  std::string error;
  REQUIRE(secred::read_spill_numeric(txt, ids, theirs, tstats, error));
  std::vector<Eq> ours;
  SpillReadStats ostats;
  REQUIRE(secred::read_spill_numeric(bin, ids, ours, ostats, error));

  REQUIRE_EQ(ours.size(), theirs.size());
  for (std::size_t i = 0; i < ours.size(); ++i) {
    REQUIRE_EQ(ours[i].size(), theirs[i].size());
    for (std::size_t j = 0; j < ours[i].size(); ++j) {
      REQUIRE_EQ(ours[i][j].first, theirs[i][j].first);
      REQUIRE_EQ(ours[i][j].second.num, theirs[i][j].second.num);
      REQUIRE_EQ(ours[i][j].second.den, theirs[i][j].second.den);
    }
  }
  REQUIRE_EQ(ostats.equations, tstats.equations);
  REQUIRE_EQ(ostats.terms, tstats.terms);
  REQUIRE_EQ(ostats.skipped_unknown, tstats.skipped_unknown);
  REQUIRE_EQ(ostats.skipped_unknown, std::uint64_t(2));
  // `4/6` and `1/2` stay unreduced in both: the readers normalise only the sign of the
  // denominator, and the solver reduces mod p later.
  REQUIRE_EQ(ours[1][0].second.den, int64_t(6));
  // The zero coefficient of the third equation is kept, after the term with the higher
  // id -- this reader drops it in the solver (`reduce_mod`), not on the way in.
  REQUIRE_EQ(ours[2].size(), std::size_t(2));
  REQUIRE_EQ(ours[2][0].second.num, int64_t(1));
  REQUIRE_EQ(ours[2][0].second.den, int64_t(2));
  REQUIRE_EQ(ours[2][1].second.num, int64_t(0));
}

TEST(spill_binary_symbolic_matches_the_text_reader) {
  const std::vector<GlobalEquation> eqs = symbolic_fixture();
  const auto ids = ids_of({{0, 0}, {1, 0}, {0, 1}, {1, 1}, {2, 2}});
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d"), GiNaC::possymbol("s"),
                                                 GiNaC::possymbol("t"), GiNaC::possymbol("m")};
  const std::string txt = write_text_spill("symdiff", eqs);
  const std::string bin = write_binary_spill("symdiff", eqs);

  std::vector<secred::SymEq> theirs;
  SpillReadStats tstats;
  std::string error;
  REQUIRE(secred::read_spill_symbolic(txt, ids, symbols, theirs, tstats, error));
  std::vector<secred::SymEq> ours;
  SpillReadStats ostats;
  REQUIRE(secred::read_spill_symbolic(bin, ids, symbols, ours, ostats, error));

  REQUIRE_EQ(ours.size(), theirs.size());
  for (std::size_t i = 0; i < ours.size(); ++i) {
    REQUIRE_EQ(ours[i].size(), theirs[i].size());
    for (std::size_t j = 0; j < ours[i].size(); ++j) {
      REQUIRE_EQ(ours[i][j].first, theirs[i][j].first);
      REQUIRE(::sym_poly_equal(ours[i][j].second.num, theirs[i][j].second.num));
      REQUIRE(::sym_poly_equal(ours[i][j].second.den, theirs[i][j].second.den));
    }
  }
  REQUIRE_EQ(ostats.equations, tstats.equations);
  REQUIRE_EQ(ostats.terms, tstats.terms);
  REQUIRE_EQ(ostats.skipped_unknown, tstats.skipped_unknown);
  // The one coefficient outside the strict grammar, written once in this fixture, so
  // the per-entry binary count and the per-occurrence text count agree.
  REQUIRE_EQ(ostats.fallbacks, std::uint64_t(1));
  REQUIRE_EQ(tstats.fallbacks, std::uint64_t(1));
}

// The production reader against the one the binary file is meant to reproduce: the
// pool's rows, boundaries and slots must be the ones a text read followed by
// `intern_sym_system` assigns, which is what the learn and the evaluator expect.
TEST(spill_binary_interned_matches_the_text_reader) {
  const std::vector<GlobalEquation> eqs = symbolic_fixture();
  const auto ids = ids_of({{0, 0}, {1, 0}, {0, 1}, {1, 1}, {2, 2}});
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d"), GiNaC::possymbol("s"),
                                                 GiNaC::possymbol("t"), GiNaC::possymbol("m")};
  const std::string txt = write_text_spill("symintern", eqs);
  const std::string bin = write_binary_spill("symintern", eqs);

  secred::SymCoeffPool reference;
  secred::SpillReadStats tstats;
  std::string error;
  REQUIRE(secred::read_spill_interned(txt, ids, symbols, reference, tstats, error));

  secred::SymCoeffPool pool;
  secred::SpillReadStats ostats;
  REQUIRE(secred::read_spill_interned(bin, ids, symbols, pool, ostats, error));

  REQUIRE(pool.terms == reference.terms);
  REQUIRE(pool.eq_begin == reference.eq_begin);
  REQUIRE_EQ(pool.coeffs.size(), reference.coeffs.size());
  REQUIRE_EQ(ostats.equations, tstats.equations);
  REQUIRE_EQ(ostats.terms, tstats.terms);
  REQUIRE_EQ(ostats.skipped_unknown, tstats.skipped_unknown);
  REQUIRE_EQ(pool.n_equations(), std::size_t(5));
  // Eight terms over six functions: `3/2*d+1/2` and `(d-2)/(2*m^2)` each occur twice and
  // intern to their first slot, in the binary read as in the text one, and the equation
  // whose only coefficient is zero still has its row -- empty.
  REQUIRE_EQ(pool.n_terms(), std::size_t(8));
  REQUIRE_EQ(pool.coeffs.size(), std::size_t(6));
  REQUIRE_EQ(pool.eq_begin.size(), std::size_t(6));
  REQUIRE_EQ(pool.eq_begin[3], pool.eq_begin[4]);
  REQUIRE_EQ(pool.terms[1].second, pool.terms[4].second);
  REQUIRE_EQ(pool.terms[2].second, pool.terms[6].second);
}

// A coefficient the grammar refuses and GiNaC cannot read is a hard error, as it is
// for the text reader -- the table is parsed through the same function.
TEST(spill_binary_unparsable_coefficient_is_an_error) {
  std::vector<GlobalEquation> eqs(1);
  eqs[0].emplace_back(RawIntegral(std::vector<int>{1, 1}), "d +");
  const std::string bin = write_binary_spill("symbad", eqs);
  const auto ids = ids_of({{0, 0}, {1, 1}});
  const std::vector<GiNaC::possymbol> symbols = {GiNaC::possymbol("d")};
  secred::SymCoeffPool pool;
  SpillReadStats stats;
  std::string error;
  REQUIRE(!secred::read_spill_interned(bin, ids, symbols, pool, stats, error));
  REQUIRE(error.find("cannot parse coefficient") != std::string::npos);
}

// The binary format carries the same spelling in its coefficient table, so a big
// coefficient reads back exactly as the text reader's does: same tag, same text, same
// residue.
TEST(spill_binary_numeric_reads_a_coefficient_beyond_int64) {
  std::vector<GlobalEquation> eqs(1);
  eqs[0].emplace_back(RawIntegral(std::vector<int>{1, 1}), "18446744073709551616");
  const std::string bin = write_binary_spill("numbig", eqs);
  const auto ids = ids_of({{1, 1}});
  std::vector<Eq> system;
  SpillReadStats stats;
  std::string error;
  REQUIRE(secred::read_spill_numeric(bin, ids, system, stats, error));
  REQUIRE_EQ(system.size(), std::size_t(1));
  REQUIRE_EQ(system[0].size(), std::size_t(1));
  const secred::Rat64 &c = system[0][0].second;
  REQUIRE_EQ(c.den, int64_t(0));
  REQUIRE(secred::rat_text(c) == "18446744073709551616");
  REQUIRE_EQ(secred::rat_mod_u64(c.num, c.den, 9223372036854775783ull),
             std::uint64_t(50));
}

// A file that starts as a binary spill and stops before its footer is a hard error:
// the term stream cannot be read without the tables the footer points at, and a run
// that died half-way must not look like a short system.
TEST(spill_binary_truncated_is_an_error) {
  const std::string bin = write_binary_spill("truncated", numeric_fixture());
  std::filesystem::resize_file(bin, 16);  // header and half an equation
  const auto ids = ids_of({{0, 0}, {1, 0}});
  std::vector<Eq> system;
  SpillReadStats stats;
  std::string error;
  REQUIRE(!secred::read_spill_numeric(bin, ids, system, stats, error));
  REQUIRE(error.find("truncated") != std::string::npos);
}

// The magic is checked, not assumed: a later version's file is refused rather than
// read as this one.
TEST(spill_binary_version_is_checked) {
  std::string bytes = "SRSP";
  bytes += '\x02';  // version 2
  bytes += std::string(4, '\0');
  const std::string p = write_spill("badversion", bytes);
  const auto ids = ids_of({{0, 0}});
  std::vector<Eq> system;
  SpillReadStats stats;
  std::string error;
  REQUIRE(!secred::read_spill_numeric(p, ids, system, stats, error));
  REQUIRE(error.find("version") != std::string::npos);
}

// An empty file is what a run that spilled nothing leaves behind (the stream is opened
// and truncated, and `Family` removes it only after the sector phase).  It is not a
// binary spill and scans as the empty text system, exactly as it did before.
TEST(spill_binary_empty_file_scans_as_text) {
  const std::string p = write_spill("emptybinary", "");
  const auto ids = ids_of({{0, 0}});
  std::vector<Eq> system;
  SpillReadStats stats;
  std::string error;
  REQUIRE(secred::read_spill_numeric(p, ids, system, stats, error));
  REQUIRE_EQ(system.size(), std::size_t(0));
  REQUIRE_EQ(stats.equations, std::uint64_t(0));
}
