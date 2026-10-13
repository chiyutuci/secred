// The scanner over the global stage's spill file: boundaries, the strict parsers and
// their fallbacks, and (Task 2-3) the symbolic grammar.
#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "reference_spill_reader.h"
#include "spill_reader.h"
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

TEST(spill_numeric_overflow_is_a_hard_failure) {
  const std::string p = write_spill("overflow",
      term_line({1, 1}, "99999999999999999999") + "\n");
  const auto ids = ids_of({{0, 0}, {1, 1}});
  std::vector<Eq> system;
  SpillReadStats stats;
  std::string error;
  REQUIRE(!secred::read_spill_numeric(p, ids, system, stats, error));
  REQUIRE(error.find("does not fit in int64") != std::string::npos);
}

// A 20-digit coefficient whose uint64 wrap-around lands back *inside* int64: a digit
// loop that accumulates first and checks after would accept the wrapped value silently.
// `std::stoll` throws on it, so both readers must treat it as the hard failure it is.
TEST(spill_numeric_wrapping_coefficient_is_a_hard_failure) {
  const std::string p = write_spill("wrap",
      term_line({1, 1}, "80000000000000000009") + "\n");
  const auto ids = ids_of({{0, 0}, {1, 1}});
  std::vector<Eq> system;
  SpillReadStats stats;
  std::string error;
  REQUIRE(!secred::read_spill_numeric(p, ids, system, stats, error));
  REQUIRE(error.find("does not fit in int64") != std::string::npos);
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

TEST(scan_spill_dum_picks_the_smallest_leading_id) {
  const std::string p = write_spill("dum",
      term_line({1, 1}, "3") + "\n" +
      term_line({0, 1}, "1") + term_line({1, 0}, "2") + "\n" +
      term_line({0, 0}, "5") + "\n");
  const auto ids = ids_of({{0, 0}, {1, 0}, {0, 1}, {1, 1}});
  REQUIRE_EQ(secred::scan_spill_dum(p, ids), 2u);   // its largest id is 0
}

// The dum scan does *not* flush a final unterminated equation (sector.cpp:953-986 did
// not either), so a truncated tail cannot change the pick.
TEST(scan_spill_dum_ignores_a_final_unterminated_equation) {
  const std::string p = write_spill("dumtail",
      term_line({1, 1}, "3") + "\n" + term_line({0, 0}, "5"));
  const auto ids = ids_of({{0, 0}, {1, 1}});
  REQUIRE_EQ(secred::scan_spill_dum(p, ids), 0u);
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
// term -- ids in order and exact num/den -- because that is the whole claim of B2.
TEST(spill_numeric_matches_the_reference_reader) {
  const std::string p = write_spill("refdiff",
      term_line({1, 1}, "3") + term_line({0, 1}, "1/2") + term_line({2, 2}, "7") + "\n" +
      term_line({1, 0}, "-2") + "\n" +
      term_line({0, 0}, "4/6") + term_line({0, 1}, "0000000000000000001") + "\n" +
      "not a term line\n" + "\n" +
      term_line({1, 1}, "99999999999999999999") + "\n");   // overflows int64
  const auto ids = ids_of({{0, 0}, {1, 0}, {0, 1}, {1, 1}});

  std::vector<Eq> theirs;
  std::uint64_t their_terms = 0;
  REQUIRE(!ref::legacy_numeric(p, ids, theirs, their_terms));   // stops at the overflow

  std::vector<Eq> ours;
  SpillReadStats stats;
  std::string error;
  REQUIRE(!secred::read_spill_numeric(p, ids, ours, stats, error));

  REQUIRE_EQ(ours.size(), theirs.size());
  for (std::size_t i = 0; i < ours.size(); ++i) {
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
}
