#include "test_runner.h"

#include "family.h"
#include "symmetry.h"

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Families for the symmetry tests.  They reproduce, in SecRed's own YAML, the
// kinematic configuration of the archived Kira goldens
// (bench/symmetry/native/<fam>/config/*.yaml): same propagator order (so line
// numbers and sector ids match), symbolic dimension d, and the mass at
// dimension 2 in Kira (the invariant `m2`) against dimension 1 here (SecRed's
// `m`), i.e. m2 = m^2 -- the only renaming the goldens' coefficient strings need.
//
// The families are deliberately NOT test_family.cpp's box_config: that one
// fixes invariants with invar_one and massless scalar products, which is a
// different kinematic point from the goldens'.
// ---------------------------------------------------------------------------

static YAML::Node symmetry_box_config(const std::string &cache_dir) {
  std::string yaml = R"(
settings:
  cache_dir: )" + cache_dir + R"(
  threads: 1

family:
  name: box
  dimension: d
  internals: [l1]
  externals: [p1, p2, p3]
  invariants:
    - [s, 2]
    - [t, 2]
    - [m, 1]
  sps_rules:
    - [p1, p1, m^2]
    - [p2, p2, m^2]
    - [p3, p3, m^2]
    - [p1, p2, t-m^2]
    - [p1, p3, s-m^2]
    - [p2, p3, -t-s-m^2]
  propagators:
    - [l1, m]
    - [l1+p1, 0]
    - [l1+p1+p2, m]
    - [l1+p1+p2+p3, 0]
)";
  return YAML::Load(yaml);
}

static Family make_box(const std::string &tag) {
  Family::symtab.clear();
  Family f(symmetry_box_config("/tmp/secred_tests/" + tag));
  f.init();
  return f;
}

// Kira's coefficient strings spell m^2 as m2; the goldens are compared as
// expressions, so the rewrite bench/symmetry/native/README.md documents is
// applied before parsing.
static GiNaC::ex kira_expr(const std::string &s) {
  GiNaC::parser p(Family::symtab);
  return p(kira_to_ginac(s));
}

static void require_row(const PropPrescription &p, unsigned k,
                        const std::vector<std::string> &expected) {
  for (unsigned j = 0; j + 1 < expected.size(); ++j)
    REQUIRE((p.f(k, j) - kira_expr(expected[j])).expand().is_zero());
  REQUIRE((p.c(k, 0) - kira_expr(expected.back())).expand().is_zero());
}

// ---------------------------------------------------------------------------
// verify() against Kira's own correspondences, read off the golden's canonica
// permutation lists (ing[p0[g]] = p1[g]; see spec 4.8).
// ---------------------------------------------------------------------------

TEST(symmetry_verify_box_4_to_1) {
  Family f = make_box("sym_box_41");
  const SymmetryTable::Inputs in = f.symmetry_inputs();
  // canonica: sector 4 -> "2" (line 2), sector 1 -> "0" (line 0)
  auto p = SymmetryTable::verify(4, 1, {-1, -1, 0, -1}, in);
  REQUIRE(p.has_value());
  REQUIRE(!p->dot_only);
  REQUIRE_EQ(p->from, 4u);
  REQUIRE_EQ(p->to, 1u);
  // golden relations, first block
  require_row(*p, 0, {"2", "0", "-1", "0", "4*t"});
  require_row(*p, 1, {"1", "1", "-1", "0", "2*t"});
  require_row(*p, 2, {"1", "0", "0", "0", "0"});
  require_row(*p, 3, {"1", "0", "-1", "1", "4*m2+2*t"});
}

TEST(symmetry_verify_box_5_to_5) {
  Family f = make_box("sym_box_55");
  const SymmetryTable::Inputs in = f.symmetry_inputs();
  // canonica: sector 5 -> "0 2" then "2 0", so ing[0] = 2, ing[2] = 0
  auto p = SymmetryTable::verify(5, 5, {2, -1, 0, -1}, in);
  REQUIRE(p.has_value());
  REQUIRE(!p->dot_only);
  require_row(*p, 0, {"0", "0", "1", "0", "0"});
  require_row(*p, 1, {"1", "-1", "1", "0", "4*m2-2*t"});
  require_row(*p, 2, {"1", "0", "0", "0", "0"});
  require_row(*p, 3, {"1", "0", "1", "-1", "-2*t"});
}

TEST(symmetry_verify_box_10_to_10) {
  Family f = make_box("sym_box_1010");
  const SymmetryTable::Inputs in = f.symmetry_inputs();
  // canonica: sector 10 -> "1 3" then "3 1"
  auto p = SymmetryTable::verify(10, 10, {-1, 3, -1, 1}, in);
  REQUIRE(p.has_value());
  REQUIRE(!p->dot_only);
  require_row(*p, 0, {"-1", "1", "0", "1", "-4*m2+2*t+2*s"});
  require_row(*p, 1, {"0", "0", "0", "1", "0"});
  require_row(*p, 2, {"0", "1", "-1", "1", "2*t+2*s"});
  require_row(*p, 3, {"0", "1", "0", "0", "0"});
}

TEST(symmetry_verify_rejects_box_dot_only_entries) {
  Family f = make_box("sym_box_dots");
  const SymmetryTable::Inputs in = f.symmetry_inputs();
  // 6 -> 3: ing from 6's "2 1" and 3's "0 1"
  REQUIRE(!SymmetryTable::verify(6, 3, {-1, 1, 0, -1}, in).has_value());
  // 12 -> 3: ing from 12's "2 3"
  REQUIRE(!SymmetryTable::verify(12, 3, {-1, -1, 0, 1}, in).has_value());
  // 7 -> 7: ing from 7's "0 2 1" and "2 0 1"
  REQUIRE(!SymmetryTable::verify(7, 7, {2, 1, 0, -1}, in).has_value());
  // 14 -> 14: ing from 14's "2 1 3" and "2 3 1"
  REQUIRE(!SymmetryTable::verify(14, 14, {-1, 3, 2, 1}, in).has_value());
}

TEST(symmetry_dot_only_rows_match_kira) {
  Family f = make_box("sym_box_faked");
  const SymmetryTable::Inputs in = f.symmetry_inputs();
  // Kira's faked rows (ReadYamlFiles.cpp:1896-1901): the unit vector at ing[k]
  // for a covered line, the unit vector at the lowest line outside `to` for an
  // uncovered one, every constant zero.
  PropPrescription p = SymmetryTable::dot_only_entry(6, 3, {-1, 1, 0, -1}, in);
  REQUIRE(p.dot_only);
  require_row(p, 0, {"0", "0", "1", "0", "0"});
  require_row(p, 1, {"0", "1", "0", "0", "0"});
  require_row(p, 2, {"1", "0", "0", "0", "0"});
  require_row(p, 3, {"0", "0", "1", "0", "0"});

  PropPrescription q = SymmetryTable::dot_only_entry(14, 14, {-1, 3, 2, 1}, in);
  require_row(q, 0, {"1", "0", "0", "0", "0"});  // first hole of sector 14: line 0
  require_row(q, 1, {"0", "0", "0", "1", "0"});
  require_row(q, 2, {"0", "0", "1", "0", "0"});
  require_row(q, 3, {"0", "1", "0", "0", "0"});
}

// rows_ok is the golden-comparison helper the later tasks use; it must accept a
// row that agrees with Kira's string only after the m2 rewrite and GiNaC
// canonicalisation (different factor order, m^2 folded).
TEST(symmetry_rows_ok_compares_as_expressions) {
  Family f = make_box("sym_box_rowsok");
  const SymmetryTable::Inputs in = f.symmetry_inputs();
  auto p = SymmetryTable::verify(4, 1, {-1, -1, 0, -1}, in);
  REQUIRE(p.has_value());
  REQUIRE(p->rows_ok({{"2", "0", "-1", "0", "4*t"},
                      {"1", "1", "-1", "0", "2*t"},
                      {"1", "0", "0", "0", "0"},
                      {"1", "0", "-1", "1", "4*m2+2*t"}},
                     4, Family::symtab));
  // a wrong number of rows, a wrong number of columns, and a wrong value all fail
  REQUIRE(!p->rows_ok({{"2", "0", "-1", "0", "4*t"}}, 4, Family::symtab));
  REQUIRE(!p->rows_ok({{"2", "0", "-1", "0"},
                       {"1", "1", "-1", "0"},
                       {"1", "0", "0", "0"},
                       {"1", "0", "-1", "1"}},
                      4, Family::symtab));
  REQUIRE(!p->rows_ok({{"2", "0", "-1", "0", "5*t"},
                       {"1", "1", "-1", "0", "2*t"},
                       {"1", "0", "0", "0", "0"},
                       {"1", "0", "-1", "1", "4*m2+2*t"}},
                      4, Family::symtab));
}

TEST(symmetry_numeric_conversion_round_trips) {
  // The numeric path (ibps_sym == nullptr) needs the same entries as exact
  // rationals; a numeric family is one whose _symbols is empty.
  Family f = make_box("sym_box_numeric");
  const SymmetryTable::Inputs in = f.symmetry_inputs();
  REQUIRE(!in.numeric);  // s, t, m and d are all symbols here
  auto p = SymmetryTable::verify(4, 1, {-1, -1, 0, -1}, in);
  REQUIRE(p.has_value());
  REQUIRE(p->f_num.empty());  // not a numeric family: nothing to convert
}
